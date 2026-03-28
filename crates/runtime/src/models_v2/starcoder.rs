//! StarCoder Model V2 - Clean implementation
//!
//! StarCoder architecture features:
//! - Multi-Query Attention (MQA) - single KV head
//! - GPT-2 style architecture with learned position embeddings
//! - Fill-in-the-Middle (FIM) token support
//! - Larger context windows

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

/// StarCoder model configuration
model_config!(StarCoderConfig {
    vocab_size: usize = 49152,
    hidden_size: usize = 6144,
    intermediate_size: usize = 24576,
    num_hidden_layers: usize = 40,
    num_attention_heads: usize = 48,
    num_key_value_heads: usize = 1,  // MQA: single KV head
    hidden_act: String = "gelu_new".to_string(),
    max_position_embeddings: usize = 8192,
    initializer_range: f32 = 0.02,
    layer_norm_eps: f32 = 1e-5,
    use_cache: bool = true,
    pad_token_id: i64 = 49152,
    bos_token_id: i64 = 49152,
    eos_token_id: i64 = 0,
    tie_word_embeddings: bool = true,
});

impl StarCoderConfig {
    pub fn from_gguf_config(gguf: &crate::weight_loader_core::GGUFModelConfig) -> Self {
        Self {
            vocab_size: gguf.vocab_size,
            hidden_size: gguf.hidden_size,
            intermediate_size: gguf.intermediate_size,
            num_hidden_layers: gguf.num_hidden_layers,
            num_attention_heads: gguf.num_attention_heads,
            num_key_value_heads: gguf.num_key_value_heads,
            max_position_embeddings: gguf.max_position_embeddings,
            ..Default::default()
        }
    }
}

pub struct StarCoderModelV2 {
    config: StarCoderConfig,
    device: Device,
    wte: Tensor,
    wpe: Tensor,
    layers: Vec<StarCoderLayer>,
    ln_f: Tensor,
    lm_head: Tensor,
}

pub struct StarCoderLayer {
    attn: StarCoderAttention,
    mlp: StarCoderMLP,
    ln_1: Tensor,
    ln_2: Tensor,
}

pub struct StarCoderAttention {
    c_attn: Tensor,  // Combined QKV projection
    c_proj: Tensor,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

pub struct StarCoderMLP {
    c_fc: Tensor,
    c_proj: Tensor,
}

impl Model for StarCoderModelV2 {
    type Config = StarCoderConfig;

    fn new(config: StarCoderConfig) -> Result<Self> {
        let device = Device::CPU;
        let wte = ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, &device)?;
        let wpe = ops_fn::zeros(&[config.max_position_embeddings, config.hidden_size], DataType::Float32, &device)?;
        let ln_f = ops_fn::zeros(&[config.hidden_size], DataType::Float32, &device)?;
        let lm_head = wte.clone();

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(StarCoderLayer::new(&config, &device)?);
        }

        Ok(Self { config, device, wte, wpe, layers, ln_f, lm_head })
    }

    fn from_weights(config: StarCoderConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;
        if let Some(w) = weights.get("transformer.wte.weight") { model.wte = w.clone(); }
        if let Some(w) = weights.get("transformer.wpe.weight") { model.wpe = w.clone(); }
        if let Some(w) = weights.get("transformer.ln_f.weight") { model.ln_f = w.clone(); }
        if model.config.tie_word_embeddings {
            model.lm_head = model.wte.clone();
        } else if let Some(w) = weights.get("lm_head.weight") {
            model.lm_head = w.clone();
        }
        for (i, layer) in model.layers.iter_mut().enumerate() { layer.load_weights(&weights, i)?; }
        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        match inputs {
            ModelInputs::Text { input_ids, .. } => {
                let shape = input_ids.shape();
                let seq_len = shape[1];

                let mut hidden = ops_fn::embedding(input_ids, &self.wte)?;
                let positions: Vec<i64> = (0..seq_len as i64).collect();
                let pos_tensor = Tensor::from_i64_slice(&positions, &[1, seq_len], &self.device)?;
                let pos_embeds = ops_fn::embedding(&pos_tensor, &self.wpe)?;
                hidden = ops_fn::add(&hidden, &pos_embeds)?;

                for layer in &self.layers {
                    hidden = layer.forward(&hidden)?;
                }

                hidden = ops_fn::layer_norm(&hidden, &self.ln_f, None, self.config.layer_norm_eps)?;

                // Tied embeddings - flatten to 2D, matmul, reshape back
                let lm_head_candle = self.lm_head.to_candle()?;
                let hidden_candle = hidden.to_candle()?.contiguous()?;
                let batch = hidden_candle.dims()[0];
                let seq = hidden_candle.dims()[1];
                let hidden_size = hidden_candle.dims()[2];
                let flat = hidden_candle.reshape(&[batch * seq, hidden_size])?;
                let logits_flat = flat.matmul(&lm_head_candle.t()?)?;
                let logits_candle = logits_flat.reshape(&[batch, seq, self.config.vocab_size])?;
                let logits = Tensor::from_candle(logits_candle);

                Ok(ModelOutputs::Logits { logits, hidden_states: None })
            }
            _ => Err(anyhow::anyhow!("StarCoder only supports text inputs")),
        }
    }

    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        use crate::tokenizer::Tokenizer;
        use rand::Rng;
        let tokenizer = Tokenizer::new();
        let mut tokens: Vec<u32> = tokenizer.encode(prompt);
        for _ in 0..config.max_new_tokens {
            let tokens_i64: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
            let input = Tensor::from_i64_slice(&tokens_i64, &[1, tokens.len()], &self.device)?;
            let outputs = self.forward(&ModelInputs::text(input))?;
            let logits = match outputs { ModelOutputs::Logits { logits, .. } => logits, _ => return Err(anyhow::anyhow!("Expected logits")) };
            let logits_candle = logits.to_candle()?;
            let last = logits_candle.narrow(1, logits_candle.dims()[1] - 1, 1)?.squeeze(1)?.squeeze(0)?;
            let logits_vec: Vec<f32> = last.to_vec1()?;
            let next = if config.do_sample && config.temperature > 0.0 {
                let scaled: Vec<f32> = logits_vec.iter().map(|&x| x / config.temperature).collect();
                let max_v = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scaled.iter().map(|&x| (x - max_v).exp()).sum();
                let probs: Vec<f32> = scaled.iter().map(|&x| (x - max_v).exp() / exp_sum).collect();
                let mut rng = rand::thread_rng();
                let r: f32 = rng.gen();
                let mut cum = 0.0;
                let mut s = 0u32;
                for (i, &p) in probs.iter().enumerate() { cum += p; if r <= cum { s = i as u32; break; } }
                s
            } else {
                logits_vec.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as u32).unwrap_or(0)
            };
            if next == config.eos_token_id { break; }
            tokens.push(next);
        }
        Ok(tokenizer.decode(&tokens))
    }

    fn config(&self) -> &Self::Config { &self.config }
    fn memory_requirements(&self) -> MemoryRequirements {
        let p = self.config.vocab_size * self.config.hidden_size + self.config.num_hidden_layers * 8 * self.config.hidden_size.pow(2);
        MemoryRequirements { gpu_memory: p * 4, cpu_memory: p, kv_cache_memory: 2 * self.config.num_hidden_layers * self.config.max_position_embeddings * self.config.hidden_size * 4, peak_memory: p * 5 }
    }
    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.wte = self.wte.to_device(device)?;
        self.wpe = self.wpe.to_device(device)?;
        self.ln_f = self.ln_f.to_device(device)?;
        self.lm_head = self.lm_head.to_device(device)?;
        for layer in &mut self.layers { layer.to_device(device)?; }
        self.device = device.clone();
        Ok(())
    }
}

impl StarCoderLayer {
    fn new(config: &StarCoderConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            attn: StarCoderAttention::new(config, device)?,
            mlp: StarCoderMLP::new(config, device)?,
            ln_1: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            ln_2: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let h = ops_fn::layer_norm(hidden_states, &self.ln_1, None, 1e-5)?;
        let attn_out = self.attn.forward(&h)?;
        let h = ops_fn::add(&residual, &attn_out)?;

        let residual = h.clone();
        let h = ops_fn::layer_norm(&h, &self.ln_2, None, 1e-5)?;
        let mlp_out = self.mlp.forward(&h)?;
        ops_fn::add(&residual, &mlp_out)
    }

    fn load_weights(&mut self, weights: &ModelWeights, idx: usize) -> Result<()> {
        let p = format!("transformer.h.{}", idx);
        if let Some(w) = weights.get(&format!("{}.attn.c_attn.weight", p)) { self.attn.c_attn = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.attn.c_proj.weight", p)) { self.attn.c_proj = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.mlp.c_fc.weight", p)) { self.mlp.c_fc = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.mlp.c_proj.weight", p)) { self.mlp.c_proj = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.ln_1.weight", p)) { self.ln_1 = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.ln_2.weight", p)) { self.ln_2 = w.clone(); }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.attn.to_device(device)?;
        self.mlp.to_device(device)?;
        self.ln_1 = self.ln_1.to_device(device)?;
        self.ln_2 = self.ln_2.to_device(device)?;
        Ok(())
    }
}

impl StarCoderAttention {
    fn new(config: &StarCoderConfig, device: &Device) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        // MQA: Q has num_heads, K/V have 1 head each
        let qkv_size = config.hidden_size + 2 * head_dim;  // Q (all heads) + K (1 head) + V (1 head)
        Ok(Self {
            c_attn: ops_fn::zeros(&[config.hidden_size, qkv_size], DataType::Float32, device)?,
            c_proj: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            num_heads: config.num_attention_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch, seq, hidden_size) = (shape[0], shape[1], shape[2]);

        let qkv = ops_fn::matmul(hidden_states, &self.c_attn)?.to_candle()?;

        // Split QKV - Q has full hidden_size, K and V each have head_dim
        let q = qkv.narrow(2, 0, hidden_size)?;
        let k = qkv.narrow(2, hidden_size, self.head_dim)?;
        let v = qkv.narrow(2, hidden_size + self.head_dim, self.head_dim)?;

        let q = q.reshape(&[batch, seq, self.num_heads, self.head_dim])?.transpose(1, 2)?;
        let k = k.reshape(&[batch, seq, 1, self.head_dim])?.transpose(1, 2)?;
        let v = v.reshape(&[batch, seq, 1, self.head_dim])?.transpose(1, 2)?;

        // Broadcast K, V to all heads
        let k = k.broadcast_as(&[batch, self.num_heads, seq, self.head_dim])?.contiguous()?;
        let v = v.broadcast_as(&[batch, self.num_heads, seq, self.head_dim])?.contiguous()?;

        let q = q.contiguous()?;
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scores = (q.matmul(&k_t)? * (self.scale as f64))?;

        // Causal mask
        let device = scores.device();
        let mask = {
            let mut m = vec![0.0f32; seq * seq];
            for i in 0..seq { for j in (i+1)..seq { m[i*seq+j] = f32::NEG_INFINITY; } }
            candle_core::Tensor::from_vec(m, &[1, 1, seq, seq], device)?
        };
        let scores = scores.broadcast_add(&mask)?;

        let attn = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?;
        let out = attn.transpose(1, 2)?.reshape(&[batch, seq, hidden_size])?;
        ops_fn::matmul(&Tensor::from_candle(out), &self.c_proj)
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.c_attn = self.c_attn.to_device(device)?;
        self.c_proj = self.c_proj.to_device(device)?;
        Ok(())
    }
}

impl StarCoderMLP {
    fn new(config: &StarCoderConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            c_fc: ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?,
            c_proj: ops_fn::zeros(&[config.intermediate_size, config.hidden_size], DataType::Float32, device)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = ops_fn::matmul(x, &self.c_fc)?;
        let h = ops_fn::gelu(&h)?;
        ops_fn::matmul(&h, &self.c_proj)
    }
    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.c_fc = self.c_fc.to_device(device)?;
        self.c_proj = self.c_proj.to_device(device)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_starcoder_creation() {
        let config = StarCoderConfig { vocab_size: 1000, hidden_size: 128, intermediate_size: 512, num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 1, ..Default::default() };
        let model = StarCoderModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 1000);
    }
    #[test]
    fn test_starcoder_forward() {
        let config = StarCoderConfig { vocab_size: 100, hidden_size: 64, intermediate_size: 256, num_hidden_layers: 1, num_attention_heads: 4, num_key_value_heads: 1, ..Default::default() };
        let model = StarCoderModelV2::new(config).unwrap();
        let inputs = ModelInputs::text(ops_fn::zeros(&[2, 8], DataType::Int64, &Device::CPU).unwrap());
        match model.forward(&inputs).unwrap() { ModelOutputs::Logits { logits, .. } => assert_eq!(logits.shape(), &[2, 8, 100]), _ => panic!() }
    }
}
