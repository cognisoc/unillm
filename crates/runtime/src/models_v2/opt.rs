//! OPT Model V2 - Clean implementation
//!
//! OPT (Open Pre-trained Transformer) architecture features:
//! - Learned absolute position embeddings
//! - Standard multi-head attention
//! - Pre-norm layer normalization

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

model_config!(OPTConfig {
    vocab_size: usize = 50272,
    hidden_size: usize = 768,
    intermediate_size: usize = 3072,
    num_hidden_layers: usize = 12,
    num_attention_heads: usize = 12,
    num_key_value_heads: usize = 12,
    hidden_act: String = "relu".to_string(),
    max_position_embeddings: usize = 2048,
    initializer_range: f32 = 0.02,
    layer_norm_eps: f32 = 1e-5,
    use_cache: bool = true,
    pad_token_id: i64 = 1,
    bos_token_id: i64 = 2,
    eos_token_id: i64 = 2,
    tie_word_embeddings: bool = false,
    word_embed_proj_dim: usize = 768,
    do_layer_norm_before: bool = true,
});

impl OPTConfig {
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

pub struct OPTModelV2 {
    config: OPTConfig,
    device: Device,
    embed_tokens: Tensor,
    embed_positions: Tensor,
    project_in: Option<Tensor>,
    project_out: Option<Tensor>,
    layers: Vec<OPTLayer>,
    final_layer_norm: Tensor,
    lm_head: Tensor,
}

pub struct OPTLayer {
    self_attn: OPTAttention,
    fc1: Tensor,
    fc2: Tensor,
    self_attn_layer_norm: Tensor,
    final_layer_norm: Tensor,
    do_layer_norm_before: bool,
}

pub struct OPTAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    out_proj: Tensor,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl Model for OPTModelV2 {
    type Config = OPTConfig;

    fn new(config: OPTConfig) -> Result<Self> {
        let device = Device::CPU;
        let embed_tokens = ops_fn::zeros(&[config.vocab_size, config.word_embed_proj_dim], DataType::Float32, &device)?;
        let embed_positions = ops_fn::zeros(&[config.max_position_embeddings + 2, config.hidden_size], DataType::Float32, &device)?;

        let project_in = if config.word_embed_proj_dim != config.hidden_size {
            Some(ops_fn::zeros(&[config.word_embed_proj_dim, config.hidden_size], DataType::Float32, &device)?)
        } else { None };

        let project_out = if config.word_embed_proj_dim != config.hidden_size {
            Some(ops_fn::zeros(&[config.hidden_size, config.word_embed_proj_dim], DataType::Float32, &device)?)
        } else { None };

        let final_layer_norm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, &device)?;
        let lm_head = ops_fn::zeros(&[config.word_embed_proj_dim, config.vocab_size], DataType::Float32, &device)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(OPTLayer::new(&config, &device)?);
        }

        Ok(Self { config, device, embed_tokens, embed_positions, project_in, project_out, layers, final_layer_norm, lm_head })
    }

    fn from_weights(config: OPTConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;
        if let Some(w) = weights.get("model.decoder.embed_tokens.weight") { model.embed_tokens = w.clone(); }
        if let Some(w) = weights.get("model.decoder.embed_positions.weight") { model.embed_positions = w.clone(); }
        if let Some(w) = weights.get("model.decoder.project_in.weight") { model.project_in = Some(ops_fn::transpose(w)?); }
        if let Some(w) = weights.get("model.decoder.project_out.weight") { model.project_out = Some(ops_fn::transpose(w)?); }
        if let Some(w) = weights.get("model.decoder.final_layer_norm.weight") { model.final_layer_norm = w.clone(); }
        if let Some(w) = weights.get("lm_head.weight") { model.lm_head = ops_fn::transpose(w)?; }
        for (i, layer) in model.layers.iter_mut().enumerate() { layer.load_weights(&weights, i)?; }
        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        match inputs {
            ModelInputs::Text { input_ids, .. } => {
                let shape = input_ids.shape();
                let seq_len = shape[1];

                let mut hidden = ops_fn::embedding(input_ids, &self.embed_tokens)?;
                if let Some(ref proj) = self.project_in {
                    hidden = ops_fn::matmul(&hidden, proj)?;
                }

                let positions: Vec<i64> = (2..(seq_len + 2) as i64).collect();
                let pos_tensor = Tensor::from_i64_slice(&positions, &[1, seq_len], &self.device)?;
                let pos_embeds = ops_fn::embedding(&pos_tensor, &self.embed_positions)?;
                hidden = ops_fn::add(&hidden, &pos_embeds)?;

                for layer in &self.layers {
                    hidden = layer.forward(&hidden)?;
                }

                if self.config.do_layer_norm_before {
                    hidden = ops_fn::layer_norm(&hidden, &self.final_layer_norm, None, self.config.layer_norm_eps)?;
                }

                if let Some(ref proj) = self.project_out {
                    hidden = ops_fn::matmul(&hidden, proj)?;
                }

                let logits = ops_fn::matmul(&hidden, &self.lm_head)?;
                Ok(ModelOutputs::Logits { logits, hidden_states: None })
            }
            _ => Err(anyhow::anyhow!("OPT only supports text inputs")),
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
        self.embed_tokens = self.embed_tokens.to_device(device)?;
        self.embed_positions = self.embed_positions.to_device(device)?;
        if let Some(ref mut p) = self.project_in { *p = p.to_device(device)?; }
        if let Some(ref mut p) = self.project_out { *p = p.to_device(device)?; }
        self.final_layer_norm = self.final_layer_norm.to_device(device)?;
        self.lm_head = self.lm_head.to_device(device)?;
        for l in &mut self.layers { l.to_device(device)?; }
        self.device = device.clone();
        Ok(())
    }
}

impl OPTLayer {
    fn new(config: &OPTConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            self_attn: OPTAttention::new(config, device)?,
            fc1: ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?,
            fc2: ops_fn::zeros(&[config.intermediate_size, config.hidden_size], DataType::Float32, device)?,
            self_attn_layer_norm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            final_layer_norm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            do_layer_norm_before: config.do_layer_norm_before,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let h = if self.do_layer_norm_before {
            ops_fn::layer_norm(hidden_states, &self.self_attn_layer_norm, None, 1e-5)?
        } else { hidden_states.clone() };
        let attn_out = self.self_attn.forward(&h)?;
        let h = ops_fn::add(&residual, &attn_out)?;
        let h = if !self.do_layer_norm_before {
            ops_fn::layer_norm(&h, &self.self_attn_layer_norm, None, 1e-5)?
        } else { h };

        let residual = h.clone();
        let h = if self.do_layer_norm_before {
            ops_fn::layer_norm(&h, &self.final_layer_norm, None, 1e-5)?
        } else { h };
        let fc1_out = ops_fn::matmul(&h, &self.fc1)?;
        let activated = {
            let x = fc1_out.to_candle()?;
            Tensor::from_candle(x.relu()?)
        };
        let fc2_out = ops_fn::matmul(&activated, &self.fc2)?;
        let h = ops_fn::add(&residual, &fc2_out)?;
        if !self.do_layer_norm_before {
            ops_fn::layer_norm(&h, &self.final_layer_norm, None, 1e-5)
        } else { Ok(h) }
    }

    fn load_weights(&mut self, weights: &ModelWeights, idx: usize) -> Result<()> {
        let p = format!("model.decoder.layers.{}", idx);
        if let Some(w) = weights.get(&format!("{}.self_attn.q_proj.weight", p)) { self.self_attn.q_proj = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.self_attn.k_proj.weight", p)) { self.self_attn.k_proj = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.self_attn.v_proj.weight", p)) { self.self_attn.v_proj = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.self_attn.out_proj.weight", p)) { self.self_attn.out_proj = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.fc1.weight", p)) { self.fc1 = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.fc2.weight", p)) { self.fc2 = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.self_attn_layer_norm.weight", p)) { self.self_attn_layer_norm = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.final_layer_norm.weight", p)) { self.final_layer_norm = w.clone(); }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.self_attn.to_device(device)?;
        self.fc1 = self.fc1.to_device(device)?;
        self.fc2 = self.fc2.to_device(device)?;
        self.self_attn_layer_norm = self.self_attn_layer_norm.to_device(device)?;
        self.final_layer_norm = self.final_layer_norm.to_device(device)?;
        Ok(())
    }
}

impl OPTAttention {
    fn new(config: &OPTConfig, device: &Device) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        Ok(Self {
            q_proj: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            k_proj: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            v_proj: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            out_proj: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            num_heads: config.num_attention_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch, seq_len, _) = (shape[0], shape[1], shape[2]);

        let q = ops_fn::matmul(hidden_states, &self.q_proj)?.to_candle()?.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?.transpose(1, 2)?;
        let k = ops_fn::matmul(hidden_states, &self.k_proj)?.to_candle()?.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?.transpose(1, 2)?;
        let v = ops_fn::matmul(hidden_states, &self.v_proj)?.to_candle()?.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?.transpose(1, 2)?;

        let q = q.contiguous()?;
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scores = (q.matmul(&k_t)? * (self.scale as f64))?;
        let device = scores.device();
        let mask = {
            let mut m = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len { for j in (i+1)..seq_len { m[i*seq_len+j] = f32::NEG_INFINITY; } }
            candle_core::Tensor::from_vec(m, &[1, 1, seq_len, seq_len], device)?
        };
        let v = v.contiguous()?;
        let attn = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(&mask)?)?.matmul(&v)?;
        let out = attn.transpose(1, 2)?.reshape(&[batch, seq_len, self.num_heads * self.head_dim])?;
        ops_fn::matmul(&Tensor::from_candle(out), &self.out_proj)
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.q_proj = self.q_proj.to_device(device)?;
        self.k_proj = self.k_proj.to_device(device)?;
        self.v_proj = self.v_proj.to_device(device)?;
        self.out_proj = self.out_proj.to_device(device)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_opt_creation() {
        let config = OPTConfig { vocab_size: 1000, hidden_size: 128, intermediate_size: 512, num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 4, ..Default::default() };
        let model = OPTModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 1000);
    }
    #[test]
    fn test_opt_forward() {
        let config = OPTConfig { vocab_size: 100, hidden_size: 64, intermediate_size: 256, num_hidden_layers: 1, num_attention_heads: 4, num_key_value_heads: 4, word_embed_proj_dim: 64, ..Default::default() };
        let model = OPTModelV2::new(config).unwrap();
        let inputs = ModelInputs::text(ops_fn::zeros(&[2, 8], DataType::Int64, &Device::CPU).unwrap());
        match model.forward(&inputs).unwrap() { ModelOutputs::Logits { logits, .. } => assert_eq!(logits.shape(), &[2, 8, 100]), _ => panic!() }
    }
}
