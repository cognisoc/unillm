//! GPT-NeoX Model V2 - Clean implementation
//!
//! GPT-NeoX architecture features:
//! - Rotary Position Embeddings (RoPE)
//! - Parallel attention + MLP
//! - Pre-norm layer normalization

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

model_config!(GPTNeoXConfig {
    vocab_size: usize = 50432,
    hidden_size: usize = 6144,
    intermediate_size: usize = 24576,
    num_hidden_layers: usize = 44,
    num_attention_heads: usize = 64,
    num_key_value_heads: usize = 64,
    hidden_act: String = "gelu".to_string(),
    max_position_embeddings: usize = 2048,
    initializer_range: f32 = 0.02,
    layer_norm_eps: f32 = 1e-5,
    use_cache: bool = true,
    pad_token_id: i64 = 0,
    bos_token_id: i64 = 0,
    eos_token_id: i64 = 0,
    tie_word_embeddings: bool = false,
    rope_theta: f32 = 10000.0,
    rotary_pct: f32 = 0.25,
    use_parallel_residual: bool = true,
});

impl GPTNeoXConfig {
    pub fn from_gguf_config(gguf: &crate::weight_loader_core::GGUFModelConfig) -> Self {
        Self {
            vocab_size: gguf.vocab_size,
            hidden_size: gguf.hidden_size,
            intermediate_size: gguf.intermediate_size,
            num_hidden_layers: gguf.num_hidden_layers,
            num_attention_heads: gguf.num_attention_heads,
            num_key_value_heads: gguf.num_key_value_heads,
            rope_theta: gguf.rope_theta,
            max_position_embeddings: gguf.max_position_embeddings,
            ..Default::default()
        }
    }
}

pub struct GPTNeoXModelV2 {
    config: GPTNeoXConfig,
    device: Device,
    embed_in: Tensor,
    layers: Vec<GPTNeoXLayer>,
    final_layer_norm: Tensor,
    embed_out: Tensor,
}

pub struct GPTNeoXLayer {
    attention: GPTNeoXAttention,
    mlp: GPTNeoXMLP,
    input_layernorm: Tensor,
    post_attention_layernorm: Tensor,
    use_parallel_residual: bool,
}

pub struct GPTNeoXAttention {
    query_key_value: Tensor,
    dense: Tensor,
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    scale: f32,
}

pub struct GPTNeoXMLP {
    dense_h_to_4h: Tensor,
    dense_4h_to_h: Tensor,
}

impl Model for GPTNeoXModelV2 {
    type Config = GPTNeoXConfig;

    fn new(config: GPTNeoXConfig) -> Result<Self> {
        let device = Device::CPU;
        let embed_in = ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, &device)?;
        let final_layer_norm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, &device)?;
        let embed_out = ops_fn::zeros(&[config.hidden_size, config.vocab_size], DataType::Float32, &device)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(GPTNeoXLayer::new(&config, &device)?);
        }

        Ok(Self { config, device, embed_in, layers, final_layer_norm, embed_out })
    }

    fn from_weights(config: GPTNeoXConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;
        if let Some(w) = weights.get("gpt_neox.embed_in.weight") { model.embed_in = w.clone(); }
        if let Some(w) = weights.get("gpt_neox.final_layer_norm.weight") { model.final_layer_norm = w.clone(); }
        if let Some(w) = weights.get("embed_out.weight") { model.embed_out = ops_fn::transpose(w)?; }
        for (i, layer) in model.layers.iter_mut().enumerate() { layer.load_weights(&weights, i)?; }
        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        match inputs {
            ModelInputs::Text { input_ids, .. } => {
                let mut hidden_states = ops_fn::embedding(input_ids, &self.embed_in)?;
                for layer in &self.layers {
                    hidden_states = layer.forward(&hidden_states, self.config.rope_theta)?;
                }
                hidden_states = ops_fn::layer_norm(&hidden_states, &self.final_layer_norm, None, self.config.layer_norm_eps)?;
                let logits = ops_fn::matmul(&hidden_states, &self.embed_out)?;
                Ok(ModelOutputs::Logits { logits, hidden_states: None })
            }
            _ => Err(anyhow::anyhow!("GPT-NeoX only supports text inputs")),
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
        self.embed_in = self.embed_in.to_device(device)?;
        self.final_layer_norm = self.final_layer_norm.to_device(device)?;
        self.embed_out = self.embed_out.to_device(device)?;
        for l in &mut self.layers { l.to_device(device)?; }
        self.device = device.clone();
        Ok(())
    }
}

impl GPTNeoXLayer {
    fn new(config: &GPTNeoXConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            attention: GPTNeoXAttention::new(config, device)?,
            mlp: GPTNeoXMLP::new(config, device)?,
            input_layernorm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            post_attention_layernorm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            use_parallel_residual: config.use_parallel_residual,
        })
    }

    fn forward(&self, hidden_states: &Tensor, rope_theta: f32) -> Result<Tensor> {
        let ln1 = ops_fn::layer_norm(hidden_states, &self.input_layernorm, None, 1e-5)?;
        let attn_out = self.attention.forward(&ln1, rope_theta)?;

        if self.use_parallel_residual {
            let ln2 = ops_fn::layer_norm(hidden_states, &self.post_attention_layernorm, None, 1e-5)?;
            let mlp_out = self.mlp.forward(&ln2)?;
            let combined = ops_fn::add(&attn_out, &mlp_out)?;
            ops_fn::add(hidden_states, &combined)
        } else {
            let h = ops_fn::add(hidden_states, &attn_out)?;
            let ln2 = ops_fn::layer_norm(&h, &self.post_attention_layernorm, None, 1e-5)?;
            let mlp_out = self.mlp.forward(&ln2)?;
            ops_fn::add(&h, &mlp_out)
        }
    }

    fn load_weights(&mut self, weights: &ModelWeights, idx: usize) -> Result<()> {
        let p = format!("gpt_neox.layers.{}", idx);
        if let Some(w) = weights.get(&format!("{}.attention.query_key_value.weight", p)) { self.attention.query_key_value = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.attention.dense.weight", p)) { self.attention.dense = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.mlp.dense_h_to_4h.weight", p)) { self.mlp.dense_h_to_4h = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.mlp.dense_4h_to_h.weight", p)) { self.mlp.dense_4h_to_h = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.input_layernorm.weight", p)) { self.input_layernorm = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.post_attention_layernorm.weight", p)) { self.post_attention_layernorm = w.clone(); }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.attention.to_device(device)?;
        self.mlp.to_device(device)?;
        self.input_layernorm = self.input_layernorm.to_device(device)?;
        self.post_attention_layernorm = self.post_attention_layernorm.to_device(device)?;
        Ok(())
    }
}

impl GPTNeoXAttention {
    fn new(config: &GPTNeoXConfig, device: &Device) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let rotary_dim = (head_dim as f32 * config.rotary_pct) as usize;
        Ok(Self {
            query_key_value: ops_fn::zeros(&[config.hidden_size, 3 * config.hidden_size], DataType::Float32, device)?,
            dense: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            num_heads: config.num_attention_heads,
            head_dim,
            rotary_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    fn forward(&self, hidden_states: &Tensor, rope_theta: f32) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch, seq_len, hidden_size) = (shape[0], shape[1], shape[2]);

        let qkv = ops_fn::matmul(hidden_states, &self.query_key_value)?.to_candle()?;
        let q = qkv.narrow(2, 0, hidden_size)?.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?.transpose(1, 2)?;
        let k = qkv.narrow(2, hidden_size, hidden_size)?.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?.transpose(1, 2)?;
        let v = qkv.narrow(2, 2 * hidden_size, hidden_size)?.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?.transpose(1, 2)?;

        let (q, k) = apply_partial_rope(&q, &k, seq_len, self.rotary_dim, rope_theta)?;

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
        let out = attn.transpose(1, 2)?.reshape(&[batch, seq_len, hidden_size])?;
        ops_fn::matmul(&Tensor::from_candle(out), &self.dense)
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.query_key_value = self.query_key_value.to_device(device)?;
        self.dense = self.dense.to_device(device)?;
        Ok(())
    }
}

fn apply_partial_rope(q: &candle_core::Tensor, k: &candle_core::Tensor, seq_len: usize, rotary_dim: usize, rope_theta: f32) -> Result<(candle_core::Tensor, candle_core::Tensor)> {
    let device = q.device();
    let half = rotary_dim / 2;
    let inv_freq: Vec<f32> = (0..half).map(|i| 1.0 / rope_theta.powf((2 * i) as f32 / rotary_dim as f32)).collect();
    let positions: Vec<f32> = (0..seq_len).map(|p| p as f32).collect();
    let mut angles = Vec::with_capacity(seq_len * half);
    for pos in &positions { for freq in &inv_freq { angles.push(pos * freq); } }
    let angles_t = candle_core::Tensor::from_vec(angles, &[seq_len, half], device)?;
    let cos = angles_t.cos()?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = angles_t.sin()?.unsqueeze(0)?.unsqueeze(0)?;

    let head_dim = q.dims()[3];
    if rotary_dim >= head_dim {
        let q1 = q.narrow(3, 0, half)?;
        let q2 = q.narrow(3, half, half)?;
        let k1 = k.narrow(3, 0, half)?;
        let k2 = k.narrow(3, half, half)?;
        let qr1 = (q1.broadcast_mul(&cos)? - q2.broadcast_mul(&sin)?)?;
        let qr2 = (q1.broadcast_mul(&sin)? + q2.broadcast_mul(&cos)?)?;
        let kr1 = (k1.broadcast_mul(&cos)? - k2.broadcast_mul(&sin)?)?;
        let kr2 = (k1.broadcast_mul(&sin)? + k2.broadcast_mul(&cos)?)?;
        Ok((candle_core::Tensor::cat(&[&qr1, &qr2], 3)?, candle_core::Tensor::cat(&[&kr1, &kr2], 3)?))
    } else {
        let q_rot = q.narrow(3, 0, rotary_dim)?;
        let q_pass = q.narrow(3, rotary_dim, head_dim - rotary_dim)?;
        let k_rot = k.narrow(3, 0, rotary_dim)?;
        let k_pass = k.narrow(3, rotary_dim, head_dim - rotary_dim)?;
        let q1 = q_rot.narrow(3, 0, half)?;
        let q2 = q_rot.narrow(3, half, half)?;
        let k1 = k_rot.narrow(3, 0, half)?;
        let k2 = k_rot.narrow(3, half, half)?;
        let qr = candle_core::Tensor::cat(&[&(q1.broadcast_mul(&cos)? - q2.broadcast_mul(&sin)?)?, &(q1.broadcast_mul(&sin)? + q2.broadcast_mul(&cos)?)?], 3)?;
        let kr = candle_core::Tensor::cat(&[&(k1.broadcast_mul(&cos)? - k2.broadcast_mul(&sin)?)?, &(k1.broadcast_mul(&sin)? + k2.broadcast_mul(&cos)?)?], 3)?;
        Ok((candle_core::Tensor::cat(&[&qr, &q_pass], 3)?, candle_core::Tensor::cat(&[&kr, &k_pass], 3)?))
    }
}

impl GPTNeoXMLP {
    fn new(config: &GPTNeoXConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            dense_h_to_4h: ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?,
            dense_4h_to_h: ops_fn::zeros(&[config.intermediate_size, config.hidden_size], DataType::Float32, device)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = ops_fn::matmul(x, &self.dense_h_to_4h)?;
        let h = ops_fn::gelu(&h)?;
        ops_fn::matmul(&h, &self.dense_4h_to_h)
    }
    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.dense_h_to_4h = self.dense_h_to_4h.to_device(device)?;
        self.dense_4h_to_h = self.dense_4h_to_h.to_device(device)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_gptneox_creation() {
        let config = GPTNeoXConfig { vocab_size: 1000, hidden_size: 128, intermediate_size: 512, num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 4, ..Default::default() };
        let model = GPTNeoXModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 1000);
    }
    #[test]
    fn test_gptneox_forward() {
        let config = GPTNeoXConfig { vocab_size: 100, hidden_size: 64, intermediate_size: 256, num_hidden_layers: 1, num_attention_heads: 4, num_key_value_heads: 4, ..Default::default() };
        let model = GPTNeoXModelV2::new(config).unwrap();
        let inputs = ModelInputs::text(ops_fn::zeros(&[2, 8], DataType::Int64, &Device::CPU).unwrap());
        match model.forward(&inputs).unwrap() { ModelOutputs::Logits { logits, .. } => assert_eq!(logits.shape(), &[2, 8, 100]), _ => panic!() }
    }
}
