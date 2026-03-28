//! BLOOM Model V2 - Clean implementation
//!
//! BLOOM architecture features:
//! - ALiBi (Attention with Linear Biases) position embeddings
//! - No bias in attention projections
//! - LayerNorm before attention

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

model_config!(BLOOMConfig {
    vocab_size: usize = 250880,
    hidden_size: usize = 2048,
    intermediate_size: usize = 8192,
    num_hidden_layers: usize = 24,
    num_attention_heads: usize = 16,
    num_key_value_heads: usize = 16,
    hidden_act: String = "gelu".to_string(),
    max_position_embeddings: usize = 2048,
    initializer_range: f32 = 0.02,
    layer_norm_eps: f32 = 1e-5,
    use_cache: bool = true,
    pad_token_id: i64 = 3,
    bos_token_id: i64 = 1,
    eos_token_id: i64 = 2,
    tie_word_embeddings: bool = true,
    apply_residual_connection_post_layernorm: bool = false,
});

impl BLOOMConfig {
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

pub struct BLOOMModelV2 {
    config: BLOOMConfig,
    device: Device,
    word_embeddings: Tensor,
    word_embeddings_layernorm: Tensor,
    layers: Vec<BLOOMLayer>,
    ln_f: Tensor,
    lm_head: Tensor,
}

pub struct BLOOMLayer {
    self_attention: BLOOMAttention,
    mlp: BLOOMMLP,
    input_layernorm: Tensor,
    post_attention_layernorm: Tensor,
}

pub struct BLOOMAttention {
    query_key_value: Tensor,
    dense: Tensor,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

pub struct BLOOMMLP {
    dense_h_to_4h: Tensor,
    dense_4h_to_h: Tensor,
}

impl Model for BLOOMModelV2 {
    type Config = BLOOMConfig;

    fn new(config: BLOOMConfig) -> Result<Self> {
        let device = Device::CPU;
        let word_embeddings = ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, &device)?;
        let word_embeddings_layernorm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, &device)?;
        let ln_f = ops_fn::zeros(&[config.hidden_size], DataType::Float32, &device)?;
        let lm_head = word_embeddings.clone();

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(BLOOMLayer::new(&config, &device)?);
        }

        Ok(Self { config, device, word_embeddings, word_embeddings_layernorm, layers, ln_f, lm_head })
    }

    fn from_weights(config: BLOOMConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;
        if let Some(w) = weights.get("transformer.word_embeddings.weight") { model.word_embeddings = w.clone(); model.lm_head = w.clone(); }
        if let Some(w) = weights.get("transformer.word_embeddings_layernorm.weight") { model.word_embeddings_layernorm = w.clone(); }
        if let Some(w) = weights.get("transformer.ln_f.weight") { model.ln_f = w.clone(); }
        if let Some(w) = weights.get("lm_head.weight") { model.lm_head = w.clone(); }
        for (i, layer) in model.layers.iter_mut().enumerate() { layer.load_weights(&weights, i)?; }
        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        match inputs {
            ModelInputs::Text { input_ids, .. } => {
                let mut hidden = ops_fn::embedding(input_ids, &self.word_embeddings)?;
                hidden = ops_fn::layer_norm(&hidden, &self.word_embeddings_layernorm, None, self.config.layer_norm_eps)?;

                let seq_len = input_ids.shape()[1];
                for layer in &self.layers {
                    hidden = layer.forward(&hidden, seq_len)?;
                }

                hidden = ops_fn::layer_norm(&hidden, &self.ln_f, None, self.config.layer_norm_eps)?;

                // Tied embeddings - flatten to 2D for matmul, then reshape back
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
            _ => Err(anyhow::anyhow!("BLOOM only supports text inputs")),
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
        self.word_embeddings = self.word_embeddings.to_device(device)?;
        self.word_embeddings_layernorm = self.word_embeddings_layernorm.to_device(device)?;
        self.ln_f = self.ln_f.to_device(device)?;
        self.lm_head = self.lm_head.to_device(device)?;
        for l in &mut self.layers { l.to_device(device)?; }
        self.device = device.clone();
        Ok(())
    }
}

impl BLOOMLayer {
    fn new(config: &BLOOMConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            self_attention: BLOOMAttention::new(config, device)?,
            mlp: BLOOMMLP::new(config, device)?,
            input_layernorm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            post_attention_layernorm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
        })
    }

    fn forward(&self, hidden_states: &Tensor, seq_len: usize) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let ln_out = ops_fn::layer_norm(hidden_states, &self.input_layernorm, None, 1e-5)?;
        let attn_out = self.self_attention.forward(&ln_out, seq_len)?;
        let h = ops_fn::add(&residual, &attn_out)?;

        let residual = h.clone();
        let ln_out = ops_fn::layer_norm(&h, &self.post_attention_layernorm, None, 1e-5)?;
        let mlp_out = self.mlp.forward(&ln_out)?;
        ops_fn::add(&residual, &mlp_out)
    }

    fn load_weights(&mut self, weights: &ModelWeights, idx: usize) -> Result<()> {
        let p = format!("transformer.h.{}", idx);
        if let Some(w) = weights.get(&format!("{}.self_attention.query_key_value.weight", p)) { self.self_attention.query_key_value = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.self_attention.dense.weight", p)) { self.self_attention.dense = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.mlp.dense_h_to_4h.weight", p)) { self.mlp.dense_h_to_4h = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.mlp.dense_4h_to_h.weight", p)) { self.mlp.dense_4h_to_h = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.input_layernorm.weight", p)) { self.input_layernorm = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.post_attention_layernorm.weight", p)) { self.post_attention_layernorm = w.clone(); }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.self_attention.to_device(device)?;
        self.mlp.to_device(device)?;
        self.input_layernorm = self.input_layernorm.to_device(device)?;
        self.post_attention_layernorm = self.post_attention_layernorm.to_device(device)?;
        Ok(())
    }
}

/// Build ALiBi attention bias
fn build_alibi_bias(num_heads: usize, seq_len: usize, device: &candle_core::Device) -> Result<candle_core::Tensor> {
    let closest_power_of_2 = 2usize.pow((num_heads as f64).log2().floor() as u32);
    let base = 2f32.powf(-(2f32.powf(-((closest_power_of_2 as f32).log2() - 3.0))));

    let mut slopes = Vec::with_capacity(num_heads);
    for i in 0..num_heads {
        let power = (i + 1) as f32;
        slopes.push(base.powf(power));
    }

    // Build position bias: slope * (j - i) for causal attention
    let mut bias_data = vec![0.0f32; num_heads * seq_len * seq_len];
    for h in 0..num_heads {
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j <= i {
                    bias_data[h * seq_len * seq_len + i * seq_len + j] = slopes[h] * (j as i32 - i as i32) as f32;
                } else {
                    bias_data[h * seq_len * seq_len + i * seq_len + j] = f32::NEG_INFINITY;
                }
            }
        }
    }

    Ok(candle_core::Tensor::from_vec(bias_data, &[1, num_heads, seq_len, seq_len], device)?)
}

impl BLOOMAttention {
    fn new(config: &BLOOMConfig, device: &Device) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        Ok(Self {
            query_key_value: ops_fn::zeros(&[config.hidden_size, 3 * config.hidden_size], DataType::Float32, device)?,
            dense: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            num_heads: config.num_attention_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    fn forward(&self, hidden_states: &Tensor, seq_len: usize) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch, seq, hidden_size) = (shape[0], shape[1], shape[2]);

        let qkv = ops_fn::matmul(hidden_states, &self.query_key_value)?.to_candle()?;
        let q = qkv.narrow(2, 0, hidden_size)?.reshape(&[batch, seq, self.num_heads, self.head_dim])?.transpose(1, 2)?;
        let k = qkv.narrow(2, hidden_size, hidden_size)?.reshape(&[batch, seq, self.num_heads, self.head_dim])?.transpose(1, 2)?;
        let v = qkv.narrow(2, 2 * hidden_size, hidden_size)?.reshape(&[batch, seq, self.num_heads, self.head_dim])?.transpose(1, 2)?;

        let q = q.contiguous()?;
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scores = (q.matmul(&k_t)? * (self.scale as f64))?;

        // Apply ALiBi bias
        let device = scores.device();
        let alibi_bias = build_alibi_bias(self.num_heads, seq_len, device)?;
        let scores = scores.broadcast_add(&alibi_bias)?;

        let v = v.contiguous()?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?;
        let out = attn.transpose(1, 2)?.reshape(&[batch, seq, hidden_size])?;
        ops_fn::matmul(&Tensor::from_candle(out), &self.dense)
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.query_key_value = self.query_key_value.to_device(device)?;
        self.dense = self.dense.to_device(device)?;
        Ok(())
    }
}

impl BLOOMMLP {
    fn new(config: &BLOOMConfig, device: &Device) -> Result<Self> {
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
    fn test_bloom_creation() {
        let config = BLOOMConfig { vocab_size: 1000, hidden_size: 128, intermediate_size: 512, num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 4, ..Default::default() };
        let model = BLOOMModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 1000);
    }
    #[test]
    fn test_bloom_forward() {
        let config = BLOOMConfig { vocab_size: 100, hidden_size: 64, intermediate_size: 256, num_hidden_layers: 1, num_attention_heads: 4, num_key_value_heads: 4, ..Default::default() };
        let model = BLOOMModelV2::new(config).unwrap();
        let inputs = ModelInputs::text(ops_fn::zeros(&[2, 8], DataType::Int64, &Device::CPU).unwrap());
        match model.forward(&inputs).unwrap() { ModelOutputs::Logits { logits, .. } => assert_eq!(logits.shape(), &[2, 8, 100]), _ => panic!() }
    }
}
