//! GPT-J Model V2 - Clean implementation using solid abstractions
//!
//! This implements the GPT-J architecture which features:
//! - Rotary Position Embeddings (RoPE)
//! - Parallel attention + MLP computation
//! - No bias in attention projections
//! - Uses unified Tensor type from tensor_core

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

/// GPT-J model configuration
model_config!(GPTJConfig {
    vocab_size: usize = 50400,
    hidden_size: usize = 4096,
    intermediate_size: usize = 16384,
    num_hidden_layers: usize = 28,
    num_attention_heads: usize = 16,
    num_key_value_heads: usize = 16,
    hidden_act: String = "gelu".to_string(),
    max_position_embeddings: usize = 2048,
    initializer_range: f32 = 0.02,
    layer_norm_eps: f32 = 1e-5,
    use_cache: bool = true,
    pad_token_id: i64 = 50256,
    bos_token_id: i64 = 50256,
    eos_token_id: i64 = 50256,
    tie_word_embeddings: bool = true,
    rope_theta: f32 = 10000.0,
    rotary_dim: usize = 64,
});

impl GPTJConfig {
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

/// Main GPT-J model
pub struct GPTJModelV2 {
    config: GPTJConfig,
    device: Device,
    wte: Tensor,
    layers: Vec<GPTJLayer>,
    ln_f: Tensor,
    lm_head: Tensor,
}

pub struct GPTJLayer {
    attn: GPTJAttention,
    mlp: GPTJMLP,
    ln_1: Tensor,
}

pub struct GPTJAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    scale: f32,
}

pub struct GPTJMLP {
    fc_in: Tensor,
    fc_out: Tensor,
}

impl Model for GPTJModelV2 {
    type Config = GPTJConfig;

    fn new(config: GPTJConfig) -> Result<Self> {
        let device = Device::CPU;

        let wte = ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, &device)?;
        let ln_f = ops_fn::zeros(&[config.hidden_size], DataType::Float32, &device)?;
        let lm_head = if config.tie_word_embeddings {
            wte.clone()
        } else {
            ops_fn::zeros(&[config.hidden_size, config.vocab_size], DataType::Float32, &device)?
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(GPTJLayer::new(&config, &device)?);
        }

        Ok(Self { config, device, wte, layers, ln_f, lm_head })
    }

    fn from_weights(config: GPTJConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;

        if let Some(wte) = weights.get("transformer.wte.weight") {
            model.wte = wte.clone();
        }
        if let Some(ln_f) = weights.get("transformer.ln_f.weight") {
            model.ln_f = ln_f.clone();
        }
        if !model.config.tie_word_embeddings {
            if let Some(lm_head) = weights.get("lm_head.weight") {
                model.lm_head = ops_fn::transpose(lm_head)?;
            }
        }

        for (i, layer) in model.layers.iter_mut().enumerate() {
            layer.load_weights(&weights, i)?;
        }

        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        match inputs {
            ModelInputs::Text { input_ids, .. } => {
                let mut hidden_states = ops_fn::embedding(input_ids, &self.wte)?;

                for layer in &self.layers {
                    hidden_states = layer.forward(&hidden_states, self.config.rope_theta)?;
                }

                hidden_states = ops_fn::layer_norm(&hidden_states, &self.ln_f, None, self.config.layer_norm_eps)?;

                let logits = if self.config.tie_word_embeddings {
                    // Flatten to 2D for matmul, then reshape back
                    let wte_candle = self.wte.to_candle()?;
                    let hidden_candle = hidden_states.to_candle()?.contiguous()?;
                    let batch = hidden_candle.dims()[0];
                    let seq = hidden_candle.dims()[1];
                    let hidden_size = hidden_candle.dims()[2];
                    let flat = hidden_candle.reshape(&[batch * seq, hidden_size])?;
                    let logits_flat = flat.matmul(&wte_candle.t()?)?;
                    let logits_candle = logits_flat.reshape(&[batch, seq, self.config.vocab_size])?;
                    Tensor::from_candle(logits_candle)
                } else {
                    ops_fn::matmul(&hidden_states, &self.lm_head)?
                };

                Ok(ModelOutputs::Logits { logits, hidden_states: None })
            }
            _ => Err(anyhow::anyhow!("GPT-J model only supports text inputs")),
        }
    }

    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        use crate::tokenizer::Tokenizer;
        use rand::Rng;

        let tokenizer = Tokenizer::new();
        let mut tokens: Vec<u32> = tokenizer.encode(prompt);

        for _ in 0..config.max_new_tokens {
            let tokens_i64: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
            let input_tensor = Tensor::from_i64_slice(&tokens_i64, &[1, tokens.len()], &self.device)?;
            let inputs = ModelInputs::text(input_tensor);
            let outputs = self.forward(&inputs)?;

            let logits = match outputs {
                ModelOutputs::Logits { logits, .. } => logits,
                _ => return Err(anyhow::anyhow!("Expected logits")),
            };

            let logits_candle = logits.to_candle()?;
            let shape = logits_candle.dims();
            let last_logits = logits_candle.narrow(1, shape[1] - 1, 1)?.squeeze(1)?.squeeze(0)?;
            let logits_vec: Vec<f32> = last_logits.to_vec1()?;

            let next_token = if config.do_sample && config.temperature > 0.0 {
                let scaled: Vec<f32> = logits_vec.iter().map(|&x| x / config.temperature).collect();
                let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scaled.iter().map(|&x| (x - max_val).exp()).sum();
                let probs: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp() / exp_sum).collect();
                let mut rng = rand::thread_rng();
                let r: f32 = rng.gen();
                let mut cum = 0.0;
                let mut sampled = 0u32;
                for (i, &p) in probs.iter().enumerate() {
                    cum += p;
                    if r <= cum { sampled = i as u32; break; }
                }
                sampled
            } else {
                logits_vec.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as u32).unwrap_or(0)
            };

            if next_token == config.eos_token_id { break; }
            tokens.push(next_token);
        }

        Ok(tokenizer.decode(&tokens))
    }

    fn config(&self) -> &Self::Config { &self.config }

    fn memory_requirements(&self) -> MemoryRequirements {
        let param_size = self.config.vocab_size * self.config.hidden_size +
                        self.config.num_hidden_layers * (4 * self.config.hidden_size.pow(2) + 2 * self.config.hidden_size * self.config.intermediate_size);
        MemoryRequirements {
            gpu_memory: param_size * 4,
            cpu_memory: param_size,
            kv_cache_memory: 2 * self.config.num_hidden_layers * self.config.max_position_embeddings * self.config.hidden_size * 4,
            peak_memory: param_size * 5,
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.wte = self.wte.to_device(device)?;
        self.ln_f = self.ln_f.to_device(device)?;
        self.lm_head = self.lm_head.to_device(device)?;
        for layer in &mut self.layers { layer.to_device(device)?; }
        self.device = device.clone();
        Ok(())
    }
}

impl GPTJLayer {
    fn new(config: &GPTJConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            attn: GPTJAttention::new(config, device)?,
            mlp: GPTJMLP::new(config, device)?,
            ln_1: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
        })
    }

    fn forward(&self, hidden_states: &Tensor, rope_theta: f32) -> Result<Tensor> {
        let normed = ops_fn::layer_norm(hidden_states, &self.ln_1, None, 1e-5)?;

        // Parallel attention + MLP (GPT-J specific)
        let attn_output = self.attn.forward(&normed, rope_theta)?;
        let mlp_output = self.mlp.forward(&normed)?;

        let combined = ops_fn::add(&attn_output, &mlp_output)?;
        ops_fn::add(hidden_states, &combined)
    }

    fn load_weights(&mut self, weights: &ModelWeights, idx: usize) -> Result<()> {
        let p = format!("transformer.h.{}", idx);
        if let Some(w) = weights.get(&format!("{}.attn.q_proj.weight", p)) { self.attn.q_proj = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.attn.k_proj.weight", p)) { self.attn.k_proj = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.attn.v_proj.weight", p)) { self.attn.v_proj = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.attn.out_proj.weight", p)) { self.attn.o_proj = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.mlp.fc_in.weight", p)) { self.mlp.fc_in = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.mlp.fc_out.weight", p)) { self.mlp.fc_out = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.ln_1.weight", p)) { self.ln_1 = w.clone(); }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.attn.to_device(device)?;
        self.mlp.to_device(device)?;
        self.ln_1 = self.ln_1.to_device(device)?;
        Ok(())
    }
}

fn apply_partial_rope(
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    seq_len: usize,
    rotary_dim: usize,
    rope_theta: f32,
) -> Result<(candle_core::Tensor, candle_core::Tensor)> {
    let device = q.device();
    let half_rotary = rotary_dim / 2;

    let inv_freq: Vec<f32> = (0..half_rotary)
        .map(|i| 1.0 / rope_theta.powf((2 * i) as f32 / rotary_dim as f32))
        .collect();
    let positions: Vec<f32> = (0..seq_len).map(|p| p as f32).collect();

    let mut angles = Vec::with_capacity(seq_len * half_rotary);
    for pos in &positions {
        for freq in &inv_freq { angles.push(pos * freq); }
    }

    let angles_t = candle_core::Tensor::from_vec(angles, &[seq_len, half_rotary], device)?;
    let cos = angles_t.cos()?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = angles_t.sin()?.unsqueeze(0)?.unsqueeze(0)?;

    // Apply RoPE only to first rotary_dim dimensions
    let q_rot = q.narrow(3, 0, rotary_dim)?;
    let q_pass = q.narrow(3, rotary_dim, q.dims()[3] - rotary_dim)?;
    let k_rot = k.narrow(3, 0, rotary_dim)?;
    let k_pass = k.narrow(3, rotary_dim, k.dims()[3] - rotary_dim)?;

    let q1 = q_rot.narrow(3, 0, half_rotary)?;
    let q2 = q_rot.narrow(3, half_rotary, half_rotary)?;
    let k1 = k_rot.narrow(3, 0, half_rotary)?;
    let k2 = k_rot.narrow(3, half_rotary, half_rotary)?;

    let q_r1 = (q1.broadcast_mul(&cos)? - q2.broadcast_mul(&sin)?)?;
    let q_r2 = (q1.broadcast_mul(&sin)? + q2.broadcast_mul(&cos)?)?;
    let k_r1 = (k1.broadcast_mul(&cos)? - k2.broadcast_mul(&sin)?)?;
    let k_r2 = (k1.broadcast_mul(&sin)? + k2.broadcast_mul(&cos)?)?;

    let q_rotated = candle_core::Tensor::cat(&[&q_r1, &q_r2], 3)?;
    let k_rotated = candle_core::Tensor::cat(&[&k_r1, &k_r2], 3)?;

    let q_out = candle_core::Tensor::cat(&[&q_rotated, &q_pass], 3)?;
    let k_out = candle_core::Tensor::cat(&[&k_rotated, &k_pass], 3)?;

    Ok((q_out, k_out))
}

impl GPTJAttention {
    fn new(config: &GPTJConfig, device: &Device) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        Ok(Self {
            q_proj: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            k_proj: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            v_proj: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            o_proj: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            num_heads: config.num_attention_heads,
            head_dim,
            rotary_dim: config.rotary_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    fn forward(&self, hidden_states: &Tensor, rope_theta: f32) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch, seq_len, _) = (shape[0], shape[1], shape[2]);

        let q = ops_fn::matmul(hidden_states, &self.q_proj)?.to_candle()?.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?.transpose(1, 2)?;
        let k = ops_fn::matmul(hidden_states, &self.k_proj)?.to_candle()?.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?.transpose(1, 2)?;
        let v = ops_fn::matmul(hidden_states, &self.v_proj)?.to_candle()?.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?.transpose(1, 2)?;

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
        let masked = scores.broadcast_add(&mask)?;
        let v = v.contiguous()?;
        let attn = candle_nn::ops::softmax_last_dim(&masked)?.matmul(&v)?;
        let out = attn.transpose(1, 2)?.reshape(&[batch, seq_len, self.num_heads * self.head_dim])?;
        ops_fn::matmul(&Tensor::from_candle(out), &self.o_proj)
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.q_proj = self.q_proj.to_device(device)?;
        self.k_proj = self.k_proj.to_device(device)?;
        self.v_proj = self.v_proj.to_device(device)?;
        self.o_proj = self.o_proj.to_device(device)?;
        Ok(())
    }
}

impl GPTJMLP {
    fn new(config: &GPTJConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            fc_in: ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?,
            fc_out: ops_fn::zeros(&[config.intermediate_size, config.hidden_size], DataType::Float32, device)?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let h = ops_fn::matmul(hidden_states, &self.fc_in)?;
        let h = ops_fn::gelu(&h)?;
        ops_fn::matmul(&h, &self.fc_out)
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.fc_in = self.fc_in.to_device(device)?;
        self.fc_out = self.fc_out.to_device(device)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gptj_creation() {
        let config = GPTJConfig {
            vocab_size: 1000, hidden_size: 128, intermediate_size: 512,
            num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 4,
            rotary_dim: 32, ..Default::default()
        };
        let model = GPTJModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 1000);
    }

    #[test]
    fn test_gptj_forward() {
        let config = GPTJConfig {
            vocab_size: 100, hidden_size: 64, intermediate_size: 256,
            num_hidden_layers: 1, num_attention_heads: 4, num_key_value_heads: 4,
            rotary_dim: 16, ..Default::default()
        };
        let model = GPTJModelV2::new(config).unwrap();
        let inputs = ModelInputs::text(ops_fn::zeros(&[2, 8], DataType::Int64, &Device::CPU).unwrap());
        let outputs = model.forward(&inputs).unwrap();
        match outputs {
            ModelOutputs::Logits { logits, .. } => assert_eq!(logits.shape(), &[2, 8, 100]),
            _ => panic!("Expected logits"),
        }
    }
}
