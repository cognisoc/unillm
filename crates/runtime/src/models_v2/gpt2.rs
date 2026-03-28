//! GPT-2 Model V2 - Clean implementation using solid abstractions
//!
//! This implements the GPT-2 architecture which features:
//! - Learned absolute position embeddings
//! - Standard multi-head attention (no GQA)
//! - Pre-norm or post-norm layer normalization
//! - Uses unified Tensor type from tensor_core

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

/// GPT-2 model configuration
model_config!(GPT2Config {
    vocab_size: usize = 50257,
    hidden_size: usize = 768,
    intermediate_size: usize = 3072,
    num_hidden_layers: usize = 12,
    num_attention_heads: usize = 12,
    num_key_value_heads: usize = 12,
    hidden_act: String = "gelu".to_string(),
    max_position_embeddings: usize = 1024,
    initializer_range: f32 = 0.02,
    layer_norm_eps: f32 = 1e-5,
    use_cache: bool = true,
    pad_token_id: i64 = 50256,
    bos_token_id: i64 = 50256,
    eos_token_id: i64 = 50256,
    tie_word_embeddings: bool = true,
    attention_dropout: f32 = 0.1,
    residual_dropout: f32 = 0.1,
});

impl GPT2Config {
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

/// Main GPT-2 model
pub struct GPT2ModelV2 {
    config: GPT2Config,
    device: Device,

    wte: Tensor,  // Token embeddings
    wpe: Tensor,  // Position embeddings
    layers: Vec<GPT2Layer>,
    ln_f: Tensor, // Final layer norm
    lm_head: Tensor,
}

/// GPT-2 transformer layer
pub struct GPT2Layer {
    attn: GPT2Attention,
    mlp: GPT2MLP,
    ln_1: Tensor,
    ln_2: Tensor,
}

/// GPT-2 attention
pub struct GPT2Attention {
    c_attn: Tensor,  // Combined QKV projection
    c_proj: Tensor,  // Output projection
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

/// GPT-2 MLP
pub struct GPT2MLP {
    c_fc: Tensor,
    c_proj: Tensor,
}

impl Model for GPT2ModelV2 {
    type Config = GPT2Config;

    fn new(config: GPT2Config) -> Result<Self> {
        let device = Device::CPU;

        let wte = ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, &device)?;
        let wpe = ops_fn::zeros(&[config.max_position_embeddings, config.hidden_size], DataType::Float32, &device)?;
        let ln_f = ops_fn::zeros(&[config.hidden_size], DataType::Float32, &device)?;

        let lm_head = if config.tie_word_embeddings {
            wte.clone()
        } else {
            ops_fn::zeros(&[config.hidden_size, config.vocab_size], DataType::Float32, &device)?
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(GPT2Layer::new(&config, &device)?);
        }

        Ok(Self {
            config,
            device,
            wte,
            wpe,
            layers,
            ln_f,
            lm_head,
        })
    }

    fn from_weights(config: GPT2Config, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;

        if let Some(wte) = weights.get("wte.weight") {
            model.wte = wte.clone();
        }
        if let Some(wpe) = weights.get("wpe.weight") {
            model.wpe = wpe.clone();
        }
        if let Some(ln_f) = weights.get("ln_f.weight") {
            model.ln_f = ln_f.clone();
        }

        // lm_head is tied to wte in GPT-2
        if model.config.tie_word_embeddings {
            model.lm_head = model.wte.clone();
        } else if let Some(lm_head) = weights.get("lm_head.weight") {
            model.lm_head = ops_fn::transpose(lm_head)?;
        }

        for (i, layer) in model.layers.iter_mut().enumerate() {
            layer.load_weights(&weights, i)?;
        }

        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        match inputs {
            ModelInputs::Text { input_ids, .. } => {
                let shape = input_ids.shape();
                let seq_len = shape[1];

                // Token embeddings
                let mut hidden_states = ops_fn::embedding(input_ids, &self.wte)?;

                // Position embeddings
                let position_ids: Vec<i64> = (0..seq_len as i64).collect();
                let position_tensor = Tensor::from_i64_slice(&position_ids, &[1, seq_len], &self.device)?;
                let position_embeds = ops_fn::embedding(&position_tensor, &self.wpe)?;
                hidden_states = ops_fn::add(&hidden_states, &position_embeds)?;

                // Transformer layers
                for layer in &self.layers {
                    hidden_states = layer.forward(&hidden_states)?;
                }

                // Final layer norm
                hidden_states = ops_fn::layer_norm(&hidden_states, &self.ln_f, None, self.config.layer_norm_eps)?;

                // LM head (tied weights)
                let logits = if self.config.tie_word_embeddings {
                    // For tied weights, we need to do hidden @ wte.T
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

                Ok(ModelOutputs::Logits {
                    logits,
                    hidden_states: None,
                })
            }
            _ => Err(anyhow::anyhow!("GPT-2 model only supports text inputs")),
        }
    }

    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        use crate::tokenizer::Tokenizer;
        use rand::Rng;

        let tokenizer = Tokenizer::new();
        let mut tokens: Vec<u32> = tokenizer.encode(prompt);

        for _ in 0..config.max_new_tokens {
            // Truncate to max position embeddings
            let start_idx = if tokens.len() > self.config.max_position_embeddings {
                tokens.len() - self.config.max_position_embeddings
            } else {
                0
            };
            let context = &tokens[start_idx..];

            let tokens_i64: Vec<i64> = context.iter().map(|&t| t as i64).collect();
            let input_tensor = Tensor::from_i64_slice(&tokens_i64, &[1, context.len()], &self.device)?;

            let inputs = ModelInputs::Text {
                input_ids: input_tensor,
                attention_mask: None,
                position_ids: None,
            };

            let outputs = self.forward(&inputs)?;

            let logits = match outputs {
                ModelOutputs::Logits { logits, .. } => logits,
                _ => return Err(anyhow::anyhow!("Expected logits output")),
            };

            let logits_candle = logits.to_candle()?;
            let shape = logits_candle.dims();

            let last_logits = if shape.len() == 3 {
                let seq_len = shape[1];
                logits_candle.narrow(1, seq_len - 1, 1)?.squeeze(1)?.squeeze(0)?
            } else {
                let seq_len = shape[0];
                logits_candle.narrow(0, seq_len - 1, 1)?.squeeze(0)?
            };

            let logits_vec: Vec<f32> = last_logits.to_vec1()?;

            let next_token = if config.do_sample && config.temperature > 0.0 {
                let scaled: Vec<f32> = logits_vec.iter().map(|&x| x / config.temperature).collect();
                let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scaled.iter().map(|&x| (x - max_val).exp()).sum();
                let probs: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp() / exp_sum).collect();

                let mut rng = rand::thread_rng();
                let random_val: f32 = rng.gen();
                let mut cumulative = 0.0;
                let mut sampled = 0u32;

                for (idx, &prob) in probs.iter().enumerate() {
                    cumulative += prob;
                    if random_val <= cumulative {
                        sampled = idx as u32;
                        break;
                    }
                }
                sampled
            } else {
                logits_vec.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as u32).unwrap_or(0)
            };

            if next_token == config.eos_token_id {
                break;
            }

            tokens.push(next_token);
        }

        Ok(tokenizer.decode(&tokens))
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn memory_requirements(&self) -> MemoryRequirements {
        let param_size = self.config.vocab_size * self.config.hidden_size +
                        self.config.max_position_embeddings * self.config.hidden_size +
                        self.config.num_hidden_layers * (
                            3 * self.config.hidden_size * self.config.hidden_size +
                            2 * self.config.hidden_size * self.config.intermediate_size
                        );

        let param_bytes = param_size * 4;
        let kv_cache_bytes = 2 * self.config.num_hidden_layers *
                           self.config.max_position_embeddings *
                           self.config.hidden_size * 4;

        MemoryRequirements {
            gpu_memory: param_bytes,
            cpu_memory: param_bytes / 4,
            kv_cache_memory: kv_cache_bytes,
            peak_memory: param_bytes + kv_cache_bytes,
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.wte = self.wte.to_device(device)?;
        self.wpe = self.wpe.to_device(device)?;
        self.ln_f = self.ln_f.to_device(device)?;
        self.lm_head = self.lm_head.to_device(device)?;

        for layer in &mut self.layers {
            layer.to_device(device)?;
        }

        self.device = device.clone();
        Ok(())
    }
}

impl GPT2Layer {
    fn new(config: &GPT2Config, device: &Device) -> Result<Self> {
        let attn = GPT2Attention::new(config, device)?;
        let mlp = GPT2MLP::new(config, device)?;
        let ln_1 = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;
        let ln_2 = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;

        Ok(Self { attn, mlp, ln_1, ln_2 })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Pre-norm attention
        let normed = ops_fn::layer_norm(hidden_states, &self.ln_1, None, 1e-5)?;
        let attn_output = self.attn.forward(&normed)?;
        let hidden_states = ops_fn::add(hidden_states, &attn_output)?;

        // Pre-norm MLP
        let normed = ops_fn::layer_norm(&hidden_states, &self.ln_2, None, 1e-5)?;
        let mlp_output = self.mlp.forward(&normed)?;
        let output = ops_fn::add(&hidden_states, &mlp_output)?;

        Ok(output)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("h.{}", layer_idx);

        if let Some(c_attn) = weights.get(&format!("{}.attn.c_attn.weight", prefix)) {
            self.attn.c_attn = ops_fn::transpose(c_attn)?;
        }
        if let Some(c_proj) = weights.get(&format!("{}.attn.c_proj.weight", prefix)) {
            self.attn.c_proj = ops_fn::transpose(c_proj)?;
        }
        if let Some(c_fc) = weights.get(&format!("{}.mlp.c_fc.weight", prefix)) {
            self.mlp.c_fc = ops_fn::transpose(c_fc)?;
        }
        if let Some(c_proj) = weights.get(&format!("{}.mlp.c_proj.weight", prefix)) {
            self.mlp.c_proj = ops_fn::transpose(c_proj)?;
        }
        if let Some(ln_1) = weights.get(&format!("{}.ln_1.weight", prefix)) {
            self.ln_1 = ln_1.clone();
        }
        if let Some(ln_2) = weights.get(&format!("{}.ln_2.weight", prefix)) {
            self.ln_2 = ln_2.clone();
        }

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

impl GPT2Attention {
    fn new(config: &GPT2Config, device: &Device) -> Result<Self> {
        let num_heads = config.num_attention_heads;
        let head_dim = config.hidden_size / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // GPT-2 uses combined QKV projection
        let c_attn = ops_fn::zeros(&[config.hidden_size, 3 * config.hidden_size], DataType::Float32, device)?;
        let c_proj = ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?;

        Ok(Self {
            c_attn,
            c_proj,
            num_heads,
            head_dim,
            scale,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch_size, seq_len, hidden_size) = (shape[0], shape[1], shape[2]);

        // Combined QKV projection
        let qkv = ops_fn::matmul(hidden_states, &self.c_attn)?;
        let qkv_candle = qkv.to_candle()?;

        // Split into Q, K, V
        let q = qkv_candle.narrow(2, 0, hidden_size)?;
        let k = qkv_candle.narrow(2, hidden_size, hidden_size)?;
        let v = qkv_candle.narrow(2, 2 * hidden_size, hidden_size)?;

        // Reshape for multi-head attention
        let q = q.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?.transpose(1, 2)?;
        let k = k.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?.transpose(1, 2)?;
        let v = v.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?.transpose(1, 2)?;

        // Attention scores
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let q = q.contiguous()?;
        let scores = q.matmul(&k_t)?;
        let scaled_scores = (scores * (self.scale as f64))?;

        // Causal mask
        let device = scaled_scores.device();
        let causal_mask = {
            let mut mask_data = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    if j > i {
                        mask_data[i * seq_len + j] = f32::NEG_INFINITY;
                    }
                }
            }
            candle_core::Tensor::from_vec(mask_data, &[1, 1, seq_len, seq_len], device)?
        };

        let masked_scores = scaled_scores.broadcast_add(&causal_mask)?;
        let attention_weights = candle_nn::ops::softmax_last_dim(&masked_scores)?;
        let v = v.contiguous()?;
        let attn_output = attention_weights.matmul(&v)?;

        // Reshape back
        let attn_output = attn_output.transpose(1, 2)?.reshape(&[batch_size, seq_len, hidden_size])?;
        let attn_output = Tensor::from_candle(attn_output);

        // Output projection
        ops_fn::matmul(&attn_output, &self.c_proj)
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.c_attn = self.c_attn.to_device(device)?;
        self.c_proj = self.c_proj.to_device(device)?;
        Ok(())
    }
}

impl GPT2MLP {
    fn new(config: &GPT2Config, device: &Device) -> Result<Self> {
        let c_fc = ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?;
        let c_proj = ops_fn::zeros(&[config.intermediate_size, config.hidden_size], DataType::Float32, device)?;

        Ok(Self { c_fc, c_proj })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let fc_output = ops_fn::matmul(hidden_states, &self.c_fc)?;
        let activated = ops_fn::gelu(&fc_output)?;
        ops_fn::matmul(&activated, &self.c_proj)
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
    fn test_gpt2_model_creation() {
        let config = GPT2Config {
            vocab_size: 1000,
            hidden_size: 128,
            intermediate_size: 512,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            max_position_embeddings: 256,
            ..Default::default()
        };

        let model = GPT2ModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 1000);
    }

    #[test]
    fn test_gpt2_forward_pass() {
        let config = GPT2Config {
            vocab_size: 100,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            max_position_embeddings: 32,
            ..Default::default()
        };

        let model = GPT2ModelV2::new(config).unwrap();
        let input_ids = ops_fn::zeros(&[2, 8], DataType::Int64, &Device::CPU).unwrap();
        let inputs = ModelInputs::text(input_ids);

        let outputs = model.forward(&inputs).unwrap();
        match outputs {
            ModelOutputs::Logits { logits, .. } => {
                assert_eq!(logits.shape(), &[2, 8, 100]);
            }
            _ => panic!("Expected logits output"),
        }
    }
}
