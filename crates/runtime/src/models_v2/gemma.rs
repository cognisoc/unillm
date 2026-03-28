//! Gemma Model V2 - Clean implementation using solid abstractions
//!
//! This implements the Gemma architecture which is used by 10+ models including:
//! - Gemma-2B, Gemma-7B, Gemma2-9B, Gemma2-27B, CodeGemma, PaliGemma
//! - Uses unified Tensor type from tensor_core
//! - Implements Model trait from model_core

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

/// Gemma model configuration using the model_config macro
model_config!(GemmaConfig {
    vocab_size: usize = 256000,
    hidden_size: usize = 3072,
    intermediate_size: usize = 24576,
    num_hidden_layers: usize = 28,
    num_attention_heads: usize = 16,
    num_key_value_heads: usize = 16,
    hidden_act: String = "gelu".to_string(),
    max_position_embeddings: usize = 8192,
    initializer_range: f32 = 0.02,
    rms_norm_eps: f32 = 1e-6,
    use_cache: bool = true,
    pad_token_id: i64 = 0,
    bos_token_id: i64 = 2,
    eos_token_id: i64 = 1,
    tie_word_embeddings: bool = true,
    rope_theta: f32 = 10000.0,
    attention_bias: bool = false,
    head_dim: usize = 256,
});

impl GemmaConfig {
    /// Create GemmaConfig from GGUF model configuration
    pub fn from_gguf_config(gguf: &crate::weight_loader_core::GGUFModelConfig) -> Self {
        Self {
            vocab_size: gguf.vocab_size,
            hidden_size: gguf.hidden_size,
            intermediate_size: gguf.intermediate_size,
            num_hidden_layers: gguf.num_hidden_layers,
            num_attention_heads: gguf.num_attention_heads,
            num_key_value_heads: gguf.num_key_value_heads,
            rms_norm_eps: gguf.rms_norm_eps,
            rope_theta: gguf.rope_theta,
            max_position_embeddings: gguf.max_position_embeddings,
            head_dim: gguf.head_dim,
            ..Default::default()
        }
    }
}

/// Main Gemma model implementation
pub struct GemmaModelV2 {
    config: GemmaConfig,
    device: Device,

    // Model components using unified Tensor type
    embed_tokens: Tensor,
    layers: Vec<GemmaLayer>,
    norm: Tensor,
    lm_head: Tensor,
}

/// Gemma transformer layer
pub struct GemmaLayer {
    self_attn: GemmaAttention,
    mlp: GemmaMLP,
    input_layernorm: Tensor,
    post_attention_layernorm: Tensor,
}

/// Gemma attention mechanism
pub struct GemmaAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    num_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    scale: f32,
}

/// Gemma MLP (feed-forward network)
pub struct GemmaMLP {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
    hidden_act: String,
}

impl Model for GemmaModelV2 {
    type Config = GemmaConfig;

    fn new(config: GemmaConfig) -> Result<Self> {
        let device = Device::CPU;

        let embed_tokens = ops_fn::zeros(
            &[config.vocab_size, config.hidden_size],
            DataType::Float32,
            &device
        )?;

        let norm = ops_fn::zeros(
            &[config.hidden_size],
            DataType::Float32,
            &device
        )?;

        // Gemma typically uses tied embeddings
        let lm_head = if config.tie_word_embeddings {
            embed_tokens.clone()
        } else {
            ops_fn::zeros(
                &[config.hidden_size, config.vocab_size],
                DataType::Float32,
                &device
            )?
        };

        // Create transformer layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(GemmaLayer::new(&config, &device)?);
        }

        Ok(Self {
            config,
            device,
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    fn from_weights(config: GemmaConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;

        // Load weights from the unified weight container
        if let Some(embed_weights) = weights.get("model.embed_tokens.weight") {
            model.embed_tokens = embed_weights.clone();
        }

        if let Some(norm_weights) = weights.get("model.norm.weight") {
            model.norm = norm_weights.clone();
        }

        // For tied embeddings, lm_head uses embed_tokens transposed
        // For separate lm_head, transpose for matmul
        if !model.config.tie_word_embeddings {
            if let Some(lm_head_weights) = weights.get("lm_head.weight") {
                model.lm_head = ops_fn::transpose(lm_head_weights)?;
            }
        }

        // Load layer weights
        for (i, layer) in model.layers.iter_mut().enumerate() {
            layer.load_weights(&weights, i)?;
        }

        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        match inputs {
            ModelInputs::Text { input_ids, attention_mask, .. } => {
                // 1. Token embedding
                let mut hidden_states = ops_fn::embedding(input_ids, &self.embed_tokens)?;

                // 2. Gemma scales embeddings by sqrt(hidden_size)
                let scale = (self.config.hidden_size as f32).sqrt();
                hidden_states = ops_fn::scale(&hidden_states, scale)?;

                // 3. Apply transformer layers (with RoPE)
                for layer in &self.layers {
                    hidden_states = layer.forward(&hidden_states, attention_mask.as_ref(), self.config.rope_theta, self.config.head_dim)?;
                }

                // 4. Final layer norm
                hidden_states = ops_fn::layer_norm(&hidden_states, &self.norm, None, self.config.rms_norm_eps)?;

                // 5. Language modeling head
                // For tied embeddings, we need to transpose embed_tokens for matmul
                let logits = if self.config.tie_word_embeddings {
                    let lm_head_t = ops_fn::transpose(&self.embed_tokens)?;
                    ops_fn::matmul(&hidden_states, &lm_head_t)?
                } else {
                    ops_fn::matmul(&hidden_states, &self.lm_head)?
                };

                Ok(ModelOutputs::Logits {
                    logits,
                    hidden_states: None,
                })
            }
            ModelInputs::Multimodal { input_ids, .. } => {
                let text_inputs = ModelInputs::Text {
                    input_ids: input_ids.clone(),
                    attention_mask: None,
                    position_ids: None,
                };
                self.forward(&text_inputs)
            }
            _ => Err(anyhow::anyhow!("Gemma model only supports text and multimodal inputs")),
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
                logits_candle
                    .narrow(1, seq_len - 1, 1)?
                    .squeeze(1)?
                    .squeeze(0)?
            } else {
                let seq_len = shape[0];
                logits_candle
                    .narrow(0, seq_len - 1, 1)?
                    .squeeze(0)?
            };

            let logits_vec: Vec<f32> = last_logits.to_vec1()?;

            let next_token = if config.do_sample && config.temperature > 0.0 {
                let scaled: Vec<f32> = logits_vec.iter()
                    .map(|&x| x / config.temperature)
                    .collect();

                let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scaled.iter().map(|&x| (x - max_val).exp()).sum();
                let probs: Vec<f32> = scaled.iter()
                    .map(|&x| (x - max_val).exp() / exp_sum)
                    .collect();

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
                let mut max_idx = 0;
                let mut max_val = logits_vec[0];
                for (idx, &val) in logits_vec.iter().enumerate() {
                    if val > max_val {
                        max_val = val;
                        max_idx = idx;
                    }
                }
                max_idx as u32
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
                        self.config.num_hidden_layers * (
                            4 * self.config.hidden_size * self.config.hidden_size +
                            3 * self.config.hidden_size * self.config.intermediate_size
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
        self.embed_tokens = self.embed_tokens.to_device(device)?;
        self.norm = self.norm.to_device(device)?;
        self.lm_head = self.lm_head.to_device(device)?;

        for layer in &mut self.layers {
            layer.to_device(device)?;
        }

        self.device = device.clone();
        Ok(())
    }
}

impl GemmaLayer {
    fn new(config: &GemmaConfig, device: &Device) -> Result<Self> {
        let self_attn = GemmaAttention::new(config, device)?;
        let mlp = GemmaMLP::new(config, device)?;

        let input_layernorm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;
        let post_attention_layernorm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>, rope_theta: f32, head_dim: usize) -> Result<Tensor> {
        // 1. Pre-attention layer norm
        let normed = ops_fn::layer_norm(hidden_states, &self.input_layernorm, None, 1e-6)?;

        // 2. Self attention (with RoPE)
        let attn_output = self.self_attn.forward(&normed, attention_mask, rope_theta, head_dim)?;

        // 3. Residual connection
        let hidden_states = ops_fn::add(hidden_states, &attn_output)?;

        // 4. Pre-MLP layer norm
        let normed = ops_fn::layer_norm(&hidden_states, &self.post_attention_layernorm, None, 1e-6)?;

        // 5. MLP
        let mlp_output = self.mlp.forward(&normed)?;

        // 6. Residual connection
        let output = ops_fn::add(&hidden_states, &mlp_output)?;

        Ok(output)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("model.layers.{}", layer_idx);

        // Load attention weights (transpose for matmul)
        if let Some(q_proj) = weights.get(&format!("{}.self_attn.q_proj.weight", prefix)) {
            self.self_attn.q_proj = ops_fn::transpose(q_proj)?;
        }
        if let Some(k_proj) = weights.get(&format!("{}.self_attn.k_proj.weight", prefix)) {
            self.self_attn.k_proj = ops_fn::transpose(k_proj)?;
        }
        if let Some(v_proj) = weights.get(&format!("{}.self_attn.v_proj.weight", prefix)) {
            self.self_attn.v_proj = ops_fn::transpose(v_proj)?;
        }
        if let Some(o_proj) = weights.get(&format!("{}.self_attn.o_proj.weight", prefix)) {
            self.self_attn.o_proj = ops_fn::transpose(o_proj)?;
        }

        // Load MLP weights (transpose for matmul)
        if let Some(gate_proj) = weights.get(&format!("{}.mlp.gate_proj.weight", prefix)) {
            self.mlp.gate_proj = ops_fn::transpose(gate_proj)?;
        }
        if let Some(up_proj) = weights.get(&format!("{}.mlp.up_proj.weight", prefix)) {
            self.mlp.up_proj = ops_fn::transpose(up_proj)?;
        }
        if let Some(down_proj) = weights.get(&format!("{}.mlp.down_proj.weight", prefix)) {
            self.mlp.down_proj = ops_fn::transpose(down_proj)?;
        }

        // Load layer norm weights
        if let Some(input_ln) = weights.get(&format!("{}.input_layernorm.weight", prefix)) {
            self.input_layernorm = input_ln.clone();
        }
        if let Some(post_ln) = weights.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
            self.post_attention_layernorm = post_ln.clone();
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.self_attn.to_device(device)?;
        self.mlp.to_device(device)?;
        self.input_layernorm = self.input_layernorm.to_device(device)?;
        self.post_attention_layernorm = self.post_attention_layernorm.to_device(device)?;
        Ok(())
    }
}

/// Apply Rotary Position Embedding (RoPE) to Q and K tensors
fn apply_rope(
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    seq_len: usize,
    head_dim: usize,
    rope_theta: f32,
) -> Result<(candle_core::Tensor, candle_core::Tensor)> {
    let device = q.device();

    let half_dim = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / rope_theta.powf((2 * i) as f32 / head_dim as f32))
        .collect();

    let positions: Vec<f32> = (0..seq_len).map(|p| p as f32).collect();

    let mut angles = Vec::with_capacity(seq_len * half_dim);
    for pos in &positions {
        for freq in &inv_freq {
            angles.push(pos * freq);
        }
    }

    let angles_tensor = candle_core::Tensor::from_vec(angles, &[seq_len, half_dim], device)?;

    let cos = angles_tensor.cos()?;
    let sin = angles_tensor.sin()?;

    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

    let q_half1 = q.narrow(3, 0, half_dim)?;
    let q_half2 = q.narrow(3, half_dim, half_dim)?;
    let k_half1 = k.narrow(3, 0, half_dim)?;
    let k_half2 = k.narrow(3, half_dim, half_dim)?;

    let q_rot1 = (q_half1.broadcast_mul(&cos)? - q_half2.broadcast_mul(&sin)?)?;
    let q_rot2 = (q_half1.broadcast_mul(&sin)? + q_half2.broadcast_mul(&cos)?)?;
    let k_rot1 = (k_half1.broadcast_mul(&cos)? - k_half2.broadcast_mul(&sin)?)?;
    let k_rot2 = (k_half1.broadcast_mul(&sin)? + k_half2.broadcast_mul(&cos)?)?;

    let q_rotated = candle_core::Tensor::cat(&[&q_rot1, &q_rot2], 3)?;
    let k_rotated = candle_core::Tensor::cat(&[&k_rot1, &k_rot2], 3)?;

    Ok((q_rotated, k_rotated))
}

impl GemmaAttention {
    fn new(config: &GemmaConfig, device: &Device) -> Result<Self> {
        let num_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q_proj = ops_fn::zeros(&[config.hidden_size, num_heads * head_dim], DataType::Float32, device)?;
        let k_proj = ops_fn::zeros(&[config.hidden_size, num_key_value_heads * head_dim], DataType::Float32, device)?;
        let v_proj = ops_fn::zeros(&[config.hidden_size, num_key_value_heads * head_dim], DataType::Float32, device)?;
        let o_proj = ops_fn::zeros(&[num_heads * head_dim, config.hidden_size], DataType::Float32, device)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_key_value_heads,
            head_dim,
            scale,
        })
    }

    fn forward(&self, hidden_states: &Tensor, _attention_mask: Option<&Tensor>, rope_theta: f32, head_dim: usize) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch_size, seq_len, _hidden_size) = if shape.len() == 3 {
            (shape[0], shape[1], shape[2])
        } else if shape.len() == 2 {
            (1, shape[0], shape[1])
        } else {
            return Err(anyhow::anyhow!("Invalid hidden_states shape: {:?}", shape));
        };

        let query_states = ops_fn::matmul(hidden_states, &self.q_proj)?;
        let key_states = ops_fn::matmul(hidden_states, &self.k_proj)?;
        let value_states = ops_fn::matmul(hidden_states, &self.v_proj)?;

        let q_candle = query_states.to_candle()?;
        let k_candle = key_states.to_candle()?;
        let v_candle = value_states.to_candle()?;

        let q_reshaped = q_candle
            .reshape(&[batch_size, seq_len, self.num_heads, head_dim])?
            .transpose(1, 2)?;

        let k_reshaped = k_candle
            .reshape(&[batch_size, seq_len, self.num_key_value_heads, head_dim])?
            .transpose(1, 2)?;

        let v_reshaped = v_candle
            .reshape(&[batch_size, seq_len, self.num_key_value_heads, head_dim])?
            .transpose(1, 2)?;

        let (q_with_rope, k_with_rope) = apply_rope(&q_reshaped, &k_reshaped, seq_len, head_dim, rope_theta)?;

        let num_groups = self.num_heads / self.num_key_value_heads;
        let (k_expanded, v_expanded) = if num_groups > 1 {
            let k_rep = k_with_rope
                .unsqueeze(2)?
                .broadcast_as(&[batch_size, self.num_key_value_heads, num_groups, seq_len, head_dim])?
                .reshape(&[batch_size, self.num_heads, seq_len, head_dim])?;
            let v_rep = v_reshaped
                .unsqueeze(2)?
                .broadcast_as(&[batch_size, self.num_key_value_heads, num_groups, seq_len, head_dim])?
                .reshape(&[batch_size, self.num_heads, seq_len, head_dim])?;
            (k_rep, v_rep)
        } else {
            (k_with_rope, v_reshaped)
        };

        let k_t = k_expanded.transpose(2, 3)?;

        let q_contiguous = q_with_rope.contiguous()?;
        let k_contiguous = k_t.contiguous()?;

        let scores = q_contiguous.matmul(&k_contiguous)?;
        let scaled_scores = (scores * (self.scale as f64))?;

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

        let v_contiguous = v_expanded.contiguous()?;
        let attn_output = attention_weights.matmul(&v_contiguous)?;

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape(&[batch_size, seq_len, self.num_heads * head_dim])?;

        let attn_output = Tensor::from_candle(attn_output);

        let output = ops_fn::matmul(&attn_output, &self.o_proj)?;

        Ok(output)
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.q_proj = self.q_proj.to_device(device)?;
        self.k_proj = self.k_proj.to_device(device)?;
        self.v_proj = self.v_proj.to_device(device)?;
        self.o_proj = self.o_proj.to_device(device)?;
        Ok(())
    }
}

impl GemmaMLP {
    fn new(config: &GemmaConfig, device: &Device) -> Result<Self> {
        let gate_proj = ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?;
        let up_proj = ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?;
        let down_proj = ops_fn::zeros(&[config.intermediate_size, config.hidden_size], DataType::Float32, device)?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            hidden_act: config.hidden_act.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let gate_output = ops_fn::matmul(hidden_states, &self.gate_proj)?;
        let up_output = ops_fn::matmul(hidden_states, &self.up_proj)?;

        let gate_activated = match self.hidden_act.as_str() {
            "gelu" | "gelu_new" => ops_fn::gelu(&gate_output)?,
            "silu" | "swish" => ops_fn::silu(&gate_output)?,
            _ => return Err(anyhow::anyhow!("Unsupported activation: {}", self.hidden_act)),
        };

        let gated = ops_fn::mul(&gate_activated, &up_output)?;

        let output = ops_fn::matmul(&gated, &self.down_proj)?;

        Ok(output)
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.gate_proj = self.gate_proj.to_device(device)?;
        self.up_proj = self.up_proj.to_device(device)?;
        self.down_proj = self.down_proj.to_device(device)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemma_model_creation() {
        let config = GemmaConfig {
            vocab_size: 1000,
            hidden_size: 128,
            intermediate_size: 512,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            num_key_value_heads: 8,
            head_dim: 16,
            ..Default::default()
        };

        let model = GemmaModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 1000);
        assert_eq!(model.config().hidden_size(), 128);
        assert_eq!(model.config().num_layers(), 2);
    }

    #[test]
    fn test_gemma_forward_pass() {
        let config = GemmaConfig {
            vocab_size: 100,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            head_dim: 16,
            ..Default::default()
        };

        let model = GemmaModelV2::new(config).unwrap();
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

    #[test]
    fn test_gemma_generation() {
        let config = GemmaConfig {
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            head_dim: 16,
            ..Default::default()
        };
        let model = GemmaModelV2::new(config).unwrap();
        let gen_config = GenerationConfig {
            max_new_tokens: 5,
            ..Default::default()
        };

        let output = model.generate("Hello", &gen_config).unwrap();
        assert!(!output.is_empty());
    }
}
