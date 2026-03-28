//! Phi Model V2 - Clean implementation using solid abstractions
//!
//! This implements the Phi architecture (Phi-1, Phi-2, Phi-3) which includes:
//! - Microsoft Phi-1/1.5/2/3 models
//! - Uses unified Tensor type from tensor_core
//! - Implements Model trait from model_core

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

/// Phi model configuration using the model_config macro
model_config!(PhiConfig {
    vocab_size: usize = 51200,
    hidden_size: usize = 2560,
    intermediate_size: usize = 10240,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 32,
    num_key_value_heads: usize = 32,
    hidden_act: String = "gelu_new".to_string(),
    max_position_embeddings: usize = 2048,
    initializer_range: f32 = 0.02,
    rms_norm_eps: f32 = 1e-5,
    use_cache: bool = true,
    pad_token_id: i64 = 0,
    bos_token_id: i64 = 1,
    eos_token_id: i64 = 2,
    tie_word_embeddings: bool = false,
    rope_theta: f32 = 10000.0,
    attention_dropout: f32 = 0.0,
    partial_rotary_factor: f32 = 0.5,
});

impl PhiConfig {
    /// Create PhiConfig from GGUF model configuration
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
            ..Default::default()
        }
    }
}

/// Main Phi model implementation
pub struct PhiModelV2 {
    config: PhiConfig,
    device: Device,

    // Model components using unified Tensor type
    embed_tokens: Tensor,
    layers: Vec<PhiLayer>,
    norm: Tensor,
    lm_head: Tensor,
}

/// Phi transformer layer
pub struct PhiLayer {
    self_attn: PhiAttention,
    mlp: PhiMLP,
    input_layernorm: Tensor,
}

/// Phi attention mechanism
pub struct PhiAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    dense: Tensor, // Phi uses 'dense' instead of 'o_proj'
    num_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    scale: f32,
    partial_rotary_factor: f32,
}

/// Phi MLP (feed-forward network) - standard 2-layer MLP, not SwiGLU
pub struct PhiMLP {
    fc1: Tensor,
    fc2: Tensor,
    hidden_act: String,
}

impl Model for PhiModelV2 {
    type Config = PhiConfig;

    fn new(config: PhiConfig) -> Result<Self> {
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
            layers.push(PhiLayer::new(&config, &device)?);
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

    fn from_weights(config: PhiConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;

        // Load weights from the unified weight container
        if let Some(embed_weights) = weights.get("model.embed_tokens.weight") {
            model.embed_tokens = embed_weights.clone();
        }

        // Phi uses final_layernorm instead of norm
        if let Some(norm_weights) = weights.get("model.final_layernorm.weight") {
            model.norm = norm_weights.clone();
        } else if let Some(norm_weights) = weights.get("model.norm.weight") {
            model.norm = norm_weights.clone();
        }

        // lm_head weight needs transpose for matmul
        if let Some(lm_head_weights) = weights.get("lm_head.weight") {
            model.lm_head = ops_fn::transpose(lm_head_weights)?;
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

                // 2. Apply transformer layers (with partial RoPE)
                for layer in &self.layers {
                    hidden_states = layer.forward(&hidden_states, attention_mask.as_ref(), self.config.rope_theta, self.config.partial_rotary_factor)?;
                }

                // 3. Final layer norm
                hidden_states = ops_fn::layer_norm(&hidden_states, &self.norm, None, self.config.rms_norm_eps)?;

                // 4. Language modeling head
                let logits = ops_fn::matmul(&hidden_states, &self.lm_head)?;

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
            _ => Err(anyhow::anyhow!("Phi model only supports text and multimodal inputs")),
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

impl PhiLayer {
    fn new(config: &PhiConfig, device: &Device) -> Result<Self> {
        let input_layernorm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;

        Ok(Self {
            self_attn: PhiAttention::new(config, device)?,
            mlp: PhiMLP::new(config, device)?,
            input_layernorm,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>, rope_theta: f32, partial_rotary_factor: f32) -> Result<Tensor> {
        // Phi uses parallel attention and MLP (like Falcon)
        let residual = hidden_states.clone();

        // Layer norm
        let normed = ops_fn::layer_norm(hidden_states, &self.input_layernorm, None, 1e-5)?;

        // Attention and MLP in parallel
        let attn_output = self.self_attn.forward(&normed, attention_mask, rope_theta, partial_rotary_factor)?;
        let mlp_output = self.mlp.forward(&normed)?;

        // Add both outputs to residual
        let combined = ops_fn::add(&attn_output, &mlp_output)?;
        let output = ops_fn::add(&residual, &combined)?;

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
        // Phi uses 'dense' for output projection
        if let Some(dense) = weights.get(&format!("{}.self_attn.dense.weight", prefix)) {
            self.self_attn.dense = ops_fn::transpose(dense)?;
        } else if let Some(o_proj) = weights.get(&format!("{}.self_attn.o_proj.weight", prefix)) {
            self.self_attn.dense = ops_fn::transpose(o_proj)?;
        }

        // Load MLP weights (transpose for matmul)
        if let Some(fc1) = weights.get(&format!("{}.mlp.fc1.weight", prefix)) {
            self.mlp.fc1 = ops_fn::transpose(fc1)?;
        }
        if let Some(fc2) = weights.get(&format!("{}.mlp.fc2.weight", prefix)) {
            self.mlp.fc2 = ops_fn::transpose(fc2)?;
        }

        // Load layer norm weight
        if let Some(input_ln) = weights.get(&format!("{}.input_layernorm.weight", prefix)) {
            self.input_layernorm = input_ln.clone();
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.self_attn.to_device(device)?;
        self.mlp.to_device(device)?;
        self.input_layernorm = self.input_layernorm.to_device(device)?;
        Ok(())
    }
}

/// Apply partial Rotary Position Embedding (RoPE) to Q and K tensors
/// Only applies RoPE to first (partial_rotary_factor * head_dim) dimensions
fn apply_partial_rope(
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    seq_len: usize,
    head_dim: usize,
    rope_theta: f32,
    partial_rotary_factor: f32,
) -> Result<(candle_core::Tensor, candle_core::Tensor)> {
    let device = q.device();

    // Calculate how much of head_dim to apply RoPE to
    let rotary_dim = ((head_dim as f32 * partial_rotary_factor) as usize / 2) * 2; // Make even
    let half_rotary_dim = rotary_dim / 2;

    if rotary_dim == 0 {
        // No rotation needed
        return Ok((q.clone(), k.clone()));
    }

    // Compute inverse frequencies for the rotary dimensions
    let inv_freq: Vec<f32> = (0..half_rotary_dim)
        .map(|i| 1.0 / rope_theta.powf((2 * i) as f32 / rotary_dim as f32))
        .collect();

    let positions: Vec<f32> = (0..seq_len).map(|p| p as f32).collect();

    let mut angles = Vec::with_capacity(seq_len * half_rotary_dim);
    for pos in &positions {
        for freq in &inv_freq {
            angles.push(pos * freq);
        }
    }

    let angles_tensor = candle_core::Tensor::from_vec(angles, &[seq_len, half_rotary_dim], device)?;

    let cos = angles_tensor.cos()?;
    let sin = angles_tensor.sin()?;

    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

    // Split into rotary and non-rotary parts
    let q_rot_part = q.narrow(3, 0, rotary_dim)?;
    let q_pass_part = q.narrow(3, rotary_dim, head_dim - rotary_dim)?;
    let k_rot_part = k.narrow(3, 0, rotary_dim)?;
    let k_pass_part = k.narrow(3, rotary_dim, head_dim - rotary_dim)?;

    // Apply rotation to rotary part
    let q_half1 = q_rot_part.narrow(3, 0, half_rotary_dim)?;
    let q_half2 = q_rot_part.narrow(3, half_rotary_dim, half_rotary_dim)?;
    let k_half1 = k_rot_part.narrow(3, 0, half_rotary_dim)?;
    let k_half2 = k_rot_part.narrow(3, half_rotary_dim, half_rotary_dim)?;

    let q_rot1 = (q_half1.broadcast_mul(&cos)? - q_half2.broadcast_mul(&sin)?)?;
    let q_rot2 = (q_half1.broadcast_mul(&sin)? + q_half2.broadcast_mul(&cos)?)?;
    let k_rot1 = (k_half1.broadcast_mul(&cos)? - k_half2.broadcast_mul(&sin)?)?;
    let k_rot2 = (k_half1.broadcast_mul(&sin)? + k_half2.broadcast_mul(&cos)?)?;

    // Concatenate rotated and non-rotated parts
    let q_rotated_part = candle_core::Tensor::cat(&[&q_rot1, &q_rot2], 3)?;
    let k_rotated_part = candle_core::Tensor::cat(&[&k_rot1, &k_rot2], 3)?;

    let q_rotated = candle_core::Tensor::cat(&[&q_rotated_part, &q_pass_part], 3)?;
    let k_rotated = candle_core::Tensor::cat(&[&k_rotated_part, &k_pass_part], 3)?;

    Ok((q_rotated, k_rotated))
}

impl PhiAttention {
    fn new(config: &PhiConfig, device: &Device) -> Result<Self> {
        let num_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads;
        let head_dim = config.hidden_size / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q_proj = ops_fn::zeros(&[config.hidden_size, num_heads * head_dim], DataType::Float32, device)?;
        let k_proj = ops_fn::zeros(&[config.hidden_size, num_key_value_heads * head_dim], DataType::Float32, device)?;
        let v_proj = ops_fn::zeros(&[config.hidden_size, num_key_value_heads * head_dim], DataType::Float32, device)?;
        let dense = ops_fn::zeros(&[num_heads * head_dim, config.hidden_size], DataType::Float32, device)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            dense,
            num_heads,
            num_key_value_heads,
            head_dim,
            scale,
            partial_rotary_factor: config.partial_rotary_factor,
        })
    }

    fn forward(&self, hidden_states: &Tensor, _attention_mask: Option<&Tensor>, rope_theta: f32, partial_rotary_factor: f32) -> Result<Tensor> {
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
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;

        let k_reshaped = k_candle
            .reshape(&[batch_size, seq_len, self.num_key_value_heads, self.head_dim])?
            .transpose(1, 2)?;

        let v_reshaped = v_candle
            .reshape(&[batch_size, seq_len, self.num_key_value_heads, self.head_dim])?
            .transpose(1, 2)?;

        // Apply partial RoPE
        let (q_with_rope, k_with_rope) = apply_partial_rope(&q_reshaped, &k_reshaped, seq_len, self.head_dim, rope_theta, partial_rotary_factor)?;

        // Handle GQA
        let num_groups = self.num_heads / self.num_key_value_heads;
        let (k_expanded, v_expanded) = if num_groups > 1 {
            let k_rep = k_with_rope
                .unsqueeze(2)?
                .broadcast_as(&[batch_size, self.num_key_value_heads, num_groups, seq_len, self.head_dim])?
                .reshape(&[batch_size, self.num_heads, seq_len, self.head_dim])?;
            let v_rep = v_reshaped
                .unsqueeze(2)?
                .broadcast_as(&[batch_size, self.num_key_value_heads, num_groups, seq_len, self.head_dim])?
                .reshape(&[batch_size, self.num_heads, seq_len, self.head_dim])?;
            (k_rep, v_rep)
        } else {
            (k_with_rope, v_reshaped)
        };

        let k_t = k_expanded.transpose(2, 3)?;

        let q_contiguous = q_with_rope.contiguous()?;
        let k_contiguous = k_t.contiguous()?;

        let scores = q_contiguous.matmul(&k_contiguous)?;
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

        let v_contiguous = v_expanded.contiguous()?;
        let attn_output = attention_weights.matmul(&v_contiguous)?;

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape(&[batch_size, seq_len, self.num_heads * self.head_dim])?;

        let attn_output = Tensor::from_candle(attn_output);

        let output = ops_fn::matmul(&attn_output, &self.dense)?;

        Ok(output)
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.q_proj = self.q_proj.to_device(device)?;
        self.k_proj = self.k_proj.to_device(device)?;
        self.v_proj = self.v_proj.to_device(device)?;
        self.dense = self.dense.to_device(device)?;
        Ok(())
    }
}

impl PhiMLP {
    fn new(config: &PhiConfig, device: &Device) -> Result<Self> {
        let fc1 = ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?;
        let fc2 = ops_fn::zeros(&[config.intermediate_size, config.hidden_size], DataType::Float32, device)?;

        Ok(Self {
            fc1,
            fc2,
            hidden_act: config.hidden_act.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // First projection with activation
        let intermediate = ops_fn::matmul(hidden_states, &self.fc1)?;

        // Apply activation (Phi typically uses GELU)
        let activated = match self.hidden_act.as_str() {
            "gelu" | "gelu_new" => ops_fn::gelu(&intermediate)?,
            "silu" | "swish" => ops_fn::silu(&intermediate)?,
            _ => return Err(anyhow::anyhow!("Unsupported activation: {}", self.hidden_act)),
        };

        // Second projection
        let output = ops_fn::matmul(&activated, &self.fc2)?;

        Ok(output)
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.fc1 = self.fc1.to_device(device)?;
        self.fc2 = self.fc2.to_device(device)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_model_creation() {
        let config = PhiConfig {
            vocab_size: 1000,
            hidden_size: 128,
            intermediate_size: 512,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            num_key_value_heads: 8,
            ..Default::default()
        };

        let model = PhiModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 1000);
        assert_eq!(model.config().hidden_size(), 128);
        assert_eq!(model.config().num_layers(), 2);
    }

    #[test]
    fn test_phi_forward_pass() {
        let config = PhiConfig {
            vocab_size: 100,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            ..Default::default()
        };

        let model = PhiModelV2::new(config).unwrap();
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
    fn test_phi_generation() {
        let config = PhiConfig {
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            ..Default::default()
        };
        let model = PhiModelV2::new(config).unwrap();
        let gen_config = GenerationConfig {
            max_new_tokens: 5,
            ..Default::default()
        };

        let output = model.generate("Hello", &gen_config).unwrap();
        assert!(!output.is_empty());
    }
}
