//! Qwen Model V2 - Clean implementation using solid abstractions
//!
//! This implements the Qwen/Qwen2 architecture which is used by 15+ models including:
//! - Qwen2, Qwen-7B, Qwen-14B, CodeQwen, Qwen-VL, Qwen-Chat
//! - Uses unified Tensor type from tensor_core
//! - Implements Model trait from model_core
//! - Supports loading via weight_loader_core

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

/// Qwen model configuration using the model_config macro
model_config!(QwenConfig {
    vocab_size: usize = 151936,
    hidden_size: usize = 4096,
    intermediate_size: usize = 11008,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 32,
    num_key_value_heads: usize = 32,
    hidden_act: String = "silu".to_string(),
    max_position_embeddings: usize = 32768,
    initializer_range: f32 = 0.02,
    rms_norm_eps: f32 = 1e-6,
    use_cache: bool = true,
    pad_token_id: i64 = 151643,
    bos_token_id: i64 = 151643,
    eos_token_id: i64 = 151643,
    tie_word_embeddings: bool = false,
    rope_theta: f32 = 1000000.0,
    use_sliding_window: bool = false,
    sliding_window: usize = 4096,
    attention_dropout: f32 = 0.0,
});

impl QwenConfig {
    /// Create QwenConfig from GGUF model configuration
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

/// Main Qwen model implementation
pub struct QwenModelV2 {
    config: QwenConfig,
    device: Device,

    // Model components using unified Tensor type
    embed_tokens: Tensor,
    layers: Vec<QwenLayer>,
    norm: Tensor,
    lm_head: Tensor,
}

/// Qwen transformer layer
pub struct QwenLayer {
    self_attn: QwenAttention,
    mlp: QwenMLP,
    input_layernorm: Tensor,
    post_attention_layernorm: Tensor,
}

/// Qwen attention mechanism
pub struct QwenAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    num_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    scale: f32,
}

/// Qwen MLP (feed-forward network)
pub struct QwenMLP {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
    hidden_act: String,
}

impl Model for QwenModelV2 {
    type Config = QwenConfig;

    fn new(config: QwenConfig) -> Result<Self> {
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
            layers.push(QwenLayer::new(&config, &device)?);
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

    fn from_weights(config: QwenConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;

        // Load weights from the unified weight container
        // Note: embedding weight is not transposed (used for index lookup)
        if let Some(embed_weights) = weights.get("model.embed_tokens.weight") {
            model.embed_tokens = embed_weights.clone();
        }

        if let Some(norm_weights) = weights.get("model.norm.weight") {
            model.norm = norm_weights.clone();
        }

        // lm_head weight needs transpose: [vocab, hidden] -> [hidden, vocab] for matmul
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

                // 2. Apply transformer layers (with RoPE)
                for layer in &self.layers {
                    hidden_states = layer.forward(&hidden_states, attention_mask.as_ref(), self.config.rope_theta)?;
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
            _ => Err(anyhow::anyhow!("Qwen model only supports text and multimodal inputs")),
        }
    }

    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        use crate::tokenizer::Tokenizer;
        use rand::Rng;

        // 1. Tokenize prompt
        let tokenizer = Tokenizer::new();
        let mut tokens: Vec<u32> = tokenizer.encode(prompt);

        // 2. Generation loop
        for _ in 0..config.max_new_tokens {
            // Create input tensor from current tokens
            let tokens_i64: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
            let input_tensor = Tensor::from_i64_slice(&tokens_i64, &[1, tokens.len()], &self.device)?;

            let inputs = ModelInputs::Text {
                input_ids: input_tensor,
                attention_mask: None,
                position_ids: None,
            };

            // 3. Forward pass
            let outputs = self.forward(&inputs)?;

            // 4. Get logits and sample next token
            let logits = match outputs {
                ModelOutputs::Logits { logits, .. } => logits,
                _ => return Err(anyhow::anyhow!("Expected logits output")),
            };

            // Get last token logits
            let logits_candle = logits.to_candle()?;
            let shape = logits_candle.dims();

            // Extract last position logits [batch, seq, vocab] -> [vocab]
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

            // Convert to probabilities and sample
            let logits_vec: Vec<f32> = last_logits.to_vec1()?;

            let next_token = if config.do_sample && config.temperature > 0.0 {
                // Temperature sampling
                let scaled: Vec<f32> = logits_vec.iter()
                    .map(|&x| x / config.temperature)
                    .collect();

                // Softmax
                let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scaled.iter().map(|&x| (x - max_val).exp()).sum();
                let probs: Vec<f32> = scaled.iter()
                    .map(|&x| (x - max_val).exp() / exp_sum)
                    .collect();

                // Sample from distribution
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
                // Greedy sampling
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

            // 5. Check for EOS
            if next_token == config.eos_token_id {
                break;
            }

            // 6. Append token
            tokens.push(next_token);
        }

        // 7. Decode and return
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

impl QwenLayer {
    fn new(config: &QwenConfig, device: &Device) -> Result<Self> {
        let self_attn = QwenAttention::new(config, device)?;
        let mlp = QwenMLP::new(config, device)?;

        let input_layernorm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;
        let post_attention_layernorm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>, rope_theta: f32) -> Result<Tensor> {
        // 1. Pre-attention layer norm
        let normed = ops_fn::layer_norm(hidden_states, &self.input_layernorm, None, 1e-6)?;

        // 2. Self attention (with RoPE)
        let attn_output = self.self_attn.forward(&normed, attention_mask, rope_theta)?;

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

        // Load attention weights (transpose for matmul: [out, in] -> [in, out])
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

        // Load MLP weights (transpose for matmul: [out, in] -> [in, out])
        if let Some(gate_proj) = weights.get(&format!("{}.mlp.gate_proj.weight", prefix)) {
            self.mlp.gate_proj = ops_fn::transpose(gate_proj)?;
        }
        if let Some(up_proj) = weights.get(&format!("{}.mlp.up_proj.weight", prefix)) {
            self.mlp.up_proj = ops_fn::transpose(up_proj)?;
        }
        if let Some(down_proj) = weights.get(&format!("{}.mlp.down_proj.weight", prefix)) {
            self.mlp.down_proj = ops_fn::transpose(down_proj)?;
        }

        // Load layer norm weights (no transpose needed - 1D tensors)
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

    // Compute inverse frequencies: 1 / (theta^(2i/d)) for i in [0, d/2)
    let half_dim = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / rope_theta.powf((2 * i) as f32 / head_dim as f32))
        .collect();

    // Create position indices [0, 1, 2, ..., seq_len-1]
    let positions: Vec<f32> = (0..seq_len).map(|p| p as f32).collect();

    // Compute angles: pos * inv_freq -> [seq_len, half_dim]
    let mut angles = Vec::with_capacity(seq_len * half_dim);
    for pos in &positions {
        for freq in &inv_freq {
            angles.push(pos * freq);
        }
    }

    let angles_tensor = candle_core::Tensor::from_vec(angles, &[seq_len, half_dim], device)?;

    // Compute cos and sin
    let cos = angles_tensor.cos()?;
    let sin = angles_tensor.sin()?;

    // Reshape for broadcasting: [1, 1, seq_len, half_dim]
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

    // Apply RoPE rotation
    let q_half1 = q.narrow(3, 0, half_dim)?;
    let q_half2 = q.narrow(3, half_dim, half_dim)?;
    let k_half1 = k.narrow(3, 0, half_dim)?;
    let k_half2 = k.narrow(3, half_dim, half_dim)?;

    // Apply rotation
    let q_rot1 = (q_half1.broadcast_mul(&cos)? - q_half2.broadcast_mul(&sin)?)?;
    let q_rot2 = (q_half1.broadcast_mul(&sin)? + q_half2.broadcast_mul(&cos)?)?;
    let k_rot1 = (k_half1.broadcast_mul(&cos)? - k_half2.broadcast_mul(&sin)?)?;
    let k_rot2 = (k_half1.broadcast_mul(&sin)? + k_half2.broadcast_mul(&cos)?)?;

    // Concatenate rotated halves
    let q_rotated = candle_core::Tensor::cat(&[&q_rot1, &q_rot2], 3)?;
    let k_rotated = candle_core::Tensor::cat(&[&k_rot1, &k_rot2], 3)?;

    Ok((q_rotated, k_rotated))
}

impl QwenAttention {
    fn new(config: &QwenConfig, device: &Device) -> Result<Self> {
        let num_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads;
        let head_dim = config.hidden_size / num_heads;
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

    fn forward(&self, hidden_states: &Tensor, _attention_mask: Option<&Tensor>, rope_theta: f32) -> Result<Tensor> {
        // Get batch and sequence length from hidden_states shape
        let shape = hidden_states.shape();
        let (batch_size, seq_len, _hidden_size) = if shape.len() == 3 {
            (shape[0], shape[1], shape[2])
        } else if shape.len() == 2 {
            (1, shape[0], shape[1])
        } else {
            return Err(anyhow::anyhow!("Invalid hidden_states shape: {:?}", shape));
        };

        // 1. Project to Q, K, V
        let query_states = ops_fn::matmul(hidden_states, &self.q_proj)?;
        let key_states = ops_fn::matmul(hidden_states, &self.k_proj)?;
        let value_states = ops_fn::matmul(hidden_states, &self.v_proj)?;

        // 2. Reshape for multi-head attention
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

        // 3. Apply RoPE (Rotary Position Embedding) to Q and K
        let (q_with_rope, k_with_rope) = apply_rope(&q_reshaped, &k_reshaped, seq_len, self.head_dim, rope_theta)?;

        // 4. Handle GQA (Grouped Query Attention) - repeat K/V heads to match Q heads
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

        // 5. Scaled dot-product attention
        let k_t = k_expanded.transpose(2, 3)?;

        let q_contiguous = q_with_rope.contiguous()?;
        let k_contiguous = k_t.contiguous()?;

        let scores = q_contiguous.matmul(&k_contiguous)?;
        let scaled_scores = (scores * (self.scale as f64))?;

        // Apply causal mask
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

        // Softmax over last dimension
        let attention_weights = candle_nn::ops::softmax_last_dim(&masked_scores)?;

        // Apply attention to values
        let v_contiguous = v_expanded.contiguous()?;
        let attn_output = attention_weights.matmul(&v_contiguous)?;

        // 6. Reshape back
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape(&[batch_size, seq_len, self.num_heads * self.head_dim])?;

        let attn_output = Tensor::from_candle(attn_output);

        // 7. Output projection
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

impl QwenMLP {
    fn new(config: &QwenConfig, device: &Device) -> Result<Self> {
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
        // 1. Gate and up projections
        let gate_output = ops_fn::matmul(hidden_states, &self.gate_proj)?;
        let up_output = ops_fn::matmul(hidden_states, &self.up_proj)?;

        // 2. Apply activation (SiLU for Qwen)
        let gate_activated = match self.hidden_act.as_str() {
            "silu" | "swish" => ops_fn::silu(&gate_output)?,
            "gelu" => ops_fn::gelu(&gate_output)?,
            _ => return Err(anyhow::anyhow!("Unsupported activation: {}", self.hidden_act)),
        };

        // 3. Element-wise multiplication (gating)
        let gated = ops_fn::mul(&gate_activated, &up_output)?;

        // 4. Down projection
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
    fn test_qwen_model_creation() {
        let config = QwenConfig {
            vocab_size: 1000,
            hidden_size: 128,
            intermediate_size: 512,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            num_key_value_heads: 8,
            ..Default::default()
        };

        let model = QwenModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 1000);
        assert_eq!(model.config().hidden_size(), 128);
        assert_eq!(model.config().num_layers(), 2);
    }

    #[test]
    fn test_qwen_forward_pass() {
        let config = QwenConfig {
            vocab_size: 100,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            ..Default::default()
        };

        let model = QwenModelV2::new(config).unwrap();
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
    fn test_qwen_generation() {
        let config = QwenConfig {
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            ..Default::default()
        };
        let model = QwenModelV2::new(config).unwrap();
        let gen_config = GenerationConfig {
            max_new_tokens: 5,
            ..Default::default()
        };

        let output = model.generate("Hello", &gen_config).unwrap();
        assert!(!output.is_empty());
    }
}
