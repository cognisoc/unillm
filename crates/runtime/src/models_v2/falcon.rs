//! Falcon Model V2 - Clean implementation using solid abstractions
//!
//! This implements the Falcon architecture which includes:
//! - Falcon-7B, Falcon-40B, Falcon-180B models
//! - Uses unified Tensor type from tensor_core
//! - Implements Model trait from model_core
//!
//! Falcon-specific characteristics:
//! - Uses `transformer.h.{i}` layer prefix instead of `model.layers.{i}`
//! - Uses `word_embeddings` instead of `embed_tokens`
//! - Uses `ln_f` instead of `norm` for final layer norm
//! - Parallel attention + MLP (like Phi)
//! - Packed QKV attention (query_key_value)

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

/// Falcon model configuration using the model_config macro
model_config!(FalconConfig {
    vocab_size: usize = 65024,
    hidden_size: usize = 4544,
    intermediate_size: usize = 18176,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 71,
    num_key_value_heads: usize = 1,  // Falcon typically uses MQA (1 kv head) or GQA
    hidden_act: String = "gelu".to_string(),
    max_position_embeddings: usize = 2048,
    initializer_range: f32 = 0.02,
    rms_norm_eps: f32 = 1e-5,
    use_cache: bool = true,
    pad_token_id: i64 = 11,
    bos_token_id: i64 = 11,
    eos_token_id: i64 = 11,
    tie_word_embeddings: bool = false,
    rope_theta: f32 = 10000.0,
    // Falcon specific
    parallel_attn: bool = true,
    bias: bool = false,
    new_decoder_architecture: bool = false,
});

impl FalconConfig {
    /// Create FalconConfig from GGUF model configuration
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

/// Main Falcon model implementation
pub struct FalconModelV2 {
    config: FalconConfig,
    device: Device,

    // Model components using unified Tensor type
    word_embeddings: Tensor,
    h: Vec<FalconDecoderLayer>,
    ln_f: Tensor,
    lm_head: Tensor,
}

/// Falcon transformer layer
pub struct FalconDecoderLayer {
    self_attention: FalconAttention,
    mlp: FalconMLP,
    input_layernorm: Tensor,
    // For non-parallel architectures
    post_attention_layernorm: Option<Tensor>,
    config: FalconConfig,
}

/// Falcon attention mechanism with packed QKV
pub struct FalconAttention {
    query_key_value: Tensor,  // Packed QKV projection
    dense: Tensor,            // Output projection
    num_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    scale: f32,
}

/// Falcon MLP (feed-forward network) - standard 2-layer MLP
pub struct FalconMLP {
    dense_h_to_4h: Tensor,
    dense_4h_to_h: Tensor,
    hidden_act: String,
}

impl Model for FalconModelV2 {
    type Config = FalconConfig;

    fn new(config: FalconConfig) -> Result<Self> {
        let device = Device::CPU;

        let word_embeddings = ops_fn::zeros(
            &[config.vocab_size, config.hidden_size],
            DataType::Float32,
            &device
        )?;

        let ln_f = ops_fn::zeros(
            &[config.hidden_size],
            DataType::Float32,
            &device
        )?;

        let lm_head = if config.tie_word_embeddings {
            word_embeddings.clone()
        } else {
            ops_fn::zeros(
                &[config.hidden_size, config.vocab_size],
                DataType::Float32,
                &device
            )?
        };

        // Create transformer layers
        let mut h = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            h.push(FalconDecoderLayer::new(&config, &device)?);
        }

        Ok(Self {
            config,
            device,
            word_embeddings,
            h,
            ln_f,
            lm_head,
        })
    }

    fn from_weights(config: FalconConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;

        // Load weights from the unified weight container
        // Falcon uses transformer.word_embeddings.weight
        if let Some(embed_weights) = weights.get("transformer.word_embeddings.weight") {
            model.word_embeddings = embed_weights.clone();
        }

        // Falcon uses transformer.ln_f.weight for final layer norm
        if let Some(norm_weights) = weights.get("transformer.ln_f.weight") {
            model.ln_f = norm_weights.clone();
        }

        // lm_head weight needs transpose for matmul: [vocab, hidden] -> [hidden, vocab]
        if let Some(lm_head_weights) = weights.get("lm_head.weight") {
            model.lm_head = ops_fn::transpose(lm_head_weights)?;
        }

        // Load layer weights
        for (i, layer) in model.h.iter_mut().enumerate() {
            layer.load_weights(&weights, i)?;
        }

        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        match inputs {
            ModelInputs::Text { input_ids, attention_mask, .. } => {
                // 1. Token embedding
                let mut hidden_states = ops_fn::embedding(input_ids, &self.word_embeddings)?;

                // 2. Apply transformer layers (with RoPE and parallel attention+MLP)
                for layer in &self.h {
                    hidden_states = layer.forward(&hidden_states, attention_mask.as_ref(), self.config.rope_theta)?;
                }

                // 3. Final layer norm
                hidden_states = ops_fn::layer_norm(&hidden_states, &self.ln_f, None, self.config.rms_norm_eps)?;

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
            _ => Err(anyhow::anyhow!("Falcon model only supports text and multimodal inputs")),
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
        // Calculate approximate memory requirements
        let param_size = self.config.vocab_size * self.config.hidden_size + // embeddings
                        self.config.num_hidden_layers * (
                            // Packed QKV + output projection
                            self.config.hidden_size * (self.config.num_attention_heads + 2 * self.config.num_key_value_heads) * (self.config.hidden_size / self.config.num_attention_heads) +
                            self.config.hidden_size * self.config.hidden_size +
                            // MLP
                            2 * self.config.hidden_size * self.config.intermediate_size
                        );

        let param_bytes = param_size * 4; // float32
        let kv_cache_bytes = 2 * self.config.num_hidden_layers *
                           self.config.max_position_embeddings *
                           self.config.num_key_value_heads *
                           (self.config.hidden_size / self.config.num_attention_heads) * 4;

        MemoryRequirements {
            gpu_memory: param_bytes,
            cpu_memory: param_bytes / 4,
            kv_cache_memory: kv_cache_bytes,
            peak_memory: param_bytes + kv_cache_bytes,
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.word_embeddings = self.word_embeddings.to_device(device)?;
        self.ln_f = self.ln_f.to_device(device)?;
        self.lm_head = self.lm_head.to_device(device)?;

        for layer in &mut self.h {
            layer.to_device(device)?;
        }

        self.device = device.clone();
        Ok(())
    }
}

impl FalconDecoderLayer {
    fn new(config: &FalconConfig, device: &Device) -> Result<Self> {
        let input_layernorm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;

        // Non-parallel architectures have a post-attention layernorm
        let post_attention_layernorm = if !config.parallel_attn {
            Some(ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?)
        } else {
            None
        };

        Ok(Self {
            self_attention: FalconAttention::new(config, device)?,
            mlp: FalconMLP::new(config, device)?,
            input_layernorm,
            post_attention_layernorm,
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>, rope_theta: f32) -> Result<Tensor> {
        let residual = hidden_states.clone();

        // Pre-attention layer norm
        let normed = ops_fn::layer_norm(hidden_states, &self.input_layernorm, None, self.config.rms_norm_eps)?;

        if self.config.parallel_attn {
            // Falcon parallel architecture: attention and MLP computed in parallel
            let attn_output = self.self_attention.forward(&normed, attention_mask, rope_theta)?;
            let mlp_output = self.mlp.forward(&normed)?;

            // Add both outputs to residual
            let combined = ops_fn::add(&attn_output, &mlp_output)?;
            ops_fn::add(&residual, &combined)
        } else {
            // Sequential architecture (Falcon-40B uses this)
            let attn_output = self.self_attention.forward(&normed, attention_mask, rope_theta)?;
            let hidden_states = ops_fn::add(&residual, &attn_output)?;

            // Post-attention layer norm
            if let Some(post_ln) = &self.post_attention_layernorm {
                let normed = ops_fn::layer_norm(&hidden_states, post_ln, None, self.config.rms_norm_eps)?;
                let mlp_output = self.mlp.forward(&normed)?;
                ops_fn::add(&hidden_states, &mlp_output)
            } else {
                // Fallback if post_attention_layernorm is not available
                let mlp_output = self.mlp.forward(&hidden_states)?;
                ops_fn::add(&hidden_states, &mlp_output)
            }
        }
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("transformer.h.{}", layer_idx);

        // Load input layernorm
        if let Some(ln) = weights.get(&format!("{}.input_layernorm.weight", prefix)) {
            self.input_layernorm = ln.clone();
        }

        // Load post-attention layernorm for sequential architectures
        if let Some(post_ln) = weights.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
            self.post_attention_layernorm = Some(post_ln.clone());
        }

        // Load attention weights
        self.self_attention.load_weights(weights, layer_idx)?;

        // Load MLP weights
        self.mlp.load_weights(weights, layer_idx)?;

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.input_layernorm = self.input_layernorm.to_device(device)?;
        if let Some(post_ln) = &self.post_attention_layernorm {
            self.post_attention_layernorm = Some(post_ln.to_device(device)?);
        }
        self.self_attention.to_device(device)?;
        self.mlp.to_device(device)?;
        Ok(())
    }
}

/// Apply Rotary Position Embedding (RoPE) to Q and K tensors
/// Input shape: [batch, heads, seq, head_dim]
/// Returns tensors with same shape but with positional information encoded
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
    // Split q and k into two halves along head_dim
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

impl FalconAttention {
    fn new(config: &FalconConfig, device: &Device) -> Result<Self> {
        let num_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads;
        let head_dim = config.hidden_size / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Packed QKV: [hidden_size, (num_heads + 2*num_kv_heads) * head_dim]
        let qkv_size = (num_heads + 2 * num_key_value_heads) * head_dim;
        let query_key_value = ops_fn::zeros(
            &[config.hidden_size, qkv_size],
            DataType::Float32,
            device
        )?;

        let dense = ops_fn::zeros(
            &[num_heads * head_dim, config.hidden_size],
            DataType::Float32,
            device
        )?;

        Ok(Self {
            query_key_value,
            dense,
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

        // 1. Compute packed QKV projection
        // [batch, seq, hidden] @ [hidden, qkv_size] -> [batch, seq, qkv_size]
        let qkv = ops_fn::matmul(hidden_states, &self.query_key_value)?;
        let qkv_candle = qkv.to_candle()?;

        // 2. Split QKV
        // qkv_size = (num_heads + 2*num_kv_heads) * head_dim
        let q_size = self.num_heads * self.head_dim;
        let kv_size = self.num_key_value_heads * self.head_dim;

        // Split along the last dimension
        let query_states = qkv_candle.narrow(2, 0, q_size)?;
        let key_states = qkv_candle.narrow(2, q_size, kv_size)?;
        let value_states = qkv_candle.narrow(2, q_size + kv_size, kv_size)?;

        // 3. Reshape for multi-head attention
        // Q: [batch, seq, num_heads * head_dim] -> [batch, num_heads, seq, head_dim]
        // K: [batch, seq, num_kv_heads * head_dim] -> [batch, num_kv_heads, seq, head_dim]
        let q_reshaped = query_states
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;

        let k_reshaped = key_states
            .reshape(&[batch_size, seq_len, self.num_key_value_heads, self.head_dim])?
            .transpose(1, 2)?;

        let v_reshaped = value_states
            .reshape(&[batch_size, seq_len, self.num_key_value_heads, self.head_dim])?
            .transpose(1, 2)?;

        // 4. Apply RoPE (Rotary Position Embedding) to Q and K
        let (q_with_rope, k_with_rope) = apply_rope(&q_reshaped, &k_reshaped, seq_len, self.head_dim, rope_theta)?;

        // 5. Handle GQA (Grouped Query Attention) / MQA - repeat K/V heads to match Q heads
        let num_groups = self.num_heads / self.num_key_value_heads;
        let (k_expanded, v_expanded) = if num_groups > 1 {
            // Repeat K and V along the head dimension
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

        // 6. Scaled dot-product attention
        // scores = Q @ K^T / sqrt(head_dim)
        let k_t = k_expanded.transpose(2, 3)?;

        let q_contiguous = q_with_rope.contiguous()?;
        let k_contiguous = k_t.contiguous()?;

        let scores = q_contiguous.matmul(&k_contiguous)?;
        let scaled_scores = (scores * (self.scale as f64))?;

        // Apply causal mask: prevent attending to future positions
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

        // 7. Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, heads * head_dim]
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape(&[batch_size, seq_len, self.num_heads * self.head_dim])?;

        let attn_output = Tensor::from_candle(attn_output);

        // 8. Output projection
        let output = ops_fn::matmul(&attn_output, &self.dense)?;

        Ok(output)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("transformer.h.{}.self_attention", layer_idx);

        // Falcon uses packed query_key_value weight
        // Transpose for matmul: [out, in] -> [in, out]
        if let Some(qkv) = weights.get(&format!("{}.query_key_value.weight", prefix)) {
            self.query_key_value = ops_fn::transpose(qkv)?;
        }

        // Output projection (dense)
        if let Some(dense) = weights.get(&format!("{}.dense.weight", prefix)) {
            self.dense = ops_fn::transpose(dense)?;
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.query_key_value = self.query_key_value.to_device(device)?;
        self.dense = self.dense.to_device(device)?;
        Ok(())
    }
}

impl FalconMLP {
    fn new(config: &FalconConfig, device: &Device) -> Result<Self> {
        let dense_h_to_4h = ops_fn::zeros(
            &[config.hidden_size, config.intermediate_size],
            DataType::Float32,
            device
        )?;
        let dense_4h_to_h = ops_fn::zeros(
            &[config.intermediate_size, config.hidden_size],
            DataType::Float32,
            device
        )?;

        Ok(Self {
            dense_h_to_4h,
            dense_4h_to_h,
            hidden_act: config.hidden_act.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // First projection with activation
        let intermediate = ops_fn::matmul(hidden_states, &self.dense_h_to_4h)?;

        // Apply activation (Falcon uses GELU)
        let activated = match self.hidden_act.as_str() {
            "gelu" | "gelu_new" => ops_fn::gelu(&intermediate)?,
            "silu" | "swish" => ops_fn::silu(&intermediate)?,
            _ => return Err(anyhow::anyhow!("Unsupported activation: {}", self.hidden_act)),
        };

        // Second projection
        let output = ops_fn::matmul(&activated, &self.dense_4h_to_h)?;

        Ok(output)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("transformer.h.{}.mlp", layer_idx);

        // Transpose for matmul: [out, in] -> [in, out]
        if let Some(w) = weights.get(&format!("{}.dense_h_to_4h.weight", prefix)) {
            self.dense_h_to_4h = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.dense_4h_to_h.weight", prefix)) {
            self.dense_4h_to_h = ops_fn::transpose(w)?;
        }

        Ok(())
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
    fn test_falcon_model_creation() {
        let config = FalconConfig {
            vocab_size: 1000,
            hidden_size: 128,
            intermediate_size: 512,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            num_key_value_heads: 1, // MQA
            ..Default::default()
        };

        let model = FalconModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 1000);
        assert_eq!(model.config().hidden_size(), 128);
        assert_eq!(model.config().num_layers(), 2);
    }

    #[test]
    fn test_falcon_forward_pass() {
        let config = FalconConfig {
            vocab_size: 100,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 8,
            num_key_value_heads: 1, // MQA
            ..Default::default()
        };

        let model = FalconModelV2::new(config).unwrap();
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
    fn test_falcon_gqa() {
        // Test with Grouped Query Attention (GQA)
        let config = FalconConfig {
            vocab_size: 100,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 8,
            num_key_value_heads: 2, // GQA with 4 groups
            ..Default::default()
        };

        let model = FalconModelV2::new(config).unwrap();
        let input_ids = ops_fn::zeros(&[1, 4], DataType::Int64, &Device::CPU).unwrap();
        let inputs = ModelInputs::text(input_ids);

        let outputs = model.forward(&inputs).unwrap();
        match outputs {
            ModelOutputs::Logits { logits, .. } => {
                assert_eq!(logits.shape(), &[1, 4, 100]);
            }
            _ => panic!("Expected logits output"),
        }
    }

    #[test]
    fn test_falcon_sequential_architecture() {
        // Test with sequential (non-parallel) attention+MLP
        let config = FalconConfig {
            vocab_size: 100,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 8,
            num_key_value_heads: 1,
            parallel_attn: false, // Sequential architecture
            ..Default::default()
        };

        let model = FalconModelV2::new(config).unwrap();
        let input_ids = ops_fn::zeros(&[1, 4], DataType::Int64, &Device::CPU).unwrap();
        let inputs = ModelInputs::text(input_ids);

        let outputs = model.forward(&inputs).unwrap();
        match outputs {
            ModelOutputs::Logits { logits, .. } => {
                assert_eq!(logits.shape(), &[1, 4, 100]);
            }
            _ => panic!("Expected logits output"),
        }
    }

    #[test]
    fn test_falcon_generation() {
        let config = FalconConfig {
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 8,
            num_key_value_heads: 1,
            ..Default::default()
        };
        let model = FalconModelV2::new(config).unwrap();
        let gen_config = GenerationConfig {
            max_new_tokens: 5,
            ..Default::default()
        };

        let output = model.generate("Hello", &gen_config).unwrap();
        assert!(!output.is_empty());
    }
}
