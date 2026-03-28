//! Baichuan Model V2 - Clean implementation using solid abstractions
//!
//! This implements the Baichuan architecture including:
//! - Baichuan-7B, Baichuan-13B, Baichuan2-7B, Baichuan2-13B
//!
//! Key differences from LLaMA:
//! - Packed QKV attention (W_pack weight) - single weight matrix for Q, K, V
//! - ALiBi (Attention with Linear Biases) instead of RoPE
//! - Standard transformer structure otherwise

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

/// Baichuan model configuration using the model_config macro
model_config!(BaichuanConfig {
    vocab_size: usize = 64000,
    hidden_size: usize = 4096,
    intermediate_size: usize = 11008,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 32,
    num_key_value_heads: usize = 32,
    hidden_act: String = "silu".to_string(),
    max_position_embeddings: usize = 4096,
    initializer_range: f32 = 0.02,
    rms_norm_eps: f32 = 1e-6,
    use_cache: bool = true,
    pad_token_id: i64 = 0,
    bos_token_id: i64 = 1,
    eos_token_id: i64 = 2,
    tie_word_embeddings: bool = false,
    // Baichuan specific - using ALiBi instead of RoPE
    use_alibi: bool = true,
    model_max_length: usize = 4096,
    z_loss_weight: f32 = 0.0,
});

impl BaichuanConfig {
    /// Create BaichuanConfig from GGUF model configuration
    pub fn from_gguf_config(gguf: &crate::weight_loader_core::GGUFModelConfig) -> Self {
        Self {
            vocab_size: gguf.vocab_size,
            hidden_size: gguf.hidden_size,
            intermediate_size: gguf.intermediate_size,
            num_hidden_layers: gguf.num_hidden_layers,
            num_attention_heads: gguf.num_attention_heads,
            num_key_value_heads: gguf.num_key_value_heads,
            rms_norm_eps: gguf.rms_norm_eps,
            max_position_embeddings: gguf.max_position_embeddings,
            ..Default::default()
        }
    }
}

/// Main Baichuan model implementation
pub struct BaichuanModelV2 {
    config: BaichuanConfig,
    device: Device,

    // Model components using unified Tensor type
    embed_tokens: Tensor,
    layers: Vec<BaichuanLayer>,
    norm: Tensor,
    lm_head: Tensor,
}

/// Baichuan transformer layer
pub struct BaichuanLayer {
    self_attn: BaichuanAttention,
    mlp: BaichuanMLP,
    input_layernorm: Tensor,
    post_attention_layernorm: Tensor,
}

/// Baichuan attention mechanism with packed QKV and ALiBi
pub struct BaichuanAttention {
    w_pack: Tensor, // Packed Q, K, V weights [hidden_size, 3 * hidden_size]
    o_proj: Tensor,
    num_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    scale: f32,
}

/// Baichuan MLP (feed-forward network)
pub struct BaichuanMLP {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
    hidden_act: String,
}

impl Model for BaichuanModelV2 {
    type Config = BaichuanConfig;

    fn new(config: Self::Config) -> Result<Self> {
        let device = Device::CPU;

        // Create model tensors with correct shapes
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
            layers.push(BaichuanLayer::new(&config, &device)?);
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

    fn from_weights(config: Self::Config, weights: ModelWeights) -> Result<Self> {
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

                // 2. Apply transformer layers (with ALiBi)
                for layer in &self.layers {
                    hidden_states = layer.forward(&hidden_states, attention_mask.as_ref())?;
                }

                // 3. Final layer norm
                hidden_states = ops_fn::layer_norm(&hidden_states, &self.norm, None, self.config.rms_norm_eps)?;

                // 4. Language modeling head
                let logits = ops_fn::matmul(&hidden_states, &self.lm_head)?;

                Ok(ModelOutputs::Logits {
                    logits,
                    hidden_states: None, // Don't return hidden states to save memory
                })
            }
            ModelInputs::Multimodal { input_ids, .. } => {
                // For multimodal inputs, just process text part for now
                let text_inputs = ModelInputs::Text {
                    input_ids: input_ids.clone(),
                    attention_mask: None,
                    position_ids: None,
                };
                self.forward(&text_inputs)
            }
            _ => Err(anyhow::anyhow!("Baichuan model only supports text and multimodal inputs")),
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
                            3 * self.config.hidden_size * self.config.hidden_size + // packed QKV attention
                            self.config.hidden_size * self.config.hidden_size + // o_proj
                            3 * self.config.hidden_size * self.config.intermediate_size // MLP
                        );

        let param_bytes = param_size * 4; // float32
        let kv_cache_bytes = 2 * self.config.num_hidden_layers *
                           self.config.max_position_embeddings *
                           self.config.hidden_size * 4; // K and V caches

        MemoryRequirements {
            gpu_memory: param_bytes,
            cpu_memory: param_bytes / 4, // Reduced for CPU
            kv_cache_memory: kv_cache_bytes,
            peak_memory: param_bytes + kv_cache_bytes,
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        // Move all tensors to the specified device
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

impl BaichuanLayer {
    fn new(config: &BaichuanConfig, device: &Device) -> Result<Self> {
        let self_attn = BaichuanAttention::new(config, device)?;
        let mlp = BaichuanMLP::new(config, device)?;

        let input_layernorm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;
        let post_attention_layernorm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // 1. Pre-attention layer norm
        let normed = ops_fn::layer_norm(hidden_states, &self.input_layernorm, None, 1e-6)?;

        // 2. Self attention (with ALiBi)
        let attn_output = self.self_attn.forward(&normed, attention_mask)?;

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

        // Load packed QKV attention weight (W_pack) - transpose for matmul: [out, in] -> [in, out]
        if let Some(w_pack) = weights.get(&format!("{}.self_attn.W_pack.weight", prefix)) {
            self.self_attn.w_pack = ops_fn::transpose(w_pack)?;
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

/// Compute ALiBi (Attention with Linear Biases) slopes for each head
/// ALiBi slopes follow a geometric sequence: 2^(-8/n), 2^(-16/n), ..., 2^(-8)
/// where n is the number of attention heads
fn compute_alibi_slopes(num_heads: usize) -> Vec<f32> {
    // For power of 2 heads, use standard geometric sequence
    // For non-power of 2, use closest power of 2 and interpolate
    let closest_power_of_2 = 2_usize.pow((num_heads as f32).log2().floor() as u32);
    let base = 2.0_f32.powf(-8.0 / closest_power_of_2 as f32);

    let mut slopes = Vec::with_capacity(num_heads);

    if num_heads == closest_power_of_2 {
        // Standard case: power of 2 heads
        for i in 1..=num_heads {
            slopes.push(base.powi(i as i32));
        }
    } else {
        // Non-power of 2: use extra slopes from double the base
        let extra_base = 2.0_f32.powf(-8.0 / (2 * closest_power_of_2) as f32);
        let num_remaining = num_heads - closest_power_of_2;

        // First add slopes from the larger base
        for i in 1..=closest_power_of_2 {
            slopes.push(base.powi(i as i32));
        }

        // Then add interleaved slopes from the extra base
        for i in 1..=num_remaining {
            slopes.push(extra_base.powi((2 * i - 1) as i32));
        }
    }

    slopes
}

/// Build ALiBi bias matrix for attention scores
/// bias[i,j] = -abs(i - j) * slope for each head
fn build_alibi_bias(
    seq_len: usize,
    num_heads: usize,
    device: &candle_core::Device,
) -> Result<candle_core::Tensor> {
    let slopes = compute_alibi_slopes(num_heads);

    // Create position difference matrix: -|i - j| for all positions
    let mut bias_data = Vec::with_capacity(num_heads * seq_len * seq_len);

    for (_head_idx, &slope) in slopes.iter().enumerate() {
        for i in 0..seq_len {
            for j in 0..seq_len {
                // ALiBi bias: -|query_pos - key_pos| * slope
                let distance = (i as i32 - j as i32).abs() as f32;
                bias_data.push(-distance * slope);
            }
        }
    }

    // Shape: [num_heads, seq_len, seq_len]
    let bias = candle_core::Tensor::from_vec(
        bias_data,
        &[num_heads, seq_len, seq_len],
        device
    )?;

    // Expand to [1, num_heads, seq_len, seq_len] for broadcasting
    Ok(bias.unsqueeze(0)?)
}

impl BaichuanAttention {
    fn new(config: &BaichuanConfig, device: &Device) -> Result<Self> {
        let num_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads;
        let head_dim = config.hidden_size / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Packed QKV: [hidden_size, 3 * hidden_size] for Q, K, V combined
        let total_kv_size = num_key_value_heads * head_dim;
        let total_q_size = num_heads * head_dim;
        let w_pack = ops_fn::zeros(
            &[config.hidden_size, total_q_size + 2 * total_kv_size],
            DataType::Float32,
            device
        )?;

        let o_proj = ops_fn::zeros(
            &[num_heads * head_dim, config.hidden_size],
            DataType::Float32,
            device
        )?;

        Ok(Self {
            w_pack,
            o_proj,
            num_heads,
            num_key_value_heads,
            head_dim,
            scale,
        })
    }

    fn forward(&self, hidden_states: &Tensor, _attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Get batch and sequence length from hidden_states shape
        let shape = hidden_states.shape();
        let (batch_size, seq_len, _hidden_size) = if shape.len() == 3 {
            (shape[0], shape[1], shape[2])
        } else if shape.len() == 2 {
            (1, shape[0], shape[1])
        } else {
            return Err(anyhow::anyhow!("Invalid hidden_states shape: {:?}", shape));
        };

        // 1. Packed QKV projection
        // [batch, seq, hidden] @ [hidden, q_size + 2*kv_size] -> [batch, seq, q_size + 2*kv_size]
        let qkv = ops_fn::matmul(hidden_states, &self.w_pack)?;
        let qkv_candle = qkv.to_candle()?;

        // 2. Split into Q, K, V
        let q_size = self.num_heads * self.head_dim;
        let kv_size = self.num_key_value_heads * self.head_dim;

        let query_states = qkv_candle.narrow(2, 0, q_size)?;
        let key_states = qkv_candle.narrow(2, q_size, kv_size)?;
        let value_states = qkv_candle.narrow(2, q_size + kv_size, kv_size)?;

        // 3. Reshape for multi-head attention
        // Q: [batch, seq, num_heads * head_dim] -> [batch, num_heads, seq, head_dim]
        // K, V: [batch, seq, num_kv_heads * head_dim] -> [batch, num_kv_heads, seq, head_dim]
        let q_reshaped = query_states
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?; // [batch, heads, seq, head_dim]

        let k_reshaped = key_states
            .reshape(&[batch_size, seq_len, self.num_key_value_heads, self.head_dim])?
            .transpose(1, 2)?; // [batch, kv_heads, seq, head_dim]

        let v_reshaped = value_states
            .reshape(&[batch_size, seq_len, self.num_key_value_heads, self.head_dim])?
            .transpose(1, 2)?; // [batch, kv_heads, seq, head_dim]

        // 4. Handle GQA (Grouped Query Attention) - repeat K/V heads to match Q heads
        let num_groups = self.num_heads / self.num_key_value_heads;
        let (k_expanded, v_expanded) = if num_groups > 1 {
            // Repeat K and V along the head dimension
            // [batch, kv_heads, seq, head_dim] -> [batch, num_heads, seq, head_dim]
            let k_rep = k_reshaped
                .unsqueeze(2)? // [batch, kv_heads, 1, seq, head_dim]
                .broadcast_as(&[batch_size, self.num_key_value_heads, num_groups, seq_len, self.head_dim])?
                .reshape(&[batch_size, self.num_heads, seq_len, self.head_dim])?;
            let v_rep = v_reshaped
                .unsqueeze(2)?
                .broadcast_as(&[batch_size, self.num_key_value_heads, num_groups, seq_len, self.head_dim])?
                .reshape(&[batch_size, self.num_heads, seq_len, self.head_dim])?;
            (k_rep, v_rep)
        } else {
            (k_reshaped, v_reshaped)
        };

        // 5. Scaled dot-product attention with ALiBi
        // scores = Q @ K^T / sqrt(head_dim)
        // [batch, heads, seq, head_dim] @ [batch, heads, head_dim, seq] -> [batch, heads, seq, seq]
        let k_t = k_expanded.transpose(2, 3)?; // [batch, heads, head_dim, seq]

        // Make tensors contiguous for matmul (required by candle)
        let q_contiguous = q_reshaped.contiguous()?;
        let k_contiguous = k_t.contiguous()?;

        let scores = q_contiguous.matmul(&k_contiguous)?;
        let scaled_scores = (scores * (self.scale as f64))?;

        // Build ALiBi bias and add to scores
        let device = scaled_scores.device();
        let alibi_bias = build_alibi_bias(seq_len, self.num_heads, device)?;
        let scores_with_alibi = scaled_scores.broadcast_add(&alibi_bias)?;

        // Apply causal mask: prevent attending to future positions
        let causal_mask = {
            let mut mask_data = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    if j > i {
                        // Future position - mask out with large negative value
                        mask_data[i * seq_len + j] = f32::NEG_INFINITY;
                    }
                }
            }
            candle_core::Tensor::from_vec(mask_data, &[1, 1, seq_len, seq_len], device)?
        };

        // Add causal mask to scores (broadcast over batch and heads)
        let masked_scores = scores_with_alibi.broadcast_add(&causal_mask)?;

        // Softmax over last dimension
        let attention_weights = candle_nn::ops::softmax_last_dim(&masked_scores)?;

        // Apply attention to values
        // [batch, heads, seq, seq] @ [batch, heads, seq, head_dim] -> [batch, heads, seq, head_dim]
        let v_contiguous = v_expanded.contiguous()?;
        let attn_output = attention_weights.matmul(&v_contiguous)?;

        // 6. Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, heads * head_dim]
        let attn_output = attn_output
            .transpose(1, 2)? // [batch, seq, heads, head_dim]
            .reshape(&[batch_size, seq_len, self.num_heads * self.head_dim])?;

        let attn_output = Tensor::from_candle(attn_output);

        // 7. Output projection
        let output = ops_fn::matmul(&attn_output, &self.o_proj)?;

        Ok(output)
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.w_pack = self.w_pack.to_device(device)?;
        self.o_proj = self.o_proj.to_device(device)?;
        Ok(())
    }
}

impl BaichuanMLP {
    fn new(config: &BaichuanConfig, device: &Device) -> Result<Self> {
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

        // 2. Apply activation (SiLU for Baichuan)
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
    fn test_baichuan_model_creation() {
        let config = BaichuanConfig {
            vocab_size: 1000,
            hidden_size: 128,
            intermediate_size: 512,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            num_key_value_heads: 8,
            ..Default::default()
        };

        let model = BaichuanModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 1000);
        assert_eq!(model.config().hidden_size(), 128);
        assert_eq!(model.config().num_layers(), 2);
    }

    #[test]
    fn test_baichuan_forward_pass() {
        let config = BaichuanConfig {
            vocab_size: 100,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            ..Default::default()
        };

        let model = BaichuanModelV2::new(config).unwrap();
        let input_ids = ops_fn::zeros(&[2, 8], DataType::Int64, &Device::CPU).unwrap();
        let inputs = ModelInputs::text(input_ids);

        let outputs = model.forward(&inputs).unwrap();
        match outputs {
            ModelOutputs::Logits { logits, .. } => {
                assert_eq!(logits.shape(), &[2, 8, 100]); // batch, seq, vocab
            }
            _ => panic!("Expected logits output"),
        }
    }

    #[test]
    fn test_alibi_slopes() {
        // Test power of 2 heads
        let slopes_8 = compute_alibi_slopes(8);
        assert_eq!(slopes_8.len(), 8);
        // First slope should be 2^(-1) = 0.5
        assert!((slopes_8[0] - 0.5).abs() < 1e-6);

        // Test non-power of 2 heads
        let slopes_12 = compute_alibi_slopes(12);
        assert_eq!(slopes_12.len(), 12);
    }

    #[test]
    fn test_baichuan_generation() {
        // Use a small config for testing
        // vocab_size must be >= basic tokenizer's vocab (~200 tokens)
        let config = BaichuanConfig {
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            ..Default::default()
        };
        let model = BaichuanModelV2::new(config).unwrap();
        let gen_config = GenerationConfig {
            max_new_tokens: 5, // Generate only a few tokens for testing
            ..Default::default()
        };

        let output = model.generate("Hello", &gen_config).unwrap();
        assert!(!output.is_empty());
    }
}
