//! ChatGLM Model V2 - Clean implementation using solid abstractions
//!
//! This implements the ChatGLM architecture including:
//! - ChatGLM-6B, ChatGLM2-6B, ChatGLM3-6B, GLM-4
//!
//! ChatGLM unique characteristics:
//! - Structure: embedding -> encoder -> output_layer
//! - Uses `transformer.embedding.word_embeddings.weight`
//! - Uses `transformer.encoder.layers.{i}`
//! - Uses `transformer.output_layer.weight`
//! - Packed QKV attention (query_key_value combined)
//! - SwiGLU activation in MLP

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

/// ChatGLM model configuration using the model_config macro
model_config!(ChatGLMConfig {
    vocab_size: usize = 65024,
    hidden_size: usize = 4096,
    intermediate_size: usize = 13696,
    num_hidden_layers: usize = 28,
    num_attention_heads: usize = 32,
    num_key_value_heads: usize = 2,
    hidden_act: String = "swiglu".to_string(),
    max_position_embeddings: usize = 8192,
    initializer_range: f32 = 0.02,
    rms_norm_eps: f32 = 1e-5,
    use_cache: bool = true,
    pad_token_id: i64 = 0,
    bos_token_id: i64 = 1,
    eos_token_id: i64 = 2,
    tie_word_embeddings: bool = false,
    rope_theta: f32 = 10000.0,
    attention_dropout: f32 = 0.0,
    // ChatGLM specific
    add_bias_linear: bool = false,
    add_qkv_bias: bool = true,
    apply_residual_connection_post_layernorm: bool = false,
    kv_channels: usize = 128,
    multi_query_attention: bool = true,
});

impl ChatGLMConfig {
    /// Create ChatGLMConfig from GGUF model configuration
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
            kv_channels: gguf.head_dim,
            ..Default::default()
        }
    }
}

/// Main ChatGLM model implementation
pub struct ChatGLMModelV2 {
    config: ChatGLMConfig,
    device: Device,

    // Model components using unified Tensor type
    // ChatGLM uses: transformer.embedding.word_embeddings, transformer.encoder, transformer.output_layer
    word_embeddings: Tensor,
    layers: Vec<ChatGLMLayer>,
    final_layernorm: Tensor,
    output_layer: Tensor,
}

/// ChatGLM transformer layer
pub struct ChatGLMLayer {
    input_layernorm: Tensor,
    self_attention: ChatGLMAttention,
    post_attention_layernorm: Tensor,
    mlp: ChatGLMMLP,
}

/// ChatGLM attention mechanism with packed QKV
pub struct ChatGLMAttention {
    // ChatGLM uses packed QKV: query_key_value combined weight
    query_key_value: Tensor,
    // Optional biases for QKV (ChatGLM often uses bias)
    qkv_bias: Option<Tensor>,
    // Output projection
    dense: Tensor,
    num_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    scale: f32,
}

/// ChatGLM MLP with SwiGLU activation
pub struct ChatGLMMLP {
    // ChatGLM uses dense_h_to_4h (combined gate+up) and dense_4h_to_h
    dense_h_to_4h: Tensor,
    dense_4h_to_h: Tensor,
    hidden_act: String,
}

impl Model for ChatGLMModelV2 {
    type Config = ChatGLMConfig;

    fn new(config: ChatGLMConfig) -> Result<Self> {
        let device = Device::CPU;

        // Create embedding layer
        let word_embeddings = ops_fn::zeros(
            &[config.vocab_size, config.hidden_size],
            DataType::Float32,
            &device
        )?;

        // Final layer norm
        let final_layernorm = ops_fn::zeros(
            &[config.hidden_size],
            DataType::Float32,
            &device
        )?;

        // Output layer (lm_head equivalent)
        let output_layer = ops_fn::zeros(
            &[config.hidden_size, config.vocab_size],
            DataType::Float32,
            &device
        )?;

        // Create transformer layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(ChatGLMLayer::new(&config, &device)?);
        }

        Ok(Self {
            config,
            device,
            word_embeddings,
            layers,
            final_layernorm,
            output_layer,
        })
    }

    fn from_weights(config: ChatGLMConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;

        // Load embedding weights (ChatGLM uses transformer.embedding.word_embeddings.weight)
        if let Some(embed_weights) = weights.get("transformer.embedding.word_embeddings.weight") {
            model.word_embeddings = embed_weights.clone();
        }

        // Load final layer norm
        if let Some(ln_weights) = weights.get("transformer.encoder.final_layernorm.weight") {
            model.final_layernorm = ln_weights.clone();
        }

        // Load output layer (transpose for matmul: [vocab, hidden] -> [hidden, vocab])
        if let Some(output_weights) = weights.get("transformer.output_layer.weight") {
            model.output_layer = ops_fn::transpose(output_weights)?;
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
                let mut hidden_states = ops_fn::embedding(input_ids, &self.word_embeddings)?;

                // 2. Apply transformer layers (with RoPE)
                for layer in &self.layers {
                    hidden_states = layer.forward(&hidden_states, attention_mask.as_ref(), self.config.rope_theta)?;
                }

                // 3. Final layer norm
                hidden_states = ops_fn::layer_norm(&hidden_states, &self.final_layernorm, None, self.config.rms_norm_eps)?;

                // 4. Output layer (language modeling head)
                let logits = ops_fn::matmul(&hidden_states, &self.output_layer)?;

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
            _ => Err(anyhow::anyhow!("ChatGLM model only supports text and multimodal inputs")),
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
                            // Packed QKV + dense projection
                            (self.config.num_attention_heads + 2 * self.config.num_key_value_heads) *
                                (self.config.hidden_size / self.config.num_attention_heads) * self.config.hidden_size +
                            self.config.hidden_size * self.config.hidden_size +
                            // MLP
                            2 * self.config.hidden_size * self.config.intermediate_size
                        );

        let param_bytes = param_size * 4; // float32
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
        self.word_embeddings = self.word_embeddings.to_device(device)?;
        self.final_layernorm = self.final_layernorm.to_device(device)?;
        self.output_layer = self.output_layer.to_device(device)?;

        for layer in &mut self.layers {
            layer.to_device(device)?;
        }

        self.device = device.clone();
        Ok(())
    }
}

impl ChatGLMLayer {
    fn new(config: &ChatGLMConfig, device: &Device) -> Result<Self> {
        let input_layernorm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;
        let post_attention_layernorm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;
        let self_attention = ChatGLMAttention::new(config, device)?;
        let mlp = ChatGLMMLP::new(config, device)?;

        Ok(Self {
            input_layernorm,
            self_attention,
            post_attention_layernorm,
            mlp,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>, rope_theta: f32) -> Result<Tensor> {
        // 1. Pre-attention layer norm
        let normed = ops_fn::layer_norm(hidden_states, &self.input_layernorm, None, 1e-5)?;

        // 2. Self attention (with RoPE and packed QKV)
        let attn_output = self.self_attention.forward(&normed, attention_mask, rope_theta)?;

        // 3. Residual connection
        let hidden_states = ops_fn::add(hidden_states, &attn_output)?;

        // 4. Pre-MLP layer norm
        let normed = ops_fn::layer_norm(&hidden_states, &self.post_attention_layernorm, None, 1e-5)?;

        // 5. MLP with SwiGLU
        let mlp_output = self.mlp.forward(&normed)?;

        // 6. Residual connection
        let output = ops_fn::add(&hidden_states, &mlp_output)?;

        Ok(output)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("transformer.encoder.layers.{}", layer_idx);

        // Load layer norms
        if let Some(w) = weights.get(&format!("{}.input_layernorm.weight", prefix)) {
            self.input_layernorm = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
            self.post_attention_layernorm = w.clone();
        }

        // Load attention weights
        self.self_attention.load_weights(weights, layer_idx)?;

        // Load MLP weights
        self.mlp.load_weights(weights, layer_idx)?;

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.input_layernorm = self.input_layernorm.to_device(device)?;
        self.post_attention_layernorm = self.post_attention_layernorm.to_device(device)?;
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

impl ChatGLMAttention {
    fn new(config: &ChatGLMConfig, device: &Device) -> Result<Self> {
        let num_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads;
        let head_dim = config.hidden_size / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // ChatGLM uses packed QKV: [hidden, (num_heads + 2*num_kv_heads) * head_dim]
        let qkv_size = (num_heads + 2 * num_key_value_heads) * head_dim;
        let query_key_value = ops_fn::zeros(
            &[config.hidden_size, qkv_size],
            DataType::Float32,
            device
        )?;

        // Optional QKV bias
        let qkv_bias = if config.add_qkv_bias {
            Some(ops_fn::zeros(&[qkv_size], DataType::Float32, device)?)
        } else {
            None
        };

        // Output projection (dense)
        let dense = ops_fn::zeros(
            &[num_heads * head_dim, config.hidden_size],
            DataType::Float32,
            device
        )?;

        Ok(Self {
            query_key_value,
            qkv_bias,
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
        let qkv = ops_fn::matmul(hidden_states, &self.query_key_value)?;

        // Add bias if present
        let qkv = if let Some(ref bias) = self.qkv_bias {
            ops_fn::add(&qkv, bias)?
        } else {
            qkv
        };

        // 2. Split QKV into Q, K, V
        let qkv_candle = qkv.to_candle()?;

        // QKV layout: [batch, seq, (num_heads + 2*num_kv_heads) * head_dim]
        // Split into: Q [batch, seq, num_heads * head_dim]
        //             K [batch, seq, num_kv_heads * head_dim]
        //             V [batch, seq, num_kv_heads * head_dim]
        let q_size = self.num_heads * self.head_dim;
        let kv_size = self.num_key_value_heads * self.head_dim;

        let q = qkv_candle.narrow(2, 0, q_size)?;
        let k = qkv_candle.narrow(2, q_size, kv_size)?;
        let v = qkv_candle.narrow(2, q_size + kv_size, kv_size)?;

        // 3. Reshape for multi-head attention
        // Q: [batch, seq, num_heads * head_dim] -> [batch, num_heads, seq, head_dim]
        let q_reshaped = q
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;

        let k_reshaped = k
            .reshape(&[batch_size, seq_len, self.num_key_value_heads, self.head_dim])?
            .transpose(1, 2)?;

        let v_reshaped = v
            .reshape(&[batch_size, seq_len, self.num_key_value_heads, self.head_dim])?
            .transpose(1, 2)?;

        // 4. Apply RoPE (Rotary Position Embedding) to Q and K
        let (q_with_rope, k_with_rope) = apply_rope(&q_reshaped, &k_reshaped, seq_len, self.head_dim, rope_theta)?;

        // 5. Handle GQA (Grouped Query Attention) - repeat K/V heads to match Q heads
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

        // 7. Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, heads * head_dim]
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape(&[batch_size, seq_len, self.num_heads * self.head_dim])?;

        let attn_output = Tensor::from_candle(attn_output);

        // 8. Output projection (dense)
        let output = ops_fn::matmul(&attn_output, &self.dense)?;

        Ok(output)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("transformer.encoder.layers.{}.self_attention", layer_idx);

        // Load packed QKV weight (transpose for matmul: [out, in] -> [in, out])
        if let Some(qkv_weight) = weights.get(&format!("{}.query_key_value.weight", prefix)) {
            self.query_key_value = ops_fn::transpose(qkv_weight)?;
        }

        // Load QKV bias if present
        if let Some(qkv_bias) = weights.get(&format!("{}.query_key_value.bias", prefix)) {
            self.qkv_bias = Some(qkv_bias.clone());
        }

        // Load output projection (dense) - transpose for matmul
        if let Some(dense_weight) = weights.get(&format!("{}.dense.weight", prefix)) {
            self.dense = ops_fn::transpose(dense_weight)?;
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.query_key_value = self.query_key_value.to_device(device)?;
        if let Some(ref mut bias) = self.qkv_bias {
            *bias = bias.to_device(device)?;
        }
        self.dense = self.dense.to_device(device)?;
        Ok(())
    }
}

impl ChatGLMMLP {
    fn new(config: &ChatGLMConfig, device: &Device) -> Result<Self> {
        // ChatGLM MLP uses SwiGLU, with combined gate+up projection
        // dense_h_to_4h: [hidden, intermediate*2] (for gate and up combined)
        // dense_4h_to_h: [intermediate, hidden]
        let dense_h_to_4h = ops_fn::zeros(
            &[config.hidden_size, config.intermediate_size * 2],
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
        // 1. Combined gate+up projection
        let h_to_4h = ops_fn::matmul(hidden_states, &self.dense_h_to_4h)?;

        // 2. Split into gate and up parts for SwiGLU
        let h_to_4h_candle = h_to_4h.to_candle()?;
        let shape = h_to_4h_candle.dims();
        let half_size = shape[shape.len() - 1] / 2;

        let gate = h_to_4h_candle.narrow(shape.len() - 1, 0, half_size)?;
        let up = h_to_4h_candle.narrow(shape.len() - 1, half_size, half_size)?;

        // 3. Apply SwiGLU: activation(gate) * up
        // ChatGLM uses SwiGLU (SiLU/Swish for the gate)
        let gate_tensor = Tensor::from_candle(gate);
        let up_tensor = Tensor::from_candle(up);

        let gate_activated = match self.hidden_act.as_str() {
            "swiglu" | "silu" | "swish" => ops_fn::silu(&gate_tensor)?,
            "gelu" => ops_fn::gelu(&gate_tensor)?,
            _ => ops_fn::silu(&gate_tensor)?, // Default to SiLU for ChatGLM
        };
        let gated = ops_fn::mul(&gate_activated, &up_tensor)?;

        // 4. Down projection
        let output = ops_fn::matmul(&gated, &self.dense_4h_to_h)?;

        Ok(output)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("transformer.encoder.layers.{}.mlp", layer_idx);

        // Load MLP weights (transpose for matmul: [out, in] -> [in, out])
        if let Some(h_to_4h) = weights.get(&format!("{}.dense_h_to_4h.weight", prefix)) {
            self.dense_h_to_4h = ops_fn::transpose(h_to_4h)?;
        }
        if let Some(h4_to_h) = weights.get(&format!("{}.dense_4h_to_h.weight", prefix)) {
            self.dense_4h_to_h = ops_fn::transpose(h4_to_h)?;
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
    fn test_chatglm_model_creation() {
        let config = ChatGLMConfig {
            vocab_size: 1000,
            hidden_size: 128,
            intermediate_size: 512,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            ..Default::default()
        };

        let model = ChatGLMModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 1000);
        assert_eq!(model.config().hidden_size(), 128);
        assert_eq!(model.config().num_layers(), 2);
    }

    #[test]
    fn test_chatglm_forward_pass() {
        let config = ChatGLMConfig {
            vocab_size: 100,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            ..Default::default()
        };

        let model = ChatGLMModelV2::new(config).unwrap();
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
    fn test_chatglm_generation() {
        let config = ChatGLMConfig {
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            ..Default::default()
        };
        let model = ChatGLMModelV2::new(config).unwrap();
        let gen_config = GenerationConfig {
            max_new_tokens: 5,
            ..Default::default()
        };

        let output = model.generate("Hello", &gen_config).unwrap();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_chatglm_from_gguf_config() {
        let gguf_config = crate::weight_loader_core::GGUFModelConfig {
            architecture: "chatglm".to_string(),
            vocab_size: 65024,
            hidden_size: 4096,
            intermediate_size: 13696,
            num_hidden_layers: 28,
            num_attention_heads: 32,
            num_key_value_heads: 2,
            head_dim: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 8192,
        };

        let config = ChatGLMConfig::from_gguf_config(&gguf_config);
        assert_eq!(config.vocab_size, 65024);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_key_value_heads, 2);
        assert_eq!(config.kv_channels, 128);
    }
}
