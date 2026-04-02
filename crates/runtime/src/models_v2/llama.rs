//! Llama Model V2 - Clean implementation using solid abstractions
//!
//! This is a complete rewrite of the Llama model using our new architecture:
//! - Uses unified Tensor type from tensor_core
//! - Implements Model trait from model_core
//! - Supports loading via weight_loader_core
//! - Clean, maintainable code with proper abstractions

use crate::model_config;
use crate::kv_cache::{KVCache, LayerKVCache};
use super::traits::*;

use anyhow::Result;
use serde::{Serialize, Deserialize};
use candle_core::quantized::QMatMul;
use candle_nn::Module;
use std::sync::Arc;

/// Llama model configuration using the model_config macro
model_config!(LlamaConfig {
    vocab_size: usize = 32000,
    hidden_size: usize = 4096,
    intermediate_size: usize = 11008,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 32,
    num_key_value_heads: usize = 32, // Changed from Option<usize>
    hidden_act: String = "silu".to_string(),
    max_position_embeddings: usize = 2048,
    initializer_range: f32 = 0.02,
    rms_norm_eps: f32 = 1e-6,
    use_cache: bool = true,
    pad_token_id: i64 = 0, // Changed from Option<i64>
    bos_token_id: i64 = 1, // Changed from Option<i64>
    eos_token_id: i64 = 2, // Changed from Option<i64>
    tie_word_embeddings: bool = false,
    rope_theta: f32 = 10000.0,
    attention_bias: bool = false,
});

impl LlamaConfig {
    /// Create LlamaConfig from GGUF model configuration
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

/// Main Llama model implementation
pub struct LlamaModelV2 {
    config: LlamaConfig,
    device: Device,

    // Model components using unified Tensor type
    embed_tokens: Tensor,
    layers: Vec<LlamaLayer>,
    norm: Tensor,
    lm_head: Tensor,
    // Quantized lm_head (optional)
    lm_head_q: Option<Arc<QMatMul>>,
}

/// Llama transformer layer
pub struct LlamaLayer {
    self_attn: LlamaAttention,
    mlp: LlamaMLP,
    input_layernorm: Tensor,
    post_attention_layernorm: Tensor,
}

/// Llama attention mechanism
pub struct LlamaAttention {
    // F32 weights (fallback)
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    // Quantized weights (optional, for efficient inference)
    q_proj_q: Option<Arc<QMatMul>>,
    k_proj_q: Option<Arc<QMatMul>>,
    v_proj_q: Option<Arc<QMatMul>>,
    o_proj_q: Option<Arc<QMatMul>>,
    // SIMD quantized weights (optional, for native SIMD inference)
    #[cfg(feature = "simd")]
    q_proj_simd: Option<Arc<crate::simd::quant::QuantizedTensor>>,
    #[cfg(feature = "simd")]
    k_proj_simd: Option<Arc<crate::simd::quant::QuantizedTensor>>,
    #[cfg(feature = "simd")]
    v_proj_simd: Option<Arc<crate::simd::quant::QuantizedTensor>>,
    #[cfg(feature = "simd")]
    o_proj_simd: Option<Arc<crate::simd::quant::QuantizedTensor>>,
    // Config
    num_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    scale: f32,
}

/// Llama MLP (feed-forward network)
pub struct LlamaMLP {
    // F32 weights (fallback)
    gate_proj: Tensor,  // Gate projection
    up_proj: Tensor,    // Up projection
    down_proj: Tensor,  // Down projection
    // Quantized weights (optional, for efficient inference)
    gate_proj_q: Option<Arc<QMatMul>>,
    up_proj_q: Option<Arc<QMatMul>>,
    down_proj_q: Option<Arc<QMatMul>>,
    // SIMD quantized weights (optional, for native SIMD inference)
    #[cfg(feature = "simd")]
    gate_proj_simd: Option<Arc<crate::simd::quant::QuantizedTensor>>,
    #[cfg(feature = "simd")]
    up_proj_simd: Option<Arc<crate::simd::quant::QuantizedTensor>>,
    #[cfg(feature = "simd")]
    down_proj_simd: Option<Arc<crate::simd::quant::QuantizedTensor>>,
    // Config
    hidden_act: String,
}

impl Model for LlamaModelV2 {
    type Config = LlamaConfig;

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
            layers.push(LlamaLayer::new(&config, &device)?);
        }

        Ok(Self {
            config,
            device,
            embed_tokens,
            layers,
            norm,
            lm_head,
            lm_head_q: None,
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
        // Load quantized lm_head if available
        model.lm_head_q = weights.get_quantized("lm_head.weight");

        // Load layer weights (including quantized)
        for (i, layer) in model.layers.iter_mut().enumerate() {
            layer.load_weights(&weights, i)?;
        }

        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        match inputs {
            ModelInputs::Text { input_ids, attention_mask, .. } => {
                // Debug: print input shape
                // println!("Forward: input_ids shape: {:?}", input_ids.shape());

                // 1. Token embedding
                let mut hidden_states = ops_fn::embedding(input_ids, &self.embed_tokens)?;

                // Debug: check embedding output (disabled for cleaner output)
                // let emb_candle = hidden_states.to_candle()?;
                // let emb_slice: Vec<f32> = emb_candle.flatten_all()?.to_vec1()?;
                // let emb_sum: f32 = emb_slice.iter().take(100).sum();
                // let emb_max = emb_slice.iter().take(100).cloned().fold(f32::NEG_INFINITY, f32::max);
                // let emb_min = emb_slice.iter().take(100).cloned().fold(f32::INFINITY, f32::min);
                // println!("After embedding: shape={:?}, sample sum={:.4}, min={:.4}, max={:.4}",
                //     hidden_states.shape(), emb_sum, emb_min, emb_max);

                // 2. Apply transformer layers (with RoPE)
                for layer in &self.layers {
                    hidden_states = layer.forward(&hidden_states, attention_mask.as_ref(), self.config.rope_theta)?;
                }

                // 3. Final layer norm
                hidden_states = ops_fn::layer_norm(&hidden_states, &self.norm, None, self.config.rms_norm_eps)?;

                // 4. Language modeling head (use quantized if available)
                let logits = self.lm_head_forward(&hidden_states)?;

                Ok(ModelOutputs::Logits {
                    logits,
                    hidden_states: None,  // Don't return hidden states to save memory
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
            _ => Err(anyhow::anyhow!("Llama model only supports text and multimodal inputs")),
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
                            4 * self.config.hidden_size * self.config.hidden_size + // attention
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

// Helper methods
impl LlamaModelV2 {
    /// Apply lm_head projection, using quantized if available
    fn lm_head_forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        if let Some(ref qmatmul) = self.lm_head_q {
            let input_candle = hidden_states.to_candle()?;
            let output = qmatmul.forward(&input_candle)
                .map_err(|e| anyhow::anyhow!("QMatMul lm_head forward failed: {}", e))?;
            Ok(Tensor::from_candle(output))
        } else {
            ops_fn::matmul(hidden_states, &self.lm_head)
        }
    }
}

// KV Cache methods for efficient autoregressive generation
impl LlamaModelV2 {
    /// Forward pass with KV cache support
    ///
    /// This method enables efficient autoregressive generation by caching
    /// Key and Value tensors from previous tokens.
    ///
    /// # Arguments
    /// * `inputs` - Model inputs (only new tokens)
    /// * `cache` - Optional mutable reference to KV cache
    ///
    /// # Returns
    /// Model outputs with logits for the new tokens
    pub fn forward_with_cache(
        &self,
        inputs: &ModelInputs,
        mut cache: Option<&mut KVCache>,
    ) -> Result<ModelOutputs> {
        match inputs {
            ModelInputs::Text { input_ids, .. } => {
                // Get position offset from cache
                let position_offset = cache.as_ref().map(|c| c.seq_len()).unwrap_or(0);

                // 1. Token embedding (only for new tokens)
                let mut hidden_states = ops_fn::embedding(input_ids, &self.embed_tokens)?;

                // 2. Apply transformer layers with cache
                for (layer_idx, layer) in self.layers.iter().enumerate() {
                    let layer_cache = cache.as_mut().map(|c| c.layer_mut(layer_idx));
                    hidden_states = layer.forward_with_cache(
                        &hidden_states,
                        layer_cache,
                        position_offset,
                        self.config.rope_theta,
                    )?;
                }

                // 3. Final layer norm
                hidden_states = ops_fn::layer_norm(&hidden_states, &self.norm, None, self.config.rms_norm_eps)?;

                // 4. Language modeling head (use quantized if available)
                let logits = self.lm_head_forward(&hidden_states)?;

                // 5. Update cache sequence length
                if let Some(cache) = cache {
                    let new_tokens = input_ids.shape().get(1).copied().unwrap_or(1);
                    cache.set_seq_len(position_offset + new_tokens);
                }

                Ok(ModelOutputs::Logits {
                    logits,
                    hidden_states: None,
                })
            }
            _ => Err(anyhow::anyhow!("forward_with_cache only supports text inputs")),
        }
    }

    /// Generate text with KV caching for efficient autoregressive generation
    ///
    /// This method is significantly faster than `generate()` for long sequences
    /// because it caches Key and Value tensors instead of recomputing them.
    ///
    /// # Arguments
    /// * `prompt` - Input text prompt
    /// * `config` - Generation configuration
    ///
    /// # Returns
    /// Generated text including the prompt
    pub fn generate_with_cache(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        use crate::tokenizer::Tokenizer;

        // 1. Tokenize prompt
        let tokenizer = Tokenizer::new();
        let prompt_tokens: Vec<u32> = tokenizer.encode(prompt);
        let mut tokens = prompt_tokens.clone();

        // 2. Initialize KV cache
        let mut cache = KVCache::new(self.config.num_hidden_layers);

        // 3. PREFILL: Process entire prompt at once
        let prompt_i64: Vec<i64> = prompt_tokens.iter().map(|&t| t as i64).collect();
        let prompt_tensor = Tensor::from_i64_slice(&prompt_i64, &[1, prompt_tokens.len()], &self.device)?;
        let prompt_inputs = ModelInputs::Text {
            input_ids: prompt_tensor,
            attention_mask: None,
            position_ids: None,
        };

        // Prefill: process all prompt tokens, populate cache
        let outputs = self.forward_with_cache(&prompt_inputs, Some(&mut cache))?;

        // Get the last token's logits from prefill to sample first new token
        let logits = match outputs {
            ModelOutputs::Logits { logits, .. } => logits,
            _ => return Err(anyhow::anyhow!("Expected logits output")),
        };

        // Sample first new token from prefill output
        let logits_candle = logits.to_candle()?;
        let shape = logits_candle.dims();
        let seq_len = if shape.len() == 3 { shape[1] } else { shape[0] };
        let last_logits = if shape.len() == 3 {
            logits_candle.narrow(1, seq_len - 1, 1)?.squeeze(1)?.squeeze(0)?
        } else {
            logits_candle.narrow(0, seq_len - 1, 1)?.squeeze(0)?
        };

        let mut next_token = self.sample_token_from_logits(&last_logits, config)?;

        // Check EOS
        if next_token == config.eos_token_id {
            return Ok(tokenizer.decode(&tokens));
        }
        tokens.push(next_token);

        // 4. DECODE: Generate tokens one at a time using cache
        for _ in 1..config.max_new_tokens {
            // Create input tensor for SINGLE new token
            let input_tensor = Tensor::from_i64_slice(
                &[next_token as i64],
                &[1, 1],
                &self.device
            )?;

            let inputs = ModelInputs::Text {
                input_ids: input_tensor,
                attention_mask: None,
                position_ids: None,
            };

            // Forward with cache: only processes the new token
            let outputs = self.forward_with_cache(&inputs, Some(&mut cache))?;

            // Get logits and sample
            let logits = match outputs {
                ModelOutputs::Logits { logits, .. } => logits,
                _ => return Err(anyhow::anyhow!("Expected logits output")),
            };

            let logits_candle = logits.to_candle()?;
            let last_logits = logits_candle.squeeze(0)?.squeeze(0)?;

            next_token = self.sample_token_from_logits(&last_logits, config)?;

            // Check EOS
            if next_token == config.eos_token_id {
                break;
            }

            tokens.push(next_token);
        }

        // 5. Decode and return
        Ok(tokenizer.decode(&tokens))
    }

    /// Sample a token from logits vector
    fn sample_token_from_logits(&self, logits: &candle_core::Tensor, config: &GenerationConfig) -> Result<u32> {
        use rand::Rng;

        let logits_vec: Vec<f32> = logits.to_vec1()?;

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

        Ok(next_token)
    }
}

impl LlamaLayer {
    fn new(config: &LlamaConfig, device: &Device) -> Result<Self> {
        let self_attn = LlamaAttention::new(config, device)?;
        let mlp = LlamaMLP::new(config, device)?;

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

    /// Forward pass with KV cache support
    fn forward_with_cache(
        &self,
        hidden_states: &Tensor,
        cache: Option<&mut LayerKVCache>,
        position_offset: usize,
        rope_theta: f32,
    ) -> Result<Tensor> {
        // 1. Pre-attention layer norm
        let normed = ops_fn::layer_norm(hidden_states, &self.input_layernorm, None, 1e-6)?;

        // 2. Self attention with cache
        let attn_output = self.self_attn.forward_with_cache(&normed, cache, position_offset, rope_theta)?;

        // 3. Residual connection
        let hidden_states = ops_fn::add(hidden_states, &attn_output)?;

        // 4. Pre-MLP layer norm
        let normed = ops_fn::layer_norm(&hidden_states, &self.post_attention_layernorm, None, 1e-6)?;

        // 5. MLP (unchanged - no caching needed)
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

        // Load quantized attention weights if available
        self.self_attn.q_proj_q = weights.get_quantized(&format!("{}.self_attn.q_proj.weight", prefix));
        self.self_attn.k_proj_q = weights.get_quantized(&format!("{}.self_attn.k_proj.weight", prefix));
        self.self_attn.v_proj_q = weights.get_quantized(&format!("{}.self_attn.v_proj.weight", prefix));
        self.self_attn.o_proj_q = weights.get_quantized(&format!("{}.self_attn.o_proj.weight", prefix));

        // Load SIMD quantized attention weights if available
        #[cfg(feature = "simd")]
        {
            self.self_attn.q_proj_simd = weights.get_simd_quantized(&format!("{}.self_attn.q_proj.weight", prefix));
            self.self_attn.k_proj_simd = weights.get_simd_quantized(&format!("{}.self_attn.k_proj.weight", prefix));
            self.self_attn.v_proj_simd = weights.get_simd_quantized(&format!("{}.self_attn.v_proj.weight", prefix));
            self.self_attn.o_proj_simd = weights.get_simd_quantized(&format!("{}.self_attn.o_proj.weight", prefix));
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

        // Load quantized MLP weights if available
        self.mlp.gate_proj_q = weights.get_quantized(&format!("{}.mlp.gate_proj.weight", prefix));
        self.mlp.up_proj_q = weights.get_quantized(&format!("{}.mlp.up_proj.weight", prefix));
        self.mlp.down_proj_q = weights.get_quantized(&format!("{}.mlp.down_proj.weight", prefix));

        // Load SIMD quantized MLP weights if available
        #[cfg(feature = "simd")]
        {
            self.mlp.gate_proj_simd = weights.get_simd_quantized(&format!("{}.mlp.gate_proj.weight", prefix));
            self.mlp.up_proj_simd = weights.get_simd_quantized(&format!("{}.mlp.up_proj.weight", prefix));
            self.mlp.down_proj_simd = weights.get_simd_quantized(&format!("{}.mlp.down_proj.weight", prefix));
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
/// Input shape: [batch, heads, seq, head_dim]
/// Returns tensors with same shape but with positional information encoded
fn apply_rope(
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    seq_len: usize,
    head_dim: usize,
    rope_theta: f32,
) -> Result<(candle_core::Tensor, candle_core::Tensor)> {
    use candle_core::{DType, Device};

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
    // q = [q1, q2], k = [k1, k2] where each half has shape [..., half_dim]
    // rotated_q = [q1*cos - q2*sin, q1*sin + q2*cos]
    // rotated_k = [k1*cos - k2*sin, k1*sin + k2*cos]

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

/// Apply Rotary Position Embedding (RoPE) with a position offset
/// This is used for KV caching where we need to apply RoPE starting from a specific position
/// Input shape: [batch, heads, seq, head_dim]
/// Returns tensors with same shape but with positional information encoded
fn apply_rope_with_offset(
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    seq_len: usize,
    head_dim: usize,
    rope_theta: f32,
    position_offset: usize,
) -> Result<(candle_core::Tensor, candle_core::Tensor)> {
    let device = q.device();

    // Compute inverse frequencies: 1 / (theta^(2i/d)) for i in [0, d/2)
    let half_dim = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / rope_theta.powf((2 * i) as f32 / head_dim as f32))
        .collect();

    // Create position indices starting from offset: [offset, offset+1, ..., offset+seq_len-1]
    let positions: Vec<f32> = (0..seq_len)
        .map(|p| (p + position_offset) as f32)
        .collect();

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

impl LlamaAttention {
    fn new(config: &LlamaConfig, device: &Device) -> Result<Self> {
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
            q_proj_q: None,
            k_proj_q: None,
            v_proj_q: None,
            o_proj_q: None,
            #[cfg(feature = "simd")]
            q_proj_simd: None,
            #[cfg(feature = "simd")]
            k_proj_simd: None,
            #[cfg(feature = "simd")]
            v_proj_simd: None,
            #[cfg(feature = "simd")]
            o_proj_simd: None,
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

        // 1. Project to Q, K, V (use quantized if available)
        // Q: [batch, seq, hidden] @ [hidden, num_heads * head_dim] -> [batch, seq, num_heads * head_dim]
        // K: [batch, seq, hidden] @ [hidden, num_kv_heads * head_dim] -> [batch, seq, num_kv_heads * head_dim]
        // V: [batch, seq, hidden] @ [hidden, num_kv_heads * head_dim] -> [batch, seq, num_kv_heads * head_dim]
        let query_states = self.quantized_matmul(hidden_states, &self.q_proj, &self.q_proj_q)?;
        let key_states = self.quantized_matmul(hidden_states, &self.k_proj, &self.k_proj_q)?;
        let value_states = self.quantized_matmul(hidden_states, &self.v_proj, &self.v_proj_q)?;

        // 2. Reshape for multi-head attention
        // Q: [batch, seq, num_heads * head_dim] -> [batch, num_heads, seq, head_dim]
        // K: [batch, seq, num_kv_heads * head_dim] -> [batch, num_kv_heads, seq, head_dim]
        let q_candle = query_states.to_candle()?;
        let k_candle = key_states.to_candle()?;
        let v_candle = value_states.to_candle()?;

        // Reshape: [batch, seq, heads*head_dim] -> [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        let q_reshaped = q_candle
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;  // [batch, heads, seq, head_dim]

        let k_reshaped = k_candle
            .reshape(&[batch_size, seq_len, self.num_key_value_heads, self.head_dim])?
            .transpose(1, 2)?;  // [batch, kv_heads, seq, head_dim]

        let v_reshaped = v_candle
            .reshape(&[batch_size, seq_len, self.num_key_value_heads, self.head_dim])?
            .transpose(1, 2)?;  // [batch, kv_heads, seq, head_dim]

        // 2.5 Apply RoPE (Rotary Position Embedding) to Q and K
        let (q_with_rope, k_with_rope) = apply_rope(&q_reshaped, &k_reshaped, seq_len, self.head_dim, rope_theta)?;

        // 3. Handle GQA (Grouped Query Attention) - repeat K/V heads to match Q heads
        let num_groups = self.num_heads / self.num_key_value_heads;
        let (k_expanded, v_expanded) = if num_groups > 1 {
            // Repeat K and V along the head dimension
            // [batch, kv_heads, seq, head_dim] -> [batch, num_heads, seq, head_dim]
            let k_rep = k_with_rope
                .unsqueeze(2)?  // [batch, kv_heads, 1, seq, head_dim]
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

        // 4. Fused scaled dot-product attention with causal masking
        // Uses ops_fn::flash_attention which fuses: Q @ K^T, scale, mask, softmax, @ V
        // Input shapes: [batch, heads, seq, head_dim]

        // Make tensors contiguous (required by some backends)
        let q_contiguous = q_with_rope.contiguous()?;
        let k_contiguous = k_expanded.contiguous()?;
        let v_contiguous = v_expanded.contiguous()?;

        // Convert to Tensor type for ops_fn
        let q_tensor = Tensor::from_candle(q_contiguous);
        let k_tensor = Tensor::from_candle(k_contiguous);
        let v_tensor = Tensor::from_candle(v_contiguous);

        // Fused attention: softmax((Q @ K^T) * scale) @ V with causal masking
        let attn_output_tensor = ops_fn::flash_attention(
            &q_tensor,
            &k_tensor,
            &v_tensor,
            self.scale,
            true,  // causal = true for autoregressive
        )?;

        // Convert back to Candle tensor for the rest of the computation
        let attn_output = attn_output_tensor.to_candle()?;

        // 5. Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, heads * head_dim]
        let attn_output = attn_output
            .transpose(1, 2)?  // [batch, seq, heads, head_dim]
            .reshape(&[batch_size, seq_len, self.num_heads * self.head_dim])?;

        let attn_output = Tensor::from_candle(attn_output);

        // 6. Output projection (use quantized if available)
        let output = self.quantized_matmul(&attn_output, &self.o_proj, &self.o_proj_q)?;

        Ok(output)
    }

    /// Forward pass with KV cache support for efficient autoregressive generation
    ///
    /// # Arguments
    /// * `hidden_states` - Input hidden states for NEW tokens only
    /// * `cache` - Optional mutable reference to layer KV cache
    /// * `position_offset` - Starting position for RoPE (cache.seq_len for incremental)
    /// * `rope_theta` - RoPE theta parameter
    fn forward_with_cache(
        &self,
        hidden_states: &Tensor,
        cache: Option<&mut LayerKVCache>,
        position_offset: usize,
        rope_theta: f32,
    ) -> Result<Tensor> {
        // Get batch and sequence length from hidden_states shape (new tokens only)
        let shape = hidden_states.shape();
        let (batch_size, new_seq_len, _hidden_size) = if shape.len() == 3 {
            (shape[0], shape[1], shape[2])
        } else if shape.len() == 2 {
            (1, shape[0], shape[1])
        } else {
            return Err(anyhow::anyhow!("Invalid hidden_states shape: {:?}", shape));
        };

        // 1. Project to Q, K, V for NEW tokens only (use quantized if available)
        let query_states = self.quantized_matmul(hidden_states, &self.q_proj, &self.q_proj_q)?;
        let key_states = self.quantized_matmul(hidden_states, &self.k_proj, &self.k_proj_q)?;
        let value_states = self.quantized_matmul(hidden_states, &self.v_proj, &self.v_proj_q)?;

        // 2. Reshape for multi-head attention
        let q_candle = query_states.to_candle()?;
        let k_candle = key_states.to_candle()?;
        let v_candle = value_states.to_candle()?;

        let q_reshaped = q_candle
            .reshape(&[batch_size, new_seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;  // [batch, heads, new_seq, head_dim]

        let k_reshaped = k_candle
            .reshape(&[batch_size, new_seq_len, self.num_key_value_heads, self.head_dim])?
            .transpose(1, 2)?;  // [batch, kv_heads, new_seq, head_dim]

        let v_reshaped = v_candle
            .reshape(&[batch_size, new_seq_len, self.num_key_value_heads, self.head_dim])?
            .transpose(1, 2)?;  // [batch, kv_heads, new_seq, head_dim]

        // 3. Apply RoPE with position offset
        let (q_with_rope, k_with_rope) = apply_rope_with_offset(
            &q_reshaped, &k_reshaped,
            new_seq_len, self.head_dim, rope_theta,
            position_offset
        )?;

        // 4. Handle GQA expansion for K before caching (expand once, cache expanded)
        let num_groups = self.num_heads / self.num_key_value_heads;
        let (k_expanded, v_expanded) = if num_groups > 1 {
            let k_rep = k_with_rope
                .unsqueeze(2)?
                .broadcast_as(&[batch_size, self.num_key_value_heads, num_groups, new_seq_len, self.head_dim])?
                .reshape(&[batch_size, self.num_heads, new_seq_len, self.head_dim])?;
            let v_rep = v_reshaped
                .unsqueeze(2)?
                .broadcast_as(&[batch_size, self.num_key_value_heads, num_groups, new_seq_len, self.head_dim])?
                .reshape(&[batch_size, self.num_heads, new_seq_len, self.head_dim])?;
            (k_rep, v_rep)
        } else {
            (k_with_rope, v_reshaped)
        };

        // 5. Update cache and get full K, V (using pre-allocated buffers)
        let (full_k, full_v, total_seq_len) = if let Some(cache) = cache {
            // Append new K,V to cache using slice_set (zero-allocation!)
            if let Err(e) = cache.append(&k_expanded, &v_expanded) {
                return Err(anyhow::anyhow!("KV cache append failed: {}", e));
            }

            // Get view of full cached K,V (narrow, no copy)
            match cache.get_kv() {
                Some((k, v)) => {
                    let total_len = k.dims()[2];
                    (k, v, total_len)
                }
                None => return Err(anyhow::anyhow!("Cache should not be empty after append")),
            }
        } else {
            // No caching, use new K,V directly
            (k_expanded, v_expanded, new_seq_len)
        };

        // 6. Scaled dot-product attention
        // Q: [batch, heads, new_seq, head_dim]
        // K: [batch, heads, total_seq, head_dim]
        // scores: [batch, heads, new_seq, total_seq]
        let k_t = full_k.transpose(2, 3)?;
        let q_contiguous = q_with_rope.contiguous()?;
        let k_contiguous = k_t.contiguous()?;

        let scores = q_contiguous.matmul(&k_contiguous)?;
        let scaled_scores = (scores * (self.scale as f64))?;

        // 7. Apply causal mask
        // For cached generation, new tokens at positions [offset, offset+new_seq)
        // can attend to all positions [0, offset+new_seq)
        let device = scaled_scores.device();
        let causal_mask = {
            let mut mask_data = vec![0.0f32; new_seq_len * total_seq_len];
            for i in 0..new_seq_len {
                let query_pos = position_offset + i;
                for j in 0..total_seq_len {
                    if j > query_pos {
                        // Future position - mask out
                        mask_data[i * total_seq_len + j] = f32::NEG_INFINITY;
                    }
                }
            }
            candle_core::Tensor::from_vec(mask_data, &[1, 1, new_seq_len, total_seq_len], device)?
        };

        let masked_scores = scaled_scores.broadcast_add(&causal_mask)?;
        let attention_weights = candle_nn::ops::softmax_last_dim(&masked_scores)?;

        // 8. Apply attention to values
        let v_contiguous = full_v.contiguous()?;
        let attn_output = attention_weights.matmul(&v_contiguous)?;

        // 9. Reshape back: [batch, heads, new_seq, head_dim] -> [batch, new_seq, heads * head_dim]
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape(&[batch_size, new_seq_len, self.num_heads * self.head_dim])?;

        let attn_output = Tensor::from_candle(attn_output);

        // 10. Output projection (use quantized if available)
        let output = self.quantized_matmul(&attn_output, &self.o_proj, &self.o_proj_q)?;

        Ok(output)
    }

    /// Helper: use SIMD/QMatMul when available, fall back to F32 matmul
    /// Priority: SIMD > QMatMul > F32
    #[cfg(feature = "simd")]
    fn quantized_matmul_simd(
        &self,
        input: &Tensor,
        weight: &Tensor,
        quantized: &Option<Arc<QMatMul>>,
        simd_quantized: &Option<Arc<crate::simd::quant::QuantizedTensor>>,
    ) -> Result<Tensor> {
        // Try SIMD first (highest priority for performance)
        if let Some(ref simd_weight) = simd_quantized {
            return simd_matmul(input, simd_weight);
        }
        // Fall back to QMatMul
        if let Some(ref qmatmul) = quantized {
            let input_candle = input.to_candle()?;
            let output = qmatmul.forward(&input_candle)
                .map_err(|e| anyhow::anyhow!("QMatMul forward failed: {}", e))?;
            return Ok(Tensor::from_candle(output));
        }
        // Fall back to F32 matmul
        ops_fn::matmul(input, weight)
    }

    /// Helper: use QMatMul when available, fall back to F32 matmul
    fn quantized_matmul(&self, input: &Tensor, weight: &Tensor, quantized: &Option<Arc<QMatMul>>) -> Result<Tensor> {
        if let Some(ref qmatmul) = quantized {
            // Use quantized matmul
            let input_candle = input.to_candle()?;
            let output = qmatmul.forward(&input_candle)
                .map_err(|e| anyhow::anyhow!("QMatMul forward failed: {}", e))?;
            Ok(Tensor::from_candle(output))
        } else {
            // Fall back to F32 matmul
            ops_fn::matmul(input, weight)
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.q_proj = self.q_proj.to_device(device)?;
        self.k_proj = self.k_proj.to_device(device)?;
        self.v_proj = self.v_proj.to_device(device)?;
        self.o_proj = self.o_proj.to_device(device)?;
        Ok(())
    }
}

/// SIMD-accelerated matrix multiplication for QuantizedTensor
#[cfg(feature = "simd")]
fn simd_matmul(
    input: &Tensor,
    weight: &crate::simd::quant::QuantizedTensor,
) -> Result<Tensor> {
    use crate::simd::get_simd_backend;
    use crate::simd::matmul::gemm::q4_gemm;

    let shape = input.shape();
    let input_candle = input.to_candle()?;
    let input_flat = input_candle.flatten_all()?;
    let input_vec: Vec<f32> = input_flat.to_vec1()?;

    let (n, k) = (weight.rows(), weight.cols());

    // Determine M (batch size * sequence length)
    let m = if shape.len() >= 2 {
        shape[..shape.len()-1].iter().product()
    } else {
        1
    };

    let mut output = vec![0.0f32; m * n];

    if m == 1 {
        // GEMV for single token (decode phase)
        get_simd_backend().q4_gemv(weight, &input_vec, &mut output);
    } else {
        // GEMM for multiple tokens (prefill phase)
        q4_gemm(weight, &input_vec, &mut output, m, k, n);
    }

    // Convert back to Tensor
    let device = input_candle.device();
    let output_candle = candle_core::Tensor::from_vec(output, &[m, n], device)?;

    // Reshape to match expected output shape
    let mut out_shape = shape[..shape.len()-1].to_vec();
    out_shape.push(n);
    let output_reshaped = output_candle.reshape(out_shape.as_slice())?;

    Ok(Tensor::from_candle(output_reshaped))
}

impl LlamaMLP {
    fn new(config: &LlamaConfig, device: &Device) -> Result<Self> {
        let gate_proj = ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?;
        let up_proj = ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?;
        let down_proj = ops_fn::zeros(&[config.intermediate_size, config.hidden_size], DataType::Float32, device)?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            gate_proj_q: None,
            up_proj_q: None,
            down_proj_q: None,
            #[cfg(feature = "simd")]
            gate_proj_simd: None,
            #[cfg(feature = "simd")]
            up_proj_simd: None,
            #[cfg(feature = "simd")]
            down_proj_simd: None,
            hidden_act: config.hidden_act.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // 1. Gate and up projections (use quantized if available)
        let gate_output = self.quantized_matmul(hidden_states, &self.gate_proj, &self.gate_proj_q)?;
        let up_output = self.quantized_matmul(hidden_states, &self.up_proj, &self.up_proj_q)?;

        // 2. Apply activation and gating (fused for SiLU, separate for others)
        let gated = match self.hidden_act.as_str() {
            "silu" | "swish" => {
                // Use fused SwiGLU: silu(gate) * up in one operation
                ops_fn::fused_swiglu(&gate_output, &up_output)?
            }
            "gelu" => {
                // GELU + element-wise multiply (not fused)
                let gate_activated = ops_fn::gelu(&gate_output)?;
                ops_fn::mul(&gate_activated, &up_output)?
            }
            _ => return Err(anyhow::anyhow!("Unsupported activation: {}", self.hidden_act)),
        };

        // 3. Down projection (use quantized if available)
        let output = self.quantized_matmul(&gated, &self.down_proj, &self.down_proj_q)?;

        Ok(output)
    }

    /// Helper: use SIMD/QMatMul when available, fall back to F32 matmul
    /// Priority: SIMD > QMatMul > F32
    #[cfg(feature = "simd")]
    fn quantized_matmul_simd(
        &self,
        input: &Tensor,
        weight: &Tensor,
        quantized: &Option<Arc<QMatMul>>,
        simd_quantized: &Option<Arc<crate::simd::quant::QuantizedTensor>>,
    ) -> Result<Tensor> {
        // Try SIMD first (highest priority for performance)
        if let Some(ref simd_weight) = simd_quantized {
            return simd_matmul(input, simd_weight);
        }
        // Fall back to QMatMul
        if let Some(ref qmatmul) = quantized {
            let input_candle = input.to_candle()?;
            let output = qmatmul.forward(&input_candle)
                .map_err(|e| anyhow::anyhow!("QMatMul forward failed: {}", e))?;
            return Ok(Tensor::from_candle(output));
        }
        // Fall back to F32 matmul
        ops_fn::matmul(input, weight)
    }

    /// Helper: use QMatMul when available, fall back to F32 matmul
    fn quantized_matmul(&self, input: &Tensor, weight: &Tensor, quantized: &Option<Arc<QMatMul>>) -> Result<Tensor> {
        if let Some(ref qmatmul) = quantized {
            // Use quantized matmul
            let input_candle = input.to_candle()?;
            let output = qmatmul.forward(&input_candle)
                .map_err(|e| anyhow::anyhow!("QMatMul forward failed: {}", e))?;
            Ok(Tensor::from_candle(output))
        } else {
            // Fall back to F32 matmul
            ops_fn::matmul(input, weight)
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.gate_proj = self.gate_proj.to_device(device)?;
        self.up_proj = self.up_proj.to_device(device)?;
        self.down_proj = self.down_proj.to_device(device)?;
        Ok(())
    }
}

// Helper functions are now available in ops_fn module

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_model_creation() {
        let config = LlamaConfig {
            vocab_size: 1000,
            hidden_size: 128,
            intermediate_size: 512,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            ..Default::default()
        };

        let model = LlamaModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 1000);
        assert_eq!(model.config().hidden_size(), 128);
        assert_eq!(model.config().num_layers(), 2);
    }

    #[test]
    fn test_llama_forward_pass() {
        let config = LlamaConfig {
            vocab_size: 100,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 4, // Must match num_attention_heads for standard attention
            ..Default::default()
        };

        let model = LlamaModelV2::new(config).unwrap();
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
    fn test_llama_generation() {
        // Use a small config for testing
        // vocab_size must be >= basic tokenizer's vocab (~200 tokens)
        let config = LlamaConfig {
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            ..Default::default()
        };
        let model = LlamaModelV2::new(config).unwrap();
        let gen_config = GenerationConfig {
            max_new_tokens: 5, // Generate only a few tokens for testing
            ..Default::default()
        };

        let output = model.generate("Hello", &gen_config).unwrap();
        assert!(!output.is_empty());
    }
}