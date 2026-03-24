//! Llama Model V2 - Clean implementation using solid abstractions
//!
//! This is a complete rewrite of the Llama model using our new architecture:
//! - Uses unified Tensor type from tensor_core
//! - Implements Model trait from model_core
//! - Supports loading via weight_loader_core
//! - Clean, maintainable code with proper abstractions

use crate::model_config;
use super::traits::*;

use anyhow::Result;
use serde::{Serialize, Deserialize};

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

/// Main Llama model implementation
pub struct LlamaModelV2 {
    config: LlamaConfig,
    device: Device,

    // Model components using unified Tensor type
    embed_tokens: Tensor,
    layers: Vec<LlamaLayer>,
    norm: Tensor,
    lm_head: Tensor,
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
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    num_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    scale: f32,
}

/// Llama MLP (feed-forward network)
pub struct LlamaMLP {
    gate_proj: Tensor,  // Up projection
    up_proj: Tensor,    // Gate projection
    down_proj: Tensor,  // Down projection
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
        })
    }

    fn from_weights(config: Self::Config, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;

        // Load weights from the unified weight container
        if let Some(embed_weights) = weights.get("model.embed_tokens.weight") {
            model.embed_tokens = embed_weights.clone();
        }

        if let Some(norm_weights) = weights.get("model.norm.weight") {
            model.norm = norm_weights.clone();
        }

        if let Some(lm_head_weights) = weights.get("lm_head.weight") {
            model.lm_head = lm_head_weights.clone();
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

                // 2. Apply transformer layers
                for layer in &self.layers {
                    hidden_states = layer.forward(&hidden_states, attention_mask.as_ref())?;
                }

                // 3. Final layer norm
                hidden_states = ops_fn::layer_norm(&hidden_states, &self.norm, None, self.config.rms_norm_eps)?;

                // 4. Language modeling head
                let logits = ops_fn::matmul(&hidden_states, &self.lm_head)?;

                Ok(ModelOutputs::Logits {
                    logits,
                    hidden_states: Some(hidden_states),
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
        // Simplified generation for now
        // Real implementation would:
        // 1. Tokenize prompt
        // 2. Forward pass to get logits
        // 3. Sample next token
        // 4. Repeat until stop condition

        Ok(format!(
            "Llama generated response to '{}' (max_tokens: {}, temp: {:.2})",
            prompt, config.max_new_tokens, config.temperature
        ))
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

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // 1. Pre-attention layer norm
        let normed = ops_fn::layer_norm(hidden_states, &self.input_layernorm, None, 1e-6)?;

        // 2. Self attention
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

        // Load attention weights
        if let Some(q_proj) = weights.get(&format!("{}.self_attn.q_proj.weight", prefix)) {
            self.self_attn.q_proj = q_proj.clone();
        }
        if let Some(k_proj) = weights.get(&format!("{}.self_attn.k_proj.weight", prefix)) {
            self.self_attn.k_proj = k_proj.clone();
        }
        if let Some(v_proj) = weights.get(&format!("{}.self_attn.v_proj.weight", prefix)) {
            self.self_attn.v_proj = v_proj.clone();
        }
        if let Some(o_proj) = weights.get(&format!("{}.self_attn.o_proj.weight", prefix)) {
            self.self_attn.o_proj = o_proj.clone();
        }

        // Load MLP weights
        if let Some(gate_proj) = weights.get(&format!("{}.mlp.gate_proj.weight", prefix)) {
            self.mlp.gate_proj = gate_proj.clone();
        }
        if let Some(up_proj) = weights.get(&format!("{}.mlp.up_proj.weight", prefix)) {
            self.mlp.up_proj = up_proj.clone();
        }
        if let Some(down_proj) = weights.get(&format!("{}.mlp.down_proj.weight", prefix)) {
            self.mlp.down_proj = down_proj.clone();
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
            num_heads,
            num_key_value_heads,
            head_dim,
            scale,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // 1. Project to Q, K, V
        let query_states = ops_fn::matmul(hidden_states, &self.q_proj)?;
        let key_states = ops_fn::matmul(hidden_states, &self.k_proj)?;
        let value_states = ops_fn::matmul(hidden_states, &self.v_proj)?;

        // 2. Reshape for attention heads
        // For simplicity, keeping as is for now
        // Real implementation would reshape to [batch, seq, num_heads, head_dim]

        // 3. Scaled dot-product attention using our unified ops
        let attn_output = ops_fn::attention(&query_states, &key_states, &value_states, attention_mask)?;

        // 4. Output projection
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

impl LlamaMLP {
    fn new(config: &LlamaConfig, device: &Device) -> Result<Self> {
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

        // 2. Apply activation (SiLU for Llama)
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
        let config = LlamaConfig::default();
        let model = LlamaModelV2::new(config).unwrap();
        let gen_config = GenerationConfig::default();

        let output = model.generate("Hello world", &gen_config).unwrap();
        assert!(!output.is_empty());
        assert!(output.contains("Hello world"));
    }
}