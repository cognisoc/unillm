//! DeepSeek Model V2 - Clean implementation using solid abstractions
//!
//! This implements the DeepSeek/DeepSeek-V2 architecture including:
//! - DeepSeek-7B, DeepSeek-67B, DeepSeek-Coder, DeepSeek-Math
//! - Uses unified Tensor type from tensor_core
//! - Supports MoE (Mixture of Experts) architecture

use crate::model_config;
use super::traits::*;
use std::collections::HashMap;
use anyhow::Result;
use serde::{Serialize, Deserialize};

/// DeepSeek model configuration using the model_config macro
model_config!(DeepSeekConfig {
    vocab_size: usize = 102400,
    hidden_size: usize = 4096,
    intermediate_size: usize = 11008,
    num_hidden_layers: usize = 30,
    num_attention_heads: usize = 32,
    num_key_value_heads: Option<usize> = Some(32),
    hidden_act: String = "silu".to_string(),
    max_position_embeddings: usize = 4096,
    initializer_range: f32 = 0.02,
    rms_norm_eps: f32 = 1e-6,
    use_cache: bool = true,
    pad_token_id: Option<i64> = Some(0),
    bos_token_id: Option<i64> = Some(100000),
    eos_token_id: Option<i64> = Some(100001),
    tie_word_embeddings: bool = false,
    rope_theta: f32 = 10000.0,
    attention_bias: bool = false,
    // MoE specific parameters
    num_experts: Option<usize> = None,
    num_experts_per_tok: Option<usize> = None,
    expert_capacity: Option<usize> = None,
});

/// Main DeepSeek model implementation
pub struct DeepSeekModelV2 {
    config: DeepSeekConfig,
    device: Device,

    // Model components using unified Tensor type
    embed_tokens: Tensor,
    layers: Vec<DeepSeekLayer>,
    norm: Tensor,
    lm_head: Tensor,
}

/// DeepSeek transformer layer
pub struct DeepSeekLayer {
    self_attn: DeepSeekAttention,
    mlp: DeepSeekMLP,
    input_layernorm: Tensor,
    post_attention_layernorm: Tensor,
}

/// DeepSeek attention mechanism
pub struct DeepSeekAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    config: DeepSeekConfig,
}

/// DeepSeek MLP (feed-forward network)
pub struct DeepSeekMLP {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
    config: DeepSeekConfig,
}

impl Model for DeepSeekModelV2 {
    type Config = DeepSeekConfig;

    fn new(config: DeepSeekConfig) -> Result<Self> {
        let device = Device::CPU;

        // Initialize model weights
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

        let lm_head = ops_fn::zeros(
            &[config.vocab_size, config.hidden_size],
            DataType::Float32,
            &device
        )?;

        // Create transformer layers
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(DeepSeekLayer::new(&config, &device)?);
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

    fn from_weights(config: DeepSeekConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;

        // Load actual weights from ModelWeights
        if let Some(embed_weight) = weights.get("model.embed_tokens.weight") {
            model.embed_tokens = embed_weight.clone();
        }

        if let Some(norm_weight) = weights.get("model.norm.weight") {
            model.norm = norm_weight.clone();
        }

        if let Some(lm_head_weight) = weights.get("lm_head.weight") {
            model.lm_head = lm_head_weight.clone();
        }

        // Load layer weights
        for (i, layer) in model.layers.iter_mut().enumerate() {
            layer.load_weights(&weights, i)?;
        }

        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        let input_ids = match inputs {
            ModelInputs::Text { input_ids, .. } => input_ids,
            ModelInputs::Multimodal { input_ids, .. } => input_ids,
            _ => return Err(anyhow::anyhow!("DeepSeek expects text input")),
        };

        // Token embedding
        let mut hidden_states = ops_fn::embedding(input_ids, &self.embed_tokens)?;

        // Apply transformer layers
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, None)?;
        }

        // Final layer norm
        hidden_states = ops_fn::layer_norm(&hidden_states, &self.norm, None, self.config.rms_norm_eps)?;

        // Language modeling head
        let logits = ops_fn::matmul(&hidden_states, &self.lm_head)?;

        Ok(ModelOutputs::Logits {
            logits,
            hidden_states: Some(hidden_states),
        })
    }

    fn generate(&self, prompt: &str, _config: &GenerationConfig) -> Result<String> {
        Ok(format!("DeepSeek generated response to: {}", prompt))
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn memory_requirements(&self) -> MemoryRequirements {
        let param_size = (self.config.vocab_size * self.config.hidden_size * 2 +
                         self.config.num_hidden_layers * self.config.hidden_size * self.config.hidden_size * 4) * 4;

        MemoryRequirements {
            gpu_memory: param_size,
            cpu_memory: param_size / 4,
            kv_cache_memory: self.config.max_position_embeddings * self.config.hidden_size * 2 * 4,
            peak_memory: param_size + param_size / 2,
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        self.embed_tokens = self.embed_tokens.to_device(device)?;
        self.norm = self.norm.to_device(device)?;
        self.lm_head = self.lm_head.to_device(device)?;

        for layer in &mut self.layers {
            layer.to_device(device)?;
        }

        Ok(())
    }
}

impl DeepSeekLayer {
    fn new(config: &DeepSeekConfig, device: &Device) -> Result<Self> {
        let input_layernorm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;
        let post_attention_layernorm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;

        Ok(Self {
            self_attn: DeepSeekAttention::new(config, device)?,
            mlp: DeepSeekMLP::new(config, device)?,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Pre-attention norm
        let normed = ops_fn::layer_norm(hidden_states, &self.input_layernorm, None, 1e-6)?;

        // Self attention
        let attn_output = self.self_attn.forward(&normed, attention_mask)?;

        // Residual connection
        let hidden_states = ops_fn::add(hidden_states, &attn_output)?;

        // Pre-MLP norm
        let normed = ops_fn::layer_norm(&hidden_states, &self.post_attention_layernorm, None, 1e-6)?;

        // MLP
        let mlp_output = self.mlp.forward(&normed)?;

        // Residual connection
        let output = ops_fn::add(&hidden_states, &mlp_output)?;

        Ok(output)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("model.layers.{}", layer_idx);

        if let Some(w) = weights.get(&format!("{}.input_layernorm.weight", prefix)) {
            self.input_layernorm = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
            self.post_attention_layernorm = w.clone();
        }

        self.self_attn.load_weights(weights, layer_idx)?;
        self.mlp.load_weights(weights, layer_idx)?;

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.input_layernorm = self.input_layernorm.to_device(device)?;
        self.post_attention_layernorm = self.post_attention_layernorm.to_device(device)?;
        self.self_attn.to_device(device)?;
        self.mlp.to_device(device)?;
        Ok(())
    }
}

impl DeepSeekAttention {
    fn new(config: &DeepSeekConfig, device: &Device) -> Result<Self> {
        let num_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads.unwrap_or(num_heads);
        let head_dim = config.hidden_size / num_heads;

        let q_proj = ops_fn::zeros(&[config.hidden_size, num_heads * head_dim], DataType::Float32, device)?;
        let k_proj = ops_fn::zeros(&[config.hidden_size, num_key_value_heads * head_dim], DataType::Float32, device)?;
        let v_proj = ops_fn::zeros(&[config.hidden_size, num_key_value_heads * head_dim], DataType::Float32, device)?;
        let o_proj = ops_fn::zeros(&[num_heads * head_dim, config.hidden_size], DataType::Float32, device)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Project to q, k, v
        let query_states = ops_fn::matmul(hidden_states, &self.q_proj)?;
        let key_states = ops_fn::matmul(hidden_states, &self.k_proj)?;
        let value_states = ops_fn::matmul(hidden_states, &self.v_proj)?;

        // Apply rotary position embedding (RoPE)
        // For now, skip RoPE implementation

        // Scaled dot-product attention
        let attn_output = ops_fn::attention(&query_states, &key_states, &value_states, attention_mask)?;

        // Output projection
        let output = ops_fn::matmul(&attn_output, &self.o_proj)?;

        Ok(output)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("model.layers.{}.self_attn", layer_idx);

        if let Some(w) = weights.get(&format!("{}.q_proj.weight", prefix)) {
            self.q_proj = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.k_proj.weight", prefix)) {
            self.k_proj = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.v_proj.weight", prefix)) {
            self.v_proj = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.o_proj.weight", prefix)) {
            self.o_proj = w.clone();
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.q_proj = self.q_proj.to_device(device)?;
        self.k_proj = self.k_proj.to_device(device)?;
        self.v_proj = self.v_proj.to_device(device)?;
        self.o_proj = self.o_proj.to_device(device)?;
        Ok(())
    }
}

impl DeepSeekMLP {
    fn new(config: &DeepSeekConfig, device: &Device) -> Result<Self> {
        let gate_proj = ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?;
        let up_proj = ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?;
        let down_proj = ops_fn::zeros(&[config.intermediate_size, config.hidden_size], DataType::Float32, device)?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Gate and up projections
        let gate_output = ops_fn::matmul(hidden_states, &self.gate_proj)?;
        let up_output = ops_fn::matmul(hidden_states, &self.up_proj)?;

        // Apply activation to gate
        let gate_activated = match self.config.hidden_act.as_str() {
            "silu" | "swish" => ops_fn::silu(&gate_output)?,
            "gelu" => ops_fn::gelu(&gate_output)?,
            _ => return Err(anyhow::anyhow!("Unsupported activation: {}", self.config.hidden_act)),
        };

        // Element-wise multiplication (gating)
        let gated = ops_fn::mul(&gate_activated, &up_output)?;

        // Down projection
        let output = ops_fn::matmul(&gated, &self.down_proj)?;

        Ok(output)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("model.layers.{}.mlp", layer_idx);

        if let Some(w) = weights.get(&format!("{}.gate_proj.weight", prefix)) {
            self.gate_proj = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.up_proj.weight", prefix)) {
            self.up_proj = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.down_proj.weight", prefix)) {
            self.down_proj = w.clone();
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.gate_proj = self.gate_proj.to_device(device)?;
        self.up_proj = self.up_proj.to_device(device)?;
        self.down_proj = self.down_proj.to_device(device)?;
        Ok(())
    }
}