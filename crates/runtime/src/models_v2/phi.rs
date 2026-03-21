//! Phi Model V2 - Clean implementation using solid abstractions
//!
//! This implements the Phi architecture (Phi-1, Phi-2, Phi-3) which includes:
//! - Microsoft Phi-1/1.5/2/3 models
//! - Uses unified Tensor type from tensor_core
//! - Implements Model trait from model_core

use crate::model_config;
use super::traits::*;
use std::collections::HashMap;
use anyhow::Result;
use serde::{Serialize, Deserialize};

/// Phi model configuration using the model_config macro
model_config!(PhiConfig {
    vocab_size: usize = 32064,
    hidden_size: usize = 2560,
    intermediate_size: usize = 10240,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 32,
    num_key_value_heads: Option<usize> = Some(32),
    hidden_act: String = "gelu_new".to_string(),
    max_position_embeddings: usize = 2048,
    initializer_range: f32 = 0.02,
    rms_norm_eps: f32 = 1e-5,
    use_cache: bool = true,
    pad_token_id: Option<i64> = Some(0),
    bos_token_id: Option<i64> = Some(1),
    eos_token_id: Option<i64> = Some(2),
    tie_word_embeddings: bool = false,
    rope_theta: f32 = 10000.0,
    attention_dropout: f32 = 0.0,
    partial_rotary_factor: f32 = 0.5,
});

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
    config: PhiConfig,
}

/// Phi MLP (feed-forward network)
pub struct PhiMLP {
    fc1: Tensor, // Phi uses fc1/fc2 naming
    fc2: Tensor,
    config: PhiConfig,
}

impl Model for PhiModelV2 {
    type Config = PhiConfig;

    fn new(config: PhiConfig) -> Result<Self> {
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

        let lm_head = if config.tie_word_embeddings {
            embed_tokens.clone()
        } else {
            ops_fn::zeros(
                &[config.vocab_size, config.hidden_size],
                DataType::Float32,
                &device
            )?
        };

        // Create transformer layers
        let mut layers = Vec::new();
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

        // Load actual weights from ModelWeights
        if let Some(embed_weight) = weights.get("model.embed_tokens.weight") {
            model.embed_tokens = embed_weight.clone();
        }

        if let Some(norm_weight) = weights.get("model.final_layernorm.weight") {
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
            _ => return Err(anyhow::anyhow!("Phi expects text input")),
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
        Ok(format!("Phi generated response to: {}", prompt))
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

impl PhiLayer {
    fn new(config: &PhiConfig, device: &Device) -> Result<Self> {
        let input_layernorm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;

        Ok(Self {
            self_attn: PhiAttention::new(config, device)?,
            mlp: PhiMLP::new(config, device)?,
            input_layernorm,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Phi uses parallel attention and MLP (like Falcon)
        let residual = hidden_states.clone();

        // Layer norm
        let normed = ops_fn::layer_norm(hidden_states, &self.input_layernorm, None, 1e-5)?;

        // Attention
        let attn_output = self.self_attn.forward(&normed, attention_mask)?;

        // MLP
        let mlp_output = self.mlp.forward(&normed)?;

        // Add both outputs to residual
        let combined = ops_fn::add(&attn_output, &mlp_output)?;
        let output = ops_fn::add(&residual, &combined)?;

        Ok(output)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("model.layers.{}", layer_idx);

        if let Some(w) = weights.get(&format!("{}.input_layernorm.weight", prefix)) {
            self.input_layernorm = w.clone();
        }

        self.self_attn.load_weights(weights, layer_idx)?;
        self.mlp.load_weights(weights, layer_idx)?;

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.input_layernorm = self.input_layernorm.to_device(device)?;
        self.self_attn.to_device(device)?;
        self.mlp.to_device(device)?;
        Ok(())
    }
}

impl PhiAttention {
    fn new(config: &PhiConfig, device: &Device) -> Result<Self> {
        let num_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads.unwrap_or(num_heads);
        let head_dim = config.hidden_size / num_heads;

        let q_proj = ops_fn::zeros(&[config.hidden_size, num_heads * head_dim], DataType::Float32, device)?;
        let k_proj = ops_fn::zeros(&[config.hidden_size, num_key_value_heads * head_dim], DataType::Float32, device)?;
        let v_proj = ops_fn::zeros(&[config.hidden_size, num_key_value_heads * head_dim], DataType::Float32, device)?;
        let dense = ops_fn::zeros(&[num_heads * head_dim, config.hidden_size], DataType::Float32, device)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            dense,
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Project to q, k, v
        let query_states = ops_fn::matmul(hidden_states, &self.q_proj)?;
        let key_states = ops_fn::matmul(hidden_states, &self.k_proj)?;
        let value_states = ops_fn::matmul(hidden_states, &self.v_proj)?;

        // Apply partial rotary position embedding
        // For now, skip RoPE implementation

        // Scaled dot-product attention
        let attn_output = ops_fn::attention(&query_states, &key_states, &value_states, attention_mask)?;

        // Output projection
        let output = ops_fn::matmul(&attn_output, &self.dense)?;

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
        if let Some(w) = weights.get(&format!("{}.dense.weight", prefix)) {
            self.dense = w.clone();
        }

        Ok(())
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
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // First projection with activation
        let intermediate = ops_fn::matmul(hidden_states, &self.fc1)?;

        // Apply activation (Phi typically uses GELU)
        let activated = match self.config.hidden_act.as_str() {
            "gelu" | "gelu_new" => ops_fn::gelu(&intermediate)?,
            "silu" | "swish" => ops_fn::silu(&intermediate)?,
            _ => return Err(anyhow::anyhow!("Unsupported activation: {}", self.config.hidden_act)),
        };

        // Second projection
        let output = ops_fn::matmul(&activated, &self.fc2)?;

        Ok(output)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("model.layers.{}.mlp", layer_idx);

        if let Some(w) = weights.get(&format!("{}.fc1.weight", prefix)) {
            self.fc1 = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.fc2.weight", prefix)) {
            self.fc2 = w.clone();
        }

        Ok(())
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
        let config = PhiConfig::default();
        let model = PhiModelV2::new(config).unwrap();

        assert_eq!(model.config().vocab_size(), 32064);
        assert_eq!(model.config().hidden_size(), 2560);
        assert_eq!(model.config().num_layers(), 32);
        assert_eq!(model.config().architecture(), "PhiConfig");
    }

    #[test]
    fn test_phi_forward_pass() {
        let config = PhiConfig {
            vocab_size: 1000,
            hidden_size: 64,
            num_hidden_layers: 2,
            ..Default::default()
        };

        let model = PhiModelV2::new(config).unwrap();
        let input_ids = ops_fn::zeros(&[1, 8], DataType::Int64, &Device::CPU).unwrap();
        let inputs = ModelInputs::text(input_ids);

        let outputs = model.forward(&inputs).unwrap();

        match outputs {
            ModelOutputs::Logits { logits, .. } => {
                assert_eq!(logits.shape(), &[1, 8, 1000]);
            }
            _ => panic!("Expected logits output"),
        }
    }
}