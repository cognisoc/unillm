//! Core Model Abstraction
//!
//! This module provides the foundational model abstraction that all
//! model implementations build upon. It defines clean interfaces that
//! hide implementation complexity.

use crate::tensor_core::{Tensor, Device};
use std::collections::HashMap;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Core model trait - the single interface all models implement
pub trait Model: Send + Sync {
    /// Model configuration type
    type Config: ModelConfig;

    /// Create new model instance
    fn new(config: Self::Config) -> Result<Self> where Self: Sized;

    /// Load model with weights
    fn from_weights(config: Self::Config, weights: ModelWeights) -> Result<Self> where Self: Sized;

    /// Forward pass - core inference method
    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs>;

    /// Generate text (high-level interface)
    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String>;

    /// Get model configuration
    fn config(&self) -> &Self::Config;

    /// Get model memory requirements
    fn memory_requirements(&self) -> MemoryRequirements;

    /// Move model to device
    fn to_device(&mut self, device: &Device) -> Result<()>;
}

/// Model configuration trait
pub trait ModelConfig: Send + Sync + std::fmt::Debug {
    /// Get model architecture name
    fn architecture(&self) -> &str;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;

    /// Get hidden dimension
    fn hidden_size(&self) -> usize;

    /// Get number of layers
    fn num_layers(&self) -> usize;

    /// Validate configuration
    fn validate(&self) -> Result<()>;
}

/// Unified model inputs - supports all model types
#[derive(Debug, Clone)]
pub enum ModelInputs {
    /// Text-only inputs (most language models)
    Text {
        input_ids: Tensor,
        attention_mask: Option<Tensor>,
        position_ids: Option<Tensor>,
    },
    /// Image-only inputs (vision models)
    Image {
        pixel_values: Tensor,
        image_mask: Option<Tensor>,
    },
    /// Multimodal inputs (vision-language models)
    Multimodal {
        input_ids: Tensor,
        pixel_values: Option<Tensor>,
        attention_mask: Option<Tensor>,
        image_mask: Option<Tensor>,
    },
    /// Audio inputs (speech models)
    Audio {
        input_features: Tensor,
        attention_mask: Option<Tensor>,
    },
}

/// Unified model outputs - supports all model types
#[derive(Debug, Clone)]
pub enum ModelOutputs {
    /// Language model logits
    Logits {
        logits: Tensor,
        hidden_states: Option<Tensor>,
    },
    /// Embeddings/representations
    Embeddings {
        embeddings: Tensor,
        pooled: Option<Tensor>,
    },
    /// Multimodal outputs
    Multimodal {
        text_logits: Option<Tensor>,
        image_logits: Option<Tensor>,
        text_embeddings: Option<Tensor>,
        image_embeddings: Option<Tensor>,
    },
    /// Sequence-to-sequence outputs
    Sequence {
        logits: Tensor,
        encoder_hidden_states: Option<Tensor>,
        decoder_hidden_states: Option<Tensor>,
    },
    /// CLIP contrastive outputs
    CLIP {
        logits_per_text: Tensor,
        logits_per_image: Tensor,
        text_embeds: Tensor,
        image_embeds: Tensor,
    },
}

/// Model weights container
#[derive(Debug, Clone)]
pub struct ModelWeights {
    /// Tensor weights by name
    pub tensors: HashMap<String, Tensor>,
    /// Metadata
    pub metadata: WeightMetadata,
    /// GGUF-specific config (populated when loading from GGUF)
    pub gguf_config: Option<crate::weight_loader_core::GGUFModelConfig>,
    /// GGUF tokenizer data (populated when loading from GGUF)
    pub gguf_tokenizer: Option<crate::weight_loader_core::GGUFTokenizer>,
}

/// Weight metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightMetadata {
    /// Model architecture
    pub architecture: String,
    /// Total parameters
    pub total_params: usize,
    /// Weight format
    pub format: WeightFormat,
    /// Precision
    pub dtype: String,
}

/// Supported weight formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WeightFormat {
    SafeTensors,
    PyTorch,
    GGUF,
    HuggingFace,
}

/// Generation configuration
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub do_sample: bool,
    pub repetition_penalty: f32,
    pub stop_sequences: Vec<String>,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 100,
            temperature: 1.0,
            top_p: 0.9,
            top_k: None,
            do_sample: true,
            repetition_penalty: 1.0,
            stop_sequences: vec![],
            eos_token_id: 2,
            pad_token_id: 0,
        }
    }
}

/// Memory requirements for a model
#[derive(Debug, Clone)]
pub struct MemoryRequirements {
    /// GPU memory in bytes
    pub gpu_memory: usize,
    /// CPU memory in bytes
    pub cpu_memory: usize,
    /// KV cache memory in bytes
    pub kv_cache_memory: usize,
    /// Peak memory during inference
    pub peak_memory: usize,
}

/// Model factory trait for creating models by name
pub trait ModelFactory: Send + Sync {
    /// Create model from configuration
    fn create_model(&self, config_json: &str) -> Result<Box<dyn Model<Config = Box<dyn ModelConfig>>>>;

    /// Get supported architecture names
    fn supported_architectures(&self) -> Vec<&str>;

    /// Check if architecture is supported
    fn supports(&self, architecture: &str) -> bool;
}

/// Model registry for managing all supported architectures
pub struct ModelRegistry {
    factories: HashMap<String, Box<dyn ModelFactory>>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Register a model factory
    pub fn register<F>(&mut self, architecture: &str, factory: F)
    where
        F: ModelFactory + 'static,
    {
        self.factories.insert(architecture.to_string(), Box::new(factory));
    }

    /// Create model by architecture name
    pub fn create_model(&self, architecture: &str, config_json: &str) -> Result<Box<dyn Model<Config = Box<dyn ModelConfig>>>> {
        let factory = self.factories.get(architecture)
            .ok_or_else(|| anyhow::anyhow!("Unsupported architecture: {}", architecture))?;

        factory.create_model(config_json)
    }

    /// Get all supported architectures
    pub fn supported_architectures(&self) -> Vec<String> {
        self.factories.keys().cloned().collect()
    }

    /// Check if architecture is supported
    pub fn supports(&self, architecture: &str) -> bool {
        self.factories.contains_key(architecture)
    }
}

/// Global model registry
static MODEL_REGISTRY: std::sync::OnceLock<std::sync::Mutex<ModelRegistry>> = std::sync::OnceLock::new();

/// Get global model registry
pub fn registry() -> &'static std::sync::Mutex<ModelRegistry> {
    MODEL_REGISTRY.get_or_init(|| std::sync::Mutex::new(ModelRegistry::new()))
}

/// Utility functions for model inputs/outputs
impl ModelInputs {
    /// Create text inputs
    pub fn text(input_ids: Tensor) -> Self {
        Self::Text {
            input_ids,
            attention_mask: None,
            position_ids: None,
        }
    }

    /// Create text inputs with attention mask
    pub fn text_with_mask(input_ids: Tensor, attention_mask: Tensor) -> Self {
        Self::Text {
            input_ids,
            attention_mask: Some(attention_mask),
            position_ids: None,
        }
    }

    /// Create image inputs
    pub fn image(pixel_values: Tensor) -> Self {
        Self::Image {
            pixel_values,
            image_mask: None,
        }
    }

    /// Create multimodal inputs
    pub fn multimodal(input_ids: Tensor, pixel_values: Option<Tensor>) -> Self {
        Self::Multimodal {
            input_ids,
            pixel_values,
            attention_mask: None,
            image_mask: None,
        }
    }

    /// Get batch size
    pub fn batch_size(&self) -> usize {
        match self {
            Self::Text { input_ids, .. } => input_ids.shape()[0],
            Self::Image { pixel_values, .. } => pixel_values.shape()[0],
            Self::Multimodal { input_ids, .. } => input_ids.shape()[0],
            Self::Audio { input_features, .. } => input_features.shape()[0],
        }
    }

    /// Get sequence length (for text inputs)
    pub fn sequence_length(&self) -> Option<usize> {
        match self {
            Self::Text { input_ids, .. } => Some(input_ids.shape()[1]),
            Self::Multimodal { input_ids, .. } => Some(input_ids.shape()[1]),
            _ => None,
        }
    }
}

impl ModelOutputs {
    /// Create logits output
    pub fn logits(logits: Tensor) -> Self {
        Self::Logits {
            logits,
            hidden_states: None,
        }
    }

    /// Create embeddings output
    pub fn embeddings(embeddings: Tensor) -> Self {
        Self::Embeddings {
            embeddings,
            pooled: None,
        }
    }

    /// Get main tensor output
    pub fn main_tensor(&self) -> &Tensor {
        match self {
            Self::Logits { logits, .. } => logits,
            Self::Embeddings { embeddings, .. } => embeddings,
            Self::Multimodal { text_logits: Some(logits), .. } => logits,
            Self::Multimodal { image_logits: Some(logits), .. } => logits,
            Self::Sequence { logits, .. } => logits,
            _ => panic!("No main tensor available"),
        }
    }
}

impl ModelWeights {
    /// Create new model weights
    pub fn new(tensors: HashMap<String, Tensor>, metadata: WeightMetadata) -> Self {
        Self { tensors, metadata, gguf_config: None, gguf_tokenizer: None }
    }

    /// Create new model weights with GGUF config
    pub fn with_gguf_config(
        tensors: HashMap<String, Tensor>,
        metadata: WeightMetadata,
        gguf_config: crate::weight_loader_core::GGUFModelConfig,
    ) -> Self {
        Self { tensors, metadata, gguf_config: Some(gguf_config), gguf_tokenizer: None }
    }

    /// Create new model weights with GGUF config and tokenizer
    pub fn with_gguf_config_and_tokenizer(
        tensors: HashMap<String, Tensor>,
        metadata: WeightMetadata,
        gguf_config: crate::weight_loader_core::GGUFModelConfig,
        gguf_tokenizer: Option<crate::weight_loader_core::GGUFTokenizer>,
    ) -> Self {
        Self { tensors, metadata, gguf_config: Some(gguf_config), gguf_tokenizer }
    }

    /// Get tensor by name
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Get tensor by name (required)
    pub fn require(&self, name: &str) -> Result<&Tensor> {
        self.tensors.get(name)
            .ok_or_else(|| anyhow::anyhow!("Required tensor '{}' not found", name))
    }

    /// Get all tensor names
    pub fn tensor_names(&self) -> Vec<&String> {
        self.tensors.keys().collect()
    }

    /// Move all tensors to device
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        for tensor in self.tensors.values_mut() {
            *tensor = tensor.to_device(device)?;
        }
        Ok(())
    }
}

// ============================================================================
// Mixture of Experts (MoE) Support
// ============================================================================

/// MLP trait for expert networks in MoE layers
pub trait MLPLayer: Send + Sync {
    /// Forward pass through the MLP
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor>;
}

/// Configuration for MoE layer
#[derive(Debug, Clone)]
pub struct MoEConfig {
    /// Number of experts
    pub num_experts: usize,
    /// Number of experts activated per token
    pub num_experts_per_tok: usize,
    /// Whether to use auxiliary load balancing loss
    pub aux_loss: bool,
    /// Router type (softmax, top-k, etc.)
    pub router_type: RouterType,
}

impl Default for MoEConfig {
    fn default() -> Self {
        Self {
            num_experts: 8,
            num_experts_per_tok: 2,
            aux_loss: true,
            router_type: RouterType::TopK,
        }
    }
}

/// Router types for MoE
#[derive(Debug, Clone)]
pub enum RouterType {
    /// Standard top-k routing
    TopK,
    /// Expert choice routing (each expert chooses tokens)
    ExpertChoice,
    /// Soft routing with weighted combination
    Soft,
}

/// Mixture of Experts layer
/// Routes tokens to top-k experts and combines their outputs
#[derive(Debug)]
pub struct MoELayer {
    /// Router weights: [hidden_size, num_experts]
    pub router_weights: Tensor,
    /// Number of experts
    pub num_experts: usize,
    /// Number of experts per token
    pub num_experts_per_tok: usize,
    /// Device
    pub device: Device,
}

impl MoELayer {
    /// Create new MoE layer
    pub fn new(
        router_weights: Tensor,
        num_experts: usize,
        num_experts_per_tok: usize,
        device: Device,
    ) -> Self {
        Self {
            router_weights,
            num_experts,
            num_experts_per_tok,
            device,
        }
    }

    /// Compute routing weights and indices for each token
    /// Returns (routing_weights, expert_indices) where:
    /// - routing_weights: [batch * seq_len, num_experts_per_tok] - normalized weights
    /// - expert_indices: [batch * seq_len, num_experts_per_tok] - expert indices
    pub fn route(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor)> {
        use crate::tensor_core::ops_fn;

        // hidden_states: [batch, seq_len, hidden_size]
        let shape = hidden_states.shape();
        let (batch, seq_len, _hidden_size) = (shape[0], shape[1], shape[2]);
        let num_tokens = batch * seq_len;

        // Reshape to [batch * seq_len, hidden_size]
        let flat_hidden = hidden_states.reshape(&[num_tokens, shape[2]])?;

        // Compute router logits: [batch * seq_len, num_experts]
        let router_logits = ops_fn::matmul(&flat_hidden, &self.router_weights)?;

        // Get top-k experts
        let (topk_weights, topk_indices) = ops_fn::topk(&router_logits, self.num_experts_per_tok, -1)?;

        // Softmax over the selected experts to get normalized weights
        let routing_weights = ops_fn::softmax(&topk_weights, -1)?;

        Ok((routing_weights, topk_indices))
    }

    /// Forward pass through MoE layer
    /// expert_fn: function that takes (hidden_states, expert_idx) and returns expert output
    pub fn forward_with_experts<F>(&self, hidden_states: &Tensor, expert_fn: F) -> Result<Tensor>
    where
        F: Fn(&Tensor, usize) -> Result<Tensor>,
    {
        use crate::tensor_core::ops_fn;

        let shape = hidden_states.shape();
        let (batch, seq_len, hidden_size) = (shape[0], shape[1], shape[2]);
        let num_tokens = batch * seq_len;

        // Get routing weights and indices
        let (routing_weights, expert_indices) = self.route(hidden_states)?;

        // Flatten hidden states for processing
        let flat_hidden = hidden_states.reshape(&[num_tokens, hidden_size])?;

        // Initialize output tensor
        let mut output = ops_fn::zeros(&[num_tokens, hidden_size], hidden_states.dtype(), &self.device)?;

        // Process each expert
        // This is a basic implementation - production would batch tokens by expert
        for expert_idx in 0..self.num_experts {
            // Find tokens routed to this expert
            // For each position in top-k, check if it matches this expert
            let expert_indices_candle = expert_indices.to_candle()?;
            let routing_weights_candle = routing_weights.to_candle()?;

            for tok_idx in 0..num_tokens {
                for k in 0..self.num_experts_per_tok {
                    let idx_val: Vec<i64> = expert_indices_candle.get(tok_idx)?.to_vec1()?;
                    if idx_val[k] as usize == expert_idx {
                        // Get this token's hidden state
                        let token_hidden = flat_hidden.to_candle()?.get(tok_idx)?;
                        let token_tensor = Tensor::from_candle(token_hidden.unsqueeze(0)?);

                        // Get expert output
                        let expert_output = expert_fn(&token_tensor, expert_idx)?;

                        // Get routing weight for this expert
                        let weight_val: Vec<f32> = routing_weights_candle.get(tok_idx)?.to_vec1()?;
                        let weight = weight_val[k];

                        // Scale by routing weight and add to output
                        let scaled_output = ops_fn::scale(&expert_output, weight)?;

                        // Add to output at this position
                        let output_candle = output.to_candle()?;
                        let current = output_candle.get(tok_idx)?;
                        let new_val = (current + scaled_output.to_candle()?.squeeze(0)?)?;

                        // Update output tensor at this position
                        // This is inefficient but works for the basic implementation
                        let mut output_data: Vec<f32> = output.to_candle()?.flatten_all()?.to_vec1()?;
                        let new_data: Vec<f32> = new_val.to_vec1()?;
                        for (i, v) in new_data.iter().enumerate() {
                            output_data[tok_idx * hidden_size + i] = *v;
                        }
                        output = Tensor::from_f32_slice(&output_data, &[num_tokens, hidden_size], &self.device)?;
                    }
                }
            }
        }

        // Reshape back to [batch, seq_len, hidden_size]
        output.reshape(&[batch, seq_len, hidden_size])
    }
}

/// Helper struct for a standard MoE expert (simple MLP)
#[derive(Debug)]
pub struct MoEExpert {
    /// Gate projection: hidden_size -> intermediate_size
    pub gate_proj: Tensor,
    /// Up projection: hidden_size -> intermediate_size
    pub up_proj: Tensor,
    /// Down projection: intermediate_size -> hidden_size
    pub down_proj: Tensor,
}

impl MoEExpert {
    pub fn new(gate_proj: Tensor, up_proj: Tensor, down_proj: Tensor) -> Self {
        Self { gate_proj, up_proj, down_proj }
    }

    /// Forward pass: SwiGLU activation
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        use crate::tensor_core::ops_fn;

        // SwiGLU: down(silu(gate(x)) * up(x))
        let gate = ops_fn::matmul(hidden_states, &self.gate_proj)?;
        let gate_activated = ops_fn::silu(&gate)?;
        let up = ops_fn::matmul(hidden_states, &self.up_proj)?;
        let gated = ops_fn::mul(&gate_activated, &up)?;
        ops_fn::matmul(&gated, &self.down_proj)
    }
}

// ============================================================================
// Model Configuration Macro
// ============================================================================

/// Helper macro for creating model configurations
#[macro_export]
macro_rules! model_config {
    ($name:ident {
        $($field:ident: $type:ty = $default:expr),* $(,)?
    }) => {
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct $name {
            $(pub $field: $type,)*
        }

        impl Default for $name {
            fn default() -> Self {
                Self {
                    $($field: $default,)*
                }
            }
        }

        impl ModelConfig for $name {
            fn architecture(&self) -> &str {
                stringify!($name)
            }

            fn vocab_size(&self) -> usize {
                self.vocab_size
            }

            fn hidden_size(&self) -> usize {
                self.hidden_size
            }

            fn num_layers(&self) -> usize {
                self.num_hidden_layers
            }

            fn validate(&self) -> Result<()> {
                if self.vocab_size() == 0 {
                    return Err(anyhow::anyhow!("vocab_size must be > 0"));
                }
                if self.hidden_size() == 0 {
                    return Err(anyhow::anyhow!("hidden_size must be > 0"));
                }
                if self.num_layers() == 0 {
                    return Err(anyhow::anyhow!("num_layers must be > 0"));
                }
                Ok(())
            }
        }
    };
}

// Example usage would be:
// model_config!(LlamaConfig {
//     vocab_size: usize = 32000,
//     hidden_size: usize = 4096,
//     // ... other fields
// });