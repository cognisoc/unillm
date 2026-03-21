//! Core Model Abstraction
//!
//! This module provides the foundational model abstraction that all
//! model implementations build upon. It defines clean interfaces that
//! hide implementation complexity.

use crate::tensor_core::{Tensor, Device, DataType};
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
}

/// Model weights container
#[derive(Debug, Clone)]
pub struct ModelWeights {
    /// Tensor weights by name
    pub tensors: HashMap<String, Tensor>,
    /// Metadata
    pub metadata: WeightMetadata,
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
        Self { tensors, metadata }
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