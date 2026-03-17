//! Core types for UniLLM runtime
//!
//! This module defines fundamental data structures used throughout the runtime.

use std::collections::HashMap;
use std::fmt;

/// Data types supported by tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Float32,
    Float16,
    BFloat16,
    Int32,
    Int64,
    Int8,
    Int4,
    Bool,
}

impl DataType {
    pub fn size_bytes(&self) -> usize {
        match self {
            DataType::Float32 | DataType::Int32 => 4,
            DataType::Float16 | DataType::BFloat16 => 2,
            DataType::Int64 => 8,
            DataType::Int8 => 1,
            DataType::Int4 => 1, // Packed
            DataType::Bool => 1,
        }
    }
}

/// Device types for tensor placement
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Device {
    CPU,
    CUDA(usize),  // GPU ID
    ROCm(usize),  // GPU ID
    Intel(usize), // XPU ID
    Metal(usize), // GPU ID
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::CPU => write!(f, "cpu"),
            Device::CUDA(id) => write!(f, "cuda:{}", id),
            Device::ROCm(id) => write!(f, "rocm:{}", id),
            Device::Intel(id) => write!(f, "intel:{}", id),
            Device::Metal(id) => write!(f, "metal:{}", id),
        }
    }
}

/// Tensor structure
#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub dtype: DataType,
    pub device: Device,
    pub data_ptr: usize, // Pointer to actual data (0 for unallocated)
    pub strides: Vec<usize>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, dtype: DataType, device: Device) -> Self {
        let strides = calculate_strides(&shape);
        Self {
            shape,
            dtype,
            device,
            data_ptr: 0,
            strides,
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn size_bytes(&self) -> usize {
        self.numel() * self.dtype.size_bytes()
    }
}

/// Model configuration
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_name: String,
    pub model_path: String,
    pub max_sequence_length: usize,
    pub vocabulary_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub dtype: DataType,
}

/// Model features that can be supported
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFeature {
    FlashAttention,
    GroupedQueryAttention,
    RotaryEmbedding,
    RMSNorm,
    LayerNorm,
    SwiGLU,
    GELU,
    PrefixCaching,
    ChunkedPrefill,
    DynamicBatching,
    ContinuousBatching,
    LongContext,
    SlidingWindow,
    Quantization,
    LoRA,
}

/// Memory requirements for a model
#[derive(Debug, Clone)]
pub struct MemoryRequirements {
    pub gpu_memory_bytes: usize,
    pub cpu_memory_bytes: usize,
    pub kv_cache_bytes: usize,
    pub peak_memory_bytes: usize,
    pub fragmentation_overhead: f32,
}

/// Model output structure
#[derive(Debug, Clone)]
pub struct ModelOutput {
    pub logits: Tensor,
    pub hidden_states: Option<Vec<Tensor>>,
    pub attention_weights: Option<Vec<Tensor>>,
    pub kv_cache_states: Option<HashMap<String, Tensor>>,
    pub auxiliary_outputs: HashMap<String, Tensor>,
}

/// Prepared inputs for model execution
#[derive(Debug, Clone)]
pub struct PreparedInputs {
    pub input_ids: Tensor,
    pub attention_mask: Option<Tensor>,
    pub position_ids: Option<Tensor>,
    pub input_embeddings: Option<Tensor>,
    pub auxiliary_inputs: HashMap<String, Tensor>,
}

/// Error types for model operations
#[derive(Debug)]
pub enum ModelError {
    InitializationFailed(String),
    ComputationFailed(String),
    InvalidInput(String),
    DeviceError(String),
    MemoryError(String),
    UnsupportedOperation(String),
    GenerationFailed(String),
    ValidationFailed(String),
    ServerError(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::InitializationFailed(msg) => write!(f, "Initialization failed: {}", msg),
            ModelError::ComputationFailed(msg) => write!(f, "Computation failed: {}", msg),
            ModelError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            ModelError::DeviceError(msg) => write!(f, "Device error: {}", msg),
            ModelError::MemoryError(msg) => write!(f, "Memory error: {}", msg),
            ModelError::UnsupportedOperation(msg) => write!(f, "Unsupported operation: {}", msg),
            ModelError::GenerationFailed(msg) => write!(f, "Generation failed: {}", msg),
            ModelError::ValidationFailed(msg) => write!(f, "Validation failed: {}", msg),
            ModelError::ServerError(msg) => write!(f, "Server error: {}", msg),
        }
    }
}

impl std::error::Error for ModelError {}

/// Result type for model operations
pub type ModelResult<T> = Result<T, ModelError>;

/// Normalization types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationType {
    LayerNorm,
    RMSNorm,
    GroupNorm,
}

/// Position embedding types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionEmbeddingType {
    Absolute,
    Relative,
    Rotary,
    ALiBi,
}

/// Generation statistics
#[derive(Debug, Clone)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub time_to_first_token_ms: f64,
    pub tokens_per_second: f64,
    pub total_time_ms: f64,
    pub cache_hit_rate: f64,
    pub memory_usage_mb: f64,
}

/// Inference inputs
#[derive(Debug, Clone)]
pub struct InferenceInputs {
    pub batch_size: usize,
    pub sequence_length: usize,
    pub attention_mask: Option<Vec<bool>>,
    pub position_ids: Option<Vec<u32>>,
}

/// Inference output
#[derive(Debug, Clone)]
pub struct InferenceOutput {
    pub text: String,
    pub logits: Option<Vec<f32>>,
    pub hidden_states: Option<Vec<Tensor>>,
    pub attention_weights: Option<Vec<Tensor>>,
    pub generation_stats: Option<GenerationStats>,
}

/// Calculate strides for a given shape (row-major order)
pub fn calculate_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_size_bytes() {
        assert_eq!(DataType::Float32.size_bytes(), 4);
        assert_eq!(DataType::Float16.size_bytes(), 2);
        assert_eq!(DataType::BFloat16.size_bytes(), 2);
        assert_eq!(DataType::Int32.size_bytes(), 4);
        assert_eq!(DataType::Int64.size_bytes(), 8);
        assert_eq!(DataType::Int8.size_bytes(), 1);
        assert_eq!(DataType::Int4.size_bytes(), 1);
        assert_eq!(DataType::Bool.size_bytes(), 1);
    }

    #[test]
    fn test_device_display() {
        assert_eq!(Device::CPU.to_string(), "cpu");
        assert_eq!(Device::CUDA(0).to_string(), "cuda:0");
        assert_eq!(Device::CUDA(5).to_string(), "cuda:5");
        assert_eq!(Device::ROCm(2).to_string(), "rocm:2");
        assert_eq!(Device::Intel(1).to_string(), "intel:1");
        assert_eq!(Device::Metal(0).to_string(), "metal:0");
    }

    #[test]
    fn test_tensor_creation() {
        let tensor = Tensor::new(vec![2, 3, 4], DataType::Float32, Device::CPU);
        assert_eq!(tensor.shape, vec![2, 3, 4]);
        assert_eq!(tensor.dtype, DataType::Float32);
        assert_eq!(tensor.device, Device::CPU);
        assert_eq!(tensor.numel(), 24);
        assert_eq!(tensor.size_bytes(), 96); // 24 * 4 bytes
        assert_eq!(tensor.strides, vec![12, 4, 1]); // Row-major strides
    }

    #[test]
    fn test_tensor_numel() {
        let tensor1 = Tensor::new(vec![5], DataType::Float32, Device::CPU);
        assert_eq!(tensor1.numel(), 5);

        let tensor2 = Tensor::new(vec![2, 3], DataType::Float32, Device::CPU);
        assert_eq!(tensor2.numel(), 6);

        let tensor3 = Tensor::new(vec![2, 3, 4], DataType::Float32, Device::CPU);
        assert_eq!(tensor3.numel(), 24);
    }

    #[test]
    fn test_tensor_size_bytes() {
        let tensor_f32 = Tensor::new(vec![10], DataType::Float32, Device::CPU);
        assert_eq!(tensor_f32.size_bytes(), 40);

        let tensor_f16 = Tensor::new(vec![10], DataType::Float16, Device::CPU);
        assert_eq!(tensor_f16.size_bytes(), 20);

        let tensor_i8 = Tensor::new(vec![10], DataType::Int8, Device::CPU);
        assert_eq!(tensor_i8.size_bytes(), 10);
    }

    #[test]
    fn test_model_config_creation() {
        let config = ModelConfig {
            model_name: "test_model".to_string(),
            model_path: "/path/to/model".to_string(),
            max_sequence_length: 2048,
            vocabulary_size: 32000,
            num_layers: 32,
            num_heads: 32,
            head_dim: 128,
            hidden_size: 4096,
            intermediate_size: 11008,
            dtype: DataType::Float16,
        };

        assert_eq!(config.model_name, "test_model");
        assert_eq!(config.max_sequence_length, 2048);
        assert_eq!(config.vocabulary_size, 32000);
        assert_eq!(config.dtype, DataType::Float16);
    }

    #[test]
    fn test_model_error_display() {
        let error = ModelError::InitializationFailed("test error".to_string());
        assert_eq!(error.to_string(), "Initialization failed: test error");

        let error = ModelError::ComputationFailed("math error".to_string());
        assert_eq!(error.to_string(), "Computation failed: math error");

        let error = ModelError::InvalidInput("bad input".to_string());
        assert_eq!(error.to_string(), "Invalid input: bad input");
    }

    #[test]
    fn test_memory_requirements() {
        let req = MemoryRequirements {
            gpu_memory_bytes: 1000000,
            cpu_memory_bytes: 500000,
            kv_cache_bytes: 200000,
            peak_memory_bytes: 1500000,
            fragmentation_overhead: 0.2,
        };

        assert_eq!(req.gpu_memory_bytes, 1000000);
        assert_eq!(req.fragmentation_overhead, 0.2);
    }
}