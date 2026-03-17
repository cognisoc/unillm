//! Core traits and interfaces for UniLLM model implementations
//!
//! This module defines the fundamental traits that all model architectures must implement
//! to provide unified inference capabilities across different transformer variants.

use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::types::*;
use kv::HybridKVCache;

/// Result type for model operations
pub type ModelResult<T> = Result<T, ModelError>;

/// Errors that can occur during model operations
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Model initialization failed: {0}")]
    InitializationFailed(String),

    #[error("Forward pass failed: {0}")]
    ForwardFailed(String),

    #[error("Invalid input shape: expected {expected}, got {actual}")]
    InvalidInputShape { expected: String, actual: String },

    #[error("Memory allocation failed: {0}")]
    MemoryAllocation(String),

    #[error("Attention computation failed: {0}")]
    AttentionFailed(String),

    #[error("GPU operation failed: {0}")]
    GpuError(String),

    #[error("Quantization error: {0}")]
    QuantizationError(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
}

/// Core trait that all model architectures must implement
#[async_trait]
pub trait ModelArchitecture: Send + Sync {
    /// Get the architecture name
    fn name(&self) -> &str;

    /// Get model configuration
    fn config(&self) -> &ModelConfig;

    /// Initialize the model with given configuration
    async fn initialize(&mut self, config: ModelConfig) -> ModelResult<()>;

    /// Forward pass through the model
    async fn forward(
        &self,
        input_ids: &[u32],
        attention_mask: Option<&[bool]>,
        position_ids: Option<&[u32]>,
        kv_cache: Option<&mut HybridKVCache>,
    ) -> ModelResult<ModelOutput>;

    /// Get the vocabulary size
    fn vocab_size(&self) -> usize;

    /// Get the hidden size
    fn hidden_size(&self) -> usize;

    /// Get the number of layers
    fn num_layers(&self) -> usize;

    /// Get the number of attention heads
    fn num_heads(&self) -> usize;

    /// Get the head dimension
    fn head_dim(&self) -> usize;

    /// Check if the model supports a specific feature
    fn supports_feature(&self, feature: ModelFeature) -> bool;

    /// Get memory requirements for a given sequence length
    fn memory_requirements(&self, sequence_length: usize, batch_size: usize) -> MemoryRequirements;

    /// Prepare inputs for inference
    fn prepare_inputs(&self, inputs: &InferenceInputs) -> ModelResult<PreparedInputs>;

    /// Post-process model outputs
    fn post_process_outputs(&self, outputs: ModelOutput) -> ModelResult<InferenceOutput>;
}

/// Features that models may support
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelFeature {
    /// Flash Attention optimization
    FlashAttention,
    /// Grouped Query Attention
    GroupedQueryAttention,
    /// Multi-Query Attention
    MultiQueryAttention,
    /// Sliding Window Attention
    SlidingWindowAttention,
    /// Rotary Position Embedding
    RotaryEmbedding,
    /// ALiBi Position Embedding
    ALiBiEmbedding,
    /// RMSNorm normalization
    RMSNorm,
    /// LayerNorm normalization
    LayerNorm,
    /// SwiGLU activation
    SwiGLU,
    /// GeGLU activation
    GeGLU,
    /// Mixture of Experts
    MixtureOfExperts,
    /// State Space Models (Mamba)
    StateSpaceModel,
    /// Prefix caching
    PrefixCaching,
    /// Chunked prefill
    ChunkedPrefill,
    /// Dynamic batching
    DynamicBatching,
    /// Continuous batching
    ContinuousBatching,
    /// Speculative decoding
    SpeculativeDecoding,
    /// Multi-modal inputs
    MultiModal,
    /// Function calling
    FunctionCalling,
    /// Long context (>32K tokens)
    LongContext,
}

/// Model output structure
#[derive(Debug, Clone)]
pub struct ModelOutput {
    /// Output logits [batch_size, sequence_length, vocab_size]
    pub logits: Tensor,
    /// Hidden states from all layers
    pub hidden_states: Option<Vec<Tensor>>,
    /// Attention weights from all layers
    pub attention_weights: Option<Vec<Tensor>>,
    /// KV cache states
    pub kv_cache_states: Option<HashMap<String, Tensor>>,
    /// Additional model-specific outputs
    pub auxiliary_outputs: HashMap<String, Tensor>,
}

/// Memory requirements for model execution
#[derive(Debug, Clone)]
pub struct MemoryRequirements {
    /// GPU memory required in bytes
    pub gpu_memory_bytes: usize,
    /// CPU memory required in bytes
    pub cpu_memory_bytes: usize,
    /// KV cache memory required in bytes
    pub kv_cache_bytes: usize,
    /// Peak memory usage during forward pass
    pub peak_memory_bytes: usize,
    /// Estimated memory fragmentation overhead
    pub fragmentation_overhead: f32,
}

/// Prepared inputs for model execution
#[derive(Debug, Clone)]
pub struct PreparedInputs {
    /// Token IDs [batch_size, sequence_length]
    pub input_ids: Tensor,
    /// Attention mask [batch_size, sequence_length]
    pub attention_mask: Option<Tensor>,
    /// Position IDs [batch_size, sequence_length]
    pub position_ids: Option<Tensor>,
    /// Input embeddings (for multimodal models)
    pub input_embeddings: Option<Tensor>,
    /// Additional model-specific inputs
    pub auxiliary_inputs: HashMap<String, Tensor>,
}

/// Tensor abstraction for different backends
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: DataType,
    /// Device location
    pub device: Device,
    /// Raw data pointer (implementation specific)
    pub data_ptr: u64,
    /// Stride information
    pub strides: Vec<usize>,
}

/// Device enumeration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Device {
    CPU,
    CUDA(u32),      // Device ID
    ROCM(u32),      // Device ID
    Intel(u32),     // XPU Device ID
    Metal(u32),     // Metal Device ID
}

/// Attention mechanism trait
#[async_trait]
pub trait AttentionMechanism: Send + Sync {
    /// Compute attention for given query, key, value tensors
    async fn compute_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
        kv_cache: Option<&mut HybridKVCache>,
        position_ids: Option<&Tensor>,
    ) -> ModelResult<AttentionOutput>;

    /// Get the attention mechanism type
    fn attention_type(&self) -> AttentionType;

    /// Check if the mechanism supports specific features
    fn supports_feature(&self, feature: AttentionFeature) -> bool;
}

/// Different types of attention mechanisms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionType {
    MultiHead,
    GroupedQuery,
    MultiQuery,
    FlashAttention,
    PagedAttention,
    RadixAttention,
    SlidingWindow,
    HybridCache, // UniLLM's innovation
}

/// Features specific to attention mechanisms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionFeature {
    CausalMask,
    BidirectionalMask,
    ALiBi,
    RotaryEmbedding,
    RelativePosition,
    SlidingWindow,
    Chunking,
    KVCaching,
    GQA,
    MQA,
}

/// Output from attention computation
#[derive(Debug, Clone)]
pub struct AttentionOutput {
    /// Attention output tensor
    pub output: Tensor,
    /// Attention weights (optional)
    pub weights: Option<Tensor>,
    /// Updated KV cache
    pub kv_cache_update: Option<HashMap<String, Tensor>>,
}

/// Feed-forward network trait
#[async_trait]
pub trait FeedForwardNetwork: Send + Sync {
    /// Forward pass through the FFN
    async fn forward(&self, input: &Tensor) -> ModelResult<Tensor>;

    /// Get the FFN type
    fn ffn_type(&self) -> FeedForwardType;

    /// Get intermediate size
    fn intermediate_size(&self) -> usize;
}

/// Types of feed-forward networks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedForwardType {
    MLP,
    SwiGLU,
    GeGLU,
    MoE,
    GLU,
    ReGLU,
}

/// Normalization layer trait
#[async_trait]
pub trait NormalizationLayer: Send + Sync {
    /// Apply normalization
    async fn normalize(&self, input: &Tensor) -> ModelResult<Tensor>;

    /// Get normalization type
    fn norm_type(&self) -> NormalizationType;
}

/// Types of normalization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationType {
    LayerNorm,
    RMSNorm,
    GroupNorm,
    BatchNorm,
}

/// Position embedding trait
#[async_trait]
pub trait PositionEmbedding: Send + Sync {
    /// Apply position embeddings
    async fn apply_position_embedding(
        &self,
        input: &Tensor,
        position_ids: &Tensor,
    ) -> ModelResult<Tensor>;

    /// Get embedding type
    fn embedding_type(&self) -> PositionEmbeddingType;

    /// Get maximum sequence length supported
    fn max_sequence_length(&self) -> usize;
}

/// Types of position embeddings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionEmbeddingType {
    Learned,
    Sinusoidal,
    Rotary,
    ALiBi,
    RelativePosition,
}

/// Tokenization trait
#[async_trait]
pub trait Tokenizer: Send + Sync {
    /// Encode text to token IDs
    async fn encode(&self, text: &str) -> ModelResult<Vec<u32>>;

    /// Decode token IDs to text
    async fn decode(&self, token_ids: &[u32]) -> ModelResult<String>;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;

    /// Get special token IDs
    fn special_tokens(&self) -> &SpecialTokens;

    /// Check if tokenizer supports a feature
    fn supports_feature(&self, feature: TokenizerFeature) -> bool;
}

/// Special token definitions
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub pad_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub bos_token_id: Option<u32>,
    pub unk_token_id: Option<u32>,
    pub sep_token_id: Option<u32>,
    pub cls_token_id: Option<u32>,
    pub mask_token_id: Option<u32>,
}

/// Tokenizer features
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerFeature {
    BytePairEncoding,
    SentencePiece,
    WordPiece,
    ChatTemplate,
    FunctionCalling,
    MultiLingual,
}

/// Quantization trait for model compression
#[async_trait]
pub trait Quantization: Send + Sync {
    /// Quantize a tensor
    async fn quantize(&self, tensor: &Tensor) -> ModelResult<QuantizedTensor>;

    /// Dequantize a tensor
    async fn dequantize(&self, tensor: &QuantizedTensor) -> ModelResult<Tensor>;

    /// Get quantization method
    fn quantization_method(&self) -> QuantizationMethod;

    /// Get compression ratio
    fn compression_ratio(&self) -> f32;
}

/// Quantized tensor representation
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized data
    pub data: Tensor,
    /// Scale factors
    pub scales: Option<Tensor>,
    /// Zero points
    pub zero_points: Option<Tensor>,
    /// Quantization parameters
    pub params: QuantizationParams,
}

/// Quantization methods supported
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMethod {
    FP16,
    BF16,
    FP8,
    INT8,
    INT4,
    GPTQ,
    AWQ,
    SmoothQuant,
    SqueezeQuant,
    GGUF,
    BitNet,
}

/// Quantization parameters
#[derive(Debug, Clone)]
pub struct QuantizationParams {
    pub bits: u8,
    pub group_size: Option<usize>,
    pub symmetric: bool,
    pub channel_wise: bool,
    pub block_size: Option<usize>,
}

/// Model parallelism trait
#[async_trait]
pub trait ModelParallelism: Send + Sync {
    /// Split model across devices
    async fn split_model(&self, num_devices: usize) -> ModelResult<Vec<ModelShard>>;

    /// Combine outputs from multiple shards
    async fn combine_outputs(&self, shard_outputs: &[ModelOutput]) -> ModelResult<ModelOutput>;

    /// Get parallelism strategy
    fn parallelism_strategy(&self) -> ParallelismStrategy;

    /// Get communication requirements
    fn communication_requirements(&self) -> CommunicationRequirements;
}

/// Model shard for distributed execution
#[derive(Debug, Clone)]
pub struct ModelShard {
    pub shard_id: usize,
    pub device: Device,
    pub layers: Vec<usize>,
    pub parameters: HashMap<String, Tensor>,
}

/// Parallelism strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelismStrategy {
    TensorParallel,
    PipelineParallel,
    DataParallel,
    Hybrid,
}

/// Communication requirements for distributed execution
#[derive(Debug, Clone)]
pub struct CommunicationRequirements {
    pub all_reduce_size: usize,
    pub all_gather_size: usize,
    pub point_to_point_size: usize,
    pub bandwidth_requirements: f32, // GB/s
    pub latency_tolerance: f32,      // ms
}

/// Model factory trait for creating model instances
pub trait ModelFactory: Send + Sync {
    /// Create a model instance
    fn create_model(
        &self,
        architecture: ModelArchitecture,
        config: ModelConfig,
    ) -> ModelResult<Box<dyn ModelArchitecture>>;

    /// List supported architectures
    fn supported_architectures(&self) -> Vec<String>;

    /// Detect architecture from model path
    fn detect_architecture(&self, model_path: &str) -> ModelResult<String>;
}

/// Performance profiler trait
#[async_trait]
pub trait PerformanceProfiler: Send + Sync {
    /// Start profiling
    async fn start_profiling(&mut self) -> ModelResult<()>;

    /// Stop profiling and get results
    async fn stop_profiling(&mut self) -> ModelResult<ProfileResults>;

    /// Profile a specific operation
    async fn profile_operation<F, T>(&mut self, name: &str, operation: F) -> ModelResult<T>
    where
        F: Future<Output = ModelResult<T>> + Send,
        T: Send;
}

/// Profiling results
#[derive(Debug, Clone)]
pub struct ProfileResults {
    pub total_time_ms: f64,
    pub operations: HashMap<String, OperationProfile>,
    pub memory_usage: MemoryProfile,
    pub gpu_utilization: f32,
}

/// Profile for individual operations
#[derive(Debug, Clone)]
pub struct OperationProfile {
    pub execution_time_ms: f64,
    pub memory_allocated: usize,
    pub gpu_kernel_time_ms: f64,
    pub cpu_time_ms: f64,
    pub call_count: u64,
}

/// Memory usage profile
#[derive(Debug, Clone)]
pub struct MemoryProfile {
    pub peak_gpu_memory: usize,
    pub peak_cpu_memory: usize,
    pub kv_cache_usage: usize,
    pub model_weights_size: usize,
}

use std::future::Future;

/// Extension trait for enhanced model capabilities
#[async_trait]
pub trait ModelExtensions: ModelArchitecture {
    /// Benchmark the model performance
    async fn benchmark(&self, config: BenchmarkConfig) -> ModelResult<BenchmarkResults>;

    /// Optimize the model for inference
    async fn optimize_for_inference(&mut self, optimization: OptimizationConfig) -> ModelResult<()>;

    /// Export model to different formats
    async fn export_model(&self, format: ExportFormat, path: &str) -> ModelResult<()>;

    /// Validate model correctness
    async fn validate(&self, validation_data: &[ValidationCase]) -> ModelResult<ValidationResults>;
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub sequence_lengths: Vec<usize>,
    pub batch_sizes: Vec<usize>,
    pub num_iterations: usize,
    pub warmup_iterations: usize,
    pub measure_memory: bool,
    pub measure_throughput: bool,
    pub measure_latency: bool,
}

/// Benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub average_latency_ms: f64,
    pub throughput_tokens_per_second: f64,
    pub peak_memory_usage: usize,
    pub per_sequence_length: HashMap<usize, PerformanceMetrics>,
    pub per_batch_size: HashMap<usize, PerformanceMetrics>,
}

/// Performance metrics for specific configurations
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub latency_ms: f64,
    pub throughput: f64,
    pub memory_usage: usize,
    pub gpu_utilization: f32,
}

/// Optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    pub enable_fusion: bool,
    pub enable_quantization: bool,
    pub quantization_method: Option<QuantizationMethod>,
    pub enable_pruning: bool,
    pub pruning_ratio: f32,
    pub enable_distillation: bool,
    pub teacher_model: Option<String>,
}

/// Export formats supported
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    ONNX,
    TensorRT,
    OpenVINO,
    CoreML,
    TorchScript,
    TensorFlowLite,
    GGUF,
    SafeTensors,
}

/// Validation case for model testing
#[derive(Debug, Clone)]
pub struct ValidationCase {
    pub input: String,
    pub expected_output: Option<String>,
    pub expected_logits: Option<Vec<f32>>,
    pub tolerance: f32,
}

/// Validation results
#[derive(Debug, Clone)]
pub struct ValidationResults {
    pub passed: usize,
    pub failed: usize,
    pub accuracy: f32,
    pub failed_cases: Vec<ValidationFailure>,
}

/// Details about validation failures
#[derive(Debug, Clone)]
pub struct ValidationFailure {
    pub case_index: usize,
    pub error: String,
    pub actual_output: String,
    pub expected_output: String,
    pub similarity_score: f32,
}