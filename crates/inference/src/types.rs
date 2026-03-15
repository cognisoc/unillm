use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Result type for inference operations
pub type InferenceResult<T> = std::result::Result<T, InferenceError>;

/// Error types for inference operations
#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),

    #[error("Processing failed: {0}")]
    ProcessingFailed(String),

    #[error("GPU error: {0}")]
    GpuError(String),

    #[error("Memory error: {0}")]
    MemoryError(String),

    #[error("Timeout error: {0}")]
    Timeout(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Request identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestId(pub Uuid);

impl RequestId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

/// Model configuration for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Data types supported for model parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    Float32,
    Float16,
    BFloat16,
    Int8,
    Int4,
}

/// Sampling parameters for text generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub max_tokens: Option<usize>,
    pub stop_sequences: Vec<String>,
    pub seed: Option<u64>,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: None,
            max_tokens: None,
            stop_sequences: Vec::new(),
            seed: None,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        }
    }
}

/// Engine health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineHealth {
    pub status: HealthStatus,
    pub gpu_memory_usage: f32,
    pub active_requests: usize,
    pub queue_length: usize,
    pub average_latency_ms: f32,
    pub throughput_tokens_per_second: f32,
    pub uptime: Duration,
    pub last_error: Option<String>,
}

/// Health status enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Initializing,
    Shutdown,
}

/// Request priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RequestPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl Default for RequestPriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Execution context for requests
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub request_id: RequestId,
    pub submitted_at: Instant,
    pub started_at: Option<Instant>,
    pub priority: RequestPriority,
    pub timeout: Option<Duration>,
    pub client_id: Option<String>,
    pub session_id: Option<String>,
}

impl ExecutionContext {
    pub fn new(request_id: RequestId) -> Self {
        Self {
            request_id,
            submitted_at: Instant::now(),
            started_at: None,
            priority: RequestPriority::Normal,
            timeout: None,
            client_id: None,
            session_id: None,
        }
    }

    pub fn with_priority(mut self, priority: RequestPriority) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    pub fn with_client_id(mut self, client_id: String) -> Self {
        self.client_id = Some(client_id);
        self
    }

    pub fn mark_started(&mut self) {
        self.started_at = Some(Instant::now());
    }

    pub fn queue_time(&self) -> Duration {
        match self.started_at {
            Some(started) => started.duration_since(self.submitted_at),
            None => self.submitted_at.elapsed(),
        }
    }

    pub fn execution_time(&self) -> Option<Duration> {
        self.started_at.map(|started| started.elapsed())
    }

    pub fn is_expired(&self) -> bool {
        if let Some(timeout) = self.timeout {
            self.submitted_at.elapsed() > timeout
        } else {
            false
        }
    }
}

/// Attention mechanism types supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionMechanism {
    MultiHead,
    GroupedQuery,
    MultiQuery,
    FlashAttention,
    PagedAttention,
    RadixAttention,
    HybridCacheAttention, // UniLLM's innovation
}

/// Model architecture types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelArchitecture {
    Llama,
    Mistral,
    Qwen,
    ChatGLM,
    Baichuan,
    Custom(String),
}

/// Token information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenInfo {
    pub token_id: u32,
    pub text: String,
    pub logprob: f32,
    pub top_logprobs: Option<Vec<(u32, f32)>>,
}

/// Generation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub time_to_first_token_ms: f32,
    pub tokens_per_second: f32,
    pub total_time_ms: f32,
    pub cache_hit_rate: f32,
    pub memory_usage_mb: f32,
}

/// Batch execution configuration
#[derive(Debug, Clone)]
pub struct BatchConfig {
    pub max_batch_size: usize,
    pub max_sequence_length: usize,
    pub batch_timeout_ms: u64,
    pub enable_chunked_prefill: bool,
    pub chunk_size: Option<usize>,
    pub enable_prefix_caching: bool,
    pub dynamic_batching: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_sequence_length: 4096,
            batch_timeout_ms: 10,
            enable_chunked_prefill: true,
            chunk_size: Some(512),
            enable_prefix_caching: true,
            dynamic_batching: true,
        }
    }
}

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    pub gpu_memory_fraction: f32,
    pub enable_memory_pool: bool,
    pub kv_cache_block_size: usize,
    pub max_kv_cache_blocks: Option<usize>,
    pub swap_space_gb: Option<f32>,
    pub enable_cpu_offload: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            gpu_memory_fraction: 0.9,
            enable_memory_pool: true,
            kv_cache_block_size: 16, // tokens per block
            max_kv_cache_blocks: None,
            swap_space_gb: None,
            enable_cpu_offload: false,
        }
    }
}