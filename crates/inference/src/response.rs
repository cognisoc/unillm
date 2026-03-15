use crate::types::{RequestId, TokenInfo, GenerationStats, InferenceError};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Main inference response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    /// Request identifier this response corresponds to
    pub request_id: RequestId,

    /// Generated text content
    pub text: String,

    /// Detailed token information if requested
    pub tokens: Option<Vec<TokenInfo>>,

    /// Generation statistics and performance metrics
    pub stats: GenerationStats,

    /// Response metadata
    pub metadata: ResponseMetadata,

    /// Whether this is the final response in a stream
    pub finished: bool,

    /// Finish reason
    pub finish_reason: FinishReason,
}

/// Response metadata for tracking and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMetadata {
    /// Model used for generation
    pub model_name: String,

    /// Generation timestamp
    pub created: u64,

    /// Response stream index (for streaming responses)
    pub index: usize,

    /// Cache information
    pub cache_info: CacheInfo,

    /// GPU and memory utilization during generation
    pub resource_usage: ResourceUsage,
}

/// Cache hit information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheInfo {
    /// Cache hit rate for this request
    pub hit_rate: f32,

    /// Number of tokens served from cache
    pub cached_tokens: usize,

    /// Cache tier used (L1 radix, L2 paged, L3 compressed)
    pub cache_tier: String,

    /// Whether prefix caching was used
    pub prefix_cached: bool,
}

/// Resource usage during generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// GPU memory usage in MB
    pub gpu_memory_mb: f32,

    /// GPU utilization percentage
    pub gpu_utilization: f32,

    /// Peak memory usage during generation
    pub peak_memory_mb: f32,

    /// Energy consumption in joules (if available)
    pub energy_consumption: Option<f32>,
}

/// Reasons why generation finished
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    /// Generation completed normally
    Completed,

    /// Hit maximum token limit
    Length,

    /// Encountered stop sequence
    Stop(String),

    /// Generation was aborted
    Aborted,

    /// Error occurred during generation
    Error(String),

    /// Request timed out
    Timeout,
}

/// Streaming response chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseChunk {
    /// Request identifier
    pub request_id: RequestId,

    /// Delta text for this chunk
    pub delta: String,

    /// Token information for this chunk
    pub token: Option<TokenInfo>,

    /// Cumulative text so far
    pub cumulative_text: String,

    /// Whether this is the final chunk
    pub finished: bool,

    /// Finish reason if finished
    pub finish_reason: Option<FinishReason>,

    /// Intermediate statistics
    pub partial_stats: Option<GenerationStats>,
}

/// Response stream for streaming inference
pub struct ResponseStream {
    receiver: mpsc::UnboundedReceiver<Result<ResponseChunk, InferenceError>>,
    request_id: RequestId,
    started_at: Instant,
}

impl ResponseStream {
    /// Create a new response stream
    pub fn new(request_id: RequestId) -> (Self, mpsc::UnboundedSender<Result<ResponseChunk, InferenceError>>) {
        let (sender, receiver) = mpsc::unbounded_channel();
        let stream = Self {
            receiver,
            request_id,
            started_at: Instant::now(),
        };
        (stream, sender)
    }

    /// Get the request ID for this stream
    pub fn request_id(&self) -> RequestId {
        self.request_id
    }

    /// Get elapsed time since stream started
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Receive the next chunk from the stream
    pub async fn next(&mut self) -> Option<Result<ResponseChunk, InferenceError>> {
        self.receiver.recv().await
    }

    /// Collect all chunks into a single response
    pub async fn collect(mut self) -> Result<InferenceResponse, InferenceError> {
        let mut cumulative_text = String::new();
        let mut tokens = Vec::new();
        let mut finish_reason = FinishReason::Completed;
        let mut final_stats = None;
        let mut final_metadata = None;

        while let Some(chunk_result) = self.next().await {
            match chunk_result {
                Ok(chunk) => {
                    cumulative_text.push_str(&chunk.delta);

                    if let Some(token) = chunk.token {
                        tokens.push(token);
                    }

                    if chunk.finished {
                        finish_reason = chunk.finish_reason.unwrap_or(FinishReason::Completed);
                        final_stats = chunk.partial_stats;
                        break;
                    }
                },
                Err(e) => return Err(e),
            }
        }

        // Generate final response
        Ok(InferenceResponse {
            request_id: self.request_id,
            text: cumulative_text,
            tokens: if tokens.is_empty() { None } else { Some(tokens) },
            stats: final_stats.unwrap_or_else(|| GenerationStats {
                prompt_tokens: 0,
                completion_tokens: tokens.len(),
                total_tokens: tokens.len(),
                time_to_first_token_ms: 0.0,
                tokens_per_second: 0.0,
                total_time_ms: self.elapsed().as_millis() as f32,
                cache_hit_rate: 0.0,
                memory_usage_mb: 0.0,
            }),
            metadata: final_metadata.unwrap_or_else(|| ResponseMetadata {
                model_name: "unknown".to_string(),
                created: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                index: 0,
                cache_info: CacheInfo {
                    hit_rate: 0.0,
                    cached_tokens: 0,
                    cache_tier: "none".to_string(),
                    prefix_cached: false,
                },
                resource_usage: ResourceUsage {
                    gpu_memory_mb: 0.0,
                    gpu_utilization: 0.0,
                    peak_memory_mb: 0.0,
                    energy_consumption: None,
                },
            }),
            finished: true,
            finish_reason,
        })
    }
}

/// Response builder for constructing responses
pub struct ResponseBuilder {
    request_id: RequestId,
    text: String,
    tokens: Option<Vec<TokenInfo>>,
    model_name: String,
    stats: Option<GenerationStats>,
    finish_reason: FinishReason,
}

impl ResponseBuilder {
    /// Create a new response builder
    pub fn new(request_id: RequestId, model_name: String) -> Self {
        Self {
            request_id,
            text: String::new(),
            tokens: None,
            model_name,
            stats: None,
            finish_reason: FinishReason::Completed,
        }
    }

    /// Set the generated text
    pub fn with_text(mut self, text: String) -> Self {
        self.text = text;
        self
    }

    /// Set token information
    pub fn with_tokens(mut self, tokens: Vec<TokenInfo>) -> Self {
        self.tokens = Some(tokens);
        self
    }

    /// Set generation statistics
    pub fn with_stats(mut self, stats: GenerationStats) -> Self {
        self.stats = Some(stats);
        self
    }

    /// Set finish reason
    pub fn with_finish_reason(mut self, finish_reason: FinishReason) -> Self {
        self.finish_reason = finish_reason;
        self
    }

    /// Build the final response
    pub fn build(self) -> InferenceResponse {
        let token_count = self.tokens.as_ref().map_or(0, |t| t.len());

        InferenceResponse {
            request_id: self.request_id,
            text: self.text,
            tokens: self.tokens,
            stats: self.stats.unwrap_or_else(|| GenerationStats {
                prompt_tokens: 0,
                completion_tokens: token_count,
                total_tokens: token_count,
                time_to_first_token_ms: 0.0,
                tokens_per_second: 0.0,
                total_time_ms: 0.0,
                cache_hit_rate: 0.0,
                memory_usage_mb: 0.0,
            }),
            metadata: ResponseMetadata {
                model_name: self.model_name,
                created: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                index: 0,
                cache_info: CacheInfo {
                    hit_rate: 0.0,
                    cached_tokens: 0,
                    cache_tier: "none".to_string(),
                    prefix_cached: false,
                },
                resource_usage: ResourceUsage {
                    gpu_memory_mb: 0.0,
                    gpu_utilization: 0.0,
                    peak_memory_mb: 0.0,
                    energy_consumption: None,
                },
            },
            finished: true,
            finish_reason: self.finish_reason,
        }
    }
}

impl InferenceResponse {
    /// Create a simple text response
    pub fn text(request_id: RequestId, text: String, model_name: String) -> Self {
        ResponseBuilder::new(request_id, model_name)
            .with_text(text)
            .build()
    }

    /// Create an error response
    pub fn error(request_id: RequestId, error: String, model_name: String) -> Self {
        ResponseBuilder::new(request_id, model_name)
            .with_finish_reason(FinishReason::Error(error))
            .build()
    }

    /// Check if generation was successful
    pub fn is_success(&self) -> bool {
        !matches!(self.finish_reason, FinishReason::Error(_) | FinishReason::Aborted | FinishReason::Timeout)
    }

    /// Get total processing time
    pub fn total_time(&self) -> Duration {
        Duration::from_millis(self.stats.total_time_ms as u64)
    }

    /// Get tokens per second rate
    pub fn tokens_per_second(&self) -> f32 {
        self.stats.tokens_per_second
    }

    /// Get cache hit rate
    pub fn cache_hit_rate(&self) -> f32 {
        self.metadata.cache_info.hit_rate
    }

    /// Get GPU utilization during generation
    pub fn gpu_utilization(&self) -> f32 {
        self.metadata.resource_usage.gpu_utilization
    }
}

/// Stream sender for sending response chunks
pub struct StreamSender {
    sender: mpsc::UnboundedSender<Result<ResponseChunk, InferenceError>>,
    request_id: RequestId,
    cumulative_text: String,
    chunk_index: usize,
}

impl StreamSender {
    /// Create a new stream sender
    pub fn new(request_id: RequestId, sender: mpsc::UnboundedSender<Result<ResponseChunk, InferenceError>>) -> Self {
        Self {
            sender,
            request_id,
            cumulative_text: String::new(),
            chunk_index: 0,
        }
    }

    /// Send a text delta chunk
    pub fn send_delta(&mut self, delta: String, token: Option<TokenInfo>) -> Result<(), InferenceError> {
        self.cumulative_text.push_str(&delta);

        let chunk = ResponseChunk {
            request_id: self.request_id,
            delta,
            token,
            cumulative_text: self.cumulative_text.clone(),
            finished: false,
            finish_reason: None,
            partial_stats: None,
        };

        self.sender.send(Ok(chunk))
            .map_err(|_| InferenceError::Internal("Failed to send chunk".to_string()))?;

        self.chunk_index += 1;
        Ok(())
    }

    /// Send the final chunk with completion information
    pub fn send_final(&mut self, finish_reason: FinishReason, stats: Option<GenerationStats>) -> Result<(), InferenceError> {
        let chunk = ResponseChunk {
            request_id: self.request_id,
            delta: String::new(),
            token: None,
            cumulative_text: self.cumulative_text.clone(),
            finished: true,
            finish_reason: Some(finish_reason),
            partial_stats: stats,
        };

        self.sender.send(Ok(chunk))
            .map_err(|_| InferenceError::Internal("Failed to send final chunk".to_string()))?;

        Ok(())
    }

    /// Send an error
    pub fn send_error(&self, error: InferenceError) -> Result<(), InferenceError> {
        self.sender.send(Err(error))
            .map_err(|_| InferenceError::Internal("Failed to send error".to_string()))?;

        Ok(())
    }
}