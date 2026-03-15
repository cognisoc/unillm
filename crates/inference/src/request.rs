use crate::types::{RequestId, SamplingParams, ExecutionContext, RequestPriority, AttentionMechanism};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Main inference request structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// Unique request identifier
    pub request_id: RequestId,

    /// Input prompt text
    pub prompt: String,

    /// Optional conversation messages for chat models
    pub messages: Option<Vec<ChatMessage>>,

    /// Sampling parameters for generation
    pub sampling_params: SamplingParams,

    /// Request metadata
    pub metadata: RequestMetadata,

    /// Execution context (not serialized)
    #[serde(skip)]
    pub context: ExecutionContext,
}

/// Chat message for conversation-based models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,  // "system", "user", "assistant"
    pub content: String,
    pub name: Option<String>,
}

/// Request metadata for tracking and optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetadata {
    /// Client identifier
    pub client_id: Option<String>,

    /// Session identifier for conversation tracking
    pub session_id: Option<String>,

    /// Request priority
    pub priority: RequestPriority,

    /// Maximum processing timeout
    pub timeout: Option<Duration>,

    /// Preferred attention mechanism
    pub attention_mechanism: Option<AttentionMechanism>,

    /// Enable streaming response
    pub stream: bool,

    /// Include logprobs in response
    pub logprobs: Option<usize>,

    /// Enable echo of input prompt
    pub echo: bool,

    /// Custom tags for request categorization
    pub tags: Vec<String>,

    /// Cache key for prefix caching
    pub cache_key: Option<String>,
}

impl Default for RequestMetadata {
    fn default() -> Self {
        Self {
            client_id: None,
            session_id: None,
            priority: RequestPriority::Normal,
            timeout: None,
            attention_mechanism: None,
            stream: false,
            logprobs: None,
            echo: false,
            tags: Vec::new(),
            cache_key: None,
        }
    }
}

/// Builder pattern for creating inference requests
pub struct RequestBuilder {
    request: InferenceRequest,
}

impl RequestBuilder {
    /// Create a new request builder with a prompt
    pub fn new(prompt: String) -> Self {
        let request_id = RequestId::new();
        let context = ExecutionContext::new(request_id);

        Self {
            request: InferenceRequest {
                request_id,
                prompt,
                messages: None,
                sampling_params: SamplingParams::default(),
                metadata: RequestMetadata::default(),
                context,
            },
        }
    }

    /// Create a request builder for chat conversations
    pub fn chat(messages: Vec<ChatMessage>) -> Self {
        let request_id = RequestId::new();
        let context = ExecutionContext::new(request_id);

        // Convert messages to a simple prompt for now
        // In a real implementation, this would use the model's chat template
        let prompt = messages.iter()
            .map(|msg| format!("{}: {}", msg.role, msg.content))
            .collect::<Vec<_>>()
            .join("\n");

        Self {
            request: InferenceRequest {
                request_id,
                prompt,
                messages: Some(messages),
                sampling_params: SamplingParams::default(),
                metadata: RequestMetadata::default(),
                context,
            },
        }
    }

    /// Set sampling parameters
    pub fn with_sampling_params(mut self, params: SamplingParams) -> Self {
        self.request.sampling_params = params;
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.request.sampling_params.temperature = temperature;
        self
    }

    /// Set maximum tokens to generate
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.request.sampling_params.max_tokens = Some(max_tokens);
        self
    }

    /// Set top-p sampling
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.request.sampling_params.top_p = top_p;
        self
    }

    /// Set top-k sampling
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.request.sampling_params.top_k = Some(top_k);
        self
    }

    /// Add stop sequences
    pub fn with_stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.request.sampling_params.stop_sequences = stop_sequences;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.request.sampling_params.seed = Some(seed);
        self
    }

    /// Set request priority
    pub fn with_priority(mut self, priority: RequestPriority) -> Self {
        self.request.metadata.priority = priority;
        self.request.context.priority = priority;
        self
    }

    /// Set request timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.request.metadata.timeout = Some(timeout);
        self.request.context = self.request.context.with_timeout(timeout);
        self
    }

    /// Set client identifier
    pub fn with_client_id(mut self, client_id: String) -> Self {
        self.request.metadata.client_id = Some(client_id.clone());
        self.request.context = self.request.context.with_client_id(client_id);
        self
    }

    /// Set session identifier
    pub fn with_session_id(mut self, session_id: String) -> Self {
        self.request.metadata.session_id = Some(session_id);
        self
    }

    /// Enable streaming response
    pub fn with_streaming(mut self) -> Self {
        self.request.metadata.stream = true;
        self
    }

    /// Include logprobs in response
    pub fn with_logprobs(mut self, logprobs: usize) -> Self {
        self.request.metadata.logprobs = Some(logprobs);
        self
    }

    /// Enable echo of input prompt
    pub fn with_echo(mut self) -> Self {
        self.request.metadata.echo = true;
        self
    }

    /// Set preferred attention mechanism
    pub fn with_attention_mechanism(mut self, mechanism: AttentionMechanism) -> Self {
        self.request.metadata.attention_mechanism = Some(mechanism);
        self
    }

    /// Add tags for request categorization
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.request.metadata.tags = tags;
        self
    }

    /// Set cache key for prefix caching
    pub fn with_cache_key(mut self, cache_key: String) -> Self {
        self.request.metadata.cache_key = Some(cache_key);
        self
    }

    /// Build the final request
    pub fn build(self) -> InferenceRequest {
        self.request
    }
}

impl InferenceRequest {
    /// Create a simple text generation request
    pub fn new(prompt: String) -> Self {
        RequestBuilder::new(prompt).build()
    }

    /// Create a chat conversation request
    pub fn chat(messages: Vec<ChatMessage>) -> Self {
        RequestBuilder::chat(messages).build()
    }

    /// Get estimated request complexity for scheduling
    pub fn estimated_complexity(&self) -> f32 {
        let prompt_length = self.prompt.len() as f32;
        let max_tokens = self.sampling_params.max_tokens.unwrap_or(100) as f32;
        let temperature_factor = if self.sampling_params.temperature > 1.0 { 1.2 } else { 1.0 };

        // Simple complexity estimation based on prompt length and generation parameters
        (prompt_length + max_tokens) * temperature_factor
    }

    /// Check if request has expired based on timeout
    pub fn is_expired(&self) -> bool {
        self.context.is_expired()
    }

    /// Get queue time since submission
    pub fn queue_time(&self) -> Duration {
        self.context.queue_time()
    }

    /// Get execution time if started
    pub fn execution_time(&self) -> Option<Duration> {
        self.context.execution_time()
    }

    /// Mark request as started
    pub fn mark_started(&mut self) {
        self.context.mark_started();
    }

    /// Extract conversation context for prefix caching
    pub fn conversation_context(&self) -> Option<String> {
        if let Some(session_id) = &self.metadata.session_id {
            // In a real implementation, this would retrieve conversation history
            // For now, return the session ID as a simple cache key
            Some(format!("session:{}", session_id))
        } else {
            None
        }
    }

    /// Generate cache key for this request
    pub fn generate_cache_key(&self) -> String {
        if let Some(cache_key) = &self.metadata.cache_key {
            cache_key.clone()
        } else if let Some(context) = self.conversation_context() {
            format!("{}:{}", context, self.prompt.len())
        } else {
            // Use a hash of the prompt for caching
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            self.prompt.hash(&mut hasher);
            format!("prompt:{:x}", hasher.finish())
        }
    }

    /// Validate request parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.prompt.is_empty() && self.messages.as_ref().map_or(true, |m| m.is_empty()) {
            return Err("Request must have either prompt or messages".to_string());
        }

        if self.sampling_params.temperature < 0.0 || self.sampling_params.temperature > 2.0 {
            return Err("Temperature must be between 0.0 and 2.0".to_string());
        }

        if self.sampling_params.top_p < 0.0 || self.sampling_params.top_p > 1.0 {
            return Err("Top-p must be between 0.0 and 1.0".to_string());
        }

        if let Some(max_tokens) = self.sampling_params.max_tokens {
            if max_tokens == 0 || max_tokens > 8192 {
                return Err("Max tokens must be between 1 and 8192".to_string());
            }
        }

        Ok(())
    }
}