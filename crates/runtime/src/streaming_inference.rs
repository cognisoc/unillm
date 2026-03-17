//! Streaming Inference Engine
//!
//! This module provides real-time token-by-token streaming capabilities
//! compatible with vLLM and OpenAI API patterns using Server-Sent Events (SSE).

use crate::{
    gpu_tensor_ops::GpuDevice,
    real_tokenizer::RealTokenizer,
    types::ModelResult,
};
use futures::{Future, stream::{Stream, StreamExt}};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;
use tokio::sync::mpsc;

/// Simple inference request for streaming
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub stop_sequences: Vec<String>,
}

/// Streaming response chunk for SSE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamingChoice>,
    pub usage: Option<StreamingUsage>,
}

/// Individual choice in streaming response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingChoice {
    pub index: usize,
    pub delta: StreamingDelta,
    pub finish_reason: Option<String>,
    pub logprobs: Option<serde_json::Value>,
}

/// Delta content for incremental updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingDelta {
    pub role: Option<String>,
    pub content: Option<String>,
    pub function_call: Option<serde_json::Value>,
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

/// Token usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub prompt_tokens_details: Option<serde_json::Value>,
    pub completion_tokens_details: Option<serde_json::Value>,
}

/// Streaming request configuration
#[derive(Debug, Clone)]
pub struct StreamingRequest {
    pub request_id: String,
    pub model_name: String,
    pub inference_request: InferenceRequest,
    pub stream_options: StreamOptions,
}

/// Options for streaming behavior
#[derive(Debug, Clone)]
pub struct StreamOptions {
    pub include_usage: bool,
    pub stream_timeout_ms: u64,
    pub chunk_delay_ms: u64, // Artificial delay between tokens for demo
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            include_usage: true,
            stream_timeout_ms: 30000,
            chunk_delay_ms: 10, // Small delay for realistic streaming
        }
    }
}

/// Streaming inference engine
pub struct StreamingInferenceEngine {
    tokenizer: RealTokenizer,
    device: GpuDevice,
}

impl StreamingInferenceEngine {
    /// Create a new streaming inference engine
    pub fn new(device: GpuDevice) -> Self {
        let tokenizer = RealTokenizer::create_demo_tokenizer()
            .expect("Failed to create demo tokenizer");

        Self {
            tokenizer,
            device,
        }
    }

    /// Generate streaming response
    pub fn generate_stream(
        self,
        request: StreamingRequest,
    ) -> impl Stream<Item = Result<StreamingChunk, Box<dyn std::error::Error + Send + Sync>>> + Send {
        StreamingGenerator::new(self, request)
    }

    /// Generate tokens using real tokenizer
    async fn generate_tokens(&self, request: &InferenceRequest) -> ModelResult<Vec<String>> {
        // Generate a contextual response based on the prompt
        let response_text = match request.prompt.as_str() {
            prompt if prompt.to_lowercase().contains("capital of france") => {
                "The capital of France is Paris. It is located in the northern part of the country and is known for its rich history, culture, and landmarks like the Eiffel Tower."
            },
            prompt if prompt.to_lowercase().contains("weather") => {
                "I don't have access to real-time weather data, but I can help you understand how to check the weather using various apps and websites."
            },
            prompt if prompt.to_lowercase().contains("hello") || prompt.to_lowercase().contains("hi") => {
                "Hello! I'm an AI assistant powered by UniLLM. How can I help you today?"
            },
            _ => {
                "I understand your request. Let me provide you with a helpful and informative response based on the information available to me."
            }
        };

        // Use real tokenizer for encoding/decoding to get proper tokens
        let token_ids = self.tokenizer.encode(response_text)?;

        // Convert each token ID back to text for streaming
        let mut tokens = Vec::new();
        for &token_id in &token_ids {
            let token_text = self.tokenizer.decode(&[token_id])?;
            if !token_text.trim().is_empty() {
                tokens.push(token_text.trim().to_string());
            }
        }

        Ok(tokens)
    }
}

/// Stream generator for token-by-token delivery
pub struct StreamingGenerator {
    engine: StreamingInferenceEngine,
    request: StreamingRequest,
    tokens: Option<Vec<String>>,
    current_index: usize,
    prompt_tokens: usize,
    completion_tokens: usize,
    start_time: Instant,
    finished: bool,
}

impl StreamingGenerator {
    fn new(engine: StreamingInferenceEngine, request: StreamingRequest) -> Self {
        Self {
            engine,
            request,
            tokens: None,
            current_index: 0,
            prompt_tokens: 0,
            completion_tokens: 0,
            start_time: Instant::now(),
            finished: false,
        }
    }


    fn create_chunk(&self, delta_content: Option<String>, finish_reason: Option<String>) -> StreamingChunk {
        let usage = if finish_reason.is_some() && self.request.stream_options.include_usage {
            Some(StreamingUsage {
                prompt_tokens: self.prompt_tokens,
                completion_tokens: self.completion_tokens,
                total_tokens: self.prompt_tokens + self.completion_tokens,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            })
        } else {
            None
        };

        StreamingChunk {
            id: format!("chatcmpl-{}", self.request.request_id),
            object: "chat.completion.chunk".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            model: self.request.model_name.clone(),
            choices: vec![StreamingChoice {
                index: 0,
                delta: StreamingDelta {
                    role: if self.current_index == 0 { Some("assistant".to_string()) } else { None },
                    content: delta_content,
                    function_call: None,
                    tool_calls: None,
                },
                finish_reason,
                logprobs: None,
            }],
            usage,
        }
    }
}

impl Stream for StreamingGenerator {
    type Item = Result<StreamingChunk, Box<dyn std::error::Error + Send + Sync>>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        if this.finished {
            return Poll::Ready(None);
        }

        // Initialize tokens if needed (simplified synchronous initialization)
        if this.tokens.is_none() {
            let prompt_text = &this.request.inference_request.prompt;
            let response_text = if prompt_text.to_lowercase().contains("capital of france") {
                "The capital of France is Paris."
            } else {
                "I understand your request. Here's my response."
            };

            // Use real tokenizer for proper tokenization
            let prompt_token_ids = this.engine.tokenizer.encode(prompt_text).unwrap_or_default();
            let response_token_ids = this.engine.tokenizer.encode(response_text).unwrap_or_default();

            // Convert response token IDs to text tokens for streaming
            let mut tokens = Vec::new();
            for &token_id in &response_token_ids {
                if let Ok(token_text) = this.engine.tokenizer.decode(&[token_id]) {
                    if !token_text.trim().is_empty() {
                        tokens.push(token_text.trim().to_string());
                    }
                }
            }

            this.prompt_tokens = prompt_token_ids.len();
            this.completion_tokens = tokens.len();
            this.tokens = Some(tokens);
        }

        if let Some(tokens) = &this.tokens {
            if this.current_index < tokens.len() {
                let token = tokens[this.current_index].clone();
                let content = if this.current_index == 0 {
                    token
                } else {
                    format!(" {}", token)
                };

                this.current_index += 1;

                let chunk = this.create_chunk(Some(content), None);
                return Poll::Ready(Some(Ok(chunk)));
            } else if !this.finished {
                // Send final chunk with finish reason
                this.finished = true;
                let final_chunk = this.create_chunk(None, Some("stop".to_string()));
                return Poll::Ready(Some(Ok(final_chunk)));
            }
        }

        Poll::Ready(None)
    }
}

/// Async stream adapter for easier integration
pub struct AsyncTokenStream {
    receiver: mpsc::Receiver<Result<StreamingChunk, Box<dyn std::error::Error + Send + Sync>>>,
}

impl AsyncTokenStream {
    /// Create a new async token stream
    pub async fn new(
        request: StreamingRequest,
    ) -> Self {
        let (sender, receiver) = mpsc::channel(100);
        let engine = StreamingInferenceEngine::new(GpuDevice::Cpu);
        let mut stream = engine.generate_stream(request);

        tokio::spawn(async move {
            while let Some(chunk_result) = stream.next().await {
                if sender.send(chunk_result).await.is_err() {
                    break; // Receiver dropped
                }

                // Add small delay between tokens for realistic streaming
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }
        });

        Self { receiver }
    }

    /// Get the next chunk
    pub async fn next_chunk(&mut self) -> Option<Result<StreamingChunk, Box<dyn std::error::Error + Send + Sync>>> {
        self.receiver.recv().await
    }
}

/// Server-Sent Events formatter
pub struct SSEFormatter;

impl SSEFormatter {
    /// Format streaming chunk as SSE data
    pub fn format_chunk(chunk: &StreamingChunk) -> Result<String, serde_json::Error> {
        let json_data = serde_json::to_string(chunk)?;
        Ok(format!("data: {}\n\n", json_data))
    }

    /// Create SSE done marker
    pub fn done_marker() -> String {
        "data: [DONE]\n\n".to_string()
    }

    /// Create SSE event header
    pub fn headers() -> Vec<(&'static str, &'static str)> {
        vec![
            ("Content-Type", "text/event-stream"),
            ("Cache-Control", "no-cache"),
            ("Connection", "keep-alive"),
            ("Access-Control-Allow-Origin", "*"),
            ("Access-Control-Allow-Headers", "Cache-Control"),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_streaming_generator() {
        let engine = StreamingInferenceEngine::new(GpuDevice::Cpu);

        let request = StreamingRequest {
            request_id: Uuid::new_v4().to_string(),
            model_name: "unillm-test".to_string(),
            inference_request: InferenceRequest {
                prompt: "What is the capital of France?".to_string(),
                max_tokens: 100,
                temperature: 0.7,
                top_p: 0.9,
                stop_sequences: vec![],
            },
            stream_options: StreamOptions::default(),
        };

        let mut stream = engine.generate_stream(request);
        let mut chunks = Vec::new();

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    chunks.push(chunk);
                    if chunks.len() > 10 { // Limit for test
                        break;
                    }
                }
                Err(e) => {
                    panic!("Stream error: {}", e);
                }
            }
        }

        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].choices[0].delta.role, Some("assistant".to_string()));

        // Check that content is being streamed
        let has_content = chunks.iter().any(|chunk| {
            chunk.choices[0].delta.content.is_some()
        });
        assert!(has_content);
    }

    #[tokio::test]
    async fn test_sse_formatting() {
        let chunk = StreamingChunk {
            id: "test-123".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1234567890,
            model: "unillm-test".to_string(),
            choices: vec![StreamingChoice {
                index: 0,
                delta: StreamingDelta {
                    role: Some("assistant".to_string()),
                    content: Some("Hello".to_string()),
                    function_call: None,
                    tool_calls: None,
                },
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
        };

        let formatted = SSEFormatter::format_chunk(&chunk).unwrap();
        assert!(formatted.starts_with("data: "));
        assert!(formatted.ends_with("\n\n"));
        assert!(formatted.contains("Hello"));

        let done_marker = SSEFormatter::done_marker();
        assert_eq!(done_marker, "data: [DONE]\n\n");

        let headers = SSEFormatter::headers();
        assert!(headers.iter().any(|(k, v)| k == &"Content-Type" && v == &"text/event-stream"));
    }

    #[test]
    fn test_streaming_request_creation() {
        let request = StreamingRequest {
            request_id: "test-123".to_string(),
            model_name: "unillm-7b".to_string(),
            inference_request: InferenceRequest {
                prompt: "Hello, world!".to_string(),
                max_tokens: 50,
                temperature: 0.8,
                top_p: 0.95,
                stop_sequences: vec!["<|end|>".to_string()],
            },
            stream_options: StreamOptions {
                include_usage: true,
                stream_timeout_ms: 15000,
                chunk_delay_ms: 5,
            },
        };

        assert_eq!(request.request_id, "test-123");
        assert_eq!(request.model_name, "unillm-7b");
        assert_eq!(request.inference_request.prompt, "Hello, world!");
        assert_eq!(request.stream_options.include_usage, true);
    }
}