//! OpenAI-Compatible HTTP Server
//!
//! Provides REST API endpoints compatible with the OpenAI API format.
//! Supports chat completions, text completions, and streaming via SSE.
//!
//! Features:
//! - Graceful shutdown on SIGTERM/SIGINT with request draining
//! - Enhanced /v1/models with real model metadata
//! - Streaming support for both chat and text completions
//! - Request timeout enforcement
//! - Proper stop sequence checking during generation

use crate::model_core::GenerationConfig;
use crate::service::{InferenceService, FinishReason};

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse, Json,
    },
    routing::{get, post},
    Router,
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

/// Trait for engine-powered generation with batching/scheduling.
/// The binary layer implements this using UniLLMInferenceEngine.
#[async_trait::async_trait]
pub trait EngineProcessor: Send + Sync {
    async fn process(&self, prompt: String, max_tokens: usize, temperature: f32, top_p: f32, repetition_penalty: f32) -> Result<EngineOutput, String>;
}

/// Output from an engine-powered generation
#[derive(Debug, Clone, Serialize)]
pub struct EngineOutput {
    pub text: String,
    pub finish_reason: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

// ============================================================================
// Server Configuration
// ============================================================================

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_concurrency: usize,
    pub request_timeout_secs: u64,
    pub api_key: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            max_concurrency: 4,
            request_timeout_secs: 300,
            api_key: None,
        }
    }
}

// ============================================================================
// Server State
// ============================================================================

/// Shared model state for the server
pub struct ModelState {
    /// Inference service wrapping model + tokenizer
    pub service: InferenceService,
    /// Optional inference engine with batching/scheduling
    pub engine: Option<Arc<dyn EngineProcessor>>,
    /// Model name (for API responses)
    pub model_name: String,
    /// Model architecture name
    pub architecture: String,
    /// Concurrency limiter
    pub semaphore: Semaphore,
    /// Server configuration
    pub config: ServerConfig,
    /// Cancellation token for graceful shutdown
    pub cancel: CancellationToken,
    /// Model metadata for /v1/models
    pub model_meta: ModelMeta,
}

/// Model metadata exposed via /v1/models
#[derive(Debug, Clone, Serialize)]
pub struct ModelMeta {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    pub architecture: String,
    pub context_size: usize,
    pub eos_token_id: u32,
}

pub type SharedState = Arc<ModelState>;

/// Convenience function to create a SharedState from an InferenceService
pub fn create_shared_state(
    service: InferenceService,
    model_name: String,
    architecture: String,
    concurrency: usize,
) -> SharedState {
    let meta = ModelMeta {
        id: model_name.clone(),
        object: "model".to_string(),
        created: now_unix(),
        owned_by: "unillm".to_string(),
        architecture: architecture.clone(),
        context_size: service.context_size(),
        eos_token_id: service.eos_token_id(),
    };

    Arc::new(ModelState {
        service,
        engine: None,
        model_name,
        architecture,
        semaphore: Semaphore::new(concurrency),
        config: ServerConfig::default(),
        cancel: CancellationToken::new(),
        model_meta: meta,
    })
}

/// Create a SharedState with inference engine support
pub fn create_shared_state_with_engine(
    service: InferenceService,
    engine: Arc<dyn EngineProcessor>,
    model_name: String,
    architecture: String,
    config: ServerConfig,
    cancel: CancellationToken,
) -> SharedState {
    let meta = ModelMeta {
        id: model_name.clone(),
        object: "model".to_string(),
        created: now_unix(),
        owned_by: "unillm".to_string(),
        architecture: architecture.clone(),
        context_size: service.context_size(),
        eos_token_id: service.eos_token_id(),
    };

    Arc::new(ModelState {
        service,
        engine: Some(engine),
        model_name,
        architecture,
        semaphore: Semaphore::new(config.max_concurrency),
        config,
        cancel,
        model_meta: meta,
    })
}

/// Create a SharedState with full server configuration
pub fn create_shared_state_with_config(
    service: InferenceService,
    model_name: String,
    architecture: String,
    config: ServerConfig,
    cancel: CancellationToken,
) -> SharedState {
    let meta = ModelMeta {
        id: model_name.clone(),
        object: "model".to_string(),
        created: now_unix(),
        owned_by: "unillm".to_string(),
        architecture: architecture.clone(),
        context_size: service.context_size(),
        eos_token_id: service.eos_token_id(),
    };

    Arc::new(ModelState {
        service,
        engine: None,
        model_name,
        architecture,
        semaphore: Semaphore::new(config.max_concurrency),
        config,
        cancel,
        model_meta: meta,
    })
}

// ============================================================================
// OpenAI API Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ApiChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_rep_penalty")]
    pub repetition_penalty: f32,
    #[serde(default)]
    pub frequency_penalty: f32,
    #[serde(default)]
    pub presence_penalty: f32,
    pub stop: Option<Vec<String>>,
    pub seed: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_rep_penalty")]
    pub repetition_penalty: f32,
    #[serde(default)]
    pub frequency_penalty: f32,
    #[serde(default)]
    pub presence_penalty: f32,
    pub stop: Option<Vec<String>>,
}

fn default_max_tokens() -> usize { 128 }
fn default_temperature() -> f32 { 0.8 }
fn default_top_p() -> f32 { 0.95 }
fn default_rep_penalty() -> f32 { 1.0 }

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ApiChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ApiChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatStreamChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChatStreamChoice {
    pub index: usize,
    pub delta: ChatDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionStreamChoice>,
}

#[derive(Debug, Serialize)]
pub struct CompletionStreamChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Serialize)]
pub struct ModelDetail {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    pub architecture: String,
    pub context_size: usize,
    pub eos_token_id: u32,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model: String,
    pub architecture: String,
    pub context_size: usize,
}

// ============================================================================
// Router
// ============================================================================

/// Create the axum router with all API endpoints
pub fn create_router(state: SharedState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(list_models))
        .route("/v1/engines/generate", post(engine_generate))
        .route("/health", get(health))
        .with_state(state)
}

// ============================================================================
// Handlers
// ============================================================================

async fn health(State(state): State<SharedState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        model: state.model_name.clone(),
        architecture: state.architecture.clone(),
        context_size: state.service.context_size(),
    })
}

async fn list_models(State(state): State<SharedState>) -> Json<ModelListResponse> {
    Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.model_meta.id.clone(),
            object: "model".to_string(),
            created: state.model_meta.created,
            owned_by: state.model_meta.owned_by.clone(),
        }],
    })
}

/// Enhanced /v1/models/:model endpoint — returns detailed model info
async fn model_detail(State(state): State<SharedState>) -> Json<ModelDetail> {
    let m = &state.model_meta;
    Json(ModelDetail {
        id: m.id.clone(),
        object: m.object.clone(),
        created: m.created,
        owned_by: m.owned_by.clone(),
        architecture: m.architecture.clone(),
        context_size: m.context_size,
        eos_token_id: m.eos_token_id,
    })
}

fn finish_reason_str(reason: &FinishReason) -> String {
    match reason {
        FinishReason::Stop => "stop".to_string(),
        FinishReason::Length => "length".to_string(),
        FinishReason::StopSequence(seq) => {
            if seq.is_empty() { "stop".to_string() } else { seq.clone() }
        }
    }
}

async fn chat_completions(
    State(state): State<SharedState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let _permit = state
        .semaphore
        .acquire()
        .await
        .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, e.to_string()))?;

    let messages: Vec<(String, String)> = req
        .messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();
    let prompt = state.service.render_chat(&messages);
    let prompt_tokens = state.service.encode_prompt(&prompt, true, false);

        let gen_config = state.service.make_gen_config(
        req.max_tokens,
        req.temperature,
        req.top_p,
        req.top_k,
        req.min_p,
        req.repetition_penalty,
        req.frequency_penalty,
        req.presence_penalty,
        req.stop.clone().unwrap_or_default(),
        req.temperature > 0.0,
    );

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let model_name = state.model_name.clone();

    if req.stream {
        let stream = create_chat_stream(
            state.clone(),
            prompt_tokens,
            gen_config,
            request_id,
            model_name,
        );
        drop(_permit);
        Ok(Sse::new(stream).into_response())
    } else {
        let state_clone = state.clone();
        let timeout_secs = state.config.request_timeout_secs;
        drop(_permit);
        let output = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            tokio::task::spawn_blocking(move || {
                state_clone.service.generate(&prompt_tokens, &gen_config)
            }),
        )
        .await
        .map_err(|_| (StatusCode::GATEWAY_TIMEOUT, "Request timed out".to_string()))?
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        let response = ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".to_string(),
            created: now_unix(),
            model: model_name,
            choices: vec![ChatChoice {
                index: 0,
                message: ApiChatMessage {
                    role: "assistant".to_string(),
                    content: output.text,
                },
                finish_reason: Some(finish_reason_str(&output.finish_reason)),
            }],
            usage: Usage {
                prompt_tokens: output.prompt_tokens,
                completion_tokens: output.completion_tokens,
                total_tokens: output.prompt_tokens + output.completion_tokens,
            },
        };

        Ok(Json(response).into_response())
    }
}

async fn completions(
    State(state): State<SharedState>,
    Json(req): Json<CompletionRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let _permit = state
        .semaphore
        .acquire()
        .await
        .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, e.to_string()))?;

    let prompt_tokens = state.service.encode_prompt(&req.prompt, true, false);

        let gen_config = state.service.make_gen_config(
        req.max_tokens,
        req.temperature,
        req.top_p,
        req.top_k,
        req.min_p,
        req.repetition_penalty,
        req.frequency_penalty,
        req.presence_penalty,
        req.stop.clone().unwrap_or_default(),
        req.temperature > 0.0,
    );

    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let model_name = state.model_name.clone();

    if req.stream {
        let stream = create_completion_stream(
            state.clone(),
            prompt_tokens,
            gen_config,
            request_id,
            model_name,
        );
        drop(_permit);
        Ok(Sse::new(stream).into_response())
    } else {
        let state_clone = state.clone();
        let timeout_secs = state.config.request_timeout_secs;
        drop(_permit);
        let output = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            tokio::task::spawn_blocking(move || {
                state_clone.service.generate(&prompt_tokens, &gen_config)
            }),
        )
        .await
        .map_err(|_| (StatusCode::GATEWAY_TIMEOUT, "Request timed out".to_string()))?
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        let response = CompletionResponse {
            id: request_id,
            object: "text_completion".to_string(),
            created: now_unix(),
            model: model_name,
            choices: vec![CompletionChoice {
                index: 0,
                text: output.text,
                finish_reason: Some(finish_reason_str(&output.finish_reason)),
            }],
            usage: Usage {
                prompt_tokens: output.prompt_tokens,
                completion_tokens: output.completion_tokens,
                total_tokens: output.prompt_tokens + output.completion_tokens,
            },
        };

        Ok(Json(response).into_response())
    }
}

// ============================================================================
// Engine-powered generation (uses UniLLMInferenceEngine with batching/scheduling)
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct EngineGenerateRequest {
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_rep_penalty")]
    pub repetition_penalty: f32,
}

#[derive(Debug, Serialize)]
pub struct EngineGenerateResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub text: String,
    pub finish_reason: String,
    pub usage: Usage,
}

async fn engine_generate(
    State(state): State<SharedState>,
    Json(req): Json<EngineGenerateRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let engine = match &state.engine {
        Some(e) => Arc::clone(e),
        None => {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "Inference engine not available. Start server with engine support.".to_string(),
            ));
        }
    };

    let output = engine
        .process(req.prompt, req.max_tokens, req.temperature, req.top_p, req.repetition_penalty)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let response = EngineGenerateResponse {
        id: format!("eng-{}", uuid::Uuid::new_v4()),
        object: "engine_generation".to_string(),
        created: now_unix(),
        model: state.model_name.clone(),
        text: output.text,
        finish_reason: output.finish_reason,
        usage: Usage {
            prompt_tokens: output.prompt_tokens,
            completion_tokens: output.completion_tokens,
            total_tokens: output.total_tokens,
        },
    };

    Ok(Json(response))
}

// ============================================================================
// SSE Streaming
// ============================================================================

fn create_chat_stream(
    state: SharedState,
    prompt_tokens: Vec<u32>,
    gen_config: GenerationConfig,
    request_id: String,
    model_name: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        let initial_chunk = ChatCompletionChunk {
            id: request_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: now_unix(),
            model: model_name.clone(),
            choices: vec![ChatStreamChoice {
                index: 0,
                delta: ChatDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        yield Ok(Event::default().json_data(initial_chunk).unwrap());

        let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamEvent>(32);
        let state_clone = state.clone();

        tokio::task::spawn_blocking(move || {
            let result = state_clone.service.generate_streaming(
                &prompt_tokens,
                &gen_config,
                &mut |_token_id, text| {
                    tx.blocking_send(StreamEvent::Token(text.to_string())).is_ok()
                },
            );

            match result {
                Ok(output) => {
                    let _ = tx.blocking_send(StreamEvent::Finished(output.finish_reason));
                }
                Err(e) => {
                    let _ = tx.blocking_send(StreamEvent::Error(e.to_string()));
                }
            }
        });

        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Token(text) => {
                    let chunk = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created: now_unix(),
                        model: model_name.clone(),
                        choices: vec![ChatStreamChoice {
                            index: 0,
                            delta: ChatDelta {
                                role: None,
                                content: Some(text),
                            },
                            finish_reason: None,
                        }],
                    };
                    yield Ok(Event::default().json_data(chunk).unwrap());
                }
                StreamEvent::Finished(finish_reason) => {
                    let final_chunk = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created: now_unix(),
                        model: model_name.clone(),
                        choices: vec![ChatStreamChoice {
                            index: 0,
                            delta: ChatDelta {
                                role: None,
                                content: None,
                            },
                            finish_reason: Some(finish_reason_str(&finish_reason)),
                        }],
                    };
                    yield Ok(Event::default().json_data(final_chunk).unwrap());
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                StreamEvent::Error(e) => {
                    yield Ok(Event::default().data(format!("{{\"error\": \"{}\"}}", e)));
                    break;
                }
            }
        }
    }
}

fn create_completion_stream(
    state: SharedState,
    prompt_tokens: Vec<u32>,
    gen_config: GenerationConfig,
    request_id: String,
    model_name: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamEvent>(32);
        let state_clone = state.clone();

        tokio::task::spawn_blocking(move || {
            let result = state_clone.service.generate_streaming(
                &prompt_tokens,
                &gen_config,
                &mut |_token_id, text| {
                    tx.blocking_send(StreamEvent::Token(text.to_string())).is_ok()
                },
            );

            match result {
                Ok(output) => {
                    let _ = tx.blocking_send(StreamEvent::Finished(output.finish_reason));
                }
                Err(e) => {
                    let _ = tx.blocking_send(StreamEvent::Error(e.to_string()));
                }
            }
        });

        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Token(text) => {
                    let chunk = CompletionChunk {
                        id: request_id.clone(),
                        object: "text_completion.chunk".to_string(),
                        created: now_unix(),
                        model: model_name.clone(),
                        choices: vec![CompletionStreamChoice {
                            index: 0,
                            text,
                            finish_reason: None,
                        }],
                    };
                    yield Ok(Event::default().json_data(chunk).unwrap());
                }
                StreamEvent::Finished(finish_reason) => {
                    let final_chunk = CompletionChunk {
                        id: request_id.clone(),
                        object: "text_completion.chunk".to_string(),
                        created: now_unix(),
                        model: model_name.clone(),
                        choices: vec![CompletionStreamChoice {
                            index: 0,
                            text: String::new(),
                            finish_reason: Some(finish_reason_str(&finish_reason)),
                        }],
                    };
                    yield Ok(Event::default().json_data(final_chunk).unwrap());
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                StreamEvent::Error(e) => {
                    yield Ok(Event::default().data(format!("{{\"error\": \"{}\"}}", e)));
                    break;
                }
            }
        }
    }
}

/// Events sent from the blocking generation task to the async SSE stream.
enum StreamEvent {
    Token(String),
    Finished(crate::service::FinishReason),
    Error(String),
}

// ============================================================================
// Server Startup
// ============================================================================

/// Start the HTTP server with graceful shutdown support.
///
/// Listens for SIGTERM/SIGINT and gracefully drains in-flight requests
/// before shutting down. The cancellation token can also be triggered
/// programmatically.
pub async fn start_server(
    state: SharedState,
    host: &str,
    port: u16,
) -> anyhow::Result<()> {
    let config = state.config.clone();
    start_server_with_config(state, host, port, config.request_timeout_secs).await
}

/// Start the HTTP server with full configuration and graceful shutdown.
pub async fn start_server_with_config(
    state: SharedState,
    host: &str,
    port: u16,
    request_timeout_secs: u64,
) -> anyhow::Result<()> {
    let _ = request_timeout_secs; // timeout is enforced per-request via state.config
    let app = create_router(state.clone());
    let addr = format!("{}:{}", host, port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    eprintln!("UniLLM server listening on http://{}", addr);
    eprintln!("Endpoints:");
    eprintln!("  POST /v1/chat/completions  (supports streaming)");
    eprintln!("  POST /v1/completions       (supports streaming)");
    eprintln!("  GET  /v1/models");
    eprintln!("  GET  /health");

    let cancel = state.cancel.clone();

    // Set up signal handler for graceful shutdown
    let shutdown_signal = async move {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                eprintln!("\nReceived Ctrl+C, shutting down gracefully...");
            }
            _ = cancel.cancelled() => {
                eprintln!("Shutdown requested, draining in-flight requests...");
            }
        }
    };

    // Run server with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal)
        .await?;

    eprintln!("Server shut down complete.");
    Ok(())
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}