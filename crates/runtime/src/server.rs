//! OpenAI-Compatible HTTP Server
//!
//! Provides REST API endpoints compatible with the OpenAI API format.
//! Supports chat completions, text completions, and streaming via SSE.

use crate::model_core::{GenerationConfig, ModelRunner, GenerationMetrics};
use crate::tokenizer::Tokenizer;
use crate::chat_template::{ChatMessage, ChatTemplateFormat};

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
use std::sync::Arc;
use std::convert::Infallible;
use tokio::sync::Semaphore;

// ============================================================================
// Server State
// ============================================================================

/// Shared model state for the server
pub struct ModelState {
    /// The loaded model runner
    pub model: Box<dyn ModelRunner>,
    /// Tokenizer
    pub tokenizer: Tokenizer,
    /// Model name (for API responses)
    pub model_name: String,
    /// Chat template format
    pub chat_template: ChatTemplateFormat,
    /// Concurrency limiter
    pub semaphore: Semaphore,
}

pub type SharedState = Arc<ModelState>;

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
    pub stream: bool,
    #[serde(default = "default_rep_penalty")]
    pub repetition_penalty: f32,
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
    pub stream: bool,
    #[serde(default = "default_rep_penalty")]
    pub repetition_penalty: f32,
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
pub struct HealthResponse {
    pub status: String,
    pub model: String,
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
    })
}

async fn list_models(State(state): State<SharedState>) -> Json<ModelListResponse> {
    Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.model_name.clone(),
            object: "model".to_string(),
            created: now_unix(),
            owned_by: "unillm".to_string(),
        }],
    })
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

    // Convert messages to chat template
    let messages: Vec<ChatMessage> = req
        .messages
        .iter()
        .map(|m| ChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
        })
        .collect();
    let prompt = state.chat_template.render(&messages);
    let prompt_tokens = state.tokenizer.encode_with_special_tokens(&prompt, false, false);
    let prompt_token_count = prompt_tokens.len();

    let gen_config = GenerationConfig {
        max_new_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        do_sample: req.temperature > 0.0,
        repetition_penalty: req.repetition_penalty,
        eos_token_id: state.model.eos_token_id(),
        ..Default::default()
    };

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let model_name = state.model_name.clone();

    if req.stream {
        // Streaming response via SSE
        let stream = create_chat_stream(
            state.clone(),
            prompt_tokens,
            gen_config,
            request_id,
            model_name,
        );
        Ok(Sse::new(stream).into_response())
    } else {
        // Non-streaming response
        let mut output_text = String::new();
        let mut completion_tokens = 0;

        let result = state.model.generate_streaming(
            &prompt_tokens,
            &gen_config,
            &mut |_token_id, text| {
                output_text.push_str(text);
                completion_tokens += 1;
                true
            },
        );

        match result {
            Ok(_) => {}
            Err(e) => return Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
        }

        let response = ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".to_string(),
            created: now_unix(),
            model: model_name,
            choices: vec![ChatChoice {
                index: 0,
                message: ApiChatMessage {
                    role: "assistant".to_string(),
                    content: output_text,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Usage {
                prompt_tokens: prompt_token_count,
                completion_tokens,
                total_tokens: prompt_token_count + completion_tokens,
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

    let prompt_tokens = state.tokenizer.encode_with_special_tokens(&req.prompt, true, false);
    let prompt_token_count = prompt_tokens.len();

    let gen_config = GenerationConfig {
        max_new_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        do_sample: req.temperature > 0.0,
        repetition_penalty: req.repetition_penalty,
        eos_token_id: state.model.eos_token_id(),
        ..Default::default()
    };

    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let model_name = state.model_name.clone();

    let mut output_text = String::new();
    let mut completion_tokens = 0;

    let result = state.model.generate_streaming(
        &prompt_tokens,
        &gen_config,
        &mut |_token_id, text| {
            output_text.push_str(text);
            completion_tokens += 1;
            true
        },
    );

    match result {
        Ok(_) => {}
        Err(e) => return Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    }

    let response = CompletionResponse {
        id: request_id,
        object: "text_completion".to_string(),
        created: now_unix(),
        model: model_name,
        choices: vec![CompletionChoice {
            index: 0,
            text: output_text,
            finish_reason: Some("stop".to_string()),
        }],
        usage: Usage {
            prompt_tokens: prompt_token_count,
            completion_tokens,
            total_tokens: prompt_token_count + completion_tokens,
        },
    };

    Ok(Json(response).into_response())
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
        // Send initial chunk with role
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

        // Generate tokens in a blocking task
        let (tx, mut rx) = tokio::sync::mpsc::channel::<Option<String>>(32);

        let state_clone = state.clone();

        tokio::task::spawn_blocking(move || {
            let _ = state_clone.model.generate_streaming(
                &prompt_tokens,
                &gen_config,
                &mut |_token_id, text| {
                    let should_continue = tx.blocking_send(Some(text.to_string())).is_ok();
                    should_continue
                },
            );
            let _ = tx.blocking_send(None); // Signal completion
        });

        // Stream tokens as SSE events
        while let Some(msg) = rx.recv().await {
            match msg {
                Some(text) => {
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
                None => {
                    // Final chunk with finish_reason
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
                            finish_reason: Some("stop".to_string()),
                        }],
                    };
                    yield Ok(Event::default().json_data(final_chunk).unwrap());
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
            }
        }
    }
}

// ============================================================================
// Server Startup
// ============================================================================

/// Start the HTTP server
pub async fn start_server(
    state: SharedState,
    host: &str,
    port: u16,
) -> anyhow::Result<()> {
    let app = create_router(state);
    let addr = format!("{}:{}", host, port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("UniLLM server listening on http://{}", addr);
    eprintln!("Endpoints:");
    eprintln!("  POST /v1/chat/completions");
    eprintln!("  POST /v1/completions");
    eprintln!("  GET  /v1/models");
    eprintln!("  GET  /health");
    axum::serve(listener, app).await?;
    Ok(())
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
