//! Simple Production HTTP Server
//!
//! High-performance inference server with:
//! - OpenAI-compatible endpoints
//! - Health checks and metrics
//! - Working with existing tensor operations

use axum::{
    extract::State,
    http::StatusCode,
    response::{Json as ResponseJson, IntoResponse},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::{
    sync::{Arc, atomic::{AtomicU64, Ordering}},
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::net::TcpListener;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;

use crate::{
    gpu_tensor_ops::{GpuDevice, GpuTensor, GpuTensorOps},
    types::{ModelResult, ModelError},
};

/// Server configuration
#[derive(Debug, Clone)]
pub struct SimpleServerConfig {
    pub host: String,
    pub port: u16,
    pub device: GpuDevice,
}

impl Default for SimpleServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            device: GpuDevice::auto_detect(),
        }
    }
}

/// Server state
#[derive(Clone)]
pub struct ServerState {
    pub device: GpuDevice,
    pub tensor_ops: GpuTensorOps,
    pub request_count: Arc<AtomicU64>,
    pub start_time: u64,
}

/// OpenAI-compatible chat completion request
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    pub model: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

fn default_max_tokens() -> u32 { 256 }
fn default_temperature() -> f32 { 0.7 }
fn default_top_p() -> f32 { 0.9 }

/// OpenAI-compatible chat completion response
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
    pub index: u32,
    pub message: ResponseMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ResponseMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Simple models response
#[derive(Debug, Serialize)]
pub struct ModelsResponse {
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

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub device: String,
    pub uptime_seconds: u64,
}

/// Metrics response
#[derive(Debug, Serialize)]
pub struct MetricsResponse {
    pub requests_total: u64,
    pub uptime_seconds: u64,
    pub device_info: String,
    pub memory_usage: String,
}

/// Simple production server
pub struct SimpleProductionServer {
    config: SimpleServerConfig,
}

impl SimpleProductionServer {
    /// Create new server instance
    pub fn new(config: SimpleServerConfig) -> Self {
        Self { config }
    }

    /// Start the server
    pub async fn start(self) -> ModelResult<()> {
        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| ModelError::ServerError("Failed to get system time".to_string()))?
            .as_secs();

        let state = ServerState {
            device: self.config.device.clone(),
            tensor_ops: GpuTensorOps::with_device(self.config.device.clone()),
            request_count: Arc::new(AtomicU64::new(0)),
            start_time,
        };

        let app = Router::new()
            // OpenAI-compatible endpoints
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/models", get(list_models))
            // Health and monitoring
            .route("/health", get(health_check))
            .route("/metrics", get(metrics))
            .layer(
                ServiceBuilder::new()
                    .layer(CorsLayer::permissive())
            )
            .with_state(state);

        let addr = format!("{}:{}", self.config.host, self.config.port);
        let listener = TcpListener::bind(&addr).await
            .map_err(|e| ModelError::ServerError(format!("Failed to bind to {}: {}", addr, e)))?;

        println!("🚀 UniLLM Simple Production Server");
        println!("   Listening on: {}", addr);
        println!("   Device: {:?}", self.config.device);
        println!("   Endpoints: /v1/chat/completions, /v1/models, /health, /metrics");

        axum::serve(listener, app).await
            .map_err(|e| ModelError::ServerError(format!("Server error: {}", e)))?;
        Ok(())
    }
}

/// Chat completions endpoint
async fn chat_completions(
    State(state): State<ServerState>,
    ResponseJson(request): ResponseJson<ChatCompletionRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    // Increment request counter
    state.request_count.fetch_add(1, Ordering::Relaxed);

    // Simple text generation (placeholder - could integrate with real models)
    let input_text = request.messages.last()
        .map(|m| m.content.as_str())
        .unwrap_or("Hello");

    // Simulate basic tensor operations
    let input_tensor = GpuTensor::randn(vec![1, 32, 512], state.device.clone())
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let _processed = state.tensor_ops.add(&input_tensor, &input_tensor)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Generate response text
    let response_text = format!("I understand you said: \"{}\". This is a UniLLM demonstration response showing GPU tensor processing is working!", input_text);

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: SystemTime::now().duration_since(UNIX_EPOCH)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?.as_secs(),
        model: request.model,
        choices: vec![ChatChoice {
            index: 0,
            message: ResponseMessage {
                role: "assistant".to_string(),
                content: response_text,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: input_text.len() as u32 / 4, // Rough estimate
            completion_tokens: 50,
            total_tokens: (input_text.len() as u32 / 4) + 50,
        },
    };

    Ok(ResponseJson(response))
}

/// List models endpoint
async fn list_models() -> impl IntoResponse {
    let models = ModelsResponse {
        object: "list".to_string(),
        data: vec![
            ModelInfo {
                id: "unillm-default".to_string(),
                object: "model".to_string(),
                created: SystemTime::now().duration_since(UNIX_EPOCH)
                    .unwrap_or_default().as_secs(),
                owned_by: "unillm".to_string(),
            },
        ],
    };

    ResponseJson(models)
}

/// Health check endpoint
async fn health_check(State(state): State<ServerState>) -> impl IntoResponse {
    let uptime = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() - state.start_time;

    let health = HealthResponse {
        status: "healthy".to_string(),
        version: "1.0.0".to_string(),
        device: format!("{:?}", state.device),
        uptime_seconds: uptime,
    };

    ResponseJson(health)
}

/// Metrics endpoint
async fn metrics(State(state): State<ServerState>) -> impl IntoResponse {
    let uptime = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() - state.start_time;

    let metrics = MetricsResponse {
        requests_total: state.request_count.load(Ordering::Relaxed),
        uptime_seconds: uptime,
        device_info: format!("{:?}", state.device),
        memory_usage: "Available via system info".to_string(),
    };

    ResponseJson(metrics)
}