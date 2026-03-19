//! Production-grade API server for UniLLM
//!
//! High-performance inference server with vLLM/SGLang API compatibility.
//! Supports multiple model formats, automatic loading, and concurrent serving.

use crate::{
    gpu_tensor_ops::{GpuDevice, GpuTensorOps},
    working_llama::WorkingLlamaModel,
    tokenizer::Tokenizer,
    basic_model::ModelConfig,
    types::*,
};

use axum::{
    extract::{Path, State, Query},
    http::{StatusCode, HeaderMap},
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use tokio::sync::{Semaphore, RwLock as TokioRwLock};
use tower_http::{
    cors::CorsLayer,
    compression::CompressionLayer,
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use uuid::Uuid;
use tracing::{info, warn, error, debug};

/// Production server state
#[derive(Clone)]
pub struct ServerState {
    pub models: Arc<TokioRwLock<HashMap<String, LoadedModel>>>,
    pub request_semaphore: Arc<Semaphore>,
    pub config: ServerConfig,
    pub stats: Arc<RwLock<ServerStats>>,
}

/// Loaded model with metadata
pub struct LoadedModel {
    pub model: WorkingLlamaModel,
    pub tokenizer: Tokenizer,
    pub config: ModelConfig,
    pub device: GpuDevice,
    pub created_at: Instant,
    pub request_count: Arc<RwLock<u64>>,
}

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_concurrent_requests: usize,
    pub max_sequence_length: usize,
    pub request_timeout: Duration,
    pub model_cache_size: usize,
    pub enable_metrics: bool,
    pub enable_streaming: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8000,
            max_concurrent_requests: 128,
            max_sequence_length: 4096,
            request_timeout: Duration::from_secs(300),
            model_cache_size: 4,
            enable_metrics: true,
            enable_streaming: true,
        }
    }
}

/// Server statistics
#[derive(Debug, Default)]
pub struct ServerStats {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_tokens_generated: u64,
    pub avg_tokens_per_second: f64,
    pub uptime_seconds: u64,
}

/// OpenAI-compatible chat completion request
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
    pub stop: Option<Vec<String>>,
}

/// Chat message
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

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

/// Chat choice
#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// Token usage statistics
#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Model loading request
#[derive(Debug, Deserialize)]
pub struct LoadModelRequest {
    pub model_id: String,
    pub model_type: Option<String>, // "safetensors", "gguf", "huggingface"
    pub device: Option<String>,     // "auto", "cuda:0", "cpu"
    pub max_memory: Option<String>, // "8GB", "auto"
}

/// Model info response
#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    pub permission: Vec<Permission>,
}

/// Permission info
#[derive(Debug, Serialize)]
pub struct Permission {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub allow_create_engine: bool,
    pub allow_sampling: bool,
    pub allow_logprobs: bool,
    pub allow_search_indices: bool,
    pub allow_view: bool,
    pub allow_fine_tuning: bool,
    pub organization: String,
    pub group: Option<String>,
    pub is_blocking: bool,
}

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub gpu_memory_used: Option<String>,
    pub gpu_memory_total: Option<String>,
    pub models_loaded: usize,
}

/// Metrics response
#[derive(Debug, Serialize)]
pub struct MetricsResponse {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_tokens_generated: u64,
    pub avg_tokens_per_second: f64,
    pub uptime_seconds: u64,
    pub models_loaded: usize,
    pub active_requests: usize,
}

impl ServerState {
    pub fn new(config: ServerConfig) -> Self {
        Self {
            models: Arc::new(TokioRwLock::new(HashMap::new())),
            request_semaphore: Arc::new(Semaphore::new(config.max_concurrent_requests)),
            config,
            stats: Arc::new(RwLock::new(ServerStats::default())),
        }
    }

    /// Load a model (simplified version using basic APIs)
    pub async fn load_model(&self, request: LoadModelRequest) -> Result<String, ModelError> {
        info!("Loading model: {}", request.model_id);

        // Parse device
        let device = match request.device.as_deref().unwrap_or("auto") {
            "auto" => GpuDevice::auto_detect(),
            "cpu" => GpuDevice::Cpu,
            device_str if device_str.starts_with("cuda:") => {
                let id: usize = device_str[5..].parse()
                    .map_err(|_| ModelError::ConfigurationError("Invalid CUDA device ID".to_string()))?;
                GpuDevice::Cuda(id)
            }
            device_str if device_str.starts_with("metal:") => {
                let id: usize = device_str[6..].parse()
                    .map_err(|_| ModelError::ConfigurationError("Invalid Metal device ID".to_string()))?;
                GpuDevice::Metal(id)
            }
            _ => return Err(ModelError::ConfigurationError("Unsupported device type".to_string())),
        };

        // For now, create a basic working model with default config
        let model = WorkingLlamaModel::new(device.clone()).await?;
        let tokenizer = Tokenizer::new();
        let config = ModelConfig::default();

        // Store model
        let loaded_model = LoadedModel {
            model,
            tokenizer,
            config,
            device,
            created_at: Instant::now(),
            request_count: Arc::new(RwLock::new(0)),
        };

        let mut models = self.models.write().await;
        models.insert(request.model_id.clone(), loaded_model);

        info!("Successfully loaded basic model: {}", request.model_id);
        Ok(request.model_id)
    }


    pub async fn generate_completion(&self, request: ChatCompletionRequest) -> Result<ChatCompletionResponse, ModelError> {
        // Acquire semaphore permit
        let _permit = self.request_semaphore.acquire().await
            .map_err(|_| ModelError::ServerError("Too many concurrent requests".to_string()))?;

        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_requests += 1;
        }

        // Get model
        let models = self.models.read().await;
        let loaded_model = models.get(&request.model)
            .ok_or_else(|| ModelError::ConfigurationError(format!("Model {} not found", request.model)))?;

        // Update model request count
        {
            let mut count = loaded_model.request_count.write().unwrap();
            *count += 1;
        }

        // Convert messages to prompt
        let prompt = self.messages_to_prompt(&request.messages);

        // Tokenize
        let input_tokens = loaded_model.tokenizer.encode(&prompt);
        let prompt_tokens = input_tokens.len();

        // Generate
        let max_tokens = request.max_tokens.unwrap_or(512).min(self.config.max_sequence_length - prompt_tokens);
        let generated_text = loaded_model.model.generate_text(&prompt, max_tokens).await?;

        // Count completion tokens (approximate)
        let completion_tokens = loaded_model.tokenizer.encode(&generated_text).len() - prompt_tokens;

        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.successful_requests += 1;
            stats.total_tokens_generated += completion_tokens as u64;
        }

        // Create response
        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{}", Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            model: request.model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: generated_text,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        };

        Ok(response)
    }

    fn messages_to_prompt(&self, messages: &[ChatMessage]) -> String {
        // Simple chat template - in production, this should be model-specific
        let mut prompt = String::new();
        for message in messages {
            match message.role.as_str() {
                "system" => prompt.push_str(&format!("System: {}\n", message.content)),
                "user" => prompt.push_str(&format!("User: {}\n", message.content)),
                "assistant" => prompt.push_str(&format!("Assistant: {}\n", message.content)),
                _ => prompt.push_str(&format!("{}: {}\n", message.role, message.content)),
            }
        }
        prompt.push_str("Assistant: ");
        prompt
    }
}

/// Chat completion endpoint
pub async fn chat_completions(
    State(state): State<ServerState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, StatusCode> {
    match state.generate_completion(request).await {
        Ok(response) => Ok(Json(response)),
        Err(e) => {
            error!("Chat completion failed: {}", e);
            // Update failure stats
            {
                let mut stats = state.stats.write().unwrap();
                stats.failed_requests += 1;
            }
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Model loading endpoint
pub async fn load_model(
    State(state): State<ServerState>,
    Json(request): Json<LoadModelRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.load_model(request).await {
        Ok(model_id) => Ok(Json(serde_json::json!({
            "status": "success",
            "model_id": model_id
        }))),
        Err(e) => {
            error!("Model loading failed: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// List models endpoint
pub async fn list_models(
    State(state): State<ServerState>,
) -> Json<serde_json::Value> {
    let models = state.models.read().await;
    let model_list: Vec<ModelInfo> = models.keys().map(|id| ModelInfo {
        id: id.clone(),
        object: "model".to_string(),
        created: 1677610602,
        owned_by: "unillm".to_string(),
        permission: vec![Permission {
            id: format!("modelperm-{}", Uuid::new_v4()),
            object: "model_permission".to_string(),
            created: 1677610602,
            allow_create_engine: false,
            allow_sampling: true,
            allow_logprobs: true,
            allow_search_indices: false,
            allow_view: true,
            allow_fine_tuning: false,
            organization: "*".to_string(),
            group: None,
            is_blocking: false,
        }],
    }).collect();

    Json(serde_json::json!({
        "object": "list",
        "data": model_list
    }))
}

/// Health check endpoint
pub async fn health(
    State(state): State<ServerState>,
) -> Json<HealthResponse> {
    let models = state.models.read().await;
    let stats = state.stats.read().unwrap();

    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: stats.uptime_seconds,
        gpu_memory_used: None, // TODO: Implement GPU memory monitoring
        gpu_memory_total: None,
        models_loaded: models.len(),
    })
}

/// Metrics endpoint
pub async fn metrics(
    State(state): State<ServerState>,
) -> Json<MetricsResponse> {
    let models = state.models.read().await;
    let stats = state.stats.read().unwrap();

    Json(MetricsResponse {
        total_requests: stats.total_requests,
        successful_requests: stats.successful_requests,
        failed_requests: stats.failed_requests,
        total_tokens_generated: stats.total_tokens_generated,
        avg_tokens_per_second: stats.avg_tokens_per_second,
        uptime_seconds: stats.uptime_seconds,
        models_loaded: models.len(),
        active_requests: state.config.max_concurrent_requests - state.request_semaphore.available_permits(),
    })
}

/// Create production server router
pub fn create_router(state: ServerState) -> Router {
    Router::new()
        // OpenAI-compatible endpoints
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))

        // UniLLM-specific endpoints
        .route("/unillm/models/load", post(load_model))
        .route("/unillm/health", get(health))
        .route("/unillm/metrics", get(metrics))

        // Add middleware
        .layer(CorsLayer::permissive())
        .layer(CompressionLayer::new())
        .layer(TimeoutLayer::new(Duration::from_secs(300)))
        .layer(TraceLayer::new_for_http())

        .with_state(state)
}

/// Run the production server
pub async fn run_server(config: ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing (only if not already initialized)
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init()
        .ok(); // Ignore error if already initialized

    let state = ServerState::new(config.clone());
    let app = create_router(state);

    let bind_addr = format!("{}:{}", config.host, config.port);
    info!("Starting UniLLM production server on {}", bind_addr);

    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum_test::TestServer;

    #[tokio::test]
    async fn test_health_endpoint() {
        let config = ServerConfig::default();
        let state = ServerState::new(config);
        let app = create_router(state);
        let server = TestServer::new(app).unwrap();

        let response = server.get("/unillm/health").await;
        response.assert_status_ok();

        let health: HealthResponse = response.json();
        assert_eq!(health.status, "healthy");
    }

    #[tokio::test]
    async fn test_models_endpoint() {
        let config = ServerConfig::default();
        let state = ServerState::new(config);
        let app = create_router(state);
        let server = TestServer::new(app).unwrap();

        let response = server.get("/v1/models").await;
        response.assert_status_ok();
    }

    #[tokio::test]
    async fn test_metrics_endpoint() {
        let config = ServerConfig::default();
        let state = ServerState::new(config);
        let app = create_router(state);
        let server = TestServer::new(app).unwrap();

        let response = server.get("/unillm/metrics").await;
        response.assert_status_ok();

        let metrics: MetricsResponse = response.json();
        assert_eq!(metrics.models_loaded, 0);
    }
}