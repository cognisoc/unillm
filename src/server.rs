//! UniLLM Inference Server
//!
//! High-performance LLM inference server with hybrid cache and GPU optimization.
//! Supports both containerized and unikernel deployment modes.

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;
use tracing::{info, warn, error};

// Import UniLLM components
use inference::{UniLLMInferenceEngine, InferenceEngine, request::RequestBuilder, types::SamplingParams};
use kv::HybridKVCache;
use scheduler::IntelligentScheduler;
// use kernels::KernelFramework;  // Temporarily disabled until kernels crate is fixed

#[cfg(feature = "unikernel")]
use kernels::{create_runtime_gpu_interface, UnikernnelGpuInterface};

#[derive(Parser)]
#[command(name = "unillm-server")]
#[command(about = "UniLLM High-Performance Inference Server")]
struct Args {
    /// Host address to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to bind to
    #[arg(long, default_value = "8080")]
    port: u16,

    /// GPU target (cuda, rocm, cpu)
    #[arg(long, default_value = "cuda")]
    gpu_target: String,

    /// Optimal batch size
    #[arg(long, default_value = "32")]
    batch_size: usize,

    /// Enable unikernel mode
    #[arg(long)]
    unikernel_mode: Option<String>,

    /// Log level
    #[arg(long, default_value = "info")]
    log_level: String,
}

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    inference_engine: Arc<UniLLMInferenceEngine>,
    #[cfg(feature = "unikernel")]
    unikernel_gpu: Option<Arc<dyn UnikernnelGpuInterface>>,
}

/// Inference request
#[derive(Deserialize)]
pub struct InferenceRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
}

/// Inference response
#[derive(Serialize)]
pub struct InferenceResponse {
    pub generated_text: String,
    pub tokens_generated: usize,
    pub inference_time_ms: u64,
    pub cache_hits: usize,
    pub gpu_utilization: f64,
}

/// Health check response
#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub gpu_target: String,
    pub runtime_mode: String,
    pub memory_usage_mb: f64,
    pub gpu_memory_usage_mb: f64,
}

/// System statistics
#[derive(Serialize)]
pub struct StatsResponse {
    pub total_requests: u64,
    pub total_tokens_generated: u64,
    pub average_latency_ms: f64,
    pub cache_hit_rate: f64,
    pub gpu_utilization: f64,
    pub memory_stats: MemoryStats,
}

#[derive(Serialize)]
pub struct MemoryStats {
    pub total_memory_mb: f64,
    pub used_memory_mb: f64,
    pub cache_memory_mb: f64,
    pub gpu_memory_mb: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(args.log_level.as_str())
        .init();

    info!("🚀 Starting UniLLM Inference Server");
    info!("GPU Target: {}", args.gpu_target);
    info!("Batch Size: {}", args.batch_size);

    // Detect runtime environment
    let runtime_mode = detect_runtime_mode(&args.unikernel_mode);
    info!("Runtime Mode: {}", runtime_mode);

    // Initialize GPU interface for unikernel mode
    #[cfg(feature = "unikernel")]
    let unikernel_gpu = if runtime_mode != "container" {
        match create_runtime_gpu_interface().await {
            Ok(gpu_interface) => {
                info!("✅ Unikernel GPU interface initialized");
                Some(Arc::new(gpu_interface) as Arc<dyn UnikernnelGpuInterface>)
            }
            Err(e) => {
                warn!("⚠️  Failed to initialize unikernel GPU interface: {}", e);
                None
            }
        }
    } else {
        None
    };

    #[cfg(not(feature = "unikernel"))]
    let unikernel_gpu: Option<i32> = None; // Placeholder type since kernels is disabled

    // Initialize UniLLM components
    info!("🔧 Initializing UniLLM components...");

    // Create kernel framework (temporarily disabled)
    // let kernel_framework = Arc::new(
    //     KernelFramework::new()
    //         .map_err(|e| format!("Failed to initialize kernel framework: {}", e))?
    // );

    // Create hybrid KV cache
    let kv_cache = Arc::new(
        HybridKVCache::new(
            1024,           // l1_max_nodes
            2048,           // l2_total_pages
            4096,           // l2_page_size
            16,             // l2_pages_per_block
            0               // base_device_ptr
        )
    );

    // Create intelligent scheduler (temporarily without kernel framework)
    // let scheduler = Arc::new(
    //     IntelligentScheduler::new(kv_cache.clone(), kernel_framework.clone())
    // );
    // Create scheduler with minimal configuration
    let scheduler = Arc::new(
        scheduler::IntelligentScheduler::new_minimal(Arc::new(std::sync::Mutex::new(
            kv::GpuIntegratedCache::new_cuda(0, 1000, 10000, 4096, 32).unwrap()
        )))
    );

    // Create inference engine
    let inference_engine = Arc::new(
        UniLLMInferenceEngine::new(
            kv_cache.clone(),
            scheduler.clone(),
        ).await
            .map_err(|e| format!("Failed to initialize inference engine: {}", e))?
    );

    info!("✅ UniLLM components initialized");

    // Create application state
    let app_state = AppState {
        inference_engine,
        #[cfg(feature = "unikernel")]
        unikernel_gpu,
    };

    // Create router
    let app = create_router(app_state);

    // Start server
    let bind_addr = format!("{}:{}", args.host, args.port);
    info!("🌐 Starting server on {}", bind_addr);

    let listener = TcpListener::bind(&bind_addr).await?;

    info!("🎉 UniLLM server ready!");
    info!("📖 Health check: http://{}/health", bind_addr);
    info!("📊 Statistics: http://{}/stats", bind_addr);
    info!("🔥 Inference: POST http://{}/v1/generate", bind_addr);

    axum::serve(listener, app).await?;

    Ok(())
}

fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/stats", get(get_stats))
        .route("/v1/generate", post(generate))
        .route("/v1/completions", post(generate)) // OpenAI compatibility
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Health check endpoint
async fn health_check(State(state): State<AppState>) -> Result<Json<HealthResponse>, StatusCode> {
    let runtime_mode = if cfg!(feature = "unikernel") {
        std::env::var("UNILLM_UNIKERNEL_MODE").unwrap_or_else(|_| "container".to_string())
    } else {
        "container".to_string()
    };

    let response = HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        gpu_target: std::env::var("UNILLM_GPU_TARGET").unwrap_or_else(|_| "unknown".to_string()),
        runtime_mode,
        memory_usage_mb: get_memory_usage_mb(),
        gpu_memory_usage_mb: get_gpu_memory_usage_mb(&state).await,
    };

    Ok(Json(response))
}

/// Statistics endpoint
async fn get_stats(State(state): State<AppState>) -> Result<Json<StatsResponse>, StatusCode> {
    let engine_stats = state.inference_engine.get_metrics();

    let response = StatsResponse {
        total_requests: engine_stats.request_stats.total_requests,
        total_tokens_generated: 0, // TODO: Need to add this field to PerformanceMetrics
        average_latency_ms: engine_stats.performance.average_latency_ms as f64,
        cache_hit_rate: engine_stats.cache.hit_rate as f64,
        gpu_utilization: engine_stats.resources.gpu_memory_utilization as f64,
        memory_stats: MemoryStats {
            total_memory_mb: get_total_memory_mb(),
            used_memory_mb: get_memory_usage_mb(),
            cache_memory_mb: 0.0, // TODO: Add cache memory tracking
            gpu_memory_mb: get_gpu_memory_usage_mb(&state).await,
        },
    };

    Ok(Json(response))
}

/// Text generation endpoint
async fn generate(
    State(state): State<AppState>,
    Json(request): Json<InferenceRequest>,
) -> Result<Json<InferenceResponse>, StatusCode> {
    let start_time = std::time::Instant::now();

    // Convert request to internal format
    let sampling_params = SamplingParams {
        max_tokens: request.max_tokens,
        temperature: request.temperature.unwrap_or(1.0),
        top_p: request.top_p.unwrap_or(0.9),
        top_k: None,
        seed: None,
        stop_sequences: vec![],
        repetition_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
    };

    let inference_request = RequestBuilder::new(request.prompt)
        .with_sampling_params(sampling_params)
        .build();

    // Perform inference
    match state.inference_engine.process_request(inference_request).await {
        Ok(result) => {
            let inference_time = start_time.elapsed();

            let response = InferenceResponse {
                generated_text: result.text,
                tokens_generated: result.stats.completion_tokens,
                inference_time_ms: inference_time.as_millis() as u64,
                cache_hits: (result.stats.cache_hit_rate * result.stats.total_tokens as f32) as usize,
                gpu_utilization: (result.stats.memory_usage_mb / 1024.0) as f64, // Approximate GPU utilization
            };

            Ok(Json(response))
        }
        Err(e) => {
            error!("Inference failed: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Detect current runtime mode
fn detect_runtime_mode(unikernel_mode: &Option<String>) -> String {
    if let Some(mode) = unikernel_mode {
        return mode.clone();
    }

    #[cfg(feature = "unikernel")]
    {
        if let Some(runtime) = kernels::unikernel_gpu::detect_unikernel_runtime() {
            return runtime;
        }
    }

    "container".to_string()
}

/// Get system memory usage in MB
fn get_memory_usage_mb() -> f64 {
    // Simplified memory usage - in production would use proper system APIs
    1024.0 // 1GB placeholder
}

/// Get total system memory in MB
fn get_total_memory_mb() -> f64 {
    // Simplified total memory - in production would use proper system APIs
    16384.0 // 16GB placeholder
}

/// Get GPU memory usage in MB
async fn get_gpu_memory_usage_mb(state: &AppState) -> f64 {
    #[cfg(feature = "unikernel")]
    if let Some(ref gpu) = state.unikernel_gpu {
        if let Ok(info) = gpu.get_memory_info().await {
            return info.allocated_memory as f64 / (1024.0 * 1024.0);
        }
    }

    // For container mode, would query GPU via standard APIs
    2048.0 // 2GB placeholder
}