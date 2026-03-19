//! UniLLM Production Server Binary
//!
//! High-performance inference server with OpenAI-compatible API.
//! Supports multiple model formats and automatic model loading.

use runtime::{
    production_server_new::{run_server, ServerConfig},
    gpu_tensor_ops::GpuDevice,
};
use std::env;
use tracing::{info, error};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing (only if not already initialized)
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init()
        .ok(); // Ignore error if already initialized

    // Parse configuration from environment variables
    let config = ServerConfig {
        host: env::var("UNILLM_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
        port: env::var("UNILLM_PORT")
            .unwrap_or_else(|_| "8000".to_string())
            .parse()
            .unwrap_or(8000),
        max_concurrent_requests: env::var("UNILLM_MAX_CONCURRENT_REQUESTS")
            .unwrap_or_else(|_| "128".to_string())
            .parse()
            .unwrap_or(128),
        max_sequence_length: env::var("UNILLM_MAX_SEQUENCE_LENGTH")
            .unwrap_or_else(|_| "4096".to_string())
            .parse()
            .unwrap_or(4096),
        ..Default::default()
    };

    info!("🚀 Starting UniLLM Production Server");
    info!("📡 Listening on {}:{}", config.host, config.port);
    info!("🔧 Max concurrent requests: {}", config.max_concurrent_requests);
    info!("📏 Max sequence length: {}", config.max_sequence_length);

    // Detect and log GPU capabilities
    let device = GpuDevice::auto_detect();
    match device {
        GpuDevice::Cuda(id) => info!("🔥 CUDA GPU detected: GPU {}", id),
        GpuDevice::Metal(id) => info!("🍎 Metal GPU detected: GPU {}", id),
        GpuDevice::Cpu => info!("💻 Using CPU backend"),
    }

    info!("✅ Server configuration loaded successfully");

    // Run server
    if let Err(e) = run_server(config).await {
        error!("❌ Server failed to start: {}", e);
        std::process::exit(1);
    }

    Ok(())
}