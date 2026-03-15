//! UniLLM Unified Inference Engine
//!
//! This crate provides the main inference engine that integrates all UniLLM components
//! into a cohesive, high-performance LLM serving system.

pub mod engine;
pub mod request;
pub mod response;
pub mod batch;
pub mod metrics;
pub mod types;

use async_trait::async_trait;
pub use engine::UniLLMInferenceEngine;
pub use request::{InferenceRequest, RequestBuilder};
pub use response::{InferenceResponse, ResponseStream};
pub use batch::{BatchProcessor, BatchOptimizer};
pub use metrics::{InferenceMetrics, PerformanceCollector};
pub use types::*;

/// Main trait for inference engines
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    /// Process a single inference request
    async fn process_request(&self, request: InferenceRequest) -> InferenceResult<InferenceResponse>;

    /// Process a batch of requests
    async fn process_batch(&self, requests: Vec<InferenceRequest>) -> InferenceResult<Vec<InferenceResponse>>;

    /// Get current performance metrics
    fn get_metrics(&self) -> InferenceMetrics;

    /// Health check for the engine
    async fn health_check(&self) -> InferenceResult<EngineHealth>;

    /// Graceful shutdown
    async fn shutdown(&self) -> InferenceResult<()>;
}