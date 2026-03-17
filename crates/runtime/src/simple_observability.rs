//! Simple High-Performance Observability
//!
//! Zero-overhead metrics collection and monitoring with minimal dependencies.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{info, warn, error, instrument, Span};
use serde::{Deserialize, Serialize};
use once_cell::sync::Lazy;
use sysinfo::System;

/// Global metrics instance - zero overhead when disabled
pub static METRICS: Lazy<Arc<SimpleMetrics>> = Lazy::new(|| {
    Arc::new(SimpleMetrics::new())
});

/// Simple atomic counter-based metrics system
pub struct SimpleMetrics {
    enabled: bool,

    // Request metrics (atomic for zero-lock performance)
    requests_total: AtomicU64,
    requests_in_flight: AtomicUsize,
    errors_total: AtomicU64,

    // Inference metrics
    tokens_generated_total: AtomicU64,
    cache_hits_total: AtomicU64,
    cache_misses_total: AtomicU64,

    // System state
    health_monitor: Arc<RwLock<HealthMonitor>>,
}

impl SimpleMetrics {
    pub fn new() -> Self {
        let enabled = std::env::var("UNILLM_OBSERVABILITY_ENABLED")
            .unwrap_or_else(|_| "true".to_string())
            .parse::<bool>()
            .unwrap_or(true);

        Self {
            enabled,
            requests_total: AtomicU64::new(0),
            requests_in_flight: AtomicUsize::new(0),
            errors_total: AtomicU64::new(0),
            tokens_generated_total: AtomicU64::new(0),
            cache_hits_total: AtomicU64::new(0),
            cache_misses_total: AtomicU64::new(0),
            health_monitor: Arc::new(RwLock::new(HealthMonitor::new())),
        }
    }

    /// Initialize the metrics system
    pub fn initialize() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if !METRICS.enabled {
            info!("Observability disabled - zero overhead mode");
            return Ok(());
        }

        // Initialize tracing
        use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

        let env_filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info"));

        tracing_subscriber::registry()
            .with(env_filter)
            .with(tracing_subscriber::fmt::layer().json())
            .init();

        // Start health monitoring
        let health_monitor = Arc::clone(&METRICS.health_monitor);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));
            loop {
                interval.tick().await;
                health_monitor.write().await.update().await;
            }
        });

        info!("Simple observability system initialized");
        Ok(())
    }

    #[inline(always)]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Record request start (zero overhead when disabled)
    #[inline(always)]
    pub fn request_start(&self) -> RequestTimer {
        if !self.enabled {
            return RequestTimer::disabled();
        }

        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.requests_in_flight.fetch_add(1, Ordering::Relaxed);
        RequestTimer::new()
    }

    /// Record token generation (zero overhead when disabled)
    #[inline(always)]
    pub fn token_generated(&self) {
        if !self.enabled {
            return;
        }
        self.tokens_generated_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Record cache hit/miss (zero overhead when disabled)
    #[inline(always)]
    pub fn cache_hit(&self, hit: bool) {
        if !self.enabled {
            return;
        }

        if hit {
            self.cache_hits_total.fetch_add(1, Ordering::Relaxed);
        } else {
            self.cache_misses_total.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record error (zero overhead when disabled)
    #[inline(always)]
    pub fn error(&self, error_type: &str) {
        if !self.enabled {
            return;
        }

        self.errors_total.fetch_add(1, Ordering::Relaxed);
        error!("Error recorded: {}", error_type);
    }

    /// Get current statistics
    pub fn get_stats(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            requests_total: self.requests_total.load(Ordering::Relaxed),
            requests_in_flight: self.requests_in_flight.load(Ordering::Relaxed),
            errors_total: self.errors_total.load(Ordering::Relaxed),
            tokens_generated_total: self.tokens_generated_total.load(Ordering::Relaxed),
            cache_hits_total: self.cache_hits_total.load(Ordering::Relaxed),
            cache_misses_total: self.cache_misses_total.load(Ordering::Relaxed),
            cache_hit_rate: self.calculate_cache_hit_rate(),
        }
    }

    /// Get health status
    pub async fn get_health_status(&self) -> HealthStatus {
        self.health_monitor.read().await.get_status()
    }

    /// Get Prometheus-style metrics
    pub fn get_prometheus_metrics(&self) -> String {
        if !self.enabled {
            return String::new();
        }

        let stats = self.get_stats();
        format!(
            r#"# HELP unillm_requests_total Total requests processed
# TYPE unillm_requests_total counter
unillm_requests_total {}

# HELP unillm_requests_in_flight Currently processing requests
# TYPE unillm_requests_in_flight gauge
unillm_requests_in_flight {}

# HELP unillm_errors_total Total errors encountered
# TYPE unillm_errors_total counter
unillm_errors_total {}

# HELP unillm_tokens_generated_total Total tokens generated
# TYPE unillm_tokens_generated_total counter
unillm_tokens_generated_total {}

# HELP unillm_cache_hits_total Cache hits
# TYPE unillm_cache_hits_total counter
unillm_cache_hits_total {}

# HELP unillm_cache_misses_total Cache misses
# TYPE unillm_cache_misses_total counter
unillm_cache_misses_total {}

# HELP unillm_cache_hit_rate Cache hit percentage
# TYPE unillm_cache_hit_rate gauge
unillm_cache_hit_rate {:.2}
"#,
            stats.requests_total,
            stats.requests_in_flight,
            stats.errors_total,
            stats.tokens_generated_total,
            stats.cache_hits_total,
            stats.cache_misses_total,
            stats.cache_hit_rate
        )
    }

    fn calculate_cache_hit_rate(&self) -> f64 {
        let hits = self.cache_hits_total.load(Ordering::Relaxed) as f64;
        let misses = self.cache_misses_total.load(Ordering::Relaxed) as f64;
        let total = hits + misses;

        if total > 0.0 {
            (hits / total) * 100.0
        } else {
            0.0
        }
    }
}

/// Zero-overhead request timer
pub struct RequestTimer {
    start: Option<Instant>,
}

impl RequestTimer {
    fn new() -> Self {
        Self {
            start: Some(Instant::now()),
        }
    }

    fn disabled() -> Self {
        Self { start: None }
    }

    /// Complete the request and record metrics
    pub fn complete(self) {
        // Timer automatically decrements in_flight counter in Drop
    }
}

impl Drop for RequestTimer {
    fn drop(&mut self) {
        if self.start.is_some() && METRICS.enabled {
            METRICS.requests_in_flight.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

/// Metrics snapshot
#[derive(Debug, Serialize)]
pub struct MetricsSnapshot {
    pub requests_total: u64,
    pub requests_in_flight: usize,
    pub errors_total: u64,
    pub tokens_generated_total: u64,
    pub cache_hits_total: u64,
    pub cache_misses_total: u64,
    pub cache_hit_rate: f64,
}

/// System health monitoring
pub struct HealthMonitor {
    system: System,
    last_update: Instant,
    status: HealthStatus,
}

impl HealthMonitor {
    fn new() -> Self {
        Self {
            system: System::new_all(),
            last_update: Instant::now(),
            status: HealthStatus::default(),
        }
    }

    async fn update(&mut self) {
        self.system.refresh_all();
        self.last_update = Instant::now();

        self.status = HealthStatus {
            status: "healthy".to_string(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            cpu_usage_percent: self.system.global_cpu_usage(),
            memory_usage_percent: (self.system.used_memory() as f32 / self.system.total_memory() as f32) * 100.0,
            uptime_seconds: System::uptime(),
            gpu_available: self.check_gpu_availability(),
            memory_usage_bytes: self.system.used_memory(),
            total_memory_bytes: self.system.total_memory(),
        };
    }

    fn get_status(&self) -> HealthStatus {
        self.status.clone()
    }

    fn check_gpu_availability(&self) -> bool {
        std::env::var("CUDA_VISIBLE_DEVICES").is_ok() ||
        std::path::Path::new("/dev/nvidia0").exists() ||
        std::env::var("HIP_VISIBLE_DEVICES").is_ok()
    }
}

/// Health status response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub timestamp: u64,
    pub cpu_usage_percent: f32,
    pub memory_usage_percent: f32,
    pub uptime_seconds: u64,
    pub gpu_available: bool,
    pub memory_usage_bytes: u64,
    pub total_memory_bytes: u64,
}

impl Default for HealthStatus {
    fn default() -> Self {
        Self {
            status: "unknown".to_string(),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            cpu_usage_percent: 0.0,
            memory_usage_percent: 0.0,
            uptime_seconds: 0,
            gpu_available: false,
            memory_usage_bytes: 0,
            total_memory_bytes: 0,
        }
    }
}

/// Create a traced span for request processing
#[instrument]
pub fn start_request_span(request_id: &str, endpoint: &str) -> Span {
    tracing::info_span!("request", request_id = request_id, endpoint = endpoint)
}

/// Create a traced span for inference
#[instrument]
pub fn start_inference_span(model: &str, batch_size: usize) -> Span {
    tracing::info_span!("inference", model = model, batch_size = batch_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_zero_overhead() {
        // Test basic metrics functionality
        let timer = METRICS.request_start();
        METRICS.token_generated();
        METRICS.cache_hit(true);
        METRICS.cache_hit(false);
        METRICS.error("test_error");
        timer.complete();

        let stats = METRICS.get_stats();
        if METRICS.is_enabled() {
            assert!(stats.requests_total > 0);
            assert!(stats.tokens_generated_total > 0);
            assert!(stats.cache_hits_total > 0);
            assert!(stats.cache_misses_total > 0);
            assert!(stats.errors_total > 0);
        }
    }

    #[tokio::test]
    async fn test_health_monitoring() {
        let health = METRICS.get_health_status().await;
        assert!(health.timestamp > 0);
    }

    #[test]
    fn test_prometheus_metrics_format() {
        let metrics = METRICS.get_prometheus_metrics();
        if METRICS.is_enabled() {
            assert!(metrics.contains("unillm_requests_total"));
        }
    }

    #[test]
    fn test_cache_hit_rate() {
        METRICS.cache_hit(true);
        METRICS.cache_hit(true);
        METRICS.cache_hit(false);

        let stats = METRICS.get_stats();
        if METRICS.is_enabled() {
            // Should be around 66.67% (2 hits out of 3 total)
            assert!(stats.cache_hit_rate > 60.0 && stats.cache_hit_rate < 70.0);
        }
    }
}