use crate::types::{RequestId, InferenceError, InferenceResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Comprehensive inference metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetrics {
    /// Timestamp of last update
    pub timestamp: u64,

    /// Uptime since engine start
    pub uptime: Duration,

    /// Request statistics
    pub request_stats: RequestStats,

    /// Performance metrics
    pub performance: PerformanceMetrics,

    /// Resource utilization
    pub resources: ResourceMetrics,

    /// Cache effectiveness
    pub cache: CacheMetrics,

    /// Error tracking
    pub errors: ErrorMetrics,

    /// Quality metrics
    pub quality: QualityMetrics,
}

/// Request processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestStats {
    /// Total requests processed
    pub total_requests: u64,

    /// Requests currently processing
    pub active_requests: u64,

    /// Requests in queue
    pub queued_requests: u64,

    /// Completed requests
    pub completed_requests: u64,

    /// Failed requests
    pub failed_requests: u64,

    /// Average queue time in milliseconds
    pub average_queue_time_ms: f32,

    /// Average processing time in milliseconds
    pub average_processing_time_ms: f32,

    /// Requests per second
    pub requests_per_second: f32,

    /// Request completion rate
    pub completion_rate: f32,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average latency in milliseconds
    pub average_latency_ms: f32,

    /// 50th percentile latency
    pub p50_latency_ms: f32,

    /// 95th percentile latency
    pub p95_latency_ms: f32,

    /// 99th percentile latency
    pub p99_latency_ms: f32,

    /// Throughput in tokens per second
    pub throughput_tokens_per_second: f32,

    /// Time to first token in milliseconds
    pub average_ttft_ms: f32,

    /// Batch processing efficiency
    pub batch_efficiency: f32,

    /// GPU utilization percentage
    pub gpu_utilization: f32,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// GPU memory usage in MB
    pub gpu_memory_mb: f32,

    /// GPU memory utilization percentage
    pub gpu_memory_utilization: f32,

    /// CPU usage percentage
    pub cpu_utilization: f32,

    /// System memory usage in MB
    pub system_memory_mb: f32,

    /// Network I/O in MB/s
    pub network_io_mbps: f32,

    /// Disk I/O in MB/s
    pub disk_io_mbps: f32,

    /// Power consumption in watts
    pub power_consumption_watts: f32,

    /// Energy efficiency (tokens per joule)
    pub energy_efficiency: f32,
}

/// Cache effectiveness metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    /// Overall cache hit rate
    pub hit_rate: f32,

    /// L1 radix cache hit rate
    pub l1_hit_rate: f32,

    /// L2 paged cache hit rate
    pub l2_hit_rate: f32,

    /// L3 compressed cache hit rate
    pub l3_hit_rate: f32,

    /// Cache memory usage in MB
    pub cache_memory_mb: f32,

    /// Cache efficiency score
    pub efficiency_score: f32,

    /// Prefix sharing effectiveness
    pub prefix_sharing_rate: f32,

    /// Cache eviction rate
    pub eviction_rate: f32,
}

/// Error tracking metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMetrics {
    /// Total errors encountered
    pub total_errors: u64,

    /// Errors per hour
    pub errors_per_hour: f32,

    /// Error rate (errors / total requests)
    pub error_rate: f32,

    /// Error breakdown by type
    pub error_types: HashMap<String, u64>,

    /// GPU errors
    pub gpu_errors: u64,

    /// Memory errors
    pub memory_errors: u64,

    /// Timeout errors
    pub timeout_errors: u64,
}

/// Quality metrics for generated content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Average token generation rate
    pub average_tokens_per_request: f32,

    /// Content diversity score
    pub diversity_score: f32,

    /// Repetition rate
    pub repetition_rate: f32,

    /// Average response coherence score
    pub coherence_score: f32,

    /// Toxicity detection rate
    pub toxicity_rate: f32,
}

/// Performance collector for gathering metrics
pub struct PerformanceCollector {
    metrics: Arc<RwLock<InferenceMetrics>>,
    latency_samples: Arc<Mutex<VecDeque<f32>>>,
    request_timeline: Arc<Mutex<VecDeque<RequestMetricRecord>>>,
    error_log: Arc<Mutex<VecDeque<ErrorRecord>>>,
    start_time: Instant,
}

/// Individual request metric record
#[derive(Debug, Clone)]
struct RequestMetricRecord {
    request_id: RequestId,
    submitted_at: Instant,
    started_at: Option<Instant>,
    completed_at: Option<Instant>,
    tokens_generated: usize,
    cache_hit_rate: f32,
    gpu_utilization: f32,
    memory_usage_mb: f32,
    error: Option<String>,
}

/// Error record for tracking failures
#[derive(Debug, Clone)]
struct ErrorRecord {
    timestamp: Instant,
    error_type: String,
    error_message: String,
    request_id: Option<RequestId>,
    severity: ErrorSeverity,
}

/// Error severity levels
#[derive(Debug, Clone, PartialEq, Eq)]
enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl InferenceMetrics {
    pub fn new() -> Self {
        Self {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            uptime: Duration::from_secs(0),
            request_stats: RequestStats::default(),
            performance: PerformanceMetrics::default(),
            resources: ResourceMetrics::default(),
            cache: CacheMetrics::default(),
            errors: ErrorMetrics::default(),
            quality: QualityMetrics::default(),
        }
    }

    pub fn update_timestamp(&mut self) {
        self.timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }
}

impl Default for RequestStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            active_requests: 0,
            queued_requests: 0,
            completed_requests: 0,
            failed_requests: 0,
            average_queue_time_ms: 0.0,
            average_processing_time_ms: 0.0,
            requests_per_second: 0.0,
            completion_rate: 0.0,
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            average_latency_ms: 0.0,
            p50_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            throughput_tokens_per_second: 0.0,
            average_ttft_ms: 0.0,
            batch_efficiency: 0.0,
            gpu_utilization: 0.0,
        }
    }
}

impl Default for ResourceMetrics {
    fn default() -> Self {
        Self {
            gpu_memory_mb: 0.0,
            gpu_memory_utilization: 0.0,
            cpu_utilization: 0.0,
            system_memory_mb: 0.0,
            network_io_mbps: 0.0,
            disk_io_mbps: 0.0,
            power_consumption_watts: 0.0,
            energy_efficiency: 0.0,
        }
    }
}

impl Default for CacheMetrics {
    fn default() -> Self {
        Self {
            hit_rate: 0.0,
            l1_hit_rate: 0.0,
            l2_hit_rate: 0.0,
            l3_hit_rate: 0.0,
            cache_memory_mb: 0.0,
            efficiency_score: 0.0,
            prefix_sharing_rate: 0.0,
            eviction_rate: 0.0,
        }
    }
}

impl Default for ErrorMetrics {
    fn default() -> Self {
        Self {
            total_errors: 0,
            errors_per_hour: 0.0,
            error_rate: 0.0,
            error_types: HashMap::new(),
            gpu_errors: 0,
            memory_errors: 0,
            timeout_errors: 0,
        }
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            average_tokens_per_request: 0.0,
            diversity_score: 0.0,
            repetition_rate: 0.0,
            coherence_score: 0.0,
            toxicity_rate: 0.0,
        }
    }
}

impl PerformanceCollector {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(InferenceMetrics::new())),
            latency_samples: Arc::new(Mutex::new(VecDeque::new())),
            request_timeline: Arc::new(Mutex::new(VecDeque::new())),
            error_log: Arc::new(Mutex::new(VecDeque::new())),
            start_time: Instant::now(),
        }
    }

    /// Record a request start
    pub async fn record_request_start(&self, request_id: RequestId) {
        let record = RequestMetricRecord {
            request_id,
            submitted_at: Instant::now(),
            started_at: None,
            completed_at: None,
            tokens_generated: 0,
            cache_hit_rate: 0.0,
            gpu_utilization: 0.0,
            memory_usage_mb: 0.0,
            error: None,
        };

        let mut timeline = self.request_timeline.lock().unwrap();
        timeline.push_back(record);

        // Keep only recent records
        if timeline.len() > 10000 {
            timeline.pop_front();
        }
    }

    /// Record request processing start
    pub async fn record_request_processing_start(&self, request_id: RequestId) {
        let mut timeline = self.request_timeline.lock().unwrap();
        if let Some(record) = timeline.iter_mut().find(|r| r.request_id == request_id) {
            record.started_at = Some(Instant::now());
        }
    }

    /// Record request completion
    pub async fn record_request_completion(
        &self,
        request_id: RequestId,
        tokens_generated: usize,
        cache_hit_rate: f32,
        gpu_utilization: f32,
        memory_usage_mb: f32,
    ) {
        let completion_time = Instant::now();

        // Update request record
        {
            let mut timeline = self.request_timeline.lock().unwrap();
            if let Some(record) = timeline.iter_mut().find(|r| r.request_id == request_id) {
                record.completed_at = Some(completion_time);
                record.tokens_generated = tokens_generated;
                record.cache_hit_rate = cache_hit_rate;
                record.gpu_utilization = gpu_utilization;
                record.memory_usage_mb = memory_usage_mb;

                // Calculate latency
                if let Some(started_at) = record.started_at {
                    let latency_ms = completion_time.duration_since(started_at).as_millis() as f32;

                    // Add to latency samples
                    let mut samples = self.latency_samples.lock().unwrap();
                    samples.push_back(latency_ms);

                    // Keep only recent samples
                    if samples.len() > 1000 {
                        samples.pop_front();
                    }
                }
            }
        }

        // Update metrics
        self.update_metrics().await;
    }

    /// Record an error
    pub async fn record_error(&self, error: InferenceError, request_id: Option<RequestId>) {
        let error_record = ErrorRecord {
            timestamp: Instant::now(),
            error_type: match &error {
                InferenceError::InvalidRequest(_) => "InvalidRequest".to_string(),
                InferenceError::ModelNotLoaded(_) => "ModelNotLoaded".to_string(),
                InferenceError::ProcessingFailed(_) => "ProcessingFailed".to_string(),
                InferenceError::GpuError(_) => "GpuError".to_string(),
                InferenceError::MemoryError(_) => "MemoryError".to_string(),
                InferenceError::Timeout(_) => "Timeout".to_string(),
                InferenceError::Internal(_) => "Internal".to_string(),
                InferenceError::Io(_) => "Io".to_string(),
                InferenceError::Serialization(_) => "Serialization".to_string(),
            },
            error_message: error.to_string(),
            request_id,
            severity: match &error {
                InferenceError::GpuError(_) | InferenceError::MemoryError(_) => ErrorSeverity::High,
                InferenceError::Internal(_) => ErrorSeverity::Medium,
                InferenceError::Timeout(_) => ErrorSeverity::Medium,
                _ => ErrorSeverity::Low,
            },
        };

        {
            let mut error_log = self.error_log.lock().unwrap();
            error_log.push_back(error_record);

            // Keep only recent errors
            if error_log.len() > 1000 {
                error_log.pop_front();
            }
        }

        // Update request record with error
        if let Some(request_id) = request_id {
            let mut timeline = self.request_timeline.lock().unwrap();
            if let Some(record) = timeline.iter_mut().find(|r| r.request_id == request_id) {
                record.error = Some(error.to_string());
            }
        }

        // Update metrics
        self.update_metrics().await;
    }

    /// Update aggregated metrics
    async fn update_metrics(&self) {
        let mut metrics = self.metrics.write().await;

        // Update uptime
        metrics.uptime = self.start_time.elapsed();

        // Update request statistics
        self.update_request_stats(&mut metrics).await;

        // Update performance metrics
        self.update_performance_metrics(&mut metrics).await;

        // Update error metrics
        self.update_error_metrics(&mut metrics).await;

        // Update timestamp
        metrics.update_timestamp();
    }

    async fn update_request_stats(&self, metrics: &mut InferenceMetrics) {
        let timeline = self.request_timeline.lock().unwrap();
        let now = Instant::now();

        let completed_requests = timeline.iter()
            .filter(|r| r.completed_at.is_some())
            .count() as u64;

        let failed_requests = timeline.iter()
            .filter(|r| r.error.is_some())
            .count() as u64;

        let active_requests = timeline.iter()
            .filter(|r| r.started_at.is_some() && r.completed_at.is_none() && r.error.is_none())
            .count() as u64;

        let queued_requests = timeline.iter()
            .filter(|r| r.started_at.is_none())
            .count() as u64;

        // Calculate average queue time
        let queue_times: Vec<f32> = timeline.iter()
            .filter_map(|r| r.started_at.map(|started| started.duration_since(r.submitted_at).as_millis() as f32))
            .collect();

        let average_queue_time_ms = if !queue_times.is_empty() {
            queue_times.iter().sum::<f32>() / queue_times.len() as f32
        } else {
            0.0
        };

        // Calculate average processing time
        let processing_times: Vec<f32> = timeline.iter()
            .filter_map(|r| {
                if let (Some(started), Some(completed)) = (r.started_at, r.completed_at) {
                    Some(completed.duration_since(started).as_millis() as f32)
                } else {
                    None
                }
            })
            .collect();

        let average_processing_time_ms = if !processing_times.is_empty() {
            processing_times.iter().sum::<f32>() / processing_times.len() as f32
        } else {
            0.0
        };

        // Calculate requests per second (last minute)
        let recent_completions = timeline.iter()
            .filter(|r| {
                if let Some(completed) = r.completed_at {
                    now.duration_since(completed) < Duration::from_secs(60)
                } else {
                    false
                }
            })
            .count() as f32;

        let requests_per_second = recent_completions / 60.0;

        // Calculate completion rate
        let total_requests = completed_requests + failed_requests + active_requests + queued_requests;
        let completion_rate = if total_requests > 0 {
            completed_requests as f32 / total_requests as f32
        } else {
            0.0
        };

        metrics.request_stats = RequestStats {
            total_requests,
            active_requests,
            queued_requests,
            completed_requests,
            failed_requests,
            average_queue_time_ms,
            average_processing_time_ms,
            requests_per_second,
            completion_rate,
        };
    }

    async fn update_performance_metrics(&self, metrics: &mut InferenceMetrics) {
        let samples = self.latency_samples.lock().unwrap();

        if samples.is_empty() {
            return;
        }

        let mut sorted_samples: Vec<f32> = samples.iter().cloned().collect();
        sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let average_latency_ms = sorted_samples.iter().sum::<f32>() / sorted_samples.len() as f32;
        let p50_latency_ms = sorted_samples[sorted_samples.len() / 2];
        let p95_latency_ms = sorted_samples[(sorted_samples.len() as f32 * 0.95) as usize];
        let p99_latency_ms = sorted_samples[(sorted_samples.len() as f32 * 0.99) as usize];

        // Calculate throughput from recent completions
        let timeline = self.request_timeline.lock().unwrap();
        let recent_tokens: usize = timeline.iter()
            .filter(|r| {
                if let Some(completed) = r.completed_at {
                    Instant::now().duration_since(completed) < Duration::from_secs(60)
                } else {
                    false
                }
            })
            .map(|r| r.tokens_generated)
            .sum();

        let throughput_tokens_per_second = recent_tokens as f32 / 60.0;

        // Calculate average time to first token (simplified)
        let average_ttft_ms = average_latency_ms * 0.1; // Assume TTFT is ~10% of total latency

        // Calculate batch efficiency (simplified)
        let batch_efficiency = 0.85; // Mock value

        // Calculate average GPU utilization
        let gpu_utilizations: Vec<f32> = timeline.iter()
            .filter(|r| r.completed_at.is_some())
            .map(|r| r.gpu_utilization)
            .collect();

        let gpu_utilization = if !gpu_utilizations.is_empty() {
            gpu_utilizations.iter().sum::<f32>() / gpu_utilizations.len() as f32
        } else {
            0.0
        };

        metrics.performance = PerformanceMetrics {
            average_latency_ms,
            p50_latency_ms,
            p95_latency_ms,
            p99_latency_ms,
            throughput_tokens_per_second,
            average_ttft_ms,
            batch_efficiency,
            gpu_utilization,
        };
    }

    async fn update_error_metrics(&self, metrics: &mut InferenceMetrics) {
        let error_log = self.error_log.lock().unwrap();
        let now = Instant::now();

        let total_errors = error_log.len() as u64;

        // Errors in the last hour
        let recent_errors = error_log.iter()
            .filter(|e| now.duration_since(e.timestamp) < Duration::from_secs(3600))
            .count() as f32;

        let errors_per_hour = recent_errors;

        // Error rate
        let error_rate = if metrics.request_stats.total_requests > 0 {
            total_errors as f32 / metrics.request_stats.total_requests as f32
        } else {
            0.0
        };

        // Error breakdown by type
        let mut error_types = HashMap::new();
        for error in error_log.iter() {
            *error_types.entry(error.error_type.clone()).or_insert(0) += 1;
        }

        let gpu_errors = error_types.get("GpuError").copied().unwrap_or(0);
        let memory_errors = error_types.get("MemoryError").copied().unwrap_or(0);
        let timeout_errors = error_types.get("Timeout").copied().unwrap_or(0);

        metrics.errors = ErrorMetrics {
            total_errors,
            errors_per_hour,
            error_rate,
            error_types,
            gpu_errors,
            memory_errors,
            timeout_errors,
        };
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> InferenceMetrics {
        self.metrics.read().await.clone()
    }

    /// Export metrics in Prometheus format
    pub async fn export_prometheus(&self) -> String {
        let metrics = self.get_metrics().await;

        format!(
            r#"# HELP unillm_requests_total Total number of requests
# TYPE unillm_requests_total counter
unillm_requests_total {{}} {}

# HELP unillm_requests_active Currently active requests
# TYPE unillm_requests_active gauge
unillm_requests_active {{}} {}

# HELP unillm_latency_avg Average latency in milliseconds
# TYPE unillm_latency_avg gauge
unillm_latency_avg {{}} {}

# HELP unillm_throughput_tokens_per_second Throughput in tokens per second
# TYPE unillm_throughput_tokens_per_second gauge
unillm_throughput_tokens_per_second {{}} {}

# HELP unillm_gpu_utilization GPU utilization percentage
# TYPE unillm_gpu_utilization gauge
unillm_gpu_utilization {{}} {}

# HELP unillm_cache_hit_rate Cache hit rate
# TYPE unillm_cache_hit_rate gauge
unillm_cache_hit_rate {{}} {}

# HELP unillm_errors_total Total number of errors
# TYPE unillm_errors_total counter
unillm_errors_total {{}} {}
"#,
            metrics.request_stats.total_requests,
            metrics.request_stats.active_requests,
            metrics.performance.average_latency_ms,
            metrics.performance.throughput_tokens_per_second,
            metrics.performance.gpu_utilization,
            metrics.cache.hit_rate,
            metrics.errors.total_errors
        )
    }
}