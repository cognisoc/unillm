use crate::{InferenceRequest, InferenceResponse, types::*};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Batch processor for optimizing throughput
pub struct BatchProcessor {
    config: BatchConfig,
    pending_batches: RwLock<VecDeque<Batch>>,
    active_batches: RwLock<HashMap<BatchId, ActiveBatch>>,
}

/// Batch optimizer for intelligent batching strategies
pub struct BatchOptimizer {
    strategies: Vec<BatchingStrategy>,
    performance_history: RwLock<Vec<BatchPerformanceRecord>>,
}

/// Unique batch identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchId(uuid::Uuid);

impl BatchId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

/// A batch of requests for processing
#[derive(Debug, Clone)]
pub struct Batch {
    pub id: BatchId,
    pub requests: Vec<InferenceRequest>,
    pub created_at: Instant,
    pub strategy: BatchingStrategy,
    pub estimated_processing_time: Duration,
    pub priority_score: f32,
}

/// Active batch being processed
#[derive(Debug)]
struct ActiveBatch {
    batch: Batch,
    started_at: Instant,
    progress: BatchProgress,
}

/// Batch processing progress
#[derive(Debug)]
struct BatchProgress {
    completed_requests: usize,
    total_requests: usize,
    estimated_remaining_time: Duration,
}

/// Batching strategies for different workload patterns
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchingStrategy {
    /// Fill batches to maximum size for highest throughput
    MaxThroughput,
    /// Optimize for lowest latency
    MinLatency,
    /// Balance throughput and latency
    Balanced,
    /// Group by similar sequence lengths
    SequenceLength,
    /// Group by cache affinity
    CacheAffinity,
    /// Adaptive strategy based on current load
    Adaptive,
}

/// Performance record for a completed batch
#[derive(Debug, Clone)]
struct BatchPerformanceRecord {
    strategy: BatchingStrategy,
    batch_size: usize,
    average_sequence_length: f32,
    processing_time: Duration,
    throughput_tokens_per_second: f32,
    average_latency_ms: f32,
    cache_hit_rate: f32,
    gpu_utilization: f32,
    timestamp: Instant,
}

/// Batch formation criteria
#[derive(Debug, Clone)]
pub struct BatchCriteria {
    pub max_batch_size: usize,
    pub max_wait_time: Duration,
    pub sequence_length_variance_threshold: f32,
    pub cache_affinity_threshold: f32,
    pub priority_grouping: bool,
}

impl Default for BatchCriteria {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_wait_time: Duration::from_millis(10),
            sequence_length_variance_threshold: 0.3,
            cache_affinity_threshold: 0.7,
            priority_grouping: true,
        }
    }
}

impl BatchProcessor {
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            pending_batches: RwLock::new(VecDeque::new()),
            active_batches: RwLock::new(HashMap::new()),
        }
    }

    /// Add requests to be batched
    pub async fn enqueue_requests(&self, requests: Vec<InferenceRequest>) -> InferenceResult<()> {
        if requests.is_empty() {
            return Ok(());
        }

        // Analyze requests for optimal batching
        let batch_groups = self.group_requests_for_batching(requests).await;

        let mut pending_batches = self.pending_batches.write().await;

        for group in batch_groups {
            let batch = self.create_batch(group).await?;
            pending_batches.push_back(batch);
        }

        Ok(())
    }

    /// Get the next batch ready for processing
    pub async fn get_next_batch(&self) -> Option<Batch> {
        let mut pending_batches = self.pending_batches.write().await;

        // Check for ready batches (either full or timed out)
        while let Some(batch) = pending_batches.front() {
            if self.is_batch_ready(batch).await {
                return pending_batches.pop_front();
            } else if batch.created_at.elapsed() > Duration::from_millis(self.config.batch_timeout_ms) {
                // Force process timed out batch
                return pending_batches.pop_front();
            } else {
                break;
            }
        }

        None
    }

    /// Mark batch as started
    pub async fn start_batch(&self, batch: Batch) -> InferenceResult<()> {
        let active_batch = ActiveBatch {
            started_at: Instant::now(),
            progress: BatchProgress {
                completed_requests: 0,
                total_requests: batch.requests.len(),
                estimated_remaining_time: batch.estimated_processing_time,
            },
            batch,
        };

        let mut active_batches = self.active_batches.write().await;
        active_batches.insert(active_batch.batch.id, active_batch);

        Ok(())
    }

    /// Mark batch as completed
    pub async fn complete_batch(&self, batch_id: BatchId, results: Vec<InferenceResponse>) -> InferenceResult<()> {
        let mut active_batches = self.active_batches.write().await;

        if let Some(active_batch) = active_batches.remove(&batch_id) {
            let processing_time = active_batch.started_at.elapsed();

            // Record performance metrics for optimization
            self.record_batch_performance(&active_batch.batch, &results, processing_time).await;
        }

        Ok(())
    }

    /// Group requests into optimal batches
    async fn group_requests_for_batching(&self, requests: Vec<InferenceRequest>) -> Vec<Vec<InferenceRequest>> {
        let mut groups = Vec::new();
        let mut remaining_requests = requests;

        // Sort by priority first
        remaining_requests.sort_by(|a, b| b.metadata.priority.cmp(&a.metadata.priority));

        while !remaining_requests.is_empty() {
            let group = self.form_optimal_batch(&mut remaining_requests).await;
            if !group.is_empty() {
                groups.push(group);
            } else {
                break;
            }
        }

        groups
    }

    /// Form an optimal batch from available requests
    async fn form_optimal_batch(&self, requests: &mut Vec<InferenceRequest>) -> Vec<InferenceRequest> {
        if requests.is_empty() {
            return Vec::new();
        }

        let mut batch = Vec::new();
        let max_size = self.config.max_batch_size;

        // Strategy: Group by sequence length similarity for efficient memory usage
        let base_request = requests.remove(0);
        let base_length = base_request.prompt.len();
        batch.push(base_request);

        // Find requests with similar characteristics
        let mut i = 0;
        while i < requests.len() && batch.len() < max_size {
            let request = &requests[i];
            let length_ratio = (request.prompt.len() as f32) / (base_length as f32);

            // Group requests with similar sequence lengths
            if length_ratio > 0.7 && length_ratio < 1.3 {
                batch.push(requests.remove(i));
            } else {
                i += 1;
            }
        }

        batch
    }

    /// Create a batch from grouped requests
    async fn create_batch(&self, requests: Vec<InferenceRequest>) -> InferenceResult<Batch> {
        if requests.is_empty() {
            return Err(InferenceError::InvalidRequest("Empty batch".to_string()));
        }

        let batch_id = BatchId::new();
        let created_at = Instant::now();

        // Determine optimal strategy for this batch
        let strategy = self.determine_batch_strategy(&requests).await;

        // Estimate processing time
        let estimated_processing_time = self.estimate_batch_processing_time(&requests, &strategy).await;

        // Calculate priority score
        let priority_score = self.calculate_batch_priority(&requests).await;

        Ok(Batch {
            id: batch_id,
            requests,
            created_at,
            strategy,
            estimated_processing_time,
            priority_score,
        })
    }

    /// Determine optimal batching strategy
    async fn determine_batch_strategy(&self, requests: &[InferenceRequest]) -> BatchingStrategy {
        // Analyze request characteristics
        let avg_length = requests.iter()
            .map(|r| r.prompt.len())
            .sum::<usize>() as f32 / requests.len() as f32;

        let has_high_priority = requests.iter()
            .any(|r| matches!(r.metadata.priority, RequestPriority::High | RequestPriority::Critical));

        let has_streaming = requests.iter().any(|r| r.metadata.stream);

        // Choose strategy based on characteristics
        if has_high_priority {
            BatchingStrategy::MinLatency
        } else if has_streaming {
            BatchingStrategy::Balanced
        } else if avg_length > 1000.0 {
            BatchingStrategy::SequenceLength
        } else {
            BatchingStrategy::MaxThroughput
        }
    }

    /// Estimate processing time for a batch
    async fn estimate_batch_processing_time(&self, requests: &[InferenceRequest], strategy: &BatchingStrategy) -> Duration {
        let total_tokens = requests.iter()
            .map(|r| r.prompt.len() + r.sampling_params.max_tokens.unwrap_or(100))
            .sum::<usize>();

        let base_time_per_token = match strategy {
            BatchingStrategy::MaxThroughput => Duration::from_micros(100),
            BatchingStrategy::MinLatency => Duration::from_micros(150),
            BatchingStrategy::Balanced => Duration::from_micros(120),
            BatchingStrategy::SequenceLength => Duration::from_micros(110),
            BatchingStrategy::CacheAffinity => Duration::from_micros(90),
            BatchingStrategy::Adaptive => Duration::from_micros(110),
        };

        // Add batch overhead
        let batch_overhead = Duration::from_millis(5);

        base_time_per_token * total_tokens as u32 + batch_overhead
    }

    /// Calculate priority score for batch scheduling
    async fn calculate_batch_priority(&self, requests: &[InferenceRequest]) -> f32 {
        let mut total_priority = 0.0;
        let mut urgency_bonus = 0.0;

        for request in requests {
            // Base priority score
            total_priority += match request.metadata.priority {
                RequestPriority::Critical => 4.0,
                RequestPriority::High => 3.0,
                RequestPriority::Normal => 2.0,
                RequestPriority::Low => 1.0,
            };

            // Urgency bonus based on wait time
            let wait_time = request.queue_time().as_secs_f32();
            urgency_bonus += wait_time * 0.1;

            // Timeout urgency
            if let Some(timeout) = request.metadata.timeout {
                let remaining_ratio = 1.0 - (wait_time / timeout.as_secs_f32());
                if remaining_ratio < 0.3 {
                    urgency_bonus += 2.0; // High urgency for near-timeout requests
                }
            }
        }

        (total_priority / requests.len() as f32) + urgency_bonus
    }

    /// Check if batch is ready for processing
    async fn is_batch_ready(&self, batch: &Batch) -> bool {
        // Batch is ready if it's full or has been waiting too long
        batch.requests.len() >= self.config.max_batch_size ||
        batch.created_at.elapsed() >= Duration::from_millis(self.config.batch_timeout_ms)
    }

    /// Record performance metrics for batch optimization
    async fn record_batch_performance(&self, batch: &Batch, results: &[InferenceResponse], processing_time: Duration) {
        // Calculate metrics from results
        let total_tokens: usize = results.iter().map(|r| r.stats.total_tokens).sum();
        let throughput = total_tokens as f32 / processing_time.as_secs_f32();
        let avg_latency = results.iter().map(|r| r.stats.total_time_ms).sum::<f32>() / results.len() as f32;
        let avg_cache_hit_rate = results.iter().map(|r| r.metadata.cache_info.hit_rate).sum::<f32>() / results.len() as f32;
        let avg_gpu_utilization = results.iter().map(|r| r.metadata.resource_usage.gpu_utilization).sum::<f32>() / results.len() as f32;

        let avg_sequence_length = batch.requests.iter()
            .map(|r| r.prompt.len() as f32)
            .sum::<f32>() / batch.requests.len() as f32;

        // Store performance record (this would be used by the optimizer)
        let _record = BatchPerformanceRecord {
            strategy: batch.strategy.clone(),
            batch_size: batch.requests.len(),
            average_sequence_length: avg_sequence_length,
            processing_time,
            throughput_tokens_per_second: throughput,
            average_latency_ms: avg_latency,
            cache_hit_rate: avg_cache_hit_rate,
            gpu_utilization: avg_gpu_utilization,
            timestamp: Instant::now(),
        };

        // In a real implementation, this would be stored for analysis
    }

    /// Get current batch statistics
    pub async fn get_stats(&self) -> BatchStats {
        let pending_batches = self.pending_batches.read().await;
        let active_batches = self.active_batches.read().await;

        let pending_requests: usize = pending_batches.iter()
            .map(|b| b.requests.len())
            .sum();

        let active_requests: usize = active_batches.iter()
            .map(|(_, b)| b.batch.requests.len())
            .sum();

        BatchStats {
            pending_batches: pending_batches.len(),
            active_batches: active_batches.len(),
            pending_requests,
            active_requests,
            average_batch_size: if !pending_batches.is_empty() {
                pending_requests as f32 / pending_batches.len() as f32
            } else {
                0.0
            },
        }
    }
}

impl BatchOptimizer {
    pub fn new() -> Self {
        Self {
            strategies: vec![
                BatchingStrategy::MaxThroughput,
                BatchingStrategy::MinLatency,
                BatchingStrategy::Balanced,
                BatchingStrategy::SequenceLength,
                BatchingStrategy::CacheAffinity,
                BatchingStrategy::Adaptive,
            ],
            performance_history: RwLock::new(Vec::new()),
        }
    }

    /// Recommend optimal batching strategy based on current conditions
    pub async fn recommend_strategy(&self, current_load: f32, average_latency: f32) -> BatchingStrategy {
        let history = self.performance_history.read().await;

        if history.is_empty() {
            return BatchingStrategy::Balanced;
        }

        // Analyze recent performance to recommend strategy
        let recent_records: Vec<_> = history.iter()
            .filter(|r| r.timestamp.elapsed() < Duration::from_secs(300))
            .collect();

        if recent_records.is_empty() {
            return BatchingStrategy::Balanced;
        }

        // Find best performing strategy for current conditions
        let mut best_strategy = BatchingStrategy::Balanced;
        let mut best_score = 0.0;

        for strategy in &self.strategies {
            let strategy_records: Vec<_> = recent_records.iter()
                .filter(|r| r.strategy == *strategy)
                .collect();

            if !strategy_records.is_empty() {
                let avg_throughput = strategy_records.iter()
                    .map(|r| r.throughput_tokens_per_second)
                    .sum::<f32>() / strategy_records.len() as f32;

                let avg_latency = strategy_records.iter()
                    .map(|r| r.average_latency_ms)
                    .sum::<f32>() / strategy_records.len() as f32;

                // Score based on throughput and latency (adjustable weights)
                let score = avg_throughput * 0.7 - avg_latency * 0.3;

                if score > best_score {
                    best_score = score;
                    best_strategy = strategy.clone();
                }
            }
        }

        best_strategy
    }

    /// Optimize batch formation criteria based on performance history
    pub async fn optimize_batch_criteria(&self, current_criteria: BatchCriteria) -> BatchCriteria {
        let history = self.performance_history.read().await;

        // Analyze optimal batch sizes
        let optimal_batch_size = if !history.is_empty() {
            let size_performance: Vec<_> = history.iter()
                .map(|r| (r.batch_size, r.throughput_tokens_per_second))
                .collect();

            // Find batch size with best throughput
            size_performance.iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(size, _)| *size)
                .unwrap_or(current_criteria.max_batch_size)
        } else {
            current_criteria.max_batch_size
        };

        BatchCriteria {
            max_batch_size: optimal_batch_size,
            ..current_criteria
        }
    }
}

/// Batch processing statistics
#[derive(Debug, Clone)]
pub struct BatchStats {
    pub pending_batches: usize,
    pub active_batches: usize,
    pub pending_requests: usize,
    pub active_requests: usize,
    pub average_batch_size: f32,
}