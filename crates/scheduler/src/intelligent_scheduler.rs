//! GPU-integrated intelligent scheduler with cache-aware batch formation
//!
//! This scheduler integrates directly with our GPU memory management system
//! to achieve superior performance through cache-aware scheduling policies.

use std::collections::{HashMap, VecDeque, BinaryHeap};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::cmp::Ordering;

use crate::{Request, RequestBatch, RequestPriority, RequestState, SchedulerStats, BatchAnalysis};

// Import our GPU-integrated cache system
use kv::{GpuIntegratedCache, GpuBackendType, TokenId, CacheHandle, CacheTier, GpuIntegratedCacheStats};

/// Intelligent scheduling policies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SchedulingPolicy {
    /// First-Come-First-Served (baseline, like vLLM)
    FCFS,
    /// Longest Prefix Match (SGLang inspired)
    LongestPrefixMatch,
    /// Cache-aware scheduling (UniLLM innovation)
    CacheAware,
    /// GPU memory optimized scheduling
    GpuMemoryOptimized,
    /// Adaptive ML-optimized policy
    AdaptiveML,
}

/// Request analysis for scheduling decisions
#[derive(Debug, Clone)]
pub struct RequestAnalysis {
    pub request_id: u32,
    pub cache_hit_potential: f64,       // 0.0-1.0, likelihood of cache hits
    pub prefix_sharing_score: f64,      // 0.0-1.0, prefix sharing potential
    pub memory_requirement: usize,      // Estimated GPU memory needed
    pub priority_boost: f64,           // Priority adjustment based on analysis
    pub estimated_latency: Duration,    // Predicted processing latency
}


/// Cache-aware batch optimizer
pub struct CacheAwareBatchOptimizer {
    gpu_cache: Arc<Mutex<GpuIntegratedCache>>,
    policy: SchedulingPolicy,
    workload_analyzer: WorkloadAnalyzer,
    performance_predictor: PerformancePredictor,
}

impl CacheAwareBatchOptimizer {
    pub fn new(gpu_cache: Arc<Mutex<GpuIntegratedCache>>, policy: SchedulingPolicy) -> Self {
        Self {
            gpu_cache,
            policy,
            workload_analyzer: WorkloadAnalyzer::new(),
            performance_predictor: PerformancePredictor::new(),
        }
    }

    /// Analyze a single request for scheduling optimization
    pub fn analyze_request(&self, request: &Request) -> RequestAnalysis {
        let tokens: Vec<TokenId> = request.prompt_tokens.iter().map(|&t| t as TokenId).collect();

        // Check cache hit potential by querying our GPU cache
        let cache_hit_potential = if let Ok(gpu_cache) = self.gpu_cache.try_lock() {
            let cache_stats = gpu_cache.get_stats();
            // Simplified heuristic - in practice would query cache for prefix matches
            if cache_stats.hybrid_stats.l1_hits > 0 {
                0.7 // High likelihood if we have L1 hits
            } else if cache_stats.hybrid_stats.l2_hits > 0 {
                0.4 // Medium likelihood for L2
            } else {
                0.1 // Low likelihood for cold cache
            }
        } else {
            0.1 // Conservative estimate if cache is locked
        };

        // Calculate prefix sharing score by analyzing token overlap
        let prefix_sharing_score = self.calculate_prefix_sharing_score(&tokens);

        // Estimate memory requirement based on sequence length and model parameters
        let memory_requirement = self.estimate_memory_requirement(request);

        // Calculate priority boost based on cache potential
        let priority_boost = match self.policy {
            SchedulingPolicy::CacheAware | SchedulingPolicy::GpuMemoryOptimized => {
                cache_hit_potential * 0.5 + prefix_sharing_score * 0.3
            },
            SchedulingPolicy::LongestPrefixMatch => prefix_sharing_score * 0.7,
            _ => 0.0,
        };

        // Predict processing latency
        let estimated_latency = self.performance_predictor.predict_latency(
            request, cache_hit_potential, memory_requirement
        );

        RequestAnalysis {
            request_id: request.id,
            cache_hit_potential,
            prefix_sharing_score,
            memory_requirement,
            priority_boost,
            estimated_latency,
        }
    }

    /// Optimize batch composition for maximum performance
    pub fn optimize_batch(&self, requests: &[Request], max_batch_size: usize, max_memory: usize) -> Vec<u32> {
        let mut analyses: Vec<_> = requests.iter()
            .map(|r| self.analyze_request(r))
            .collect();

        match self.policy {
            SchedulingPolicy::FCFS => {
                // Simple FCFS - take requests in order until limits reached
                self.optimize_fcfs(&analyses, max_batch_size, max_memory)
            },
            SchedulingPolicy::LongestPrefixMatch => {
                // Group by prefix similarity, prioritize shared prefixes
                self.optimize_prefix_matching(&analyses, max_batch_size, max_memory)
            },
            SchedulingPolicy::CacheAware => {
                // Prioritize requests with high cache hit potential
                self.optimize_cache_aware(&analyses, max_batch_size, max_memory)
            },
            SchedulingPolicy::GpuMemoryOptimized => {
                // Balance cache hits with memory efficiency
                self.optimize_gpu_memory(&analyses, max_batch_size, max_memory)
            },
            SchedulingPolicy::AdaptiveML => {
                // Use ML model to select optimal batch
                self.optimize_adaptive(&analyses, max_batch_size, max_memory)
            },
        }
    }

    fn optimize_fcfs(&self, analyses: &[RequestAnalysis], max_batch_size: usize, max_memory: usize) -> Vec<u32> {
        let mut selected = Vec::new();
        let mut total_memory = 0;

        for analysis in analyses.iter().take(max_batch_size) {
            if total_memory + analysis.memory_requirement <= max_memory {
                selected.push(analysis.request_id);
                total_memory += analysis.memory_requirement;
            } else {
                break;
            }
        }

        selected
    }

    fn optimize_prefix_matching(&self, analyses: &[RequestAnalysis], max_batch_size: usize, max_memory: usize) -> Vec<u32> {
        // Sort by prefix sharing score (highest first)
        let mut sorted_analyses = analyses.to_vec();
        sorted_analyses.sort_by(|a, b| b.prefix_sharing_score.partial_cmp(&a.prefix_sharing_score).unwrap_or(Ordering::Equal));

        let mut selected = Vec::new();
        let mut total_memory = 0;

        for analysis in sorted_analyses.iter().take(max_batch_size) {
            if total_memory + analysis.memory_requirement <= max_memory {
                selected.push(analysis.request_id);
                total_memory += analysis.memory_requirement;
            }
        }

        selected
    }

    fn optimize_cache_aware(&self, analyses: &[RequestAnalysis], max_batch_size: usize, max_memory: usize) -> Vec<u32> {
        // Sort by cache hit potential (highest first)
        let mut sorted_analyses = analyses.to_vec();
        sorted_analyses.sort_by(|a, b| b.cache_hit_potential.partial_cmp(&a.cache_hit_potential).unwrap_or(Ordering::Equal));

        let mut selected = Vec::new();
        let mut total_memory = 0;

        for analysis in sorted_analyses.iter().take(max_batch_size) {
            if total_memory + analysis.memory_requirement <= max_memory {
                selected.push(analysis.request_id);
                total_memory += analysis.memory_requirement;
            }
        }

        selected
    }

    fn optimize_gpu_memory(&self, analyses: &[RequestAnalysis], max_batch_size: usize, max_memory: usize) -> Vec<u32> {
        // Score based on cache potential and memory efficiency
        let mut scored_analyses: Vec<_> = analyses.iter()
            .map(|a| {
                let cache_score = a.cache_hit_potential * 0.6;
                let memory_efficiency = 1.0 - (a.memory_requirement as f64 / max_memory as f64);
                let memory_score = memory_efficiency * 0.4;
                let total_score = cache_score + memory_score;
                (a, total_score)
            })
            .collect();

        // Sort by total score (highest first)
        scored_analyses.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let mut selected = Vec::new();
        let mut total_memory = 0;

        for (analysis, _score) in scored_analyses.iter().take(max_batch_size) {
            if total_memory + analysis.memory_requirement <= max_memory {
                selected.push(analysis.request_id);
                total_memory += analysis.memory_requirement;
            }
        }

        selected
    }

    fn optimize_adaptive(&self, analyses: &[RequestAnalysis], max_batch_size: usize, max_memory: usize) -> Vec<u32> {
        // For now, use cache-aware strategy
        // In practice, this would use an ML model trained on historical performance
        self.optimize_cache_aware(analyses, max_batch_size, max_memory)
    }

    fn calculate_prefix_sharing_score(&self, tokens: &[TokenId]) -> f64 {
        // Simplified heuristic - in practice would analyze against active sequences
        // Higher score for longer sequences (more likely to have sharing opportunities)
        let length_factor = (tokens.len() as f64).log10() / 4.0; // Normalize to ~0-1 range
        length_factor.min(1.0).max(0.0)
    }

    fn estimate_memory_requirement(&self, request: &Request) -> usize {
        // Estimate GPU memory needed for this request
        // Based on sequence length, model parameters, and KV cache requirements
        let sequence_length = request.total_length();
        let head_dim = 128;  // Typical attention head dimension
        let num_heads = 32;  // Typical number of attention heads
        let bytes_per_element = 2; // FP16

        // KV cache memory: 2 (K+V) * sequence_length * head_dim * num_heads * bytes_per_element
        let kv_memory = 2 * sequence_length * head_dim * num_heads * bytes_per_element;

        // Add some overhead for intermediate computations
        (kv_memory as f64 * 1.2) as usize
    }
}

/// Workload analyzer to understand access patterns
pub struct WorkloadAnalyzer {
    request_history: VecDeque<RequestAnalysis>,
    prefix_patterns: HashMap<Vec<TokenId>, usize>, // Track common prefixes
}

impl WorkloadAnalyzer {
    pub fn new() -> Self {
        Self {
            request_history: VecDeque::new(),
            prefix_patterns: HashMap::new(),
        }
    }

    pub fn analyze_request(&mut self, analysis: RequestAnalysis) {
        // Track request analysis for workload learning
        self.request_history.push_back(analysis);

        // Keep only recent history (sliding window)
        if self.request_history.len() > 1000 {
            self.request_history.pop_front();
        }
    }

    pub fn get_workload_characteristics(&self) -> WorkloadCharacteristics {
        let avg_cache_hit_potential = self.request_history.iter()
            .map(|a| a.cache_hit_potential)
            .sum::<f64>() / self.request_history.len() as f64;

        let avg_prefix_sharing = self.request_history.iter()
            .map(|a| a.prefix_sharing_score)
            .sum::<f64>() / self.request_history.len() as f64;

        let avg_memory_requirement = self.request_history.iter()
            .map(|a| a.memory_requirement)
            .sum::<usize>() / self.request_history.len();

        WorkloadCharacteristics {
            avg_cache_hit_potential,
            avg_prefix_sharing_score: avg_prefix_sharing,
            avg_memory_requirement,
            request_rate: self.calculate_request_rate(),
        }
    }

    fn calculate_request_rate(&self) -> f64 {
        // Calculate requests per second based on recent history
        if self.request_history.len() < 2 {
            return 0.0;
        }

        let time_span = Duration::from_secs(60); // Look at last minute
        let recent_count = self.request_history.iter()
            .filter(|a| {
                // Simplified - would need to track timestamps
                true
            })
            .count();

        recent_count as f64 / time_span.as_secs() as f64
    }
}

/// Workload characteristics for adaptive scheduling
#[derive(Debug, Clone)]
pub struct WorkloadCharacteristics {
    pub avg_cache_hit_potential: f64,
    pub avg_prefix_sharing_score: f64,
    pub avg_memory_requirement: usize,
    pub request_rate: f64,
}

/// Performance predictor for latency estimation
pub struct PerformancePredictor {
    baseline_latencies: HashMap<usize, Duration>, // sequence_length -> baseline latency
}

impl PerformancePredictor {
    pub fn new() -> Self {
        let mut baseline_latencies = HashMap::new();

        // Initialize with some baseline estimates (would be learned from data)
        baseline_latencies.insert(128, Duration::from_millis(10));   // Short sequences
        baseline_latencies.insert(512, Duration::from_millis(25));   // Medium sequences
        baseline_latencies.insert(2048, Duration::from_millis(80));  // Long sequences
        baseline_latencies.insert(8192, Duration::from_millis(300)); // Very long sequences

        Self {
            baseline_latencies,
        }
    }

    pub fn predict_latency(&self, request: &Request, cache_hit_potential: f64, memory_requirement: usize) -> Duration {
        let sequence_length = request.total_length();

        // Find closest baseline
        let baseline_latency = self.baseline_latencies.iter()
            .min_by_key(|(len, _)| (**len as isize - sequence_length as isize).abs())
            .map(|(_, latency)| *latency)
            .unwrap_or(Duration::from_millis(50));

        // Adjust for cache hits (cache hits reduce latency)
        let cache_factor = 1.0 - (cache_hit_potential * 0.3); // Up to 30% reduction

        // Adjust for memory pressure (higher memory usage can increase latency)
        let memory_factor = 1.0 + (memory_requirement as f64 / 1_000_000_000.0) * 0.1; // Slight increase for high memory

        let adjusted_latency = baseline_latency.mul_f64(cache_factor * memory_factor);
        adjusted_latency
    }
}

/// GPU-integrated intelligent scheduler
pub struct IntelligentScheduler {
    // Core scheduling components
    pending_requests: VecDeque<Request>,
    active_requests: HashMap<u32, Request>,
    completed_requests: VecDeque<Request>,

    // GPU integration
    gpu_cache: Arc<Mutex<GpuIntegratedCache>>,
    batch_optimizer: CacheAwareBatchOptimizer,

    // Adaptive policy management
    current_policy: SchedulingPolicy,
    policy_performance: HashMap<SchedulingPolicy, f64>,
    last_policy_switch: Instant,

    // Configuration
    max_batch_size: usize,
    max_gpu_memory: usize,
    batching_window: Duration,

    // State tracking
    next_request_id: u32,
    next_batch_id: u32,
    last_batch_time: Instant,

    // Statistics
    stats: IntelligentSchedulerStats,
}

impl IntelligentScheduler {
    pub fn new(
        gpu_cache: Arc<Mutex<GpuIntegratedCache>>,
        max_batch_size: usize,
        max_gpu_memory: usize,
        batching_window_ms: u64,
    ) -> Self {
        let batch_optimizer = CacheAwareBatchOptimizer::new(
            Arc::clone(&gpu_cache),
            SchedulingPolicy::CacheAware, // Start with cache-aware policy
        );

        let mut policy_performance = HashMap::new();
        policy_performance.insert(SchedulingPolicy::CacheAware, 1.0); // Initial assumption

        Self {
            pending_requests: VecDeque::new(),
            active_requests: HashMap::new(),
            completed_requests: VecDeque::new(),
            gpu_cache,
            batch_optimizer,
            current_policy: SchedulingPolicy::CacheAware,
            policy_performance,
            last_policy_switch: Instant::now(),
            max_batch_size,
            max_gpu_memory,
            batching_window: Duration::from_millis(batching_window_ms),
            next_request_id: 0,
            next_batch_id: 0,
            last_batch_time: Instant::now(),
            stats: IntelligentSchedulerStats::new(),
        }
    }

    /// Add a new request with GPU cache integration
    pub fn add_request(&mut self, prompt_tokens: Vec<u32>, max_output_length: usize, priority: RequestPriority) -> u32 {
        let request_id = self.next_request_id;
        self.next_request_id += 1;

        let mut request = Request::new(request_id, prompt_tokens, max_output_length, priority);

        // Analyze request for cache potential immediately
        let analysis = self.batch_optimizer.analyze_request(&request);

        // Store analysis for later use (could extend Request struct)
        // For now, just track stats
        self.stats.total_requests += 1;
        self.stats.avg_cache_hit_potential =
            (self.stats.avg_cache_hit_potential * (self.stats.total_requests - 1) as f64 + analysis.cache_hit_potential)
            / self.stats.total_requests as f64;

        self.pending_requests.push_back(request);

        println!("Added request {} with cache hit potential: {:.2}, prefix score: {:.2}",
                 request_id, analysis.cache_hit_potential, analysis.prefix_sharing_score);

        request_id
    }

    /// Create optimized batch using cache-aware scheduling
    pub fn create_optimal_batch(&mut self) -> Option<RequestBatch> {
        if self.pending_requests.is_empty() {
            return None;
        }

        let start_time = Instant::now();

        // Convert pending requests to vector for analysis
        let requests: Vec<_> = self.pending_requests.iter().collect();

        // Use batch optimizer to select optimal requests
        let requests_slice: Vec<Request> = requests.iter().map(|r| (*r).clone()).collect();
        let selected_ids = self.batch_optimizer.optimize_batch(
            &requests_slice,
            self.max_batch_size,
            self.max_gpu_memory
        );

        if selected_ids.is_empty() {
            return None;
        }

        // Extract selected requests from pending queue
        let mut batch_requests = Vec::new();
        let mut remaining_requests = VecDeque::new();

        for request in self.pending_requests.drain(..) {
            if selected_ids.contains(&request.id) {
                batch_requests.push(request);
            } else {
                remaining_requests.push_back(request);
            }
        }

        self.pending_requests = remaining_requests;

        if batch_requests.is_empty() {
            return None;
        }

        // Create batch
        let batch_id = self.next_batch_id;
        self.next_batch_id += 1;

        let batch = RequestBatch::new(batch_id, batch_requests);

        // Track requests as active
        for request in &batch.requests {
            self.active_requests.insert(request.id, request.clone());
        }

        self.last_batch_time = Instant::now();

        // Update statistics
        let batch_formation_time = start_time.elapsed();
        self.stats.batches_created += 1;
        self.stats.batch_formation_time = batch_formation_time;
        self.stats.avg_batch_size = (self.stats.avg_batch_size * (self.stats.batches_created - 1) as f64 + batch.size() as f64) / self.stats.batches_created as f64;

        // Try to prefetch GPU memory for the batch
        if let Ok(mut gpu_cache) = self.gpu_cache.try_lock() {
            // This would be implemented to prefetch memory for the batch
            // gpu_cache.prefetch_batch_memory(&batch);
        }

        println!("Created optimal batch {} with {} requests in {:.2}ms (policy: {:?})",
                 batch_id, batch.size(), batch_formation_time.as_secs_f64() * 1000.0, self.current_policy);

        Some(batch)
    }

    /// Evaluate and potentially switch scheduling policy
    pub fn evaluate_and_adapt_policy(&mut self) {
        // Only consider policy changes every 30 seconds to avoid thrashing
        if self.last_policy_switch.elapsed() < Duration::from_secs(30) {
            return;
        }

        // Evaluate current policy performance
        let current_performance = self.calculate_policy_performance();
        self.policy_performance.insert(self.current_policy, current_performance);

        // Consider switching to a different policy
        let best_policy = self.select_best_policy();

        if best_policy != self.current_policy {
            println!("Switching scheduling policy from {:?} to {:?} (performance: {:.3})",
                     self.current_policy, best_policy, current_performance);

            self.current_policy = best_policy;
            self.batch_optimizer.policy = best_policy;
            self.last_policy_switch = Instant::now();
            self.stats.policy_switches += 1;
        }
    }

    fn calculate_policy_performance(&self) -> f64 {
        // Simple performance metric based on recent batch formation time and cache efficiency
        let batch_efficiency = if self.stats.batch_formation_time.as_millis() > 0 {
            1000.0 / self.stats.batch_formation_time.as_millis() as f64 // Inversely related to formation time
        } else {
            1.0
        };

        let cache_efficiency = self.stats.avg_cache_hit_potential * 2.0; // Weight cache hits

        (batch_efficiency + cache_efficiency) / 2.0
    }

    fn select_best_policy(&self) -> SchedulingPolicy {
        // Return policy with best performance, with some exploration
        self.policy_performance.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
            .map(|(policy, _)| *policy)
            .unwrap_or(SchedulingPolicy::CacheAware)
    }

    /// Get comprehensive statistics including GPU integration
    pub fn get_comprehensive_stats(&self) -> IntelligentSchedulerStats {
        let mut stats = self.stats.clone();

        // Add GPU cache statistics
        if let Ok(gpu_cache) = self.gpu_cache.try_lock() {
            stats.gpu_cache_stats = Some(gpu_cache.get_stats());
        }

        stats
    }

    // Standard scheduler interface methods
    pub fn should_create_batch(&self) -> bool {
        !self.pending_requests.is_empty()
            && self.last_batch_time.elapsed() >= self.batching_window
    }

    pub fn pending_count(&self) -> usize {
        self.pending_requests.len()
    }

    pub fn active_count(&self) -> usize {
        self.active_requests.len()
    }

    pub fn complete_request(&mut self, request_id: u32) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(mut request) = self.active_requests.remove(&request_id) {
            request.state = RequestState::Completed;
            request.completed_at = Some(Instant::now());
            self.completed_requests.push_back(request);
            self.stats.completed_requests += 1;
        }
        Ok(())
    }
}

/// Extended statistics for intelligent scheduler
#[derive(Debug, Clone)]
pub struct IntelligentSchedulerStats {
    // Basic stats
    pub total_requests: usize,
    pub completed_requests: usize,
    pub batches_created: usize,
    pub avg_batch_size: f64,

    // Performance stats
    pub batch_formation_time: Duration,
    pub avg_cache_hit_potential: f64,
    pub policy_switches: usize,
    pub current_policy: SchedulingPolicy,

    // GPU integration stats
    pub gpu_cache_stats: Option<GpuIntegratedCacheStats>,
}

impl IntelligentSchedulerStats {
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            completed_requests: 0,
            batches_created: 0,
            avg_batch_size: 0.0,
            batch_formation_time: Duration::from_millis(0),
            avg_cache_hit_potential: 0.0,
            policy_switches: 0,
            current_policy: SchedulingPolicy::CacheAware,
            gpu_cache_stats: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kv::GpuIntegratedCacheBuilder;

    #[test]
    fn test_cache_aware_batch_optimizer() {
        // This would require setting up a mock GPU cache
        println!("Cache-aware batch optimizer test - would need GPU cache mock");
    }

    #[test]
    fn test_request_analysis() {
        let request = Request::new(
            1,
            vec![1, 2, 3, 4, 5],
            100,
            RequestPriority::Normal
        );

        // Mock GPU cache for testing
        // In practice would use actual GpuIntegratedCache
        println!("Request analysis test - request has {} prompt tokens", request.prompt_tokens.len());
    }

    #[test]
    fn test_scheduling_policies() {
        // Test different scheduling policies
        let policies = vec![
            SchedulingPolicy::FCFS,
            SchedulingPolicy::LongestPrefixMatch,
            SchedulingPolicy::CacheAware,
            SchedulingPolicy::GpuMemoryOptimized,
        ];

        for policy in policies {
            println!("Testing policy: {:?}", policy);
        }
    }
}

impl IntelligentScheduler {
    /// Create a new scheduler with minimal configuration
    pub fn new_minimal(kv_cache: Arc<Mutex<GpuIntegratedCache>>) -> Self {
        Self {
            pending_requests: VecDeque::new(),
            active_requests: HashMap::new(),
            completed_requests: VecDeque::new(),
            gpu_cache: kv_cache.clone(),
            batch_optimizer: CacheAwareBatchOptimizer {
                gpu_cache: kv_cache.clone(),
                policy: SchedulingPolicy::FCFS,
                workload_analyzer: WorkloadAnalyzer::new(),
                performance_predictor: PerformancePredictor::new(),
            },
            current_policy: SchedulingPolicy::FCFS,
            policy_performance: HashMap::new(),
            last_policy_switch: Instant::now(),
            max_batch_size: 32,
            max_gpu_memory: 1024 * 1024 * 1024,
            batching_window: Duration::from_millis(100),
            next_request_id: 1,
            next_batch_id: 1,
            last_batch_time: Instant::now(),
            stats: IntelligentSchedulerStats::new(),
        }
    }

    /// Schedule a request (stub implementation)
    pub fn schedule_request(&self, _request: &InferenceRequest, _cache_analysis: &kv::CacheAnalysis) -> Result<ScheduleInfo, String> {
        Ok(ScheduleInfo {
            gpu_id: 0,
            memory_pressure: 0.5,
            batch_position: Some(0),
            estimated_latency: std::time::Duration::from_millis(100),
        })
    }
}

// Temporary stub until cross-crate imports are fixed
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub prompt: String,
    pub max_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct ScheduleInfo {
    pub gpu_id: usize,
    pub memory_pressure: f32,
    pub batch_position: Option<usize>,
    pub estimated_latency: std::time::Duration,
}