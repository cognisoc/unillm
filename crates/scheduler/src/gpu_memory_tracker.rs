//! GPU memory tracking and allocation optimization
//!
//! This module provides real-time GPU memory monitoring and predictive allocation
//! to prevent out-of-memory conditions and optimize batch sizing.

use crate::types::{Request, MemoryFeasibility, OptimalBatchSize, AllocationEvent};
use kv::{GpuIntegratedCache, GpuIntegratedCacheStats};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};
use std::time::{Duration, Instant};

/// Tracks GPU memory usage and predicts optimal allocation strategies
pub struct GpuMemoryTracker {
    /// Reference to GPU cache for direct memory monitoring
    gpu_cache: Arc<Mutex<GpuIntegratedCache>>,
    /// Current available memory (updated atomically)
    available_memory: AtomicUsize,
    /// Circular buffer of recent allocation events
    allocation_history: Mutex<CircularBuffer<AllocationEvent>>,
    /// Fragmentation analysis component
    fragmentation_monitor: FragmentationAnalyzer,
    /// Out-of-memory prediction system
    oom_predictor: OOMPredictor,
    /// Memory watermark levels
    memory_watermarks: MemoryWatermarks,
    /// Allocation tracking metrics
    allocation_metrics: AllocationMetrics,
}

impl GpuMemoryTracker {
    /// Create a new GPU memory tracker
    pub fn new(gpu_cache: Arc<Mutex<GpuIntegratedCache>>, total_memory: usize) -> Self {
        let available_memory = {
            let cache_guard = gpu_cache.lock().unwrap();
            let stats = cache_guard.get_stats();
            stats.total_gpu_memory - stats.allocated_gpu_memory
        };

        Self {
            gpu_cache,
            available_memory: AtomicUsize::new(available_memory),
            allocation_history: Mutex::new(CircularBuffer::new(1000)),
            fragmentation_monitor: FragmentationAnalyzer::new(),
            oom_predictor: OOMPredictor::new(),
            memory_watermarks: MemoryWatermarks::new(total_memory),
            allocation_metrics: AllocationMetrics::new(),
        }
    }

    /// Check if a batch can be accommodated in available GPU memory
    pub fn can_accommodate_batch(&self, requests: &[Request]) -> MemoryFeasibility {
        let start_time = Instant::now();

        // Calculate total memory requirement
        let memory_requirement = self.calculate_batch_memory_requirement(requests);
        let current_available = self.available_memory.load(Ordering::Acquire);

        // Check basic feasibility
        if memory_requirement.total_bytes > current_available {
            return MemoryFeasibility {
                feasible: false,
                confidence: 0.95,
                available_memory: current_available,
                required_memory: memory_requirement.total_bytes,
                fragmentation_overhead: 0,
                estimated_allocation_time: Duration::from_millis(0),
                memory_pressure_level: self.calculate_memory_pressure(),
                recommendation: AllocationRecommendation::Reject,
                analysis_time: start_time.elapsed(),
            };
        }

        // Account for fragmentation overhead
        let fragmentation_overhead = self.fragmentation_monitor.estimate_overhead(memory_requirement.total_bytes);
        let effective_requirement = memory_requirement.total_bytes + fragmentation_overhead;

        let feasible = effective_requirement < current_available;
        let pressure_level = self.calculate_memory_pressure();

        // Get allocation time estimate
        let estimated_allocation_time = self.estimate_allocation_time(&memory_requirement);

        // Generate recommendation based on analysis
        let recommendation = self.generate_allocation_recommendation(
            feasible,
            pressure_level,
            &memory_requirement,
            current_available
        );

        MemoryFeasibility {
            feasible,
            confidence: self.calculate_feasibility_confidence(feasible, pressure_level),
            available_memory: current_available,
            required_memory: effective_requirement,
            fragmentation_overhead,
            estimated_allocation_time,
            memory_pressure_level: pressure_level,
            recommendation,
            analysis_time: start_time.elapsed(),
        }
    }

    /// Suggest optimal batch size given current memory constraints
    pub fn suggest_batch_size(&self, requests: &[Request]) -> OptimalBatchSize {
        let start_time = Instant::now();
        let current_available = self.available_memory.load(Ordering::Acquire);
        let pressure_level = self.calculate_memory_pressure();

        // Start with conservative estimate if under pressure
        let max_safe_memory = if pressure_level > 0.7 {
            (current_available as f64 * 0.6) as usize // Use only 60% when under pressure
        } else if pressure_level > 0.5 {
            (current_available as f64 * 0.8) as usize // Use 80% at moderate pressure
        } else {
            (current_available as f64 * 0.9) as usize // Use 90% when plenty of memory
        };

        // Binary search for optimal batch size
        let optimal_size = self.find_optimal_batch_size(requests, max_safe_memory);

        // Calculate expected performance metrics
        let optimal_requests = &requests[..optimal_size.min(requests.len())];
        let memory_requirement = self.calculate_batch_memory_requirement(optimal_requests);
        let expected_throughput = self.estimate_throughput(optimal_requests, &memory_requirement);
        let expected_latency = self.estimate_batch_latency(optimal_requests, &memory_requirement);

        OptimalBatchSize {
            recommended_size: optimal_size,
            max_possible_size: self.calculate_max_possible_size(requests, current_available),
            memory_efficiency: memory_requirement.total_bytes as f64 / max_safe_memory as f64,
            expected_throughput,
            expected_latency,
            confidence: self.calculate_batch_size_confidence(optimal_size, requests.len()),
            reasoning: self.generate_sizing_reasoning(optimal_size, requests.len(), pressure_level),
            analysis_time: start_time.elapsed(),
        }
    }

    /// Update memory tracking after allocation
    pub fn record_allocation(&self, event: AllocationEvent) {
        // Update available memory atomically
        self.available_memory.fetch_sub(event.size, Ordering::AcqRel);

        // Record in allocation history
        {
            let mut history = self.allocation_history.lock().unwrap();
            history.push(event.clone()); // Clone for history
        }

        // Update fragmentation analysis
        self.fragmentation_monitor.record_allocation(&event);

        // Update allocation metrics
        self.allocation_metrics.record_allocation(&event);

        // Train OOM predictor
        self.oom_predictor.update_with_allocation(&event);
    }

    /// Update memory tracking after deallocation
    pub fn record_deallocation(&self, size: usize, allocation_id: u64) {
        // Update available memory atomically
        self.available_memory.fetch_add(size, Ordering::AcqRel);

        let event = AllocationEvent {
            timestamp: Instant::now(),
            event_type: AllocationEventType::Deallocate,
            size,
            allocation_id: Some(allocation_id),
            sequence_id: None,
            fragmentation_before: self.fragmentation_monitor.current_fragmentation_ratio(),
            fragmentation_after: 0.0, // Will be updated after processing
        };

        // Record deallocation
        {
            let mut history = self.allocation_history.lock().unwrap();
            history.push(event);
        }

        // Update fragmentation analysis
        self.fragmentation_monitor.record_deallocation(size, allocation_id);

        // Update metrics
        self.allocation_metrics.record_deallocation(size);
    }

    /// Get current memory statistics
    pub fn get_memory_stats(&self) -> GpuMemoryStats {
        let cache_stats = {
            let cache_guard = self.gpu_cache.lock().unwrap();
            cache_guard.get_stats()
        };

        let history_stats = {
            let history = self.allocation_history.lock().unwrap();
            history.get_statistics()
        };

        GpuMemoryStats {
            total_memory: cache_stats.total_gpu_memory,
            allocated_memory: cache_stats.allocated_gpu_memory,
            available_memory: self.available_memory.load(Ordering::Acquire),
            fragmentation_ratio: self.fragmentation_monitor.current_fragmentation_ratio(),
            memory_pressure: self.calculate_memory_pressure(),
            allocation_rate: history_stats.allocations_per_second,
            deallocation_rate: history_stats.deallocations_per_second,
            average_allocation_size: history_stats.average_allocation_size,
            peak_memory_usage: self.allocation_metrics.peak_usage(),
            oom_risk_score: self.oom_predictor.current_risk_score(),
            watermark_status: self.memory_watermarks.current_status(),
        }
    }

    // Private helper methods

    fn calculate_batch_memory_requirement(&self, requests: &[Request]) -> BatchMemoryRequirement {
        let mut total_prompt_tokens = 0;
        let mut total_generation_tokens = 0;
        let mut sequence_count = requests.len();

        for request in requests {
            total_prompt_tokens += request.prompt_tokens.len();
            total_generation_tokens += request.max_new_tokens;
        }

        let total_tokens = total_prompt_tokens + total_generation_tokens;

        // Estimate memory requirements
        let bytes_per_token = 32; // 16 bytes K + 16 bytes V for bf16
        let kv_cache_bytes = total_tokens * bytes_per_token;

        // Additional overhead for metadata, attention matrices, etc.
        let metadata_overhead = sequence_count * 1024; // 1KB per sequence
        let attention_overhead = (total_tokens * total_tokens * 2) / sequence_count; // Simplified attention memory

        let total_bytes = kv_cache_bytes + metadata_overhead + attention_overhead;

        BatchMemoryRequirement {
            total_bytes,
            kv_cache_bytes,
            metadata_bytes: metadata_overhead,
            attention_bytes: attention_overhead,
            sequence_count,
            total_tokens,
            average_tokens_per_sequence: total_tokens / sequence_count,
        }
    }

    fn calculate_memory_pressure(&self) -> f64 {
        let current_available = self.available_memory.load(Ordering::Acquire);
        let cache_stats = {
            let cache_guard = self.gpu_cache.lock().unwrap();
            cache_guard.get_stats()
        };

        let utilization = cache_stats.allocated_gpu_memory as f64 / cache_stats.total_gpu_memory as f64;

        // Pressure increases exponentially as we approach memory limits
        if utilization > 0.95 {
            1.0 // Critical pressure
        } else if utilization > 0.9 {
            0.8 + (utilization - 0.9) * 4.0 // High pressure
        } else if utilization > 0.8 {
            0.5 + (utilization - 0.8) * 3.0 // Medium pressure
        } else if utilization > 0.7 {
            0.2 + (utilization - 0.7) * 3.0 // Low pressure
        } else {
            0.0 // No pressure
        }
    }

    fn estimate_allocation_time(&self, requirement: &BatchMemoryRequirement) -> Duration {
        // Base allocation time (empirically measured)
        let base_time_ns = 1000; // 1μs base
        let size_factor = (requirement.total_bytes as f64).log2() * 100.0; // Log scaling with size
        let fragmentation_penalty = self.fragmentation_monitor.current_fragmentation_ratio() * 500.0;

        Duration::from_nanos((base_time_ns as f64 + size_factor + fragmentation_penalty) as u64)
    }

    fn find_optimal_batch_size(&self, requests: &[Request], max_memory: usize) -> usize {
        let mut left = 1;
        let mut right = requests.len();
        let mut optimal = 1;

        while left <= right {
            let mid = (left + right) / 2;
            let batch = &requests[..mid];
            let requirement = self.calculate_batch_memory_requirement(batch);
            let fragmentation_overhead = self.fragmentation_monitor.estimate_overhead(requirement.total_bytes);

            if requirement.total_bytes + fragmentation_overhead <= max_memory {
                optimal = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        optimal
    }

    fn calculate_max_possible_size(&self, requests: &[Request], available: usize) -> usize {
        // Calculate theoretical maximum without safety margins
        self.find_optimal_batch_size(requests, available)
    }

    fn estimate_throughput(&self, requests: &[Request], requirement: &BatchMemoryRequirement) -> f64 {
        // Simplified throughput estimation based on token count and memory efficiency
        let base_throughput = requirement.total_tokens as f64 / 100.0; // 100 tokens/second baseline
        let memory_efficiency_factor = 1.0 - (requirement.total_bytes as f64 / self.available_memory.load(Ordering::Acquire) as f64).min(1.0) * 0.2;

        base_throughput * memory_efficiency_factor
    }

    fn estimate_batch_latency(&self, _requests: &[Request], requirement: &BatchMemoryRequirement) -> Duration {
        // Simplified latency estimation
        let base_latency_ms = requirement.average_tokens_per_sequence as f64 * 0.01; // 10μs per token
        let memory_pressure_penalty = self.calculate_memory_pressure() * 5.0; // Up to 5ms penalty

        Duration::from_millis((base_latency_ms + memory_pressure_penalty) as u64)
    }

    fn calculate_feasibility_confidence(&self, feasible: bool, pressure_level: f64) -> f64 {
        if feasible {
            if pressure_level < 0.3 { 0.95 }
            else if pressure_level < 0.6 { 0.85 }
            else { 0.70 }
        } else {
            if pressure_level > 0.9 { 0.95 }
            else { 0.80 }
        }
    }

    fn calculate_batch_size_confidence(&self, optimal_size: usize, total_requests: usize) -> f64 {
        let size_ratio = optimal_size as f64 / total_requests as f64;
        if size_ratio > 0.8 { 0.9 }
        else if size_ratio > 0.5 { 0.8 }
        else { 0.7 }
    }

    fn generate_allocation_recommendation(
        &self,
        feasible: bool,
        pressure_level: f64,
        _requirement: &BatchMemoryRequirement,
        _available: usize,
    ) -> AllocationRecommendation {
        match (feasible, pressure_level > 0.7) {
            (true, false) => AllocationRecommendation::Proceed,
            (true, true) => AllocationRecommendation::ProceedWithCaution,
            (false, _) => AllocationRecommendation::Reject,
        }
    }

    fn generate_sizing_reasoning(&self, optimal: usize, total: usize, pressure: f64) -> String {
        if optimal == total {
            "All requests can be accommodated efficiently".to_string()
        } else if pressure > 0.7 {
            format!("Reduced batch size due to high memory pressure ({}%)", (pressure * 100.0) as u8)
        } else {
            format!("Optimal size balances memory efficiency and throughput ({}/{})", optimal, total)
        }
    }
}

// Supporting data structures and implementations

#[derive(Debug, Clone)]
pub struct BatchMemoryRequirement {
    pub total_bytes: usize,
    pub kv_cache_bytes: usize,
    pub metadata_bytes: usize,
    pub attention_bytes: usize,
    pub sequence_count: usize,
    pub total_tokens: usize,
    pub average_tokens_per_sequence: usize,
}

#[derive(Debug, Clone)]
pub enum AllocationEventType {
    Allocate,
    Deallocate,
    Reallocate,
}

#[derive(Debug)]
pub enum AllocationRecommendation {
    Proceed,
    ProceedWithCaution,
    Reject,
}

#[derive(Debug)]
pub struct GpuMemoryStats {
    pub total_memory: usize,
    pub allocated_memory: usize,
    pub available_memory: usize,
    pub fragmentation_ratio: f64,
    pub memory_pressure: f64,
    pub allocation_rate: f64,
    pub deallocation_rate: f64,
    pub average_allocation_size: usize,
    pub peak_memory_usage: usize,
    pub oom_risk_score: f64,
    pub watermark_status: WatermarkStatus,
}

// Placeholder implementations for supporting components

pub struct CircularBuffer<T> {
    buffer: VecDeque<T>,
    capacity: usize,
}

impl<T> CircularBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, item: T) {
        if self.buffer.len() == self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(item);
    }

    pub fn get_statistics(&self) -> AllocationHistoryStats {
        AllocationHistoryStats {
            allocations_per_second: 10.0,
            deallocations_per_second: 8.0,
            average_allocation_size: 1024,
        }
    }
}

pub struct AllocationHistoryStats {
    pub allocations_per_second: f64,
    pub deallocations_per_second: f64,
    pub average_allocation_size: usize,
}

pub struct FragmentationAnalyzer {}

impl FragmentationAnalyzer {
    pub fn new() -> Self { Self {} }
    pub fn estimate_overhead(&self, _size: usize) -> usize { 0 }
    pub fn current_fragmentation_ratio(&self) -> f64 { 0.1 }
    pub fn record_allocation(&self, _event: &AllocationEvent) {}
    pub fn record_deallocation(&self, _size: usize, _id: u64) {}
}

pub struct OOMPredictor {}

impl OOMPredictor {
    pub fn new() -> Self { Self {} }
    pub fn current_risk_score(&self) -> f64 { 0.1 }
    pub fn update_with_allocation(&self, _event: &AllocationEvent) {}
}

pub struct MemoryWatermarks {
    _total_memory: usize,
}

impl MemoryWatermarks {
    pub fn new(total_memory: usize) -> Self {
        Self { _total_memory: total_memory }
    }

    pub fn current_status(&self) -> WatermarkStatus {
        WatermarkStatus::Normal
    }
}

#[derive(Debug)]
pub enum WatermarkStatus {
    Normal,
    Warning,
    Critical,
}

pub struct AllocationMetrics {}

impl AllocationMetrics {
    pub fn new() -> Self { Self {} }
    pub fn record_allocation(&self, _event: &AllocationEvent) {}
    pub fn record_deallocation(&self, _size: usize) {}
    pub fn peak_usage(&self) -> usize { 0 }
}