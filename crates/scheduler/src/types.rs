//! Type definitions for the intelligent scheduler system

use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Token identifier type
pub type TokenId = u32;
/// Sequence identifier type
pub type SequenceId = u32;

/// A request for LLM inference
#[derive(Debug, Clone)]
pub struct Request {
    /// Unique request ID
    pub id: u64,
    /// Input prompt tokens
    pub prompt_tokens: Vec<TokenId>,
    /// Maximum number of new tokens to generate
    pub max_new_tokens: usize,
    /// Current generation progress
    pub current_tokens: usize,
    /// Request priority
    pub priority: RequestPriority,
    /// Creation timestamp
    pub created_at: Instant,
    /// KV cache sequence ID if allocated
    pub sequence_id: Option<SequenceId>,
}

impl Request {
    pub fn new(
        id: u64,
        prompt_tokens: Vec<TokenId>,
        max_new_tokens: usize,
        priority: RequestPriority,
    ) -> Self {
        Self {
            id,
            prompt_tokens,
            max_new_tokens,
            current_tokens: 0,
            priority,
            created_at: Instant::now(),
            sequence_id: None,
        }
    }

    pub fn total_expected_tokens(&self) -> usize {
        self.prompt_tokens.len() + self.max_new_tokens
    }
}

/// Request priority levels for scheduling
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RequestPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Scheduling policy options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SchedulingPolicy {
    /// First-Come-First-Serve (vLLM style)
    FCFS,
    /// Longest prefix match first (SGLang inspired)
    LongestPrefixMatch,
    /// Cache-aware scheduling (UniLLM innovation)
    CacheAware,
    /// GPU memory optimized scheduling
    GpuMemoryOptimized,
    /// ML-based adaptive policy
    AdaptiveML,
}

/// Analysis of a request batch for optimization
#[derive(Debug)]
pub struct BatchAnalysis {
    /// Number of requests in the batch
    pub request_count: usize,
    /// Groups of requests sharing prefixes
    pub prefix_sharing_groups: Vec<PrefixSharingGroup>,
    /// Expected cache hit rate for the batch
    pub expected_cache_hit_rate: f64,
    /// Total memory requirement
    pub memory_requirement: usize,
    /// Memory pressure score (0.0 = no pressure, 1.0 = critical)
    pub memory_pressure_score: f64,
    /// Expected batch processing latency
    pub expected_latency: Duration,
    /// Expected throughput
    pub throughput_prediction: f64,
    /// Optimization opportunities identified
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
    /// Time taken for analysis
    pub analysis_time: Duration,
    /// Confidence score for the analysis
    pub confidence_score: f64,
}

/// Group of requests that share a common prefix
#[derive(Debug, Clone)]
pub struct PrefixSharingGroup {
    /// Indices of requests in this group
    pub requests: Vec<usize>,
    /// Length of common prefix
    pub common_prefix_length: usize,
    /// Cache hit potential for this group
    pub cache_hit_potential: f64,
    /// Memory savings from prefix sharing
    pub memory_savings: usize,
}

/// An optimization opportunity identified during analysis
#[derive(Debug)]
pub struct OptimizationOpportunity {
    /// Type of optimization
    pub op_type: OptimizationType,
    /// Request indices involved
    pub request_indices: Vec<usize>,
    /// Potential savings in tokens/bytes
    pub potential_savings: usize,
    /// Confidence in this optimization
    pub confidence: f64,
}

/// Types of optimizations that can be applied
#[derive(Debug)]
pub enum OptimizationType {
    /// Prefix sharing between requests
    PrefixSharing,
    /// Memory coalescing
    MemoryCoalescing,
    /// Cache locality improvements
    CacheLocality,
}

/// Result of batch optimization
#[derive(Debug)]
pub struct OptimizationResult {
    /// Number of requests reordered
    pub requests_reordered: usize,
    /// Improvement in cache hit rate
    pub cache_hit_improvement: f64,
    /// Memory efficiency gain
    pub memory_efficiency_gain: f64,
    /// Expected latency reduction
    pub expected_latency_reduction: Duration,
    /// Prefix sharing improvement ratio
    pub prefix_sharing_improvement: f64,
    /// Time taken for optimization
    pub optimization_time: Duration,
    /// Whether optimization was successful
    pub success: bool,
}

/// Memory constraints and availability
#[derive(Debug)]
pub struct MemoryConstraints {
    /// Available GPU memory in bytes
    pub available_memory: usize,
    /// Memory pressure level (0.0-1.0)
    pub pressure_level: f64,
    /// Fragmentation factor
    pub fragmentation_factor: f64,
    /// Maximum single allocation size
    pub max_allocation_size: usize,
}

/// Memory feasibility analysis for a batch
#[derive(Debug)]
pub struct MemoryFeasibility {
    /// Whether the batch can be accommodated
    pub feasible: bool,
    /// Confidence in the feasibility assessment
    pub confidence: f64,
    /// Available memory at analysis time
    pub available_memory: usize,
    /// Required memory for the batch
    pub required_memory: usize,
    /// Estimated fragmentation overhead
    pub fragmentation_overhead: usize,
    /// Estimated allocation time
    pub estimated_allocation_time: Duration,
    /// Current memory pressure level
    pub memory_pressure_level: f64,
    /// Recommendation for allocation
    pub recommendation: AllocationRecommendation,
    /// Time taken for analysis
    pub analysis_time: Duration,
}

/// Recommendation for memory allocation
#[derive(Debug)]
pub enum AllocationRecommendation {
    /// Proceed with allocation
    Proceed,
    /// Proceed but with caution
    ProceedWithCaution,
    /// Reject the allocation
    Reject,
}

/// Optimal batch size analysis
#[derive(Debug)]
pub struct OptimalBatchSize {
    /// Recommended batch size
    pub recommended_size: usize,
    /// Maximum possible batch size
    pub max_possible_size: usize,
    /// Memory efficiency ratio
    pub memory_efficiency: f64,
    /// Expected throughput
    pub expected_throughput: f64,
    /// Expected latency
    pub expected_latency: Duration,
    /// Confidence in the recommendation
    pub confidence: f64,
    /// Reasoning for the recommendation
    pub reasoning: String,
    /// Time taken for analysis
    pub analysis_time: Duration,
}

/// Memory allocation event for tracking
#[derive(Debug, Clone)]
pub struct AllocationEvent {
    /// Event timestamp
    pub timestamp: Instant,
    /// Type of allocation event
    pub event_type: AllocationEventType,
    /// Size of allocation in bytes
    pub size: usize,
    /// Optional allocation ID
    pub allocation_id: Option<u64>,
    /// Optional sequence ID
    pub sequence_id: Option<u32>,
    /// Fragmentation before allocation
    pub fragmentation_before: f64,
    /// Fragmentation after allocation
    pub fragmentation_after: f64,
}

/// Types of allocation events
#[derive(Debug, Clone)]
pub enum AllocationEventType {
    /// Memory allocation
    Allocate,
    /// Memory deallocation
    Deallocate,
    /// Memory reallocation
    Reallocate,
}

/// Scheduler performance metrics
#[derive(Debug, Clone)]
pub struct SchedulerMetrics {
    /// Requests processed per second
    pub requests_per_second: f64,
    /// Average prompt length
    pub average_prompt_length: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Memory pressure level
    pub memory_pressure: f64,
    /// Time taken for batch formation
    pub batch_formation_time: Duration,
    /// Active request count
    pub active_requests: usize,
    /// Completed request count
    pub completed_requests: usize,
    /// Failed request count
    pub failed_requests: usize,
    /// Timestamp of metrics
    pub timestamp: Instant,
}

/// Policy decision from adaptive engine
#[derive(Debug)]
pub struct PolicyDecision {
    /// Action to take
    pub action: PolicyAction,
    /// Confidence in the decision
    pub confidence: f64,
    /// Reasoning for the decision
    pub reasoning: String,
    /// Expected performance impact
    pub expected_impact: f64,
    /// Time required for policy transition
    pub transition_time: Duration,
    /// Summary of workload analysis
    pub analysis_summary: String,
}

/// Policy action recommendations
#[derive(Debug)]
pub enum PolicyAction {
    /// Keep current policy
    KeepCurrent,
    /// Change to a specific policy
    ChangeTo(SchedulingPolicy),
}

/// Workload profile for adaptive scheduling
#[derive(Debug)]
pub struct WorkloadProfile {
    /// Average request rate
    pub request_rate: f64,
    /// Average prompt length
    pub average_prompt_length: f64,
    /// Cache hit patterns
    pub cache_patterns: HashMap<String, f64>,
    /// Memory usage patterns
    pub memory_patterns: Vec<f64>,
    /// Request size distribution
    pub size_distribution: Vec<usize>,
    /// Temporal patterns
    pub temporal_patterns: Vec<Duration>,
}

impl WorkloadProfile {
    pub fn new() -> Self {
        Self {
            request_rate: 0.0,
            average_prompt_length: 0.0,
            cache_patterns: HashMap::new(),
            memory_patterns: Vec::new(),
            size_distribution: Vec::new(),
            temporal_patterns: Vec::new(),
        }
    }

    pub fn update_from_metrics(&mut self, metrics: &SchedulerMetrics) {
        self.request_rate = metrics.requests_per_second;
        self.average_prompt_length = metrics.average_prompt_length;

        // Update patterns (simplified)
        self.memory_patterns.push(metrics.memory_pressure);
        if self.memory_patterns.len() > 100 {
            self.memory_patterns.remove(0);
        }
    }
}