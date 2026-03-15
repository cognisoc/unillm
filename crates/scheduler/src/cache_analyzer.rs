//! Cache-aware analysis for intelligent batch formation
//!
//! This module provides deep integration with the GPU cache system to optimize
//! request scheduling based on cache hit potential and memory constraints.

use crate::types::{
    Request, BatchAnalysis, OptimizationResult, MemoryConstraints,
    PrefixSharingGroup, OptimizationOpportunity, OptimizationType
};
use kv::{GpuIntegratedCache, TokenId, SequenceId, GpuIntegratedCacheStats};
use std::collections::{HashMap, BTreeMap};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Analyzes cache state and predicts performance for request batches
pub struct CacheAwareAnalyzer {
    /// Reference to the GPU-integrated cache system
    gpu_cache: Arc<Mutex<GpuIntegratedCache>>,
    /// Prefix tree for fast common sequence detection
    prefix_detector: RadixPrefixMatcher,
    /// Memory pressure monitoring and prediction
    memory_monitor: MemoryPressureMonitor,
    /// Access pattern predictor for cache warming
    access_predictor: AccessPatternPredictor,
    /// Performance history for analysis refinement
    analysis_history: AnalysisHistory,
}

impl CacheAwareAnalyzer {
    /// Create a new cache analyzer with GPU integration
    pub fn new(gpu_cache: Arc<Mutex<GpuIntegratedCache>>) -> Self {
        Self {
            gpu_cache,
            prefix_detector: RadixPrefixMatcher::new(),
            memory_monitor: MemoryPressureMonitor::new(),
            access_predictor: AccessPatternPredictor::new(),
            analysis_history: AnalysisHistory::new(),
        }
    }

    /// Analyze a batch of requests for cache optimization opportunities
    pub fn analyze_request_batch(&self, requests: &[Request]) -> BatchAnalysis {
        let start_time = Instant::now();

        // Get current cache state from GPU system
        let cache_stats = {
            let cache_guard = self.gpu_cache.lock().unwrap();
            cache_guard.get_stats()
        };

        // Analyze prefix sharing opportunities
        let prefix_analysis = self.analyze_prefix_sharing(requests);

        // Predict cache hit rates for each request
        let cache_predictions = self.predict_cache_hits(requests, &cache_stats);

        // Analyze memory requirements and constraints
        let memory_analysis = self.analyze_memory_requirements(requests, &cache_stats);

        // Calculate expected performance metrics
        let performance_prediction = self.predict_batch_performance(
            requests,
            &prefix_analysis,
            &cache_predictions,
            &memory_analysis
        );

        let analysis_time = start_time.elapsed();

        BatchAnalysis {
            request_count: requests.len(),
            prefix_sharing_groups: prefix_analysis.sharing_groups,
            expected_cache_hit_rate: cache_predictions.average_hit_rate,
            memory_requirement: memory_analysis.total_memory_needed,
            memory_pressure_score: memory_analysis.pressure_score,
            expected_latency: performance_prediction.expected_latency,
            throughput_prediction: performance_prediction.expected_throughput,
            optimization_opportunities: prefix_analysis.optimization_ops,
            analysis_time,
            confidence_score: self.calculate_confidence_score(&cache_predictions, &memory_analysis),
        }
    }

    /// Optimize batch composition for maximum cache utilization
    pub fn optimize_batch_composition(&self, requests: &mut Vec<Request>) -> OptimizationResult {
        let original_order = requests.clone();
        let start_time = Instant::now();

        // Phase 1: Group requests by prefix sharing potential
        let prefix_groups = self.group_by_prefix_potential(requests);

        // Phase 2: Reorder within groups for memory locality
        let memory_optimized_groups = self.optimize_memory_locality(prefix_groups);

        // Phase 3: Balance groups for optimal GPU utilization
        let balanced_batch = self.balance_gpu_utilization(memory_optimized_groups);

        // Update the original request vector
        *requests = balanced_batch;

        // Calculate optimization metrics
        let original_analysis = self.analyze_request_batch(&original_order);
        let optimized_analysis = self.analyze_request_batch(requests);

        let optimization_time = start_time.elapsed();

        OptimizationResult {
            requests_reordered: requests.len(),
            cache_hit_improvement: optimized_analysis.expected_cache_hit_rate - original_analysis.expected_cache_hit_rate,
            memory_efficiency_gain: self.calculate_memory_efficiency_gain(&original_analysis, &optimized_analysis),
            expected_latency_reduction: original_analysis.expected_latency - optimized_analysis.expected_latency,
            prefix_sharing_improvement: optimized_analysis.prefix_sharing_groups.len() as f64 / original_analysis.prefix_sharing_groups.len().max(1) as f64,
            optimization_time,
            success: true,
        }
    }

    /// Analyze prefix sharing opportunities among requests
    fn analyze_prefix_sharing(&self, requests: &[Request]) -> PrefixAnalysis {
        let mut sharing_groups = Vec::new();
        let mut optimization_ops = Vec::new();
        let mut processed = vec![false; requests.len()];

        for i in 0..requests.len() {
            if processed[i] {
                continue;
            }

            let mut group = PrefixSharingGroup {
                requests: vec![i],
                common_prefix_length: requests[i].prompt_tokens.len(),
                cache_hit_potential: 0.0,
                memory_savings: 0,
            };

            // Find other requests that share a prefix with this one
            for j in (i + 1)..requests.len() {
                if processed[j] {
                    continue;
                }

                let common_prefix_len = self.find_common_prefix_length(
                    &requests[i].prompt_tokens,
                    &requests[j].prompt_tokens
                );

                if common_prefix_len >= 32 { // Minimum prefix length for sharing
                    group.requests.push(j);
                    group.common_prefix_length = group.common_prefix_length.min(common_prefix_len);
                    processed[j] = true;

                    optimization_ops.push(OptimizationOpportunity {
                        op_type: OptimizationType::PrefixSharing,
                        request_indices: vec![i, j],
                        potential_savings: common_prefix_len * 2, // Memory savings in tokens
                        confidence: 0.9,
                    });
                }
            }

            if group.requests.len() > 1 {
                group.cache_hit_potential = self.calculate_cache_hit_potential(&group, requests);
                group.memory_savings = group.common_prefix_length * (group.requests.len() - 1);
                sharing_groups.push(group);
            }

            processed[i] = true;
        }

        let total_sharing_potential = optimization_ops.iter().map(|op| op.potential_savings).sum();

        PrefixAnalysis {
            sharing_groups,
            optimization_ops,
            total_sharing_potential,
        }
    }

    /// Predict cache hit rates for requests
    fn predict_cache_hits(&self, requests: &[Request], cache_stats: &GpuIntegratedCacheStats) -> CachePrediction {
        let mut hit_predictions = Vec::new();
        let mut total_hit_rate = 0.0;

        for request in requests {
            // Check if prefix exists in current cache
            let prefix_hit_probability = {
                let cache_guard = self.gpu_cache.lock().unwrap();
                self.calculate_prefix_hit_probability(&request.prompt_tokens, &cache_guard)
            };

            // Factor in cache pressure and eviction likelihood
            let cache_pressure_factor = 1.0 - (cache_stats.memory_usage_percent / 100.0).powf(2.0);
            let adjusted_hit_rate = prefix_hit_probability * cache_pressure_factor;

            hit_predictions.push(RequestCachePrediction {
                request_id: request.id,
                l1_hit_probability: adjusted_hit_rate,
                l2_hit_probability: adjusted_hit_rate * 0.7, // L2 typically has lower hit rate
                l3_hit_probability: adjusted_hit_rate * 0.3, // L3 much lower
                overall_hit_probability: adjusted_hit_rate,
                confidence: 0.85,
            });

            total_hit_rate += adjusted_hit_rate;
        }

        CachePrediction {
            request_predictions: hit_predictions,
            average_hit_rate: total_hit_rate / requests.len() as f64,
            cache_efficiency_score: cache_stats.l1_hit_rate * 0.5 + cache_stats.l2_hit_rate * 0.3 + cache_stats.l3_hit_rate * 0.2,
        }
    }

    /// Analyze memory requirements and constraints
    fn analyze_memory_requirements(&self, requests: &[Request], cache_stats: &GpuIntegratedCacheStats) -> MemoryAnalysis {
        let total_tokens: usize = requests.iter().map(|r| r.prompt_tokens.len() + r.max_new_tokens).sum();
        let bytes_per_token = 32; // Estimate: 16 bytes K + 16 bytes V
        let total_memory_needed = total_tokens * bytes_per_token;

        let available_memory = cache_stats.total_gpu_memory - cache_stats.allocated_gpu_memory;
        let memory_utilization = cache_stats.allocated_gpu_memory as f64 / cache_stats.total_gpu_memory as f64;

        // Calculate pressure score (0.0 = no pressure, 1.0 = critical)
        let pressure_score = if memory_utilization > 0.9 {
            1.0
        } else if memory_utilization > 0.8 {
            (memory_utilization - 0.8) * 10.0
        } else {
            0.0
        };

        MemoryAnalysis {
            total_memory_needed,
            available_memory,
            memory_utilization,
            pressure_score,
            fragmentation_factor: cache_stats.fragmentation_factor,
            allocation_feasible: total_memory_needed < available_memory,
        }
    }

    /// Predict batch performance metrics
    fn predict_batch_performance(
        &self,
        requests: &[Request],
        prefix_analysis: &PrefixAnalysis,
        cache_predictions: &CachePrediction,
        memory_analysis: &MemoryAnalysis,
    ) -> PerformancePrediction {
        // Base latency calculation
        let avg_prompt_length: f64 = requests.iter().map(|r| r.prompt_tokens.len()).sum::<usize>() as f64 / requests.len() as f64;
        let base_prefill_latency = Duration::from_nanos((avg_prompt_length * 50_000.0) as u64); // 50μs per token estimate

        // Cache hit adjustment
        let cache_speedup_factor = 1.0 + (cache_predictions.average_hit_rate * 0.3); // 30% speedup from cache hits
        let cache_adjusted_latency = base_prefill_latency.as_nanos() as f64 / cache_speedup_factor;

        // Memory pressure penalty
        let memory_penalty = if memory_analysis.pressure_score > 0.5 {
            1.0 + (memory_analysis.pressure_score - 0.5) * 0.4 // Up to 20% penalty
        } else {
            1.0
        };

        let final_latency = Duration::from_nanos((cache_adjusted_latency * memory_penalty) as u64);

        // Throughput calculation
        let base_throughput = requests.len() as f64 / final_latency.as_secs_f64();
        let prefix_sharing_boost = 1.0 + (prefix_analysis.sharing_groups.len() as f64 * 0.1);
        let expected_throughput = base_throughput * prefix_sharing_boost;

        PerformancePrediction {
            expected_latency: final_latency,
            expected_throughput,
            cache_contribution: cache_predictions.average_hit_rate * 0.3,
            memory_pressure_impact: memory_penalty - 1.0,
            prefix_sharing_benefit: prefix_sharing_boost - 1.0,
        }
    }

    /// Calculate confidence score for the analysis
    fn calculate_confidence_score(&self, cache_predictions: &CachePrediction, memory_analysis: &MemoryAnalysis) -> f64 {
        let cache_confidence = cache_predictions.request_predictions.iter()
            .map(|p| p.confidence)
            .sum::<f64>() / cache_predictions.request_predictions.len() as f64;

        let memory_confidence = if memory_analysis.allocation_feasible { 0.9 } else { 0.3 };

        (cache_confidence + memory_confidence) / 2.0
    }

    // Helper methods
    fn find_common_prefix_length(&self, tokens1: &[TokenId], tokens2: &[TokenId]) -> usize {
        tokens1.iter()
            .zip(tokens2.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }

    fn calculate_prefix_hit_probability(&self, tokens: &[TokenId], _cache: &GpuIntegratedCache) -> f64 {
        // Simplified implementation - in practice would query actual cache
        if tokens.len() > 100 { 0.7 } else if tokens.len() > 50 { 0.5 } else { 0.3 }
    }

    fn calculate_cache_hit_potential(&self, _group: &PrefixSharingGroup, _requests: &[Request]) -> f64 {
        0.8 // Simplified - would calculate based on actual cache state
    }

    fn group_by_prefix_potential(&self, requests: &[Request]) -> Vec<Vec<Request>> {
        // Simplified grouping - would implement more sophisticated algorithm
        vec![requests.to_vec()]
    }

    fn optimize_memory_locality(&self, groups: Vec<Vec<Request>>) -> Vec<Vec<Request>> {
        // Return as-is for now - would implement memory locality optimization
        groups
    }

    fn balance_gpu_utilization(&self, groups: Vec<Vec<Request>>) -> Vec<Request> {
        // Flatten groups for now - would implement GPU utilization balancing
        groups.into_iter().flatten().collect()
    }

    fn calculate_memory_efficiency_gain(&self, _original: &BatchAnalysis, _optimized: &BatchAnalysis) -> f64 {
        0.15 // Simplified - would calculate actual efficiency gain
    }
}

// Supporting data structures

#[derive(Debug)]
pub struct PrefixAnalysis {
    pub sharing_groups: Vec<PrefixSharingGroup>,
    pub optimization_ops: Vec<OptimizationOpportunity>,
    pub total_sharing_potential: usize,
}

#[derive(Debug)]
pub struct RequestCachePrediction {
    pub request_id: u64,
    pub l1_hit_probability: f64,
    pub l2_hit_probability: f64,
    pub l3_hit_probability: f64,
    pub overall_hit_probability: f64,
    pub confidence: f64,
}

#[derive(Debug)]
pub struct CachePrediction {
    pub request_predictions: Vec<RequestCachePrediction>,
    pub average_hit_rate: f64,
    pub cache_efficiency_score: f64,
}

#[derive(Debug)]
pub struct MemoryAnalysis {
    pub total_memory_needed: usize,
    pub available_memory: usize,
    pub memory_utilization: f64,
    pub pressure_score: f64,
    pub fragmentation_factor: f64,
    pub allocation_feasible: bool,
}

#[derive(Debug)]
pub struct PerformancePrediction {
    pub expected_latency: Duration,
    pub expected_throughput: f64,
    pub cache_contribution: f64,
    pub memory_pressure_impact: f64,
    pub prefix_sharing_benefit: f64,
}

// Placeholder implementations for supporting components
pub struct RadixPrefixMatcher {
    // Would contain radix tree for fast prefix matching
}

impl RadixPrefixMatcher {
    pub fn new() -> Self {
        Self {}
    }
}

pub struct MemoryPressureMonitor {
    // Would track memory pressure over time
}

impl MemoryPressureMonitor {
    pub fn new() -> Self {
        Self {}
    }
}

pub struct AccessPatternPredictor {
    // Would predict future access patterns
}

impl AccessPatternPredictor {
    pub fn new() -> Self {
        Self {}
    }
}

pub struct AnalysisHistory {
    // Would track analysis accuracy over time
}

impl AnalysisHistory {
    pub fn new() -> Self {
        Self {}
    }
}