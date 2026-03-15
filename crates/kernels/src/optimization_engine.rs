use crate::hardware_detection::{GpuArchitecture, HardwareInfo};
use crate::types::{GpuDriverError, GpuDriverResult, KernelParameters};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationMetrics {
    pub throughput_ops_per_sec: f64,
    pub latency_ms: f64,
    pub memory_bandwidth_utilization: f64,
    pub compute_utilization: f64,
    pub cache_hit_rate: f64,
    pub power_consumption_watts: f64,
    pub energy_efficiency: f64, // ops per joule
    pub kernel_execution_time_us: f64,
    pub memory_transfer_time_us: f64,
    pub queue_time_us: f64,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct WorkloadCharacteristics {
    pub batch_size: usize,
    pub sequence_length: usize,
    pub model_size: String,  // "7B", "13B", "70B", etc.
    pub attention_type: String, // "multi_head", "grouped_query", "multi_query"
    pub cache_pattern: String, // "sequential", "prefix_sharing", "random"
    pub memory_pressure: MemoryPressureLevel,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum MemoryPressureLevel {
    Low,    // < 50% GPU memory usage
    Medium, // 50-80% GPU memory usage
    High,   // > 80% GPU memory usage
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfiguration {
    // Block and grid configuration
    pub block_size: u32,
    pub grid_size_x: u32,
    pub grid_size_y: u32,
    pub grid_size_z: u32,

    // Memory management
    pub shared_memory_per_block: u32,
    pub memory_coalescing_factor: u32,
    pub cache_line_size: u32,

    // Kernel-specific optimizations
    pub use_tensor_cores: bool,
    pub warp_size: u32,
    pub register_usage_limit: u32,

    // Cache integration
    pub cache_prefetch_distance: u32,
    pub cache_eviction_policy: String,
    pub l1_cache_priority: f32,

    // Performance tuning
    pub kernel_fusion_enabled: bool,
    pub async_memory_transfers: bool,
    pub overlap_computation_memory: bool,

    // Power optimization
    pub target_power_efficiency: f32,
    pub dynamic_frequency_scaling: bool,
}

impl Default for OptimizationConfiguration {
    fn default() -> Self {
        Self {
            block_size: 256,
            grid_size_x: 1,
            grid_size_y: 1,
            grid_size_z: 1,
            shared_memory_per_block: 32768, // 32KB
            memory_coalescing_factor: 4,
            cache_line_size: 128,
            use_tensor_cores: false,
            warp_size: 32,
            register_usage_limit: 255,
            cache_prefetch_distance: 2,
            cache_eviction_policy: "lru".to_string(),
            l1_cache_priority: 0.8,
            kernel_fusion_enabled: true,
            async_memory_transfers: true,
            overlap_computation_memory: true,
            target_power_efficiency: 0.5, // balanced
            dynamic_frequency_scaling: false,
        }
    }
}

#[derive(Debug)]
pub struct PerformanceHistory {
    configurations: Vec<OptimizationConfiguration>,
    metrics: Vec<OptimizationMetrics>,
    timestamps: Vec<Instant>,
    workload_contexts: Vec<WorkloadCharacteristics>,
}

impl PerformanceHistory {
    pub fn new() -> Self {
        Self {
            configurations: Vec::new(),
            metrics: Vec::new(),
            timestamps: Vec::new(),
            workload_contexts: Vec::new(),
        }
    }

    pub fn add_measurement(
        &mut self,
        config: OptimizationConfiguration,
        metrics: OptimizationMetrics,
        workload: WorkloadCharacteristics,
    ) {
        self.configurations.push(config);
        self.metrics.push(metrics);
        self.timestamps.push(Instant::now());
        self.workload_contexts.push(workload);

        // Keep only last 1000 measurements
        if self.configurations.len() > 1000 {
            self.configurations.remove(0);
            self.metrics.remove(0);
            self.timestamps.remove(0);
            self.workload_contexts.remove(0);
        }
    }

    pub fn get_best_configuration(&self, workload: &WorkloadCharacteristics) -> Option<OptimizationConfiguration> {
        let mut best_score = 0.0;
        let mut best_config = None;

        for (i, (config, metrics, context)) in self.configurations.iter()
            .zip(self.metrics.iter())
            .zip(self.workload_contexts.iter())
            .enumerate() {

            // Calculate similarity score to current workload
            let similarity = calculate_workload_similarity(workload, context);
            if similarity < 0.7 { // Only consider similar workloads
                continue;
            }

            // Performance score (higher is better)
            let performance_score = metrics.throughput_ops_per_sec / metrics.latency_ms * metrics.energy_efficiency;
            let weighted_score = performance_score * similarity;

            if weighted_score > best_score {
                best_score = weighted_score;
                best_config = Some(config.clone());
            }
        }

        best_config
    }
}

fn calculate_workload_similarity(a: &WorkloadCharacteristics, b: &WorkloadCharacteristics) -> f64 {
    let batch_similarity = 1.0 - ((a.batch_size as f64 - b.batch_size as f64).abs() / a.batch_size.max(b.batch_size) as f64);
    let seq_similarity = 1.0 - ((a.sequence_length as f64 - b.sequence_length as f64).abs() / a.sequence_length.max(b.sequence_length) as f64);
    let model_similarity = if a.model_size == b.model_size { 1.0 } else { 0.5 };
    let attention_similarity = if a.attention_type == b.attention_type { 1.0 } else { 0.3 };
    let cache_similarity = if a.cache_pattern == b.cache_pattern { 1.0 } else { 0.7 };
    let memory_similarity = if a.memory_pressure == b.memory_pressure { 1.0 } else { 0.8 };

    (batch_similarity + seq_similarity + model_similarity + attention_similarity + cache_similarity + memory_similarity) / 6.0
}

pub struct OptimizationEngine {
    hardware_info: HardwareInfo,
    performance_history: Arc<Mutex<PerformanceHistory>>,
    optimization_cache: Arc<Mutex<HashMap<WorkloadCharacteristics, OptimizationConfiguration>>>,
    learning_rate: f64,
    exploration_rate: f64,
}

impl OptimizationEngine {
    pub fn new(hardware_info: HardwareInfo) -> Self {
        Self {
            hardware_info,
            performance_history: Arc::new(Mutex::new(PerformanceHistory::new())),
            optimization_cache: Arc::new(Mutex::new(HashMap::new())),
            learning_rate: 0.1,
            exploration_rate: 0.1,
        }
    }

    pub fn optimize_for_workload(&self, workload: &WorkloadCharacteristics) -> GpuDriverResult<OptimizationConfiguration> {
        // Check cache first
        {
            let cache = self.optimization_cache.lock().unwrap();
            if let Some(cached_config) = cache.get(workload) {
                return Ok(cached_config.clone());
            }
        }

        // Get base configuration for hardware
        let mut config = self.get_hardware_optimized_base_config()?;

        // Apply workload-specific optimizations
        self.apply_workload_optimizations(&mut config, workload)?;

        // Check performance history for similar workloads
        {
            let history = self.performance_history.lock().unwrap();
            if let Some(historical_config) = history.get_best_configuration(workload) {
                // Blend historical configuration with computed one
                config = self.blend_configurations(&config, &historical_config, 0.7);
            }
        }

        // Apply exploration for learning
        if rand::random::<f64>() < self.exploration_rate {
            self.apply_exploration_mutations(&mut config);
        }

        // Cache the result
        {
            let mut cache = self.optimization_cache.lock().unwrap();
            cache.insert(workload.clone(), config.clone());
        }

        Ok(config)
    }

    fn get_hardware_optimized_base_config(&self) -> GpuDriverResult<OptimizationConfiguration> {
        let mut config = OptimizationConfiguration::default();

        // GPU architecture specific optimizations
        match self.hardware_info.gpu_architecture {
            GpuArchitecture::Ada => {
                config.block_size = 256;
                config.shared_memory_per_block = 49152; // 48KB shared memory
                config.use_tensor_cores = true;
                config.warp_size = 32;
                config.memory_coalescing_factor = 8; // Ada has better coalescing
                config.cache_line_size = 128;
            },
            GpuArchitecture::Ampere => {
                config.block_size = 256;
                config.shared_memory_per_block = 49152; // 48KB shared memory
                config.use_tensor_cores = true;
                config.warp_size = 32;
                config.memory_coalescing_factor = 4;
                config.cache_line_size = 128;
            },
            GpuArchitecture::Turing => {
                config.block_size = 256;
                config.shared_memory_per_block = 32768; // 32KB shared memory
                config.use_tensor_cores = true;
                config.warp_size = 32;
                config.memory_coalescing_factor = 4;
                config.cache_line_size = 128;
            },
            GpuArchitecture::Volta => {
                config.block_size = 256;
                config.shared_memory_per_block = 32768; // 32KB shared memory
                config.use_tensor_cores = true;
                config.warp_size = 32;
                config.memory_coalescing_factor = 4;
                config.cache_line_size = 128;
            },
            GpuArchitecture::RDNA3 | GpuArchitecture::RDNA2 => {
                // AMD GPU optimizations
                config.block_size = 256;
                config.shared_memory_per_block = 32768; // 32KB LDS
                config.use_tensor_cores = false; // AMD uses different approach
                config.warp_size = 64; // AMD wavefront size
                config.memory_coalescing_factor = 4;
                config.cache_line_size = 64; // AMD cache line
            },
            _ => {
                // Conservative defaults for unknown architectures
                config.block_size = 128;
                config.shared_memory_per_block = 16384; // 16KB
                config.use_tensor_cores = false;
                config.warp_size = 32;
                config.memory_coalescing_factor = 2;
                config.cache_line_size = 64;
            }
        }

        // Memory bandwidth optimizations
        if self.hardware_info.memory_bandwidth_gb_s > 800.0 {
            // High bandwidth GPUs (H100, A100)
            config.async_memory_transfers = true;
            config.overlap_computation_memory = true;
            config.cache_prefetch_distance = 4;
        } else if self.hardware_info.memory_bandwidth_gb_s > 500.0 {
            // Medium bandwidth GPUs
            config.async_memory_transfers = true;
            config.overlap_computation_memory = true;
            config.cache_prefetch_distance = 2;
        } else {
            // Lower bandwidth GPUs
            config.async_memory_transfers = false;
            config.overlap_computation_memory = false;
            config.cache_prefetch_distance = 1;
        }

        // Compute capability optimizations
        if self.hardware_info.compute_units > 80 {
            // Large GPUs
            config.grid_size_x = (self.hardware_info.compute_units / 4) as u32;
            config.kernel_fusion_enabled = true;
        } else {
            // Smaller GPUs
            config.grid_size_x = (self.hardware_info.compute_units / 2) as u32;
            config.kernel_fusion_enabled = false;
        }

        Ok(config)
    }

    fn apply_workload_optimizations(&self, config: &mut OptimizationConfiguration, workload: &WorkloadCharacteristics) -> GpuDriverResult<()> {
        // Batch size optimizations
        match workload.batch_size {
            1..=4 => {
                // Small batch optimization
                config.block_size = 128;
                config.grid_size_x = workload.batch_size as u32;
                config.kernel_fusion_enabled = false;
            },
            5..=32 => {
                // Medium batch optimization
                config.block_size = 256;
                config.grid_size_x = (workload.batch_size / 2) as u32;
                config.kernel_fusion_enabled = true;
            },
            _ => {
                // Large batch optimization
                config.block_size = 512;
                config.grid_size_x = (workload.batch_size / 4) as u32;
                config.kernel_fusion_enabled = true;
                config.overlap_computation_memory = true;
            }
        }

        // Sequence length optimizations
        if workload.sequence_length > 2048 {
            // Long sequences need more memory management
            config.cache_prefetch_distance = 8;
            config.l1_cache_priority = 0.9;
            config.async_memory_transfers = true;
        } else if workload.sequence_length > 512 {
            config.cache_prefetch_distance = 4;
            config.l1_cache_priority = 0.8;
        } else {
            config.cache_prefetch_distance = 2;
            config.l1_cache_priority = 0.7;
        }

        // Memory pressure adaptations
        match workload.memory_pressure {
            MemoryPressureLevel::High => {
                config.shared_memory_per_block = config.shared_memory_per_block / 2;
                config.cache_eviction_policy = "aggressive_lru".to_string();
                config.kernel_fusion_enabled = false; // Reduce memory usage
                config.l1_cache_priority = 0.9; // Prioritize L1 cache
            },
            MemoryPressureLevel::Medium => {
                config.cache_eviction_policy = "lru".to_string();
                config.l1_cache_priority = 0.8;
            },
            MemoryPressureLevel::Low => {
                config.cache_eviction_policy = "lazy_lru".to_string();
                config.l1_cache_priority = 0.7;
                config.cache_prefetch_distance *= 2; // More aggressive prefetching
            }
        }

        // Attention type optimizations
        match workload.attention_type.as_str() {
            "grouped_query" => {
                // GQA has fewer KV heads, optimize accordingly
                config.memory_coalescing_factor *= 2;
                config.cache_prefetch_distance *= 2;
            },
            "multi_query" => {
                // MQA has single KV head, very different access pattern
                config.memory_coalescing_factor *= 4;
                config.l1_cache_priority = 0.95;
            },
            _ => {
                // Standard multi-head attention
            }
        }

        // Cache pattern optimizations
        match workload.cache_pattern.as_str() {
            "prefix_sharing" => {
                config.l1_cache_priority = 0.95; // Heavy L1 cache usage
                config.cache_prefetch_distance = 1; // Sequential access
            },
            "random" => {
                config.l1_cache_priority = 0.6; // L1 less effective
                config.cache_prefetch_distance = 8; // Aggressive prefetching
            },
            _ => {
                // Sequential access pattern
                config.cache_prefetch_distance = 4;
            }
        }

        Ok(())
    }

    fn blend_configurations(&self, base: &OptimizationConfiguration, historical: &OptimizationConfiguration, base_weight: f64) -> OptimizationConfiguration {
        let hist_weight = 1.0 - base_weight;

        OptimizationConfiguration {
            block_size: if base_weight > 0.5 { base.block_size } else { historical.block_size },
            grid_size_x: ((base.grid_size_x as f64 * base_weight) + (historical.grid_size_x as f64 * hist_weight)) as u32,
            grid_size_y: if base_weight > 0.5 { base.grid_size_y } else { historical.grid_size_y },
            grid_size_z: if base_weight > 0.5 { base.grid_size_z } else { historical.grid_size_z },
            shared_memory_per_block: ((base.shared_memory_per_block as f64 * base_weight) + (historical.shared_memory_per_block as f64 * hist_weight)) as u32,
            memory_coalescing_factor: if base_weight > 0.5 { base.memory_coalescing_factor } else { historical.memory_coalescing_factor },
            cache_line_size: if base_weight > 0.5 { base.cache_line_size } else { historical.cache_line_size },
            use_tensor_cores: if base_weight > 0.5 { base.use_tensor_cores } else { historical.use_tensor_cores },
            warp_size: if base_weight > 0.5 { base.warp_size } else { historical.warp_size },
            register_usage_limit: ((base.register_usage_limit as f64 * base_weight) + (historical.register_usage_limit as f64 * hist_weight)) as u32,
            cache_prefetch_distance: ((base.cache_prefetch_distance as f64 * base_weight) + (historical.cache_prefetch_distance as f64 * hist_weight)) as u32,
            cache_eviction_policy: if base_weight > 0.5 { base.cache_eviction_policy.clone() } else { historical.cache_eviction_policy.clone() },
            l1_cache_priority: (base.l1_cache_priority * base_weight as f32) + (historical.l1_cache_priority * hist_weight as f32),
            kernel_fusion_enabled: if base_weight > 0.5 { base.kernel_fusion_enabled } else { historical.kernel_fusion_enabled },
            async_memory_transfers: if base_weight > 0.5 { base.async_memory_transfers } else { historical.async_memory_transfers },
            overlap_computation_memory: if base_weight > 0.5 { base.overlap_computation_memory } else { historical.overlap_computation_memory },
            target_power_efficiency: (base.target_power_efficiency * base_weight as f32) + (historical.target_power_efficiency * hist_weight as f32),
            dynamic_frequency_scaling: if base_weight > 0.5 { base.dynamic_frequency_scaling } else { historical.dynamic_frequency_scaling },
        }
    }

    fn apply_exploration_mutations(&self, config: &mut OptimizationConfiguration) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Randomly mutate some parameters for exploration
        if rng.gen_bool(0.3) {
            config.block_size = match config.block_size {
                128 => 256,
                256 => if rng.gen_bool(0.5) { 128 } else { 512 },
                512 => 256,
                _ => 256,
            };
        }

        if rng.gen_bool(0.2) {
            config.cache_prefetch_distance = (config.cache_prefetch_distance as f64 * rng.gen_range(0.5..2.0)) as u32;
            config.cache_prefetch_distance = config.cache_prefetch_distance.max(1).min(16);
        }

        if rng.gen_bool(0.2) {
            config.l1_cache_priority = (config.l1_cache_priority + rng.gen_range(-0.1..0.1)).clamp(0.1, 1.0);
        }

        if rng.gen_bool(0.1) {
            config.kernel_fusion_enabled = !config.kernel_fusion_enabled;
        }
    }

    pub fn record_performance(&self, config: OptimizationConfiguration, metrics: OptimizationMetrics, workload: WorkloadCharacteristics) {
        let mut history = self.performance_history.lock().unwrap();
        history.add_measurement(config, metrics, workload);
    }

    pub fn get_performance_history(&self) -> Arc<Mutex<PerformanceHistory>> {
        Arc::clone(&self.performance_history)
    }

    pub fn update_learning_parameters(&mut self, learning_rate: f64, exploration_rate: f64) {
        self.learning_rate = learning_rate.clamp(0.0, 1.0);
        self.exploration_rate = exploration_rate.clamp(0.0, 1.0);
    }

    pub fn clear_cache(&self) {
        let mut cache = self.optimization_cache.lock().unwrap();
        cache.clear();
    }
}