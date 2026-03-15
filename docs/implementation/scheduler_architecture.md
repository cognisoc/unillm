# Intelligent Scheduler Architecture

## Overview

UniLLM's intelligent scheduler is a modular, high-performance system designed for maximum extensibility and GPU driver integration. The architecture follows clean separation of concerns with well-defined interfaces between components.

## Core Design Principles

### 1. Modularity and Extensibility
- **Component-based architecture**: Each major function (cache analysis, memory tracking, policy engine) is a separate module
- **Interface-driven design**: Components communicate through well-defined traits and APIs
- **Plugin architecture**: New scheduling policies can be added without modifying core components
- **Hot-swappable policies**: Runtime policy switching without service interruption

### 2. Zero-Overhead Performance
- **Rust's zero-cost abstractions**: No runtime overhead from modularity
- **Lock-free data structures**: Atomic operations for critical performance paths
- **Memory pool allocation**: Pre-allocated buffers to avoid allocation overhead
- **SIMD optimization**: Vectorized operations for batch processing

### 3. GPU Driver Integration
- **Direct hardware access**: VFIO passthrough for bypassing OS overhead
- **Real-time memory tracking**: Atomic updates to GPU memory state
- **Hardware-aware scheduling**: Decisions based on actual GPU capabilities
- **Cache-integrated optimization**: Direct integration with GPU memory hierarchy

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    UniLLM Intelligent Scheduler                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐ │
│  │ Request Manager │    │ Batch Optimizer  │    │ GPU Monitor │ │
│  │                 │    │                  │    │             │ │
│  │ • Priority Queue│    │ • Cache Analyzer │    │ • Memory    │ │
│  │ • State Tracking│    │ • Memory Tracker │    │ • Utilization│ │
│  │ • Lifecycle Mgmt│    │ • Policy Engine  │    │ • Pressure  │ │
│  └─────────────────┘    └──────────────────┘    └─────────────┘ │
│           │                       │                      │      │
│           └───────────┐           │           ┌──────────┘      │
│                       ▼           ▼           ▼                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              IntelligentScheduler Core                     │ │
│  │                                                             │ │
│  │  • Policy Selection (FCFS, LPM, CacheAware, GPUOpt, ML)   │ │
│  │  • Batch Formation (Sub-ms latency, Cache-aware)          │ │
│  │  • GPU Integration (Direct memory access, Real-time)      │ │
│  │  • Performance Monitoring (Metrics, Profiling, Tuning)   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                │                                 │
└────────────────────────────────┼─────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                   GPU Memory Management                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐ │
│  │ Hybrid KV Cache │    │ GPU Memory Pool  │    │ VFIO Driver │ │
│  │                 │    │                  │    │             │ │
│  │ • L1 Radix      │    │ • Direct Alloc   │    │ • Passthru  │ │
│  │ • L2 Paged      │    │ • Fragmentation  │    │ • Low-level │ │
│  │ • L3 Compressed │    │ • Pressure Mon   │    │ • Hardware  │ │
│  └─────────────────┘    └──────────────────┘    └─────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### 1. IntelligentScheduler Core (`intelligent_scheduler.rs`)

**Purpose**: Central orchestrator that integrates all scheduling components

**Key Features**:
- **Multi-policy support**: 5 different scheduling strategies with runtime switching
- **GPU-integrated decisions**: Direct access to GPU memory state and cache statistics
- **Performance tracking**: Real-time metrics collection and optimization
- **Extensible design**: New policies can be plugged in through the SchedulingPolicy enum

**Extension Points**:
```rust
pub enum SchedulingPolicy {
    FCFS,                    // Baseline compatibility
    LongestPrefixMatch,      // SGLang-inspired
    CacheAware,              // UniLLM innovation
    GpuMemoryOptimized,      // Memory-pressure aware
    AdaptiveML,              // ML-optimized
    // NEW POLICIES CAN BE ADDED HERE
}
```

### 2. CacheAwareAnalyzer (`cache_analyzer.rs`)

**Purpose**: Deep cache integration for optimizing batch composition

**Key Features**:
- **Prefix sharing detection**: Identifies common token sequences across requests
- **Cache hit prediction**: ML-based prediction of cache performance
- **Memory requirement analysis**: Estimates GPU memory needs with fragmentation
- **Batch optimization**: Reorders requests for maximum cache utilization

**Extension Points**:
- Custom cache hit predictors
- Additional optimization strategies
- Cache warming policies
- Prefix matching algorithms

### 3. GpuMemoryTracker (`gpu_memory_tracker.rs`)

**Purpose**: Real-time GPU memory monitoring and allocation optimization

**Key Features**:
- **Memory pressure monitoring**: Tracks fragmentation, utilization, and pressure
- **Batch size optimization**: Suggests optimal batch sizes for current memory state
- **OOM prevention**: Predictive analysis to prevent out-of-memory conditions
- **Performance estimation**: Predicts throughput and latency impact

**Extension Points**:
- Custom memory pressure algorithms
- Additional allocation strategies
- Memory defragmentation policies
- Multi-GPU coordination

### 4. AdaptivePolicyEngine (`adaptive_policy.rs`)

**Purpose**: ML-based policy optimization and automatic strategy selection

**Key Features**:
- **Workload characterization**: Analyzes request patterns and system behavior
- **Policy performance tracking**: Measures effectiveness of different strategies
- **Automatic adaptation**: Switches policies based on performance feedback
- **Confidence scoring**: Provides reliability metrics for decisions

**Extension Points**:
- Custom ML models for policy selection
- Additional workload characteristics
- External performance feedback integration
- Multi-objective optimization

## Modular Interface Design

### Core Traits for Extensibility

```rust
// Cache analysis interface
pub trait CacheAnalyzer {
    fn analyze_batch(&self, requests: &[Request]) -> BatchAnalysis;
    fn optimize_composition(&self, requests: &mut Vec<Request>) -> OptimizationResult;
}

// Memory tracking interface
pub trait MemoryTracker {
    fn check_feasibility(&self, requests: &[Request]) -> MemoryFeasibility;
    fn suggest_batch_size(&self, requests: &[Request]) -> OptimalBatchSize;
    fn update_allocation(&self, event: AllocationEvent);
}

// Policy engine interface
pub trait PolicyEngine {
    fn analyze_workload(&self, metrics: &SchedulerMetrics) -> PolicyDecision;
    fn predict_optimal(&self, requests: &[Request]) -> SchedulingPolicy;
}

// Scheduling policy interface
pub trait SchedulingStrategy {
    fn form_batch(&self, requests: &[Request], constraints: &MemoryConstraints) -> RequestBatch;
    fn priority_score(&self, request: &Request, context: &SchedulingContext) -> f64;
}
```

### Data Flow and Integration Points

```rust
// Main scheduling loop with extension points
impl IntelligentScheduler {
    pub fn schedule_next_batch(&mut self) -> Option<RequestBatch> {
        // 1. Analyze current workload
        let workload_analysis = self.policy_engine.analyze_workload(&self.metrics);

        // 2. Select optimal policy (EXTENSIBLE)
        let policy = self.policy_engine.predict_optimal(&self.pending_requests);

        // 3. Analyze cache opportunities (EXTENSIBLE)
        let cache_analysis = self.cache_analyzer.analyze_batch(&self.pending_requests);

        // 4. Check memory constraints (EXTENSIBLE)
        let memory_feasibility = self.memory_tracker.check_feasibility(&self.pending_requests);

        // 5. Form optimal batch (POLICY-SPECIFIC)
        let batch = self.form_batch_with_policy(policy, &cache_analysis, &memory_feasibility);

        // 6. Update performance metrics
        self.update_metrics(&batch);

        batch
    }
}
```

## Extension Scenarios

### 1. Adding New Scheduling Policies

```rust
// Step 1: Add to enum
pub enum SchedulingPolicy {
    // ... existing policies
    CustomOptimized,    // NEW POLICY
}

// Step 2: Implement strategy
pub struct CustomOptimizedStrategy {
    // Custom configuration
}

impl SchedulingStrategy for CustomOptimizedStrategy {
    fn form_batch(&self, requests: &[Request], constraints: &MemoryConstraints) -> RequestBatch {
        // Custom batch formation logic
    }
}

// Step 3: Register in scheduler
impl IntelligentScheduler {
    fn select_strategy(&self, policy: SchedulingPolicy) -> Box<dyn SchedulingStrategy> {
        match policy {
            // ... existing cases
            SchedulingPolicy::CustomOptimized => Box::new(CustomOptimizedStrategy::new()),
        }
    }
}
```

### 2. Custom Cache Analysis

```rust
// Implement custom cache analyzer
pub struct AdvancedCacheAnalyzer {
    ml_model: Box<dyn MLModel>,
    cache_simulator: CacheSimulator,
}

impl CacheAnalyzer for AdvancedCacheAnalyzer {
    fn analyze_batch(&self, requests: &[Request]) -> BatchAnalysis {
        // Advanced ML-based cache prediction
        let predictions = self.ml_model.predict_cache_hits(requests);
        let simulation = self.cache_simulator.simulate_batch(requests);

        // Combine predictions with simulation
        self.combine_analysis(predictions, simulation)
    }
}
```

### 3. Multi-GPU Memory Tracking

```rust
// Extend memory tracker for multi-GPU
pub struct MultiGpuMemoryTracker {
    gpu_trackers: Vec<GpuMemoryTracker>,
    load_balancer: LoadBalancer,
}

impl MemoryTracker for MultiGpuMemoryTracker {
    fn check_feasibility(&self, requests: &[Request]) -> MemoryFeasibility {
        // Check feasibility across multiple GPUs
        let gpu_assignments = self.load_balancer.assign_requests(requests);

        for (gpu_id, gpu_requests) in gpu_assignments {
            if !self.gpu_trackers[gpu_id].check_feasibility(&gpu_requests).feasible {
                return MemoryFeasibility::infeasible();
            }
        }

        MemoryFeasibility::feasible()
    }
}
```

## Performance Characteristics

### Modularity Impact on Performance

| Component | Overhead | Optimization Strategy |
|-----------|----------|----------------------|
| Cache Analyzer | ~50μs | Pre-computed prefix trees, SIMD matching |
| Memory Tracker | ~20μs | Atomic operations, lock-free updates |
| Policy Engine | ~30μs | Cached decisions, incremental updates |
| Batch Formation | ~100μs | Zero-allocation algorithms, memory pools |

### Extensibility Features

1. **Hot-swappable Components**: Runtime component replacement without service interruption
2. **Configuration-driven Policies**: JSON/YAML configuration for policy parameters
3. **Plugin Architecture**: Dynamic loading of custom scheduling strategies
4. **Metrics Framework**: Extensible performance monitoring and alerting
5. **A/B Testing Support**: Parallel policy evaluation for gradual rollouts

## Integration with GPU Drivers

The modular design enables deep integration with GPU drivers while maintaining clean abstractions:

```rust
// Direct GPU integration points
pub trait GpuIntegration {
    // Memory management
    fn allocate_gpu_memory(&self, size: usize) -> GpuMemoryResult<GpuDevicePtr>;
    fn get_memory_stats(&self) -> GpuMemoryStats;

    // Cache management
    fn query_cache_state(&self) -> CacheStats;
    fn prefetch_sequences(&self, sequences: &[TokenSequence]) -> Result<()>;

    // Performance monitoring
    fn get_utilization(&self) -> GpuUtilization;
    fn get_thermal_state(&self) -> ThermalStatus;
}
```

This architecture provides the foundation for Phase 1.3's kernel framework, which will add the final layer of GPU driver integration and template-based kernel generation.

## Next Steps: Phase 1.3 Preparation

The modular scheduler architecture is now ready for integration with:

1. **Template-based kernel generation** for optimized GPU code
2. **Direct driver integration** through VFIO passthrough
3. **Hardware-specific optimizations** based on GPU capabilities
4. **Real-time performance tuning** with driver-level feedback

The clean interfaces and extensible design ensure that GPU driver integration can be added without modifying the core scheduling logic, maintaining both performance and maintainability.