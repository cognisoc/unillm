# UniLLM Benchmarking Framework

## Overview

This directory contains our benchmarking infrastructure for measuring UniLLM's performance against vLLM and SGLang. We maintain comprehensive performance tracking throughout development to ensure we meet our competitive targets.

## Benchmarking Strategy

### Core Metrics

**Latency Metrics:**
- Time to First Token (TTFT)
- Time Per Output Token (TPOT)
- End-to-end request latency
- P50, P90, P95, P99 latency distributions

**Throughput Metrics:**
- Requests per second
- Tokens per second (input + output)
- Concurrent request handling capacity
- Batch processing efficiency

**Memory Metrics:**
- Peak memory usage
- Memory allocation efficiency
- Cache hit rates
- Memory bandwidth utilization

**Scaling Metrics:**
- Multi-GPU scaling efficiency
- Load balancing effectiveness
- Communication overhead
- NUMA performance characteristics

### Test Scenarios

**1. Single-Stream Latency**
- Measure individual request latency
- Various sequence lengths (128, 512, 2048, 8192 tokens)
- Different model sizes (7B, 13B, 70B parameters)
- Multiple data types (FP16, BF16, FP8, INT8)

**2. High-Throughput Batching**
- Concurrent request processing
- Batch sizes: 1, 8, 16, 32, 64, 128
- Mixed sequence lengths
- Continuous batching performance

**3. Memory Efficiency**
- KV cache memory usage
- Prefix sharing effectiveness
- Memory fragmentation analysis
- Long context handling (32K+ tokens)

**4. Multi-GPU Scaling**
- Tensor parallelism efficiency
- Pipeline parallelism performance
- Expert parallelism (for MoE models)
- Communication pattern analysis

### Competitive Baselines

**vLLM Comparison:**
- Same hardware configuration
- Same model and quantization settings
- Same sequence length distributions
- Same batch sizes and concurrency levels

**SGLang Comparison:**
- Focus on prefix caching scenarios
- Structured output generation
- Complex batching patterns
- Cache-intensive workloads

## Performance Targets

### Phase 1 Targets (Foundation)
| Metric | Target vs vLLM | Target vs SGLang | Measurement Method |
|--------|---------------|------------------|-------------------|
| **TTFT** | +20-30% | +15-25% | Single request latency |
| **TPOT** | +10-20% | +10-15% | Token generation speed |
| **Memory Usage** | +30-40% | +20-30% | Peak allocation tracking |
| **Cache Hit Rate** | +25-35% | +15-20% | Cache performance monitoring |

### Phase 2 Targets (Advanced Features)
| Metric | Target vs vLLM | Target vs SGLang | Measurement Method |
|--------|---------------|------------------|-------------------|
| **Batch Throughput** | +15-25% | +10-20% | Requests per second |
| **Memory Efficiency** | +40-50% | +30-40% | Memory per token ratio |
| **Scheduling Latency** | +50-70% | +30-50% | Batch formation time |

### Phase 3 Targets (Enterprise Scale)
| Metric | Target vs vLLM | Target vs SGLang | Measurement Method |
|--------|---------------|------------------|-------------------|
| **Multi-GPU Efficiency** | 95%+ | 95%+ | Scaling factor measurement |
| **Load Balancing** | +10-20% | +10-15% | GPU utilization variance |
| **Fault Recovery** | <100ms | <100ms | Recovery time measurement |

## Benchmarking Infrastructure

### Hardware Configurations

**Single GPU:**
- NVIDIA H100 (80GB)
- NVIDIA A100 (80GB)
- AMD MI300X (192GB)
- AMD MI250X (128GB)

**Multi-GPU:**
- 2x, 4x, 8x GPU configurations
- NVLink/Infinity Fabric connectivity
- PCIe Gen5 configurations

**System Specifications:**
- Intel Xeon or AMD EPYC processors
- DDR5 memory (512GB-2TB)
- NVMe storage (10TB+)
- 100Gbps networking

### Software Stack

**Baseline Implementations:**
- vLLM v0.6.x (latest stable)
- SGLang v0.4.x (latest stable)
- PyTorch 2.5+ with CUDA 12.4+
- ROCm 6.x for AMD testing

**Testing Framework:**
```rust
// Benchmarking framework structure
pub struct BenchmarkSuite {
    pub latency_tests: LatencyBenchmarks,
    pub throughput_tests: ThroughputBenchmarks,
    pub memory_tests: MemoryBenchmarks,
    pub scaling_tests: ScalingBenchmarks,
}

pub trait BenchmarkTest {
    fn setup(&mut self) -> Result<()>;
    fn run(&mut self) -> BenchmarkResults;
    fn teardown(&mut self) -> Result<()>;
}

pub struct BenchmarkResults {
    pub metrics: HashMap<String, f64>,
    pub latency_distribution: Vec<f64>,
    pub throughput_timeline: Vec<(f64, f64)>,
    pub memory_usage: MemoryProfile,
    pub system_metrics: SystemMetrics,
}
```

### Test Automation

**Continuous Benchmarking:**
- Automated nightly performance regression tests
- Performance tracking across git commits
- Automated comparison with baseline implementations
- Performance alert system for regressions

**Test Orchestration:**
```yaml
# benchmark_config.yaml
tests:
  - name: single_stream_latency
    models: [llama-7b, llama-13b, llama-70b]
    sequence_lengths: [128, 512, 2048, 8192]
    batch_sizes: [1]
    iterations: 100

  - name: throughput_scaling
    models: [llama-7b]
    sequence_lengths: [512]
    batch_sizes: [1, 8, 16, 32, 64]
    concurrent_requests: [1, 10, 50, 100]
    iterations: 50
```

## Results Documentation

### Directory Structure
```
benchmarks/
├── README.md                    # This file
├── baseline/                    # Baseline performance results
│   ├── vllm_results.md         # vLLM benchmark results
│   ├── sglang_results.md       # SGLang benchmark results
│   └── hardware_profiles.md    # Hardware configuration details
├── phase1/                      # Phase 1 implementation results
│   ├── memory_management.md    # KV cache performance
│   ├── scheduling.md           # Scheduler performance
│   └── kernels.md             # Kernel optimization results
├── phase2/                      # Phase 2 implementation results
├── phase3/                      # Phase 3 implementation results
├── phase4/                      # Phase 4 implementation results
├── competitive/                 # Head-to-head comparisons
│   ├── vs_vllm.md              # Detailed vLLM comparison
│   ├── vs_sglang.md            # Detailed SGLang comparison
│   └── summary.md              # Executive summary
└── scripts/                     # Benchmarking automation
    ├── run_benchmarks.py       # Main benchmark runner
    ├── compare_results.py      # Results comparison tool
    └── generate_reports.py     # Automated report generation
```

### Reporting Standards

Each benchmark result document includes:

1. **Test Configuration**: Hardware, software, model details
2. **Methodology**: How the test was conducted
3. **Raw Results**: Complete performance data
4. **Analysis**: Key insights and observations
5. **Comparison**: Performance vs baseline implementations
6. **Conclusions**: Success against targets, areas for improvement

### Performance Tracking

**Regression Detection:**
- Automated alerts for >5% performance degradation
- Git bisect automation for regression identification
- Performance trend analysis and prediction

**Progress Monitoring:**
- Weekly performance summary reports
- Monthly competitive analysis updates
- Quarterly roadmap progress reviews

## Getting Started

### Running Benchmarks

```bash
# Run basic performance suite
cd benchmarks/scripts
python run_benchmarks.py --suite basic --baseline vllm

# Compare with previous results
python compare_results.py --current phase1/latest --baseline baseline/vllm

# Generate performance report
python generate_reports.py --phase phase1 --format markdown
```

### Adding New Tests

1. Create test configuration in `benchmark_config.yaml`
2. Implement test logic following `BenchmarkTest` trait
3. Add result documentation template
4. Update automation scripts if needed
5. Run validation tests before committing

This benchmarking framework ensures we maintain competitive performance throughout development while providing clear evidence of our achievements against established baselines.