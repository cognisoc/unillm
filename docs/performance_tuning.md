# ⚡ UniLLM Performance Tuning Guide

Comprehensive guide to optimizing UniLLM performance for maximum throughput, minimum latency, and efficient resource utilization.

## 📋 Table of Contents

1. [Performance Overview](#performance-overview)
2. [Throughput Optimization](#throughput-optimization)
3. [Latency Optimization](#latency-optimization)
4. [Memory Optimization](#memory-optimization)
5. [Cache Optimization](#cache-optimization)
6. [GPU Optimization](#gpu-optimization)
7. [Unikernel Performance](#unikernel-performance)
8. [Monitoring and Profiling](#monitoring-and-profiling)
9. [Advanced Tuning](#advanced-tuning)

## 🎯 Performance Overview

UniLLM provides multiple knobs for performance tuning, optimized for different workload characteristics:

### Performance Modes

| Mode | Use Case | Batch Size | Memory | Latency | Throughput |
|------|----------|------------|---------|---------|------------|
| **Latency-Optimized** | Real-time chat | 1-4 | High | **Minimal** | Low |
| **Balanced** | General purpose | 16-32 | Medium | Medium | **Medium** |
| **Throughput-Optimized** | Batch processing | 64-128 | Low | High | **Maximum** |

### Key Performance Metrics

```bash
# Check current performance
curl http://localhost:8080/stats | jq '{
  throughput_rps: .performance.throughput_rps,
  latency_p50: .performance.p50_latency_ms,
  latency_p95: .performance.p95_latency_ms,
  cache_hit_rate: .cache_stats.hit_rate,
  gpu_utilization: .gpu_stats.utilization
}'
```

## 🚀 Throughput Optimization

Maximize requests per second for batch processing workloads.

### Configuration

```bash
# Enable throughput mode
export UNILLM_THROUGHPUT_OPTIMIZED=true
export UNILLM_OPTIMAL_BATCH_SIZE=128
export UNILLM_BATCH_TIMEOUT_MS=100
export UNILLM_ENABLE_KERNEL_FUSION=true
export UNILLM_PREFETCH_ENABLED=false  # Reduce memory pressure
```

### Batch Size Tuning

```python
# Automatic batch size optimization
def optimize_batch_size(gpu_memory_gb, target_throughput_rps):
    """Find optimal batch size for maximum throughput."""

    # Start with GPU-specific baseline
    baseline_batch_sizes = {
        "rtx4090": 32,
        "h100": 128,
        "mi300x": 128,
        "a100": 96
    }

    gpu_type = detect_gpu_type()
    base_batch_size = baseline_batch_sizes.get(gpu_type, 32)

    # Scale based on available memory
    memory_factor = gpu_memory_gb / 24  # Normalize to RTX 4090
    optimal_batch_size = int(base_batch_size * memory_factor)

    # Benchmark different batch sizes
    batch_sizes = [optimal_batch_size // 2, optimal_batch_size, optimal_batch_size * 2]

    best_throughput = 0
    best_batch_size = optimal_batch_size

    for batch_size in batch_sizes:
        throughput = benchmark_throughput(batch_size)
        if throughput > best_throughput:
            best_throughput = throughput
            best_batch_size = batch_size

    return best_batch_size
```

### Memory Pool Optimization

```bash
# Pre-allocate GPU memory pools
export UNILLM_ENABLE_MEMORY_POOL=true
export UNILLM_MEMORY_POOL_SIZE_GB=20
export UNILLM_MEMORY_POOL_WARMUP=true

# Reduce fragmentation
export UNILLM_MEMORY_ALIGNMENT=512
export UNILLM_ENABLE_MEMORY_DEFRAG=true
```

### Kernel Optimization

```bash
# Enable aggressive kernel fusion
export UNILLM_ENABLE_KERNEL_FUSION=true
export UNILLM_FUSION_THRESHOLD=0.1
export UNILLM_MAX_FUSED_KERNELS=8

# Use optimized CUDA graphs (H100, A100)
export UNILLM_ENABLE_CUDA_GRAPHS=true
export UNILLM_CUDA_GRAPH_WARMUP=10
```

### Multi-GPU Scaling

```bash
# Tensor parallelism
export UNILLM_TENSOR_PARALLEL_SIZE=4
export UNILLM_PIPELINE_PARALLEL_SIZE=1

# Load balancing
export UNILLM_GPU_LOAD_BALANCE_STRATEGY=dynamic
export UNILLM_ENABLE_GPU_AFFINITY=true
```

### Expected Results

**RTX 4090 Throughput Optimization:**
```
Baseline:     245 tokens/sec
Optimized:    312 tokens/sec (+27%)
Peak:         340 tokens/sec (+39%)
```

**H100 Throughput Optimization:**
```
Baseline:     1,160 tokens/sec
Optimized:    1,580 tokens/sec (+36%)
Peak:         1,750 tokens/sec (+51%)
```

## ⚡ Latency Optimization

Minimize time-to-first-token and end-to-end latency.

### Configuration

```bash
# Enable latency mode
export UNILLM_LATENCY_OPTIMIZED=true
export UNILLM_OPTIMAL_BATCH_SIZE=4
export UNILLM_BATCH_TIMEOUT_MS=1
export UNILLM_PREFETCH_ENABLED=true
export UNILLM_ENABLE_ADAPTIVE_BATCHING=true
```

### Cache Optimization for Latency

```bash
# Optimize for cache hits
export UNILLM_L1_CACHE_SIZE_MB=2048
export UNILLM_CACHE_POLICY=aggressive
export UNILLM_PREFETCH_DEPTH=3
export UNILLM_ENABLE_SPECULATIVE_DECODING=true
```

### Memory Optimization

```bash
# Minimize memory allocation overhead
export UNILLM_PREALLOCATE_MEMORY=true
export UNILLM_MEMORY_POOL_WARMUP=true
export UNILLM_ENABLE_ZERO_COPY=true
```

### Scheduling Optimization

```bash
# Prioritize low-latency requests
export UNILLM_SCHEDULER_POLICY=shortest_job_first
export UNILLM_ENABLE_PREEMPTION=true
export UNILLM_LATENCY_SLA_MS=100
```

### Kernel Optimization

```bash
# Use fastest kernels
export UNILLM_KERNEL_SELECTION=fastest
export UNILLM_ENABLE_FLASH_ATTENTION=true
export UNILLM_ATTENTION_BACKEND=fastest
```

### Expected Results

**RTX 4090 Latency Optimization:**
```
Baseline P50:     180ms
Optimized P50:    95ms (-47%)
Baseline P95:     320ms
Optimized P95:    160ms (-50%)
```

**H100 Latency Optimization:**
```
Baseline P50:     120ms
Optimized P50:    55ms (-54%)
Baseline P95:     240ms
Optimized P95:    110ms (-54%)
```

## 💾 Memory Optimization

Optimize memory usage for large models and constrained environments.

### Memory Hierarchy Tuning

```bash
# L1 Cache (GPU Memory - Radix Tree)
export UNILLM_L1_CACHE_SIZE_MB=1024
export UNILLM_L1_CACHE_POLICY=lru
export UNILLM_RADIX_TREE_DEPTH=8

# L2 Cache (GPU Memory - Paged)
export UNILLM_L2_CACHE_SIZE_MB=2048
export UNILLM_L2_BLOCK_SIZE=256
export UNILLM_L2_CACHE_POLICY=adaptive

# L3 Cache (System Memory - Compressed)
export UNILLM_L3_CACHE_SIZE_MB=8192
export UNILLM_L3_COMPRESSION_RATIO=0.6
export UNILLM_L3_COMPRESSION_ALGORITHM=lz4
```

### Memory Pool Configuration

```bash
# GPU Memory Pools
export UNILLM_GPU_MEMORY_FRACTION=0.85
export UNILLM_GPU_MEMORY_POOL_SIZE_MB=18432  # 18GB for RTX 4090
export UNILLM_ENABLE_MEMORY_DEFRAG=true

# System Memory Pools
export UNILLM_SYSTEM_MEMORY_POOL_SIZE_GB=8
export UNILLM_ENABLE_HUGE_PAGES=true
export UNILLM_HUGE_PAGE_SIZE=2M
```

### Model Loading Optimization

```bash
# Lazy loading
export UNILLM_LAZY_MODEL_LOADING=true
export UNILLM_MODEL_SHARDING=true
export UNILLM_SHARD_SIZE_MB=512

# Memory mapping
export UNILLM_USE_MEMORY_MAPPING=true
export UNILLM_MMAP_POPULATE=true
```

### Memory Monitoring

```python
def monitor_memory_usage():
    """Monitor and optimize memory usage in real-time."""

    while True:
        stats = get_unillm_stats()

        gpu_usage = stats['gpu_stats']['memory_usage_mb']
        gpu_total = stats['gpu_stats']['memory_total_mb']
        gpu_utilization = gpu_usage / gpu_total

        system_usage = stats['memory_stats']['used_memory_mb']
        system_total = stats['memory_stats']['total_memory_mb']
        system_utilization = system_usage / system_total

        # Alert if memory usage is high
        if gpu_utilization > 0.9:
            print(f"⚠️ High GPU memory usage: {gpu_utilization:.1%}")
            # Trigger cache eviction
            evict_gpu_cache()

        if system_utilization > 0.8:
            print(f"⚠️ High system memory usage: {system_utilization:.1%}")
            # Reduce L3 cache size
            resize_l3_cache(0.8)

        time.sleep(30)
```

## 🎯 Cache Optimization

Optimize the hybrid cache system for maximum hit rates.

### Cache Configuration

```bash
# Hybrid Cache Tuning
export UNILLM_CACHE_STRATEGY=hybrid
export UNILLM_ENABLE_ADAPTIVE_CACHE=true
export UNILLM_CACHE_HIT_THRESHOLD=0.75

# Radix Cache (L1)
export UNILLM_RADIX_CACHE_SIZE_MB=1024
export UNILLM_RADIX_MIN_PREFIX_LENGTH=4
export UNILLM_RADIX_MAX_DEPTH=16

# Paged Cache (L2)
export UNILLM_PAGED_CACHE_SIZE_MB=2048
export UNILLM_PAGE_SIZE_TOKENS=256
export UNILLM_ENABLE_PAGE_SWAPPING=true

# Compressed Cache (L3)
export UNILLM_COMPRESSED_CACHE_SIZE_MB=8192
export UNILLM_COMPRESSION_LEVEL=6
export UNILLM_COMPRESSION_THREADS=4
```

### Cache Warming

```python
def warm_cache_with_common_patterns():
    """Pre-populate cache with common request patterns."""

    common_prefixes = [
        "Translate the following text to",
        "Summarize the following article:",
        "Answer the following question:",
        "Generate code for the following task:",
        "Explain the concept of",
        "What is the difference between"
    ]

    for prefix in common_prefixes:
        # Generate variations
        for i in range(10):
            prompt = f"{prefix} example {i}"

            # Send warm-up request
            response = requests.post("http://localhost:8080/v1/generate", json={
                "prompt": prompt,
                "max_tokens": 1,  # Minimal generation
                "cache_policy": "force"
            })

            print(f"Warmed cache for: {prefix}")

    print("Cache warming complete")
```

### Cache Analytics

```bash
# Monitor cache performance
watch -n 5 'curl -s http://localhost:8080/stats | jq "
{
  l1_hit_rate: .cache_stats.l1_hit_rate,
  l2_hit_rate: .cache_stats.l2_hit_rate,
  l3_hit_rate: .cache_stats.l3_hit_rate,
  overall_hit_rate: .cache_stats.hit_rate,
  eviction_rate: .cache_stats.eviction_rate
}"'
```

### Cache Optimization Strategies

```python
def optimize_cache_sizes(workload_profile):
    """Optimize cache sizes based on workload characteristics."""

    if workload_profile == "chat":
        # Chat workloads benefit from large L1 cache
        return {
            "l1_size_mb": 2048,
            "l2_size_mb": 1024,
            "l3_size_mb": 4096
        }
    elif workload_profile == "translation":
        # Translation has common prefixes
        return {
            "l1_size_mb": 1536,
            "l2_size_mb": 2048,
            "l3_size_mb": 6144
        }
    elif workload_profile == "code_generation":
        # Code generation has diverse patterns
        return {
            "l1_size_mb": 1024,
            "l2_size_mb": 2048,
            "l3_size_mb": 8192
        }
    else:
        # Balanced configuration
        return {
            "l1_size_mb": 1024,
            "l2_size_mb": 2048,
            "l3_size_mb": 6144
        }
```

## 🎮 GPU Optimization

Hardware-specific optimizations for maximum GPU utilization.

### NVIDIA Optimization

```bash
# RTX 4090 Specific
export UNILLM_CUDA_ARCH=8.9
export UNILLM_ENABLE_TENSOR_CORES=true
export UNILLM_TENSOR_CORE_PRECISION=mixed
export UNILLM_SM_COUNT=128

# H100 Specific
export UNILLM_CUDA_ARCH=9.0
export UNILLM_ENABLE_FP8=true
export UNILLM_ENABLE_TRANSFORMER_ENGINE=true
export UNILLM_MIG_ENABLED=false  # For full GPU utilization

# CUDA Driver Optimization
export UNILLM_USE_CUDA_DRIVER=true
export UNILLM_CUDA_LAUNCH_BLOCKING=0
export UNILLM_CUDA_DEVICE_ORDER=FASTEST_FIRST
```

### AMD Optimization

```bash
# MI300X Specific
export UNILLM_HIP_TARGETS=gfx942
export UNILLM_ENABLE_MATRIX_CORES=true
export UNILLM_ROCM_VERSION=6.5
export UNILLM_CU_COUNT=304

# ROCm Driver Optimization
export UNILLM_USE_HIP_DRIVER=true
export HIP_LAUNCH_BLOCKING=0
export HSA_ENABLE_SDMA=1
export ROCR_VISIBLE_DEVICES=0
```

### Multi-GPU Optimization

```bash
# Tensor Parallelism
export UNILLM_TENSOR_PARALLEL_SIZE=4
export UNILLM_PIPELINE_PARALLEL_SIZE=1
export UNILLM_SEQUENCE_PARALLEL=true

# Communication Optimization
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # If using Ethernet
export NCCL_SOCKET_IFNAME=eth0

# Load Balancing
export UNILLM_GPU_LOAD_BALANCE=dynamic
export UNILLM_WORKLOAD_SPLITTING=adaptive
```

### Kernel Optimization

```python
def generate_optimized_kernels(gpu_arch):
    """Generate GPU-specific optimized kernels."""

    templates = {
        "attention": "attention_kernel.cu.j2",
        "matmul": "matmul_kernel.cu.j2",
        "softmax": "softmax_kernel.cu.j2"
    }

    optimization_params = {
        "rtx4090": {
            "block_size": 256,
            "grid_size": 128,
            "shared_memory_kb": 48,
            "registers_per_thread": 64
        },
        "h100": {
            "block_size": 512,
            "grid_size": 256,
            "shared_memory_kb": 227,
            "registers_per_thread": 128
        },
        "mi300x": {
            "block_size": 64,
            "grid_size": 304,
            "shared_memory_kb": 32,
            "registers_per_thread": 32
        }
    }

    params = optimization_params.get(gpu_arch, optimization_params["rtx4090"])

    for kernel_name, template_path in templates.items():
        generate_kernel(template_path, params, f"{kernel_name}_{gpu_arch}.cu")
```

## 🔥 Unikernel Performance

Maximize the performance advantages of unikernel deployment.

### Nanos Optimization

```bash
# Nanos-specific configuration
export NANOS_GPU_DIRECT_ACCESS=true
export NANOS_MEMORY_POOL_SIZE=32G
export NANOS_CPU_AFFINITY=0-7
export NANOS_IRQ_AFFINITY=8-15

# Boot optimization
export NANOS_FAST_BOOT=true
export NANOS_PRELOAD_LIBRARIES=true
export NANOS_DISABLE_SERVICES="ssh,cron,systemd"
```

### Memory Efficiency

```bash
# Unikernel memory optimization
export UNILLM_UNIKERNEL_MEMORY_EFFICIENT=true
export UNILLM_DISABLE_SWAP=true
export UNILLM_MEMORY_OVERCOMMIT=false

# Direct hardware access
export UNILLM_DIRECT_GPU_ACCESS=true
export UNILLM_BYPASS_DRIVER_OVERHEAD=true
```

### Networking Optimization

```bash
# High-performance networking
export UNILLM_NETWORK_MODE=kernel_bypass
export UNILLM_USE_DPDK=true
export UNILLM_NETWORK_THREADS=4

# Connection pooling
export UNILLM_CONNECTION_POOL_SIZE=1000
export UNILLM_KEEP_ALIVE_TIMEOUT=60
```

### Performance Comparison

**Memory Usage (RTX 4090):**
```
Container Mode:  1.9GB
Unikernel Mode:  0.8GB  (-58%)
Memory per req:  15.6MB vs 6.4MB
```

**Cold Start Performance:**
```
Container Boot:  2.1s
Unikernel Boot:  0.15s  (-93%)
Ready for inference: 14x faster
```

**Steady-State Performance:**
```
Container RPS:   289
Unikernel RPS:   312  (+8%)
Latency P50:     160ms vs 145ms (-9%)
```

## 📊 Monitoring and Profiling

### Real-time Performance Dashboard

```python
def create_performance_dashboard():
    """Create a real-time performance monitoring dashboard."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import dash
    from dash import dcc, html

    app = dash.Dash(__name__)

    @app.callback(
        dash.dependencies.Output('performance-graph', 'figure'),
        [dash.dependencies.Input('interval-component', 'n_intervals')]
    )
    def update_graph(n):
        # Get latest stats
        stats = get_unillm_stats()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Throughput', 'Latency', 'Cache Hit Rate', 'GPU Utilization')
        )

        # Add traces for each metric
        fig.add_trace(
            go.Scatter(y=[stats['performance']['throughput_rps']]),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(y=[stats['performance']['p50_latency_ms']]),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(y=[stats['cache_stats']['hit_rate']]),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(y=[stats['gpu_stats']['utilization']]),
            row=2, col=2
        )

        return fig

    app.layout = html.Div([
        dcc.Graph(id='performance-graph'),
        dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
    ])

    return app

# Launch dashboard
dashboard = create_performance_dashboard()
dashboard.run_server(port=8081)
```

### Automated Performance Tuning

```python
def auto_tune_performance(target_metric='throughput'):
    """Automatically tune performance parameters."""

    # Parameter search space
    param_space = {
        'batch_size': [16, 32, 64, 128],
        'l1_cache_size': [512, 1024, 2048],
        'l2_cache_size': [1024, 2048, 4096],
        'gpu_memory_fraction': [0.7, 0.8, 0.9],
    }

    best_score = 0
    best_params = {}

    for batch_size in param_space['batch_size']:
        for l1_size in param_space['l1_cache_size']:
            for l2_size in param_space['l2_cache_size']:
                for gpu_fraction in param_space['gpu_memory_fraction']:

                    params = {
                        'UNILLM_OPTIMAL_BATCH_SIZE': batch_size,
                        'UNILLM_L1_CACHE_SIZE_MB': l1_size,
                        'UNILLM_L2_CACHE_SIZE_MB': l2_size,
                        'UNILLM_GPU_MEMORY_FRACTION': gpu_fraction
                    }

                    # Apply parameters and test
                    score = benchmark_with_params(params, target_metric)

                    if score > best_score:
                        best_score = score
                        best_params = params

                    print(f"Params: {params}, Score: {score}")

    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score}")

    return best_params
```

## 🚀 Advanced Tuning

### Dynamic Parameter Adjustment

```python
class DynamicTuner:
    """Dynamically adjust parameters based on workload."""

    def __init__(self):
        self.current_params = get_default_params()
        self.performance_history = []

    def adjust_parameters(self, workload_stats):
        """Adjust parameters based on current workload."""

        # Analyze workload characteristics
        avg_prompt_length = workload_stats['avg_prompt_length']
        avg_completion_length = workload_stats['avg_completion_length']
        request_rate = workload_stats['request_rate']
        cache_hit_rate = workload_stats['cache_hit_rate']

        # Adjust batch size based on request rate
        if request_rate > 10:  # High load
            new_batch_size = min(self.current_params['batch_size'] * 1.2, 128)
        elif request_rate < 2:  # Low load
            new_batch_size = max(self.current_params['batch_size'] * 0.8, 4)
        else:
            new_batch_size = self.current_params['batch_size']

        # Adjust cache sizes based on hit rate
        if cache_hit_rate < 0.5:  # Low hit rate
            l1_size = min(self.current_params['l1_cache_size'] * 1.3, 4096)
        else:
            l1_size = self.current_params['l1_cache_size']

        # Apply changes
        new_params = {
            'batch_size': int(new_batch_size),
            'l1_cache_size': int(l1_size)
        }

        self.apply_params(new_params)
        self.current_params.update(new_params)

    def apply_params(self, params):
        """Apply new parameters to UniLLM."""
        for key, value in params.items():
            env_var = f"UNILLM_{key.upper()}"
            os.environ[env_var] = str(value)

        # Restart service if needed
        restart_unillm_service()

# Usage
tuner = DynamicTuner()

while True:
    workload_stats = get_workload_stats()
    tuner.adjust_parameters(workload_stats)
    time.sleep(300)  # Adjust every 5 minutes
```

### A/B Testing Framework

```python
class PerformanceABTest:
    """A/B test different performance configurations."""

    def __init__(self, config_a, config_b):
        self.config_a = config_a
        self.config_b = config_b
        self.results_a = []
        self.results_b = []

    def run_test(self, duration_seconds=3600):
        """Run A/B test for specified duration."""

        start_time = time.time()
        current_config = 'a'

        while time.time() - start_time < duration_seconds:
            # Switch between configurations every 5 minutes
            if int((time.time() - start_time) / 300) % 2 == 0:
                if current_config != 'a':
                    self.apply_config(self.config_a)
                    current_config = 'a'
            else:
                if current_config != 'b':
                    self.apply_config(self.config_b)
                    current_config = 'b'

            # Collect metrics
            metrics = get_performance_metrics()

            if current_config == 'a':
                self.results_a.append(metrics)
            else:
                self.results_b.append(metrics)

            time.sleep(60)  # Sample every minute

    def analyze_results(self):
        """Analyze A/B test results."""

        import numpy as np
        from scipy import stats

        # Calculate means
        throughput_a = np.mean([r['throughput_rps'] for r in self.results_a])
        throughput_b = np.mean([r['throughput_rps'] for r in self.results_b])

        latency_a = np.mean([r['p50_latency_ms'] for r in self.results_a])
        latency_b = np.mean([r['p50_latency_ms'] for r in self.results_b])

        # Statistical significance
        throughput_pvalue = stats.ttest_ind(
            [r['throughput_rps'] for r in self.results_a],
            [r['throughput_rps'] for r in self.results_b]
        ).pvalue

        print(f"Configuration A - Throughput: {throughput_a:.1f} RPS, Latency: {latency_a:.1f}ms")
        print(f"Configuration B - Throughput: {throughput_b:.1f} RPS, Latency: {latency_b:.1f}ms")
        print(f"Throughput difference p-value: {throughput_pvalue:.4f}")

        if throughput_pvalue < 0.05:
            winner = 'A' if throughput_a > throughput_b else 'B'
            print(f"Configuration {winner} is significantly better")
        else:
            print("No significant difference between configurations")

        return {
            'winner': winner if 'winner' in locals() else 'tie',
            'throughput_improvement': abs(throughput_b - throughput_a),
            'latency_improvement': abs(latency_a - latency_b)
        }

# Example usage
config_a = {'batch_size': 32, 'l1_cache_size': 1024}
config_b = {'batch_size': 64, 'l1_cache_size': 2048}

ab_test = PerformanceABTest(config_a, config_b)
ab_test.run_test(duration_seconds=1800)  # 30 minutes
results = ab_test.analyze_results()
```

---

This performance tuning guide provides comprehensive strategies for optimizing UniLLM across different workloads and hardware configurations. For specific optimization scenarios or advanced tuning techniques, refer to the GPU support guide and deployment documentation.