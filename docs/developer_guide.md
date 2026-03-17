# 🛠️ UniLLM Developer Guide

This comprehensive guide covers everything developers need to know to work with, extend, and contribute to UniLLM.

## 📋 Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Architecture Overview](#architecture-overview)
3. [Building and Testing](#building-and-testing)
4. [API Reference](#api-reference)
5. [Extending UniLLM](#extending-unillm)
6. [Unikernel Development](#unikernel-development)
7. [Performance Optimization](#performance-optimization)
8. [Contributing Guidelines](#contributing-guidelines)

## 🔧 Development Environment Setup

### Prerequisites

**Required Tools:**
```bash
# Rust toolchain (latest stable)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Build essentials
sudo apt-get update
sudo apt-get install build-essential cmake ninja-build pkg-config libssl-dev

# Docker (for container builds)
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Python 3.8+ (for build system)
sudo apt-get install python3 python3-pip
```

**GPU Development Libraries:**

For NVIDIA GPUs:
```bash
# CUDA Toolkit (12.8 recommended)
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_550.54.15_linux.run
sudo sh cuda_12.8.0_550.54.15_linux.run

# Set environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

For AMD GPUs:
```bash
# ROCm (6.5 recommended)
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.5 jammy main" | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-dev hip-dev
```

### Project Setup

```bash
# Clone the repository
git clone https://github.com/unillm/unillm.git
cd unillm

# Install dependencies
make install-deps

# Verify setup
make check-deps

# Build for development
cargo build --features cuda,hip

# Run tests
make test
```

### Development Tools Setup

**VS Code Configuration (.vscode/settings.json):**
```json
{
    "rust-analyzer.cargo.features": ["cuda", "hip", "unikernel"],
    "rust-analyzer.checkOnSave.command": "clippy",
    "rust-analyzer.checkOnSave.extraArgs": ["--", "-D", "warnings"],
    "files.associations": {
        "*.hbs": "handlebars"
    }
}
```

**Git Hooks Setup:**
```bash
# Install pre-commit hooks
cp scripts/pre-commit .git/hooks/
chmod +x .git/hooks/pre-commit
```

## 🏗️ Architecture Overview

### Core Components

```
UniLLM/
├── crates/
│   ├── kv/                     # Hybrid KV Cache System
│   │   ├── hybrid_cache.rs     # L1 Radix + L2 Paged + L3 Compressed
│   │   ├── radix_cache.rs      # SGLang-inspired prefix sharing
│   │   └── paged_allocator.rs  # vLLM-inspired block allocation
│   │
│   ├── scheduler/              # Intelligent Scheduling
│   │   ├── intelligent_scheduler.rs  # Cache-aware scheduling
│   │   ├── stream_overlap.rs         # Concurrent stream processing
│   │   └── gpu_memory_tracker.rs     # Memory optimization
│   │
│   ├── kernels/                # GPU Kernel Framework
│   │   ├── template_engine.rs   # Template-based kernel generation
│   │   ├── cuda_driver.rs       # Direct CUDA driver interface
│   │   ├── hip_driver.rs        # Direct HIP driver interface
│   │   └── unikernel_gpu.rs     # Unikernel GPU abstraction
│   │
│   └── inference/              # Unified Inference Engine
│       ├── engine.rs           # Main inference coordinator
│       ├── batch_processor.rs  # Request batching
│       └── stream_processor.rs # Streaming responses
│
└── src/                        # Main Applications
    ├── server.rs               # HTTP inference server
    ├── client.rs               # Command-line client
    └── benchmark.rs            # Performance benchmarking
```

### Key Design Principles

1. **Modularity**: Each crate has a single responsibility and clean interfaces
2. **Performance**: Zero-cost abstractions and direct hardware access
3. **Safety**: Rust's memory safety without performance overhead
4. **Flexibility**: Support for multiple GPU vendors and deployment modes
5. **Extensibility**: Plugin architecture for new algorithms and hardware

## 🔨 Building and Testing

### Build Configurations

**Development Build:**
```bash
# Fast compilation for development
cargo build --features cuda

# With debug info
cargo build --profile release-with-debug --features cuda,hip
```

**Production Build:**
```bash
# Optimized release build
cargo build --release --features cuda,hip

# GPU-optimized build
cargo build --profile gpu-optimized --features cuda,hip
```

**Feature Flags:**
```bash
# GPU vendor support
--features cuda              # NVIDIA CUDA support
--features hip               # AMD ROCm/HIP support
--features cuda,hip          # Multi-vendor support

# Unikernel support
--features unikernel         # Base unikernel support
--features nanos             # Nanos unikernel
--features unikraft          # Unikraft unikernel

# Additional features
--features benchmarking      # Include benchmark suite
--features debug-optimizations  # Debug-friendly optimizations
```

### Testing

**Unit Tests:**
```bash
# Run all tests
cargo test --workspace

# GPU-specific tests (requires GPU)
cargo test --features cuda test_cuda
cargo test --features hip test_hip

# Specific module tests
cargo test -p kv
cargo test -p scheduler
```

**Integration Tests:**
```bash
# Full system tests
make test

# Performance tests
make benchmark

# Memory leak tests
cargo test --features cuda -- --test-threads=1 --nocapture test_memory_leak
```

**Continuous Integration:**
```bash
# CI test suite
make ci-test

# Build all variants
make ci-build
```

## 📡 API Reference

### REST API Endpoints

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "gpu_target": "cuda",
  "runtime_mode": "container",
  "memory_usage_mb": 1024.5,
  "gpu_memory_usage_mb": 2048.0
}
```

#### Text Generation
```http
POST /v1/generate
Content-Type: application/json

{
  "prompt": "The future of AI is",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**Response:**
```json
{
  "generated_text": "The future of AI is bright and full of possibilities...",
  "tokens_generated": 87,
  "inference_time_ms": 245,
  "cache_hits": 15,
  "gpu_utilization": 0.85
}
```

#### Statistics
```http
GET /stats
```

**Response:**
```json
{
  "total_requests": 1500,
  "total_tokens_generated": 150000,
  "average_latency_ms": 230.5,
  "cache_hit_rate": 0.75,
  "gpu_utilization": 0.82,
  "memory_stats": {
    "total_memory_mb": 16384.0,
    "used_memory_mb": 8192.0,
    "cache_memory_mb": 2048.0,
    "gpu_memory_mb": 12288.0
  }
}
```

### Rust API

#### Basic Inference
```rust
use unillm::{UniLLMInferenceEngine, InferenceRequest, InferenceResult};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the inference engine
    let engine = UniLLMInferenceEngine::new_with_defaults().await?;

    // Create a request
    let request = InferenceRequest {
        prompt_tokens: "Hello, how are you?".split_whitespace().count(),
        max_output_length: 50,
        temperature: 0.7,
        top_p: 0.9,
        stop_sequences: vec![],
        stream: false,
    };

    // Process the request
    let result = engine.process_request(request).await?;

    println!("Generated: {}", result.generated_text);
    println!("Tokens: {}", result.tokens_generated);
    println!("Time: {}ms", result.inference_time_ms);

    Ok(())
}
```

#### Custom GPU Configuration
```rust
use unillm::kernels::{KernelFramework, HardwareDetector};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Detect hardware
    let hardware_info = HardwareDetector::detect_hardware()?;
    println!("Detected GPU: {:?}", hardware_info.gpu_architecture);

    // Create optimized kernel framework
    let kernel_framework = KernelFramework::new()?;

    // Get optimization configuration
    let workload = WorkloadCharacteristics {
        batch_size: 32,
        sequence_length: 512,
        model_size: ModelSize::Large,
        precision: Precision::FP16,
    };

    let config = kernel_framework.get_optimized_configuration(&workload)?;
    println!("Optimization config: {:?}", config);

    Ok(())
}
```

#### Unikernel Mode Detection
```rust
#[cfg(feature = "unikernel")]
use unillm::kernels::unikernel_gpu::{detect_unikernel_runtime, create_runtime_gpu_interface};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Detect runtime environment
    if let Some(runtime) = detect_unikernel_runtime() {
        println!("Running in unikernel mode: {}", runtime);

        // Create unikernel GPU interface
        let gpu_interface = create_runtime_gpu_interface().await?;
        let memory_info = gpu_interface.get_memory_info().await?;

        println!("GPU Memory: {:.1} GB available",
                memory_info.available_memory as f64 / (1024.0 * 1024.0 * 1024.0));
    } else {
        println!("Running in container mode");
    }

    Ok(())
}
```

## 🔍 Extending UniLLM

### Adding New GPU Support

1. **Implement GPU Driver Interface:**
```rust
// crates/kernels/src/my_gpu_driver.rs
use crate::types::{GpuDriverResult, GpuContext, GpuMemoryHandle};

pub struct MyGpuDriverInterface {
    context: GpuContext,
    device_id: u32,
}

impl MyGpuDriverInterface {
    pub fn new(device_id: u32) -> GpuDriverResult<Self> {
        // Initialize GPU driver
        let context = initialize_my_gpu_driver(device_id)?;

        Ok(Self { context, device_id })
    }

    pub fn allocate_memory(&self, size: usize) -> GpuDriverResult<GpuMemoryHandle> {
        // Implement memory allocation
        todo!("Implement memory allocation for your GPU")
    }

    pub fn launch_kernel(&self, kernel_name: &str, params: &[u8]) -> GpuDriverResult<()> {
        // Implement kernel launch
        todo!("Implement kernel launch for your GPU")
    }
}
```

2. **Add Hardware Detection:**
```rust
// crates/kernels/src/hardware_detection.rs
impl HardwareDetector {
    fn detect_my_gpu() -> Option<GpuArchitecture> {
        // Check for your GPU hardware
        if my_gpu_is_present() {
            Some(GpuArchitecture::MyGpu {
                device_name: get_my_gpu_name(),
                compute_capability: get_my_gpu_capability(),
                memory_gb: get_my_gpu_memory(),
            })
        } else {
            None
        }
    }
}
```

3. **Add Build Configuration:**
```toml
# Cargo.toml
[features]
my_gpu = ["kernels/my_gpu"]

# crates/kernels/Cargo.toml
[features]
my_gpu = ["dep:my_gpu_sdk"]

[dependencies]
my_gpu_sdk = { version = "1.0", optional = true }
```

### Custom Cache Policies

```rust
// crates/kv/src/custom_cache_policy.rs
use crate::types::{CachePolicy, EvictionStrategy, AccessPattern};

pub struct CustomCachePolicy {
    // Your custom cache policy state
}

impl CachePolicy for CustomCachePolicy {
    fn should_cache(&self, pattern: &AccessPattern) -> bool {
        // Implement your caching logic
        pattern.frequency > 0.5 && pattern.recency < Duration::from_secs(300)
    }

    fn get_eviction_candidate(&self, blocks: &[CacheBlock]) -> Option<BlockId> {
        // Implement your eviction strategy
        blocks.iter()
              .min_by_key(|block| block.last_access)
              .map(|block| block.id)
    }
}
```

### Custom Schedulers

```rust
// crates/scheduler/src/custom_scheduler.rs
use crate::types::{Request, BatchSchedulingDecision, SchedulingPolicy};

pub struct CustomScheduler {
    // Your scheduler state
}

impl SchedulingPolicy for CustomScheduler {
    async fn schedule_batch(&mut self, requests: &[Request]) -> BatchSchedulingDecision {
        // Implement your scheduling algorithm
        let mut selected = Vec::new();
        let mut total_memory = 0;

        for request in requests {
            let memory_required = estimate_memory_requirement(request);
            if total_memory + memory_required <= self.memory_limit {
                selected.push(request.clone());
                total_memory += memory_required;
            }
        }

        BatchSchedulingDecision {
            selected_requests: selected,
            memory_allocation: total_memory,
            estimated_latency: self.estimate_latency(&selected),
        }
    }
}
```

## 🔥 Unikernel Development

### Building Unikernel Support

1. **Add Unikernel Feature Flag:**
```rust
#[cfg(feature = "unikernel")]
mod unikernel_specific_code {
    // Code that only runs in unikernel mode
}

#[cfg(not(feature = "unikernel"))]
mod container_specific_code {
    // Code that only runs in container mode
}
```

2. **Runtime Detection:**
```rust
pub fn detect_runtime_environment() -> RuntimeEnvironment {
    if std::env::var("UNILLM_UNIKERNEL_MODE").is_ok() {
        RuntimeEnvironment::Unikernel
    } else if std::path::Path::new("/.dockerenv").exists() {
        RuntimeEnvironment::Container
    } else {
        RuntimeEnvironment::Bare
    }
}
```

3. **Memory Management:**
```rust
#[cfg(feature = "unikernel")]
fn allocate_gpu_memory(size: usize) -> Result<*mut u8, AllocationError> {
    // Direct physical memory allocation in unikernel
    unsafe {
        let ptr = nanos_gpu_alloc(size);
        if ptr.is_null() {
            Err(AllocationError::OutOfMemory)
        } else {
            Ok(ptr)
        }
    }
}

#[cfg(not(feature = "unikernel"))]
fn allocate_gpu_memory(size: usize) -> Result<*mut u8, AllocationError> {
    // Standard GPU memory allocation
    cuda_malloc(size)
}
```

### Nanos Unikernel Integration

```rust
// Platform-specific code for Nanos
#[cfg(feature = "nanos")]
mod nanos_integration {
    use super::*;

    pub fn initialize_nanos_gpu() -> Result<NanosGpuContext, Error> {
        // Load GPU klib
        let klib_name = std::env::var("NANOS_GPU_KLIB")
            .unwrap_or_else(|_| "nvidia-535.54.03".to_string());

        // Initialize Nanos GPU interface
        let context = unsafe {
            nanos_gpu_init(&klib_name)?
        };

        Ok(NanosGpuContext { context })
    }

    pub fn nanos_gpu_launch_kernel(
        context: &NanosGpuContext,
        kernel: &str,
        params: &[u8]
    ) -> Result<(), Error> {
        unsafe {
            nanos_gpu_kernel_launch(context.context, kernel.as_ptr(), params.as_ptr(), params.len())
        }
    }
}
```

## ⚡ Performance Optimization

### Profiling and Benchmarking

```rust
// Enable detailed profiling
use unillm::profiling::{ProfileScope, ProfileData};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Enable profiling
    let _profile_scope = ProfileScope::new("inference_pipeline");

    let engine = UniLLMInferenceEngine::new_with_defaults().await?;

    // Run benchmark
    let mut total_time = Duration::ZERO;
    let num_requests = 100;

    for i in 0..num_requests {
        let start = Instant::now();

        let request = InferenceRequest {
            prompt_tokens: 50,
            max_output_length: 50,
            temperature: 0.7,
            top_p: 0.9,
            stop_sequences: vec![],
            stream: false,
        };

        let _result = engine.process_request(request).await?;

        total_time += start.elapsed();

        if i % 10 == 0 {
            println!("Completed {} requests", i);
        }
    }

    println!("Average latency: {:.2}ms",
             total_time.as_millis() as f64 / num_requests as f64);

    // Get detailed profile data
    let profile_data = ProfileData::get_current();
    println!("Profile data: {:#?}", profile_data);

    Ok(())
}
```

### Memory Optimization

```rust
// Tune cache sizes based on available memory
use unillm::memory::{MemoryManager, CacheConfiguration};

fn optimize_cache_configuration() -> CacheConfiguration {
    let total_memory = MemoryManager::get_total_memory();
    let gpu_memory = MemoryManager::get_gpu_memory();

    // Use 25% of GPU memory for L1 cache
    let l1_size = (gpu_memory as f64 * 0.25) as usize;

    // Use 50% of GPU memory for L2 cache
    let l2_size = (gpu_memory as f64 * 0.50) as usize;

    // Use system memory for L3 cache
    let l3_size = (total_memory as f64 * 0.10) as usize;

    CacheConfiguration {
        l1_radix_size: l1_size,
        l2_paged_size: l2_size,
        l3_compressed_size: l3_size,
        compression_ratio: 0.6,
        eviction_policy: EvictionPolicy::LeastRecentlyUsed,
    }
}
```

### GPU Kernel Optimization

```rust
// Template-based kernel optimization
use unillm::kernels::{KernelTemplate, OptimizationHints};

fn generate_optimized_kernel(gpu_arch: &GpuArchitecture) -> String {
    let template = KernelTemplate::new("attention_kernel.hbs");

    let optimization_hints = match gpu_arch {
        GpuArchitecture::Nvidia { compute_capability, .. } => {
            OptimizationHints {
                use_tensor_cores: compute_capability >= &(7, 0),
                block_size: if compute_capability >= &(8, 0) { 256 } else { 128 },
                shared_memory_size: 48 * 1024, // 48KB
                register_usage: RegisterUsage::High,
            }
        },
        GpuArchitecture::Amd { .. } => {
            OptimizationHints {
                use_tensor_cores: false,
                block_size: 64,
                shared_memory_size: 32 * 1024, // 32KB
                register_usage: RegisterUsage::Medium,
            }
        },
    };

    template.render(&optimization_hints)
}
```

## 🤝 Contributing Guidelines

### Code Style

1. **Rust Style:**
   - Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
   - Use `rustfmt` for formatting: `cargo fmt`
   - Use `clippy` for linting: `cargo clippy -- -D warnings`

2. **Commit Messages:**
   ```
   feat: add support for Intel XPU
   fix: resolve memory leak in cache eviction
   docs: update API reference for v2 endpoints
   perf: optimize CUDA kernel launch overhead
   test: add integration tests for unikernel mode
   ```

3. **Documentation:**
   - All public APIs must have documentation
   - Include examples in doc comments
   - Update relevant documentation files

### Testing Requirements

1. **Unit Tests:**
   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;

       #[test]
       fn test_cache_allocation() {
           // Test implementation
       }

       #[tokio::test]
       async fn test_async_inference() {
           // Async test implementation
       }
   }
   ```

2. **Integration Tests:**
   ```bash
   # tests/integration_test.rs
   # Full system integration tests
   ```

3. **Benchmark Tests:**
   ```rust
   #[cfg(feature = "benchmarking")]
   mod benches {
       use criterion::{black_box, criterion_group, criterion_main, Criterion};

       fn benchmark_inference(c: &mut Criterion) {
           c.bench_function("inference_latency", |b| {
               b.iter(|| {
                   // Benchmark implementation
               })
           });
       }

       criterion_group!(benches, benchmark_inference);
       criterion_main!(benches);
   }
   ```

### Pull Request Process

1. **Before Submitting:**
   ```bash
   # Run full test suite
   make ci-test

   # Check formatting
   cargo fmt --check

   # Run clippy
   cargo clippy --all-targets --all-features -- -D warnings

   # Update documentation
   cargo doc --no-deps --all-features
   ```

2. **PR Template:**
   - Clear description of changes
   - Link to related issues
   - Test results and performance impact
   - Documentation updates
   - Breaking changes noted

3. **Review Process:**
   - Code review from maintainers
   - CI pipeline must pass
   - Performance benchmarks if applicable
   - Documentation review

### Development Workflow

```bash
# 1. Set up feature branch
git checkout -b feature/my-new-feature

# 2. Make changes with tests
# ... edit code ...

# 3. Test locally
make test
make benchmark

# 4. Commit changes
git add .
git commit -m "feat: add my new feature"

# 5. Push and create PR
git push origin feature/my-new-feature
# Create PR on GitHub

# 6. Address review feedback
# ... make changes ...
git add .
git commit -m "fix: address review feedback"
git push origin feature/my-new-feature
```

---

This developer guide provides the foundation for working with UniLLM. For specific questions or advanced topics, please refer to the other documentation files or open an issue on GitHub.