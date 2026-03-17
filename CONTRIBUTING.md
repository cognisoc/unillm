# 🤝 Contributing to UniLLM

Thank you for your interest in contributing to UniLLM! This document provides guidelines for contributing to the world's first dual-mode LLM inference engine.

## 📋 Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Contribution Types](#contribution-types)
5. [Coding Standards](#coding-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)
9. [Community](#community)

## 📜 Code of Conduct

UniLLM is committed to fostering an open and welcoming environment. We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Expected Behavior

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Trolling, insulting/derogatory comments, and personal attacks
- Public or private harassment
- Publishing others' private information without explicit permission
- Other conduct which could reasonably be considered inappropriate

## 🚀 Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Rust**: Latest stable toolchain
- **GPU Development**: CUDA 12.8+ or ROCm 6.5+
- **Docker**: For container testing
- **Git**: For version control

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR-USERNAME/unillm.git
   cd unillm
   ```

2. **Install Dependencies**
   ```bash
   make install-deps
   make check-deps
   ```

3. **Set Up Development Environment**
   ```bash
   # Install Rust components
   rustup component add rustfmt clippy

   # Set up Git hooks
   cp scripts/pre-commit .git/hooks/
   chmod +x .git/hooks/pre-commit
   ```

4. **Verify Setup**
   ```bash
   cargo build --features cuda,hip
   cargo test --workspace
   make test
   ```

## 🔄 Development Workflow

### Branch Strategy

We use a feature branch workflow:

- `main`: Stable, production-ready code
- `develop`: Integration branch for features
- `feature/name`: Individual feature development
- `hotfix/name`: Critical bug fixes

### Creating a Feature Branch

```bash
# Create and switch to feature branch
git checkout -b feature/my-awesome-feature

# Make your changes
# ... code, test, commit ...

# Push branch
git push origin feature/my-awesome-feature

# Create pull request
```

### Commit Message Format

Use conventional commit format:

```
type(scope): short description

Longer description if needed

Fixes #123
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions or changes
- `ci`: CI/CD changes

**Examples:**
```
feat(kernels): add support for Intel XPU acceleration
fix(cache): resolve memory leak in L3 cache eviction
docs(api): update REST API documentation for v2 endpoints
perf(gpu): optimize CUDA kernel launch overhead by 15%
```

## 🎯 Contribution Types

### Code Contributions

#### New GPU Support
Adding support for new GPU architectures:

```rust
// 1. Implement driver interface
// crates/kernels/src/my_gpu_driver.rs
pub struct MyGpuDriver {
    device: MyGpuDevice,
    context: MyGpuContext,
}

impl GpuDriverInterface for MyGpuDriver {
    fn allocate_memory(&self, size: usize) -> Result<GpuMemoryHandle> {
        // Implementation
    }
    // ... other methods
}

// 2. Add hardware detection
// crates/kernels/src/hardware_detection.rs
impl HardwareDetector {
    fn detect_my_gpu() -> Option<GpuArchitecture> {
        // Detection logic
    }
}

// 3. Add build configuration
// Cargo.toml
[features]
my_gpu = ["dep:my_gpu_sdk"]
```

#### Performance Optimizations
```rust
// Example: Cache optimization
#[cfg(feature = "performance-optimization")]
impl CachePolicy {
    fn optimize_for_workload(&self, workload: &WorkloadProfile) -> CacheConfiguration {
        match workload.pattern {
            WorkloadPattern::ChatBot => self.chat_optimized_config(),
            WorkloadPattern::Translation => self.translation_optimized_config(),
            WorkloadPattern::CodeGen => self.codegen_optimized_config(),
        }
    }
}
```

#### Bug Fixes
```rust
// Example: Memory leak fix
impl MemoryPool {
    fn deallocate(&mut self, handle: MemoryHandle) -> Result<()> {
        // Ensure proper cleanup
        self.allocated_blocks.remove(&handle.id);
        self.free_blocks.insert(handle.size, handle.ptr);
        Ok(())
    }
}
```

### Documentation Contributions

- **API Documentation**: Improve inline documentation
- **User Guides**: Update deployment and usage guides
- **Examples**: Add code examples and tutorials
- **Troubleshooting**: Document common issues and solutions

### Testing Contributions

- **Unit Tests**: Test individual components
- **Integration Tests**: Test system interactions
- **Performance Tests**: Benchmark optimizations
- **Hardware Tests**: Test on specific GPU models

### Infrastructure Contributions

- **CI/CD**: Improve build and deployment pipelines
- **Docker**: Optimize container builds
- **Monitoring**: Add observability features

## 📏 Coding Standards

### Rust Style Guidelines

Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/):

#### Code Formatting
```bash
# Format code before committing
cargo fmt

# Check formatting
cargo fmt -- --check
```

#### Linting
```bash
# Run Clippy with strict settings
cargo clippy --all-targets --all-features -- -D warnings

# Fix common issues
cargo clippy --fix
```

#### Documentation
```rust
/// Calculate optimal batch size for the given GPU architecture.
///
/// # Arguments
///
/// * `gpu_arch` - The GPU architecture information
/// * `memory_gb` - Available GPU memory in GB
/// * `model_size` - Size of the model in parameters
///
/// # Returns
///
/// Optimal batch size for maximum throughput
///
/// # Examples
///
/// ```rust
/// use unillm::optimization::calculate_optimal_batch_size;
/// use unillm::types::{GpuArchitecture, ModelSize};
///
/// let gpu_arch = GpuArchitecture::NvidiaRTX4090;
/// let batch_size = calculate_optimal_batch_size(&gpu_arch, 24, ModelSize::Large);
/// assert_eq!(batch_size, 32);
/// ```
pub fn calculate_optimal_batch_size(
    gpu_arch: &GpuArchitecture,
    memory_gb: usize,
    model_size: ModelSize,
) -> usize {
    // Implementation
}
```

#### Error Handling
```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("GPU memory allocation failed: {size} bytes")]
    GpuMemoryError { size: usize },

    #[error("Invalid batch size: {batch_size}, must be between 1 and {max_size}")]
    InvalidBatchSize { batch_size: usize, max_size: usize },

    #[error("Model loading failed")]
    ModelLoadError(#[from] std::io::Error),
}
```

#### Async Code
```rust
use tokio;

impl InferenceEngine {
    /// Process inference request asynchronously
    pub async fn process_request(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        // Use proper error propagation
        let batch = self.scheduler.schedule_request(request).await?;
        let result = self.execute_batch(batch).await?;
        self.cache.store_result(&result).await?;
        Ok(result.into_response())
    }
}
```

### Python Style (Build Scripts)

For Python scripts (build.py, etc.), follow PEP 8:

```python
def build_gpu_optimized_image(
    gpu_target: str,
    cuda_version: Optional[str] = None,
    optimization_level: str = "O3"
) -> str:
    """
    Build GPU-optimized UniLLM container image.

    Args:
        gpu_target: Target GPU (rtx4090, h100, mi300x, etc.)
        cuda_version: CUDA version (auto-detected if None)
        optimization_level: Compiler optimization level

    Returns:
        Docker image tag

    Raises:
        BuildError: If build fails
        UnsupportedGPUError: If GPU not supported
    """
    if gpu_target not in SUPPORTED_GPUS:
        raise UnsupportedGPUError(f"GPU {gpu_target} not supported")

    # Implementation
    return f"unillm:{gpu_target}-optimized"
```

## 🧪 Testing Guidelines

### Test Categories

1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Benchmark critical paths
4. **Hardware Tests**: Test GPU-specific functionality

### Writing Tests

#### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_size_calculation() {
        let gpu_arch = GpuArchitecture::NvidiaRTX4090;
        let batch_size = calculate_optimal_batch_size(&gpu_arch, 24, ModelSize::Large);
        assert_eq!(batch_size, 32);
    }

    #[tokio::test]
    async fn test_async_inference() {
        let engine = InferenceEngine::new_test().await;
        let request = InferenceRequest::new("Hello world", 50);
        let response = engine.process_request(request).await.unwrap();
        assert!(!response.generated_text.is_empty());
    }
}
```

#### Integration Tests
```rust
// tests/integration_test.rs
#[tokio::test]
async fn test_end_to_end_inference() {
    // Start server
    let server = start_test_server().await;

    // Make request
    let response = reqwest::post(&format!("{}/v1/generate", server.url()))
        .json(&json!({
            "prompt": "Test prompt",
            "max_tokens": 10
        }))
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());

    // Cleanup
    server.shutdown().await;
}
```

#### Performance Tests
```rust
#[cfg(feature = "benchmarking")]
mod benches {
    use criterion::{black_box, criterion_group, criterion_main, Criterion};

    fn benchmark_inference_latency(c: &mut Criterion) {
        let engine = InferenceEngine::new_test();

        c.bench_function("inference_latency", |b| {
            b.iter(|| {
                let request = InferenceRequest::new("Benchmark prompt", 10);
                black_box(engine.process_request_sync(request))
            })
        });
    }

    criterion_group!(benches, benchmark_inference_latency);
    criterion_main!(benches);
}
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test categories
cargo test --lib                    # Unit tests
cargo test --test integration_test  # Integration tests
cargo test --features benchmarking  # Performance tests

# Run GPU-specific tests (requires GPU)
make test-cuda
make test-rocm

# Run with coverage
cargo tarpaulin --all-features --out Html
```

## 📚 Documentation

### Documentation Types

1. **Code Documentation**: Inline Rust docs
2. **API Documentation**: REST API reference
3. **User Guides**: Deployment and usage guides
4. **Developer Guides**: Contributing and architecture docs

### Writing Documentation

#### Inline Documentation
```rust
/// High-performance hybrid cache combining radix trees and paged allocation.
///
/// The `HybridCache` implements a three-tier cache system:
/// - L1: Radix tree for prefix sharing (GPU memory)
/// - L2: Paged allocation for efficiency (GPU memory)
/// - L3: Compressed storage for capacity (system memory)
///
/// # Examples
///
/// ```rust
/// use unillm::cache::HybridCache;
///
/// let cache = HybridCache::new(1024 * 1024 * 1024)?; // 1GB
/// cache.store("common prefix", &tokens).await?;
/// let cached_tokens = cache.get("common prefix").await?;
/// ```
pub struct HybridCache {
    // Implementation
}
```

#### README Updates
When adding new features, update the main README.md:

```markdown
## 🆕 New Features

### Intel XPU Support (v0.2.0)
UniLLM now supports Intel Arc GPUs and Data Center GPUs:

```bash
# Build for Intel Arc A770
make build-arc-a770

# Enable XPU features
cargo build --features xpu
```

Performance improvements:
- Arc A770: 25% faster than CPU inference
- Data Center GPU Max: Competitive with NVIDIA A100
```

### Building Documentation

```bash
# Generate Rust documentation
cargo doc --no-deps --all-features --open

# Serve documentation locally
python3 -m http.server 8000 --directory target/doc

# Check documentation
cargo doc --document-private-items
```

## 🔄 Pull Request Process

### Before Submitting

1. **Code Quality**
   ```bash
   # Format code
   cargo fmt

   # Run linter
   cargo clippy --all-targets --all-features -- -D warnings

   # Run tests
   make test
   ```

2. **Documentation**
   ```bash
   # Update documentation
   cargo doc --no-deps --all-features

   # Check documentation builds
   mdbook build docs/
   ```

3. **Performance**
   ```bash
   # Run benchmarks if changes affect performance
   make benchmark

   # Profile changes
   cargo build --profile release-with-debug
   ```

### PR Template

When creating a pull request, use this template:

```markdown
## Description
Brief description of the changes and their purpose.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance tests pass (if applicable)
- [ ] Manual testing completed

## Hardware Tested
- [ ] NVIDIA GPU (specify model): _______________
- [ ] AMD GPU (specify model): _______________
- [ ] Intel GPU (specify model): _______________
- [ ] CPU-only mode

## Performance Impact
- Throughput change: +/- ___%
- Latency change: +/- ___ms
- Memory usage change: +/- ___MB

## Documentation
- [ ] Code comments updated
- [ ] API documentation updated
- [ ] User guides updated (if applicable)
- [ ] README updated (if applicable)

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

## Related Issues
Closes #___
Related to #___
```

### Review Process

1. **Automated Checks**
   - CI/CD pipeline must pass
   - Code formatting and linting
   - All tests must pass
   - Documentation builds successfully

2. **Code Review**
   - At least one maintainer review required
   - Focus on correctness, performance, and maintainability
   - Architecture and design review for large changes

3. **Performance Review**
   - Benchmark results for performance-critical changes
   - Memory usage analysis
   - GPU utilization impact

4. **Merge Requirements**
   - All checks passing
   - Approved by maintainer
   - Up-to-date with main branch
   - Squash merge preferred for features

## 👥 Community

### Communication Channels

- **GitHub Discussions**: General questions and feature discussions
- **GitHub Issues**: Bug reports and feature requests
- **Discord**: Real-time chat and development discussions
- **Email**: maintainers@unillm.org for private matters

### Getting Help

- **Documentation**: Check the comprehensive docs in `/docs`
- **Examples**: See example code in `/examples`
- **Troubleshooting**: Review the troubleshooting guide
- **Community**: Ask questions in GitHub Discussions

### Reporting Issues

When reporting bugs, include:

1. **Environment Information**
   ```bash
   # Run diagnostic script
   ./scripts/unillm-diagnostics.sh
   ```

2. **Reproduction Steps**
   - Minimal example to reproduce the issue
   - Expected vs actual behavior
   - Any error messages or logs

3. **System Details**
   - OS and version
   - GPU model and driver version
   - UniLLM version
   - Container or unikernel mode

### Feature Requests

When requesting features:

1. **Use Case**: Describe why the feature is needed
2. **Proposed Solution**: Outline how you envision it working
3. **Alternatives**: Consider other approaches
4. **Impact**: Who would benefit from this feature

## 🎉 Recognition

Contributors are recognized in:

- **CONTRIBUTORS.md**: All contributors listed
- **Release Notes**: Significant contributions highlighted
- **Documentation**: Author attribution for major docs
- **Community**: Shout-outs in Discord and discussions

## 📄 License

By contributing to UniLLM, you agree that your contributions will be licensed under the Apache License 2.0.

---

Thank you for contributing to UniLLM! Together, we're building the future of LLM inference. 🚀