# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Commands

### Building and Testing
```bash
# Main compilation check
cargo check

# Test all workspace crates
cargo test

# Test specific crates
cargo test --lib -p runtime
cargo test --lib -p inference
cargo test --lib -p kv
cargo test --lib -p scheduler

# Build with GPU features (when available)
cargo build --features cuda,metal -p runtime

# Run basic inference test
cargo run --bin test_basic_inference -p runtime

# Build optimized release
cargo build --release --profile gpu-optimized
```

### Development Commands
```bash
# Check individual workspace member
cargo check -p runtime

# Run tests with output
cargo test --lib -p runtime -- --nocapture

# Test a specific function
cargo test test_generation_config --lib -p runtime

# Build documentation
cargo doc --open --no-deps
```

## Architecture Overview

UniLLM uses a **three-layer abstraction system** that forms the foundation of all development:

### Layer 1: TensorCore (`crates/runtime/src/tensor_core.rs`)
- **Purpose**: Unified tensor operations across CPU/GPU/Metal devices
- **Key Types**: `Tensor`, `Device`, `TensorOps` trait, `ops_fn` module
- **Pattern**: All tensor operations go through `ops_fn::operation()` functional interface
- **Device Agnostic**: Same code works on CPU, CUDA, Metal automatically

### Layer 2: ModelCore (`crates/runtime/src/model_core.rs`)
- **Purpose**: Universal model interface and configuration system
- **Key Types**: `Model` trait, `ModelConfig` trait, `model_config!` macro
- **Pattern**: All models implement `Model` trait with consistent `forward()` and `generate()` methods
- **Configuration**: Use `model_config!` macro for automatic trait implementations

### Layer 3: WeightLoaderCore (`crates/runtime/src/weight_loader_core.rs`)
- **Purpose**: Format-agnostic model weight loading
- **Supports**: SafeTensors, GGUF, PyTorch formats
- **Pattern**: `WeightLoader::from_format()` or `WeightLoader::auto_detect()`

## Model Implementation Pattern

All models in `crates/runtime/src/models_v2/` follow this exact pattern:

```rust
use crate::model_config;
use super::traits::*;

// 1. Define configuration with automatic implementations
model_config!(YourModelConfig {
    vocab_size: usize = 32000,
    hidden_size: usize = 4096,
    num_hidden_layers: usize = 32,
    // ... other fields with defaults
});

// 2. Implement the universal Model trait
impl Model for YourModelV2 {
    type Config = YourModelConfig;

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        // Model-specific implementation using tensor_core ops
    }

    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        // High-level text generation
    }

    // ... other required methods
}
```

## Workspace Structure

- **`crates/runtime/`** - Main inference runtime with core abstractions
- **`crates/inference/`** - High-level inference engine components
- **`crates/kv/`** - KV cache and memory management
- **`crates/scheduler/`** - Request scheduling and batching
- **`crates/kernels/`** - GPU kernels (temporarily disabled)

## Key Implementation Notes

### Currently Working
- Basic tensor operations (CPU only)
- LLaMA model architecture (skeleton implementation)
- Model configuration system via `model_config!` macro
- Inference pipeline framework
- Weight loading infrastructure (SafeTensors format)

### In Development
- GPU acceleration backends (CUDA, Metal)
- Real model weight loading from files
- Additional model architectures (Qwen, Gemma, Phi, etc.)
- Production inference server

### Code Patterns to Follow

1. **Use TensorCore for all tensor operations**:
   ```rust
   let result = ops_fn::matmul(&tensor_a, &tensor_b)?;
   ```

2. **Define models using model_config! macro**:
   ```rust
   model_config!(ModelConfig {
       field: type = default_value,
   });
   ```

3. **Implement universal Model trait consistently**

4. **Handle errors with ModelResult<T>** and ModelError types

5. **Use Device enum for hardware abstraction**:
   ```rust
   let device = Device::auto(); // CPU, CUDA(0), Metal(0)
   ```

## Testing Guidelines

- All tests live in `#[cfg(test)]` modules within source files
- Focus on unit tests for individual components
- Use `cargo test --lib -p runtime` for fast iteration
- Current test suite covers basic functionality with placeholder data
- Tests expect errors when using dummy tensor data (this is normal)

## Development Focus Areas

1. **GPU Backend Implementation** - CUDA and Metal acceleration
2. **Real Model Loading** - Actual SafeTensors/GGUF file loading
3. **Model Architecture Completion** - Enable disabled model families
4. **Production Server** - Complete inference server implementation
5. **Performance Optimization** - Memory management and batching

## Special Notes

- The codebase has clean separation between abstractions and implementations
- Compiler warnings (~76) are expected due to incomplete implementations
- Focus on the three-layer abstraction system for all new development
- GPU detection works but acceleration is not yet implemented
- All models should use the same consistent patterns established in the abstractions