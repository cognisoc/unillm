# Development Setup

This guide covers setting up a development environment for UniLLM.

## Prerequisites

### Required

- **Rust** 1.70+ (install via [rustup](https://rustup.rs))
- **Git** for version control
- **8GB+ RAM** for running models

### Optional

- **CUDA Toolkit** 11.0+ for NVIDIA GPU support
- **Xcode Command Line Tools** for Metal on macOS

## Installation

### Clone the Repository

```bash
git clone https://github.com/unillm/unillm.git
cd unillm
```

### Build

```bash
# Debug build (faster compilation)
cargo build

# Release build (faster execution)
cargo build --release

# With GPU features
cargo build --features cuda,metal
```

### Verify Installation

```bash
# Run tests
cargo test --lib -p runtime

# Check compilation
cargo check
```

## Project Structure

```
unillm/
├── crates/
│   ├── runtime/           # Main inference runtime
│   │   ├── src/
│   │   │   ├── lib.rs           # Crate root
│   │   │   ├── tensor_core.rs   # Layer 1
│   │   │   ├── model_core.rs    # Layer 2
│   │   │   ├── weight_loader_core.rs  # Layer 3
│   │   │   ├── inference.rs     # Inference pipeline
│   │   │   ├── tokenizer.rs     # Tokenization
│   │   │   ├── ollama.rs        # Ollama integration
│   │   │   └── models_v2/       # Model implementations
│   │   │       ├── mod.rs       # Model exports
│   │   │       ├── traits.rs    # Shared traits
│   │   │       ├── llama.rs     # LLaMA
│   │   │       └── ...          # Other models
│   │   └── Cargo.toml
│   ├── inference/         # High-level inference
│   ├── kv/                # KV cache management
│   ├── scheduler/         # Request scheduling
│   └── kernels/           # GPU kernels
├── documentation/         # MkDocs documentation
├── tests/                 # Integration tests
├── Cargo.toml             # Workspace manifest
└── CLAUDE.md              # Development guide
```

## Development Workflow

### Building

```bash
# Full build
cargo build

# Single crate
cargo build -p runtime

# With features
cargo build --features cuda -p runtime
```

### Testing

```bash
# All tests
cargo test

# Specific crate
cargo test --lib -p runtime

# Specific test
cargo test test_generation_config --lib -p runtime

# With output
cargo test --lib -p runtime -- --nocapture
```

### Running Examples

```bash
# Basic inference
cargo run --bin test_basic_inference -p runtime

# Ollama integration
cargo run --bin test_ollama -p runtime

# With specific model
cargo run --bin test_ollama -p runtime -- --model llama2:7b
```

## IDE Setup

### VS Code

Install recommended extensions:

```json
// .vscode/extensions.json
{
  "recommendations": [
    "rust-lang.rust-analyzer",
    "tamasfe.even-better-toml",
    "serayuzgur.crates"
  ]
}
```

Settings:

```json
// .vscode/settings.json
{
  "rust-analyzer.cargo.features": "all",
  "rust-analyzer.checkOnSave.command": "clippy",
  "editor.formatOnSave": true
}
```

### CLion / RustRover

- Enable `rust-analyzer` in preferences
- Set up cargo watch for auto-checking

## Common Tasks

### Adding a Dependency

```bash
# Add to workspace
cargo add dependency_name

# Add to specific crate
cargo add dependency_name -p runtime
```

### Formatting

```bash
# Format all code
cargo fmt

# Check formatting
cargo fmt --check
```

### Linting

```bash
# Run clippy
cargo clippy

# With all features
cargo clippy --all-features

# Treat warnings as errors
cargo clippy -- -D warnings
```

### Documentation

```bash
# Generate docs
cargo doc --open --no-deps

# With private items
cargo doc --open --no-deps --document-private-items
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RUST_LOG` | Log level | `info` |
| `UNILLM_CACHE_DIR` | Model cache | `~/.cache/unillm` |
| `CUDA_VISIBLE_DEVICES` | GPU selection | All GPUs |

```bash
# Enable debug logging
export RUST_LOG=debug

# Set cache directory
export UNILLM_CACHE_DIR=/path/to/cache

# Use specific GPU
export CUDA_VISIBLE_DEVICES=0
```

## Debugging

### Debug Builds

```bash
# Debug build (includes debug info)
cargo build

# Run with debugger
rust-gdb target/debug/test_ollama
rust-lldb target/debug/test_ollama
```

### Logging

```rust
use log::{debug, info, warn, error};

// In your code
debug!("Debug message: {:?}", value);
info!("Info message");
warn!("Warning message");
error!("Error message");
```

Enable with:

```bash
RUST_LOG=debug cargo run --bin test_ollama -p runtime
```

### Profiling

```bash
# With perf (Linux)
perf record cargo run --release --bin test_ollama -p runtime
perf report

# With Instruments (macOS)
cargo instruments -t time --bin test_ollama -p runtime
```

## GPU Development

### CUDA Setup

```bash
# Install CUDA toolkit
# Ubuntu
sudo apt install nvidia-cuda-toolkit

# Verify
nvcc --version
nvidia-smi
```

### Metal Setup (macOS)

Metal support comes with Xcode Command Line Tools:

```bash
xcode-select --install
```

### Building with GPU

```bash
# CUDA only
cargo build --features cuda

# Metal only
cargo build --features metal

# Both
cargo build --features cuda,metal
```

## Troubleshooting

### Common Issues

**Compilation fails with linking errors**

```bash
# Ensure Rust is up to date
rustup update stable
```

**Out of memory during compilation**

```bash
# Reduce parallelism
cargo build -j 2
```

**GPU not detected**

```bash
# Check CUDA installation
nvidia-smi

# Check Metal (macOS)
system_profiler SPDisplaysDataType
```

### Getting Help

1. Check existing issues
2. Read error messages carefully
3. Search documentation
4. Ask in discussions

## Next Steps

- [Adding Models](adding-models.md) - Implement new model architectures
- [Code Style](code-style.md) - Follow our coding conventions
- [Architecture](../architecture/index.md) - Understand the system design
