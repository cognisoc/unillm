# UniLLM

**High-performance Rust-based LLM inference engine with solid abstractions**

UniLLM is a modern, high-performance inference runtime built in Rust, designed with clean architecture and solid abstractions. It provides a unified interface for running large language models across different architectures and deployment targets.

## ✨ Key Features

### 🏗️ Solid Architecture
- **Clean Abstractions** - TensorCore, ModelCore, WeightLoaderCore layers
- **Unified Interface** - Single Model trait for all architectures
- **Type Safety** - Compile-time guarantees prevent runtime errors
- **Memory Safety** - Rust prevents entire classes of production bugs

### 🤖 Model Support
- **45+ Model Architectures** across 10 categories
- **Core LLMs** - LLaMA, Qwen, Gemma, Phi, DeepSeek, Mistral, Mixtral
- **GPT Family** - GPT-2, GPT-J, GPT-NeoX, OPT, BLOOM, MPT
- **Code Models** - StarCoder, CodeLlama
- **MoE Models** - DeepSeek-MoE, DBRX, Grok, Arctic, Jamba
- **RWKV/Linear** - RWKV-4, RWKV-6, RecurrentGemma
- **Vision-Language** - Qwen2-VL, Phi-3-Vision, InternVL, CogVLM, Idefics, Florence, LLaVA, CLIP
- **Audio/Speech** - Wav2Vec2, HuBERT, MusicGen, Encodec, Whisper
- **Additional** - Yi, Falcon, Baichuan, InternLM, ChatGLM, BERT, T5, Mamba, MiniCPM, OLMo, Granite
- **Consistent Implementation** - All models use the same solid abstractions
- **Easy Extension** - Add new models with minimal boilerplate

### ⚡ Performance Ready
- **Device Abstraction** - Unified CPU/GPU interface
- **Async Runtime** - Built on Tokio for high concurrency
- **Zero-Cost Abstractions** - Rust's performance guarantees
- **Modular Design** - Clean separation enables optimization

## 🚀 Quick Start

### Prerequisites
- Rust 1.70+
- Git

### Build and Test
```bash
git clone https://github.com/your-org/unillm.git
cd unillm
cargo check  # Verify everything compiles
cargo test   # Run tests
```

### Try the Runtime
```bash
# Basic compilation check
cargo check -p runtime

# Example server (work in progress)
cargo run --bin production_server -p runtime
```

## 📁 Project Structure

```
├── crates/
│   ├── runtime/          # 🧠 Main inference runtime
│   │   ├── src/
│   │   │   ├── lib.rs              # Clean module structure
│   │   │   ├── tensor_core.rs      # Unified tensor operations
│   │   │   ├── model_core.rs       # Model trait and abstractions
│   │   │   ├── weight_loader_core.rs # Format-agnostic weight loading
│   │   │   ├── models_v2/          # 18+ model implementations
│   │   │   ├── inference.rs        # Inference pipeline
│   │   │   ├── tokenizer.rs        # Tokenization utilities
│   │   │   └── bin/                # Example binaries
│   │   └── Cargo.toml
│   ├── inference/        # 🔄 Inference engine components
│   ├── kv/              # 💾 KV cache and memory management
│   └── scheduler/       # ⚡ Request scheduling and batching
├── docs/                # 📚 Documentation
└── Cargo.toml          # 📦 Workspace configuration
```

## 🏛️ Architecture Overview

### Three-Layer Abstraction System

1. **TensorCore** - Unified tensor operations and device management
   - Device-agnostic tensor operations
   - Memory management abstractions
   - Backend-specific implementations (CPU, CUDA, Metal)

2. **ModelCore** - Clean model interfaces and configuration
   - Universal Model trait for all architectures
   - Consistent configuration system via `model_config!` macro
   - Format-agnostic weight loading

3. **WeightLoaderCore** - Model weight loading and management
   - SafeTensors, GGUF, and other format support
   - Lazy loading and memory optimization
   - Device transfer capabilities

### Model Implementation Pattern

All models follow the same clean pattern:

```rust
use crate::model_config;
use super::traits::*;

// Define configuration with automatic trait implementations
model_config!(LlamaConfig {
    vocab_size: usize = 32000,
    hidden_size: usize = 4096,
    num_hidden_layers: usize = 32,
    // ... other fields
});

// Implement the unified Model trait
impl Model for LlamaModelV2 {
    type Config = LlamaConfig;

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        // Model-specific implementation
    }

    // ... other methods
}
```

## 🛠️ Development

### Current Status

✅ **Core Infrastructure Complete**
- Clean module structure with TensorCore, ModelCore, WeightLoaderCore
- 47 model architectures implemented across 10 categories
- Project compiles with 166 passing tests

✅ **GGUF Integration Working**
- Ollama registry integration for model downloads
- GGUF weight loading with dequantization (Q4_0, Q8_0, etc.)
- GGUF tokenizer extraction (vocabulary, special tokens)
- Tensor name mapping (GGUF → HuggingFace format)

✅ **LLaMA Inference Pipeline**
- Complete forward pass with real tensor operations
- RoPE (Rotary Position Embedding) implemented
- Causal attention masking for autoregressive generation
- Grouped Query Attention (GQA) support
- Text generation with greedy sampling

🚧 **In Progress**
- KV caching for efficient autoregressive generation
- Additional sampling methods (temperature, top-p, top-k)

📋 **Next Priorities**
1. **KV Cache** - Critical for efficient multi-token generation
2. **GPU Acceleration** - CUDA and Metal backends
3. **Production Server** - HTTP API with streaming responses
4. **Model Testing** - End-to-end validation with real weights

📋 **Future**
- Continuous batching
- Speculative decoding
- Quantized inference (use quantized weights directly)
- Distributed inference

### Adding New Models

Thanks to the solid abstractions, adding a new model is straightforward:

1. Create configuration with `model_config!` macro
2. Implement the `Model` trait
3. Define model-specific layers and operations
4. Export from `models_v2/mod.rs`

The abstractions handle device management, weight loading, and interface consistency automatically.

### Building and Testing

```bash
# Check everything compiles
cargo check

# Test specific crates
cargo test -p runtime
cargo test -p inference
cargo test -p scheduler

# Build optimized release
cargo build --release
```

## 🎯 Design Principles

1. **Solid Abstractions** - Clean, consistent interfaces across all components
2. **Type Safety** - Leverage Rust's type system for correctness
3. **Performance** - Zero-cost abstractions and efficient implementations
4. **Modularity** - Clear separation of concerns enables parallel development
5. **Extensibility** - Easy to add new models, backends, and features

## 📚 Documentation

- [Developer Guide](docs/developer_guide.md) - Getting started with development
- [API Reference](docs/api_reference.md) - Detailed API documentation
- [Architecture](ARCHITECTURE.md) - Deep dive into system design

## 🤝 Contributing

We welcome contributions! The clean architecture makes it easy to work on different components:

**High-Impact Areas:**
- Model implementations (add your favorite architecture)
- GPU backends (CUDA, Metal, ROCm)
- Inference optimizations
- Testing and validation

**Getting Started:**
1. Pick a component (runtime, inference, scheduler, etc.)
2. Read the module documentation
3. Look at existing implementations for patterns
4. Submit PRs with tests

## 📄 License

Apache-2.0 License - See LICENSE file for details.

---

**Built with Rust 🦀 • Designed for Performance ⚡ • Solid Abstractions 🏗️**
