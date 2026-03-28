# UniLLM

**High-performance Rust-based LLM inference engine with solid abstractions**

UniLLM is a modern, high-performance inference runtime built in Rust, designed with clean architecture and solid abstractions. It provides a unified interface for running large language models across different architectures and deployment targets.

---

## Key Features

<div class="grid cards" markdown>

-   :material-layers-triple:{ .lg .middle } **Solid Architecture**

    ---

    Clean three-layer abstraction system: TensorCore, ModelCore, and WeightLoaderCore

-   :material-robot:{ .lg .middle } **45+ Model Architectures**

    ---

    Support for LLMs, MoE, Vision-Language, Audio models with consistent interfaces

-   :material-lightning-bolt:{ .lg .middle } **Performance Ready**

    ---

    Device abstraction for CPU/GPU, async runtime, zero-cost abstractions

-   :material-puzzle:{ .lg .middle } **Easy Extension**

    ---

    Add new models with minimal boilerplate using the `model_config!` macro

</div>

---

## Supported Models

UniLLM supports **45+ model architectures** across 10 categories:

| Category | Models |
|----------|--------|
| **Core LLMs** | LLaMA, Qwen, Gemma, Phi, DeepSeek, Mistral, Mixtral |
| **GPT Family** | GPT-2, GPT-J, GPT-NeoX, OPT, BLOOM, MPT |
| **Code Models** | StarCoder, CodeLlama |
| **MoE Models** | DeepSeek-MoE, DBRX, Grok, Arctic, Jamba |
| **RWKV/Linear** | RWKV-4, RWKV-6, RecurrentGemma |
| **Vision-Language** | Qwen2-VL, Phi-3-Vision, InternVL, CogVLM, LLaVA, CLIP |
| **Audio/Speech** | Wav2Vec2, HuBERT, MusicGen, Encodec, Whisper |
| **Additional** | Yi, Falcon, Baichuan, InternLM, ChatGLM, BERT, T5, Mamba |

---

## Quick Start

### Installation

```bash
git clone https://github.com/anthropics/unillm.git
cd unillm
cargo build --release
```

### Basic Usage

```rust
use unillm::models_v2::llama::{LlamaModelV2, LlamaConfig};
use unillm::{Model, ModelInputs, GenerationConfig};

// Create model with default configuration
let config = LlamaConfig::default();
let model = LlamaModelV2::new(config)?;

// Generate text
let gen_config = GenerationConfig::default();
let response = model.generate("Hello, world!", &gen_config)?;
println!("{}", response);
```

### Using with Ollama

```bash
# Run inference with a model from Ollama registry
cargo run --bin test_ollama -p runtime -- --model tinyllama
```

---

## Architecture Overview

UniLLM uses a **three-layer abstraction system**:

```
┌─────────────────────────────────────────────────────────────────┐
│                           UniLLM                                 │
│                                                                  │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   TensorCore    │ │   ModelCore     │ │WeightLoaderCore │   │
│  │                 │ │                 │ │                 │   │
│  │ • Tensor        │ │ • Model trait   │ │ • WeightLoader  │   │
│  │ • TensorOps     │ │ • ModelConfig   │ │ • ModelWeights  │   │
│  │ • Device        │ │ • model_config! │ │ • Format detect │   │
│  │ • ops_fn        │ │ • ModelInputs   │ │ • GGUF, SafeT.  │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
│                                                                  │
│  ────────────────────────────────────────────────────────────   │
│                      Model Implementations                       │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │ LlamaV2 │ │ QwenV2  │ │MixtralV2│ │ LLaVAV2 │ │WhisperV2│  │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

[Learn more about the architecture :material-arrow-right:](architecture/index.md)

---

## Current Status

!!! success "Production Ready Components"
    - 47 model architectures implemented
    - 166 passing tests
    - GGUF and SafeTensors weight loading
    - Ollama registry integration
    - Full LLaMA inference pipeline

!!! warning "In Development"
    - KV caching for efficient generation
    - GPU acceleration (CUDA, Metal)
    - Production HTTP server

---

## Next Steps

<div class="grid cards" markdown>

-   [**Getting Started**](getting-started/index.md)

    Learn how to install and run your first model

-   [**User Guide**](user-guide/index.md)

    Detailed guides for loading models and running inference

-   [**Model Catalog**](models/index.md)

    Browse all supported model architectures

-   [**API Reference**](api-reference/index.md)

    Complete API documentation

</div>
