# Architecture Overview

UniLLM is built on a three-layer abstraction system that provides consistency, extensibility, and performance across 47 model architectures.

## Design Philosophy

### Goals

1. **Unified Interface** - Same API for all models, regardless of architecture
2. **Hardware Agnostic** - Same code runs on CPU, CUDA, and Metal
3. **Format Agnostic** - Load models from GGUF, SafeTensors, or PyTorch
4. **Minimal Overhead** - Direct tensor operations without runtime abstraction costs
5. **Easy Extension** - Add new models with minimal boilerplate

### Non-Goals

- Training (inference only)
- Dynamic computation graphs
- Automatic differentiation

## The Three Layers

UniLLM's architecture consists of three core abstraction layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Application                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: WeightLoaderCore                                   │
│  ├─ GGUF Loading (with dequantization)                      │
│  ├─ SafeTensors Loading                                     │
│  └─ Ollama Registry Integration                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: ModelCore                                          │
│  ├─ Model trait (universal interface)                       │
│  ├─ ModelConfig trait (configuration)                       │
│  ├─ model_config! macro (auto implementations)              │
│  └─ ModelInputs/ModelOutputs (unified I/O)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: TensorCore                                         │
│  ├─ Tensor type (universal tensor)                          │
│  ├─ Device enum (CPU, CUDA, Metal)                          │
│  ├─ ops_fn module (functional operations)                   │
│  └─ TensorOps trait (backend abstraction)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Backend: Candle (CPU) / CUDA / Metal                        │
└─────────────────────────────────────────────────────────────┘
```

### Layer 1: TensorCore

The foundation layer providing device-agnostic tensor operations.

- **Tensor**: Universal tensor type wrapping backend-specific implementations
- **Device**: Hardware abstraction (CPU, CUDA, Metal)
- **ops_fn**: Functional interface for all tensor operations
- **TensorOps**: Trait implemented by each backend

[Learn more about TensorCore](three-layers.md#tensorcore)

### Layer 2: ModelCore

The model abstraction layer providing consistent interfaces.

- **Model trait**: Universal interface implemented by all models
- **ModelConfig trait**: Configuration interface
- **model_config! macro**: Automatic trait implementations
- **ModelInputs/Outputs**: Unified input/output types

[Learn more about ModelCore](three-layers.md#modelcore)

### Layer 3: WeightLoaderCore

Format-agnostic weight loading with automatic dequantization.

- **WeightLoader**: Load from any supported format
- **ModelWeights**: Container with tensor access
- **Metadata extraction**: Configuration from GGUF metadata
- **Ollama integration**: Download from Ollama registry

[Learn more about WeightLoaderCore](three-layers.md#weightloadercore)

## Model Categories

UniLLM supports 47 model architectures across 10 categories:

| Category | Models | Examples |
|----------|--------|----------|
| Core LLMs | 5 | LLaMA, Qwen, Gemma, Phi, Mistral |
| GPT Family | 4 | GPT-2, GPT-J, GPT-NeoX, OPT |
| Code Models | 3 | StarCoder, CodeLlama, CodeGen |
| MoE Models | 6 | Mixtral, DeepSeek-MoE, Dbrx, Grok |
| RWKV Family | 3 | RWKV-4, RWKV-6, RecurrentGemma |
| Embedding | 4 | BERT, RoBERTa, XLM-RoBERTa, MPNet |
| Vision-Language | 8 | CLIP, LLaVA, Qwen2-VL, CogVLM |
| Multimodal | 3 | Flamingo, BLIP-2, PaLI |
| Audio | 4 | Whisper, Wav2Vec2, HuBERT, Encodec |
| Specialized | 7 | T5, BART, Falcon, Bloom, etc. |

## Code Organization

```
crates/
├── runtime/                    # Main inference runtime
│   ├── src/
│   │   ├── lib.rs             # Crate root, public exports
│   │   ├── tensor_core.rs     # Layer 1: TensorCore
│   │   ├── model_core.rs      # Layer 2: ModelCore
│   │   ├── weight_loader_core.rs  # Layer 3: WeightLoaderCore
│   │   ├── inference.rs       # Inference pipeline
│   │   ├── tokenizer.rs       # Tokenization
│   │   ├── ollama.rs          # Ollama integration
│   │   └── models_v2/         # Model implementations
│   │       ├── mod.rs         # Model exports
│   │       ├── traits.rs      # Shared traits
│   │       ├── llama.rs       # LLaMA implementation
│   │       ├── qwen.rs        # Qwen implementation
│   │       └── ...            # 45 more models
│   └── Cargo.toml
├── inference/                  # High-level inference engine
├── kv/                         # KV cache management
├── scheduler/                  # Request scheduling
└── kernels/                    # GPU kernels (future)
```

## Data Flow

A typical inference request flows through the system:

```
1. User Request
   └─> "Hello, world!"

2. Tokenization
   └─> [1, 15496, 11, 1917, 0]

3. Model Input
   └─> ModelInputs::Text { input_ids, ... }

4. Forward Pass (per layer)
   └─> Embedding → Attention → FFN → Normalize

5. Model Output
   └─> ModelOutputs::Logits { logits, ... }

6. Sampling
   └─> Apply temperature, top-p, top-k
   └─> Sample next token

7. Decode
   └─> "Hello, world! How"

8. Repeat 4-7 until max_tokens or EOS
```

## Next Steps

- [Three-Layer Architecture](three-layers.md) - Deep dive into each layer
- [Model Implementation Pattern](model-pattern.md) - How models are implemented
- [Design Decisions](design-decisions.md) - Key architectural choices
