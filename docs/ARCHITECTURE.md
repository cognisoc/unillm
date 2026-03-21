# UniLLM Architecture Documentation

## Overview

UniLLM implements a layered architecture with solid abstractions that provide clean separation of concerns and enable scalable development. This document describes the core design principles and abstraction layers.

## Design Principles

1. **Single Source of Truth**: One type for each concept (one Tensor type, one Model trait, etc.)
2. **Clean Interfaces**: Abstract away implementation details behind well-defined traits
3. **Device Agnostic**: CPU/GPU/Metal dispatch handled transparently
4. **Format Agnostic**: Support multiple weight formats through unified interfaces
5. **Composable**: Small, focused abstractions that work together seamlessly

## Core Abstraction Layers

### Layer 1: TensorCore - Unified Tensor System

**Purpose**: Provides a single, unified tensor abstraction that works across all devices and operations.

**Key Components**:
```rust
// Single tensor type for the entire system
pub struct Tensor {
    data: Arc<dyn TensorStorage>,  // Device-agnostic storage
    shape: Vec<usize>,             // Tensor dimensions
    dtype: DataType,               // Data type (F32, F16, etc.)
    device: Device,                // Device location (CPU, CUDA, Metal)
}

// Unified operations interface
pub trait TensorOps {
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor>;
    fn layer_norm(&self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor>;
    // ... all tensor operations through single interface
}

// Device abstraction
pub enum Device {
    CPU,
    CUDA(usize),  // GPU ID
    Metal(usize), // GPU ID
}
```

**Benefits**:
- No type confusion - one `Tensor` type everywhere
- Automatic device dispatch - operations work on any device
- Memory safety - `Arc<dyn TensorStorage>` handles lifetimes
- Clean functional interface via `ops_fn` module

### Layer 2: ModelCore - Universal Model Interface

**Purpose**: Provides a single model interface that works for all 18+ supported architectures.

**Key Components**:
```rust
// Universal model interface
pub trait Model: Send + Sync {
    type Config: ModelConfig;

    fn new(config: Self::Config) -> Result<Self>;
    fn from_weights(config: Self::Config, weights: ModelWeights) -> Result<Self>;
    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs>;
    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String>;
    fn to_device(&mut self, device: &Device) -> Result<()>;
}

// Configuration interface with automatic implementations
pub trait ModelConfig {
    fn architecture(&self) -> &str;
    fn vocab_size(&self) -> usize;
    fn hidden_size(&self) -> usize;
    fn num_layers(&self) -> usize;
}

// Unified I/O types
pub enum ModelInputs {
    Text { input_ids: Tensor, attention_mask: Option<Tensor>, ... },
    Image { pixel_values: Tensor, ... },
    Multimodal { input_ids: Tensor, pixel_values: Option<Tensor>, ... },
    Audio { input_features: Tensor, ... },
}

pub enum ModelOutputs {
    Logits { logits: Tensor, hidden_states: Option<Tensor> },
    Embeddings { embeddings: Tensor, pooled: Option<Tensor> },
    Multimodal { text_logits: Option<Tensor>, image_features: Option<Tensor>, ... },
}
```

**Magic: `model_config!` Macro**:
```rust
// This single macro call...
model_config!(LlamaConfig {
    vocab_size: usize = 32000,
    hidden_size: usize = 4096,
    num_hidden_layers: usize = 32,
    // ... other fields
});

// ...generates all of this automatically:
impl Default for LlamaConfig { /* ... */ }
impl ModelConfig for LlamaConfig { /* ... */ }
impl Clone for LlamaConfig { /* ... */ }
impl Send for LlamaConfig { /* ... */ }
impl Sync for LlamaConfig { /* ... */ }
```

**Benefits**:
- Single interface for all 18+ model families
- Consistent configuration system
- Automatic trait implementations
- Type-safe model loading and inference

### Layer 3: WeightLoaderCore - Format-Agnostic Weight Loading

**Purpose**: Loads model weights from any format (SafeTensors, GGUF, PyTorch) into unified `ModelWeights`.

**Key Components**:
```rust
// Format-agnostic weight loading
pub struct WeightLoader;

impl WeightLoader {
    pub fn from_safetensors<P: AsRef<Path>>(path: P) -> Result<ModelWeights>;
    pub fn from_gguf<P: AsRef<Path>>(path: P) -> Result<ModelWeights>;
    pub fn from_pytorch<P: AsRef<Path>>(path: P) -> Result<ModelWeights>;
    pub fn auto_detect<P: AsRef<Path>>(path: P) -> Result<ModelWeights>;
}

// Unified weight container
pub struct ModelWeights {
    tensors: HashMap<String, Tensor>,  // weight name -> tensor
    metadata: WeightMetadata,          // format info, etc.
}
```

**Benefits**:
- Models don't care about weight format
- Automatic format detection
- Consistent weight access interface
- Lazy loading support

## Model Implementation Pattern

All 18 model families follow the same clean pattern:

```rust
// 1. Define configuration with macro
model_config!(YourModelConfig {
    vocab_size: usize = 50000,
    hidden_size: usize = 1024,
    num_hidden_layers: usize = 24,
    // ... model-specific fields
});

// 2. Define model structure
pub struct YourModelV2 {
    config: YourModelConfig,
    device: Device,
    embed_tokens: Tensor,
    layers: Vec<YourModelLayer>,
    norm: Tensor,
    lm_head: Tensor,
}

// 3. Implement universal Model trait
impl Model for YourModelV2 {
    type Config = YourModelConfig;

    fn new(config: YourModelConfig) -> Result<Self> {
        // Create model with zero tensors
    }

    fn from_weights(config: YourModelConfig, weights: ModelWeights) -> Result<Self> {
        // Load actual weights into model
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        // Model-specific forward pass using ops_fn
    }

    // ... other trait methods
}
```

## Supported Model Families

All models use identical patterns:

| Family | Models | Status |
|--------|--------|---------|
| **Llama** | Llama, Llama2, Code Llama, Llama3 | Implemented |
| **Qwen** | Qwen, Qwen1.5, Qwen2, QwenVL | Implemented |
| **Gemma** | Gemma-2B, Gemma-7B | Implemented |
| **Phi** | Phi-1, Phi-1.5, Phi-2, Phi-3 | Implemented |
| **DeepSeek** | DeepSeek-Coder, DeepSeek-Chat | Implemented |
| **Yi** | Yi-6B, Yi-34B, Yi-Chat, Yi-Coder | Implemented |
| **Baichuan** | Baichuan-7B, Baichuan-13B | Implemented |
| **InternLM** | InternLM, InternLM2 | Implemented |
| **ChatGLM** | ChatGLM-6B, ChatGLM2-6B, ChatGLM3 | Implemented |
| **Falcon** | Falcon-7B, Falcon-40B | Implemented |
| **BERT** | BERT, RoBERTa, DeBERTa | Implemented |
| **T5** | T5, UL2, Flan-T5 | Implemented |
| **Whisper** | Whisper (all sizes) | Implemented |
| **CLIP** | CLIP vision-language | Implemented |
| **LLaVA** | LLaVA multimodal | Implemented |
| **Mamba** | Mamba state-space models | Implemented |
| **MiniCPM** | MiniCPM efficient models | Implemented |

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                           UniLLM                                │
│                                                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
│  │   TensorCore    │ │   ModelCore     │ │WeightLoaderCore │  │
│  │                 │ │                 │ │                 │  │
│  │ • Tensor        │ │ • Model trait   │ │ • WeightLoader  │  │
│  │ • TensorOps     │ │ • ModelConfig   │ │ • ModelWeights  │  │
│  │ • Device        │ │ • model_config! │ │ • Format detect │  │
│  │ • ops_fn        │ │ • ModelInputs   │ │ • SafeTensors   │  │
│  └─────────────────┘ │ • ModelOutputs  │ │ • GGUF, PyTorch │  │
│                      └─────────────────┘ └─────────────────┘  │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│                      Model Implementations                      │
│                                                                 │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │ LlamaV2 │ │ QwenV2  │ │ GemmaV2 │ │  PhiV2  │ │   ...   │  │
│  │         │ │         │ │         │ │         │ │         │  │
│  │All use  │ │All use  │ │All use  │ │All use  │ │All use  │  │
│  │same     │ │same     │ │same     │ │same     │ │same     │  │
│  │pattern  │ │pattern  │ │pattern  │ │pattern  │ │pattern  │  │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘  │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│                    Application Layer                            │
│                                                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
│  │ InferencePipeline│ │   Tokenizer     │ │    Sampler      │  │
│  │                 │ │                 │ │                 │  │
│  │ • End-to-end    │ │ • Text ↔ tokens │ │ • Greedy        │  │
│  │ • Generation    │ │ • Vocab mgmt    │ │ • Temperature   │  │
│  │ • Builder       │ │ • Batch support │ │ • Top-p/Top-k   │  │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Model Loading Flow
```
1. Configuration → model_config!(YourConfig { ... })
2. Weights → WeightLoader::from_safetensors("model.safetensors")
3. Model → YourModelV2::from_weights(config, weights)
4. Ready → model.forward(&inputs) or model.generate("text", &config)
```

### Inference Flow
```
1. Text → tokenizer.encode("Hello world") → tokens: [1, 15496, 3186, 2]
2. Tensor → ops_fn::zeros(&[1, 4], DataType::Int64, &Device::CPU)
3. ModelInputs → ModelInputs::Text { input_ids: tensor, ... }
4. Forward → model.forward(&inputs) → ModelOutputs::Logits { logits, ... }
5. Sampling → sampler.sample_greedy(&logits) → next_token: u32
6. Decoding → tokenizer.decode(&generated_tokens) → "Hello world!"
```

## Key Design Decisions

### 1. Single Tensor Type
**Decision**: One `Tensor` struct for the entire system
**Alternative**: Separate types for different devices/dtypes
**Rationale**: Eliminates type confusion and conversion overhead

### 2. Trait-Based Operations
**Decision**: `TensorOps` trait with `ops_fn` functional interface
**Alternative**: Methods directly on `Tensor` struct
**Rationale**: Allows device-specific optimizations while keeping clean API

### 3. Universal Model Interface
**Decision**: Single `Model` trait for all architectures
**Alternative**: Architecture-specific traits
**Rationale**: Enables polymorphism and consistent patterns

### 4. Automatic Configuration Generation
**Decision**: `model_config!` macro generates trait implementations
**Alternative**: Manual trait implementations for each model
**Rationale**: Eliminates boilerplate and ensures consistency

### 5. Format-Agnostic Weight Loading
**Decision**: Unified `ModelWeights` container
**Alternative**: Format-specific weight types
**Rationale**: Models don't need to know about weight formats

## Performance Characteristics

### Memory Management
- **Zero-Copy**: Tensors use `Arc<dyn TensorStorage>` for efficient sharing
- **Device-Aware**: Memory allocated on appropriate device
- **Lazy Loading**: Weights loaded only when needed
- **Reference Counting**: Automatic cleanup when tensors go out of scope

### Dispatch Overhead
- **Single Dispatch**: `TensorOps` trait uses dynamic dispatch once per operation
- **Device Caching**: Device context reused across operations
- **Batch Operations**: Multiple operations combined where possible

### Compilation
- **Monomorphization**: Rust generates optimized code for each model type
- **Link-Time Optimization**: LTO enables cross-crate optimizations
- **Zero-Cost Abstractions**: Trait dispatch compiled away where possible

## Extension Points

### Adding New Model Architectures
1. Define configuration with `model_config!`
2. Implement model structure
3. Implement `Model` trait
4. Export from `models_v2/mod.rs`

### Adding New Tensor Operations
1. Add method to `TensorOps` trait
2. Implement in `CpuTensorOpsImpl` (and GPU variants)
3. Add functional wrapper in `ops_fn`

### Adding New Weight Formats
1. Implement format parser
2. Add method to `WeightLoader`
3. Update `auto_detect` logic

### Adding New Devices
1. Add variant to `Device` enum
2. Implement `TensorOps` for new device
3. Add device detection logic

## Future Optimizations

### Compiled Kernels
- **Target**: Replace dynamic dispatch with compiled kernels
- **Approach**: Use macro system to generate device-specific code
- **Benefit**: Eliminate runtime dispatch overhead

### Memory Pool
- **Target**: Pre-allocate memory pools per device
- **Approach**: Custom allocator integrated with `TensorStorage`
- **Benefit**: Reduce allocation overhead

### Operation Fusion
- **Target**: Combine multiple operations into single kernels
- **Approach**: Graph optimization at compile time
- **Benefit**: Reduce memory bandwidth and latency

---

This architecture provides a solid foundation that scales from single models to production deployments while maintaining clean abstractions and consistent patterns across all components.