# API Reference

Complete API documentation for UniLLM's core abstractions and components.

## Overview

UniLLM is built on three core abstraction layers:

| Layer | Purpose | Key Types |
|-------|---------|-----------|
| [**TensorCore**](tensor-core.md) | Tensor operations & device management | `Tensor`, `Device`, `ops_fn` |
| [**ModelCore**](model-core.md) | Model interfaces & configuration | `Model`, `ModelConfig`, `model_config!` |
| [**WeightLoader**](weight-loader.md) | Format-agnostic weight loading | `WeightLoader`, `ModelWeights` |
| [**Inference**](inference.md) | Inference pipeline & sampling | `InferencePipeline`, `Sampler` |

## Quick Reference

### Creating a Model

```rust
use unillm::models_v2::llama::{LlamaModelV2, LlamaConfig};
use unillm::Model;

// With default config
let model = LlamaModelV2::new(LlamaConfig::default())?;

// With weights
let weights = WeightLoader::from_gguf("model.gguf")?;
let model = LlamaModelV2::from_weights(config, weights)?;
```

### Running Inference

```rust
use unillm::{ModelInputs, ModelOutputs};

let inputs = ModelInputs::Text {
    input_ids: tensor,
    attention_mask: None,
    position_ids: None,
};

let outputs = model.forward(&inputs)?;
```

### Generating Text

```rust
use unillm::GenerationConfig;

let gen_config = GenerationConfig {
    max_new_tokens: 100,
    temperature: 0.7,
    ..Default::default()
};

let response = model.generate("Hello", &gen_config)?;
```

### Tensor Operations

```rust
use unillm::tensor_core::{ops_fn, DataType, Device};

let a = ops_fn::zeros(&[2, 3], DataType::Float32, &Device::CPU)?;
let b = ops_fn::ones(&[3, 4], DataType::Float32, &Device::CPU)?;
let c = ops_fn::matmul(&a, &b)?;  // [2, 4]
```

## Type Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                        Traits                                │
├─────────────────────────────────────────────────────────────┤
│  Model            - Universal model interface                │
│  ModelConfig      - Configuration interface                  │
│  TensorOps        - Tensor operation interface               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Structs                               │
├─────────────────────────────────────────────────────────────┤
│  Tensor           - Universal tensor type                    │
│  Device           - Hardware device (CPU, CUDA, Metal)       │
│  ModelWeights     - Weight container                         │
│  GenerationConfig - Generation parameters                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Enums                                 │
├─────────────────────────────────────────────────────────────┤
│  ModelInputs      - Text, Image, Multimodal, Audio           │
│  ModelOutputs     - Logits, Embeddings, Multimodal           │
│  DataType         - F32, F16, BF16, I64, etc.                │
└─────────────────────────────────────────────────────────────┘
```

## Module Structure

```rust
// Core types
pub use tensor_core::{Tensor, Device, DataType, ops_fn};
pub use model_core::{Model, ModelConfig, ModelInputs, ModelOutputs};
pub use weight_loader_core::{WeightLoader, ModelWeights};

// Model implementations
pub use models_v2::llama::{LlamaModelV2, LlamaConfig};
pub use models_v2::qwen::{QwenModelV2, QwenConfig};
pub use models_v2::mixtral::{MixtralModelV2, MixtralConfig};
// ... 44 more model implementations

// Inference
pub use inference::{InferencePipeline, GenerationConfig};
pub use sampler::Sampler;
pub use tokenizer::Tokenizer;
```

## Error Handling

UniLLM uses `anyhow::Result` for error handling:

```rust
use anyhow::Result;

fn run_model() -> Result<String> {
    let model = LlamaModelV2::new(LlamaConfig::default())?;
    let response = model.generate("Hello", &GenerationConfig::default())?;
    Ok(response)
}
```

## Sections

<div class="grid cards" markdown>

-   [**TensorCore**](tensor-core.md)

    Tensor operations, devices, and data types

-   [**ModelCore**](model-core.md)

    Model trait, configuration, and I/O types

-   [**WeightLoader**](weight-loader.md)

    Weight loading from various formats

-   [**Inference Pipeline**](inference.md)

    End-to-end inference and sampling

</div>
