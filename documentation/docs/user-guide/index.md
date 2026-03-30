# User Guide

This guide covers everything you need to know to use UniLLM effectively.

## Overview

UniLLM provides a unified interface for working with large language models. This guide covers:

- **Loading Models** - How to load models from different formats
- **Running Inference** - Forward pass and batching
- **Text Generation** - Autoregressive generation with sampling
- **Configuration** - Customizing model and generation settings

## Core Concepts

### The Model Trait

All models in UniLLM implement the `Model` trait:

```rust
pub trait Model: Send + Sync {
    type Config: ModelConfig;

    fn new(config: Self::Config) -> Result<Self>;
    fn from_weights(config: Self::Config, weights: ModelWeights) -> Result<Self>;
    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs>;
    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String>;
    fn to_device(&mut self, device: &Device) -> Result<()>;
}
```

This provides a consistent interface across all 47 supported architectures.

### Input/Output Types

UniLLM uses unified input/output types:

```rust
// Input types
enum ModelInputs {
    Text { input_ids, attention_mask, position_ids },
    Image { pixel_values, image_mask },
    Multimodal { input_ids, pixel_values, ... },
    Audio { input_features, attention_mask },
}

// Output types
enum ModelOutputs {
    Logits { logits, hidden_states },
    Embeddings { embeddings, pooled },
    Multimodal { text_logits, image_features, ... },
}
```

## Quick Reference

| Task | API |
|------|-----|
| Create model | `Model::new(config)` |
| Load weights | `WeightLoader::from_gguf(path)` |
| Forward pass | `model.forward(&inputs)` |
| Generate text | `model.generate(prompt, &gen_config)` |
| Move to GPU | `model.to_device(&Device::CUDA(0))` |

## Sections

<div class="grid cards" markdown>

-   [**Loading Models**](loading-models.md)

    Load models from GGUF, SafeTensors, or Ollama

-   [**Running Inference**](inference.md)

    Execute forward passes and batch processing

-   [**Text Generation**](generation.md)

    Generate text with sampling strategies

-   [**Configuration**](configuration.md)

    Customize model and generation settings

</div>
