# Phi

Phi is Microsoft's family of small language models that achieve strong performance despite their compact size, making them ideal for edge deployment.

## Overview

| Property | Value |
|----------|-------|
| **Architecture** | Decoder-only Transformer |
| **Parameters** | 1.3B (Phi-1), 2.7B (Phi-2), 3.8B (Phi-3) |
| **Context Length** | 2K-128K (version dependent) |
| **Attention** | Multi-head / Grouped Query |
| **Position Encoding** | RoPE |
| **Activation** | GELU (Phi-1/2), SwiGLU (Phi-3) |
| **Normalization** | LayerNorm (Phi-1/2), RMSNorm (Phi-3) |

## Quick Start

```rust
use unillm::models_v2::phi::{PhiModelV2, PhiConfig};
use unillm::weight_loader_core::WeightLoader;
use unillm::{Model, GenerationConfig};

// Load model
let weights = WeightLoader::from_gguf("phi-3-mini.gguf")?;
let config = PhiConfig::from_gguf_metadata(weights.metadata())?;
let model = PhiModelV2::from_weights(config, weights)?;

// Generate
let response = model.generate(
    "Write a Python function to calculate factorial:",
    &GenerationConfig::default(),
)?;
```

## Configuration

```rust
model_config!(PhiConfig {
    vocab_size: usize = 32064,
    hidden_size: usize = 3072,
    intermediate_size: usize = 8192,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 32,
    num_key_value_heads: usize = 32,
    max_position_embeddings: usize = 4096,
    rope_theta: f32 = 10000.0,
    rms_norm_eps: f32 = 1e-5,
    partial_rotary_factor: f32 = 0.5,
});
```

### Model Sizes

| Variant | hidden_size | num_layers | num_heads | Context |
|---------|-------------|------------|-----------|---------|
| Phi-1 | 2048 | 24 | 32 | 2K |
| Phi-1.5 | 2048 | 24 | 32 | 2K |
| Phi-2 | 2560 | 32 | 32 | 2K |
| Phi-3 Mini | 3072 | 32 | 32 | 4K-128K |
| Phi-3 Small | 4096 | 32 | 32 | 8K-128K |
| Phi-3 Medium | 5120 | 40 | 40 | 4K-128K |

## Features

### Partial Rotary Embedding

Phi uses partial RoPE application:

```rust
let config = PhiConfig {
    partial_rotary_factor: 0.5,  // Apply RoPE to half of head dim
    ..Default::default()
};
```

### Extended Context (Phi-3)

Long context with RoPE scaling:

```rust
let config = PhiConfig {
    max_position_embeddings: 131072,  // 128K context
    rope_theta: 10000.0,
    ..Default::default()
};
```

## Phi Versions

### Phi-1 (1.3B)

- Code-focused training
- Strong at Python generation
- 2K context

### Phi-2 (2.7B)

- General purpose
- Textbooks and synthetic data
- Better reasoning than Phi-1

### Phi-3 (3.8B-14B)

- Latest release (2024)
- Up to 128K context
- Competitive with much larger models
- Available in Mini, Small, Medium

### Phi-3-Vision

Multimodal variant. See [Phi-3-Vision documentation](../vision-language/phi3-vision.md).

## Loading from Ollama

```rust
use unillm::ollama::OllamaRegistry;

// Phi-2
let path = OllamaRegistry::pull("phi:2.7b")?;

// Phi-3
let path = OllamaRegistry::pull("phi3:mini")?;
let path = OllamaRegistry::pull("phi3:medium")?;

// Quantized
let path = OllamaRegistry::pull("phi3:mini-q4_0")?;
```

## Generation Examples

### Code Generation

```rust
let config = GenerationConfig {
    temperature: 0.2,
    max_new_tokens: 256,
    ..Default::default()
};

let prompt = "def fibonacci(n):
    '''Return the nth Fibonacci number'''";

let response = model.generate(prompt, &config)?;
```

### Instruction Following (Phi-3)

```rust
let prompt = "<|user|>
Explain quantum entanglement in simple terms.
<|end|>
<|assistant|>
";

let config = GenerationConfig {
    stop_sequences: vec!["<|end|>".to_string()],
    ..Default::default()
};

let response = model.generate(prompt, &config)?;
```

## Memory Requirements

| Model | F32 | F16 | Q8_0 | Q4_K_M |
|-------|-----|-----|------|--------|
| Phi-2 | 11 GB | 5.5 GB | 2.8 GB | 1.6 GB |
| Phi-3 Mini | 15 GB | 7.5 GB | 3.8 GB | 2.2 GB |
| Phi-3 Medium | 56 GB | 28 GB | 14 GB | 8 GB |

## Use Cases

### Ideal For

- **Edge/mobile deployment** - Very small footprint
- **Code completion** - Strong coding ability
- **Resource-constrained** - Runs on CPU
- **Quick prototyping** - Fast inference

### Performance Comparison

Phi-3 Mini often matches or exceeds models 3-5x its size:

| Task | Phi-3 Mini | LLaMA 2 7B | Mistral 7B |
|------|------------|------------|------------|
| MMLU | 68.8 | 45.3 | 60.1 |
| HumanEval | 59.1 | 12.8 | 30.5 |
| GSM8K | 75.6 | 14.6 | 52.2 |

## Best Practices

1. **Use Phi-3** for latest capabilities
2. **Lower temperature** for code generation
3. **Use 128K context** carefully (memory intensive)
4. **Quantize aggressively** - Phi handles Q4 well

## References

- [Phi-1 Paper](https://arxiv.org/abs/2306.11644)
- [Phi-2 Blog](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)
- [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219)
