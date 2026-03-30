# Mistral

Mistral is a high-performance 7B language model that introduced sliding window attention for efficient long-context processing.

## Overview

| Property | Value |
|----------|-------|
| **Architecture** | Decoder-only Transformer |
| **Parameters** | 7B |
| **Context Length** | 8K (standard), 32K (extended) |
| **Attention** | Sliding Window + Grouped Query |
| **Position Encoding** | RoPE |
| **Activation** | SwiGLU |
| **Normalization** | RMSNorm |

## Quick Start

```rust
use unillm::models_v2::mistral::{MistralModelV2, MistralConfig};
use unillm::weight_loader_core::WeightLoader;
use unillm::{Model, GenerationConfig};

// Load model
let weights = WeightLoader::from_gguf("mistral-7b.gguf")?;
let config = MistralConfig::from_gguf_metadata(weights.metadata())?;
let model = MistralModelV2::from_weights(config, weights)?;

// Generate
let response = model.generate(
    "Explain the theory of relativity:",
    &GenerationConfig::default(),
)?;
```

## Configuration

```rust
model_config!(MistralConfig {
    vocab_size: usize = 32000,
    hidden_size: usize = 4096,
    intermediate_size: usize = 14336,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 32,
    num_key_value_heads: usize = 8,
    max_position_embeddings: usize = 32768,
    sliding_window: usize = 4096,
    rope_theta: f32 = 10000.0,
    rms_norm_eps: f32 = 1e-5,
});
```

### Model Specifications

| Property | Value |
|----------|-------|
| hidden_size | 4096 |
| num_layers | 32 |
| num_attention_heads | 32 |
| num_key_value_heads | 8 |
| head_dim | 128 |
| intermediate_size | 14336 |

## Features

### Sliding Window Attention

Efficient attention for long sequences:

```rust
let config = MistralConfig {
    sliding_window: 4096,  // Each token attends to 4K previous tokens
    max_position_embeddings: 32768,  // But supports 32K total
    ..Default::default()
};
```

The sliding window allows:
- **Memory efficiency** - Fixed attention window size
- **Long context** - Process sequences beyond window with rolling cache
- **Speed** - Less computation per attention layer

### Grouped Query Attention

Mistral uses 8 KV heads for 32 query heads (4:1 ratio):

```rust
let config = MistralConfig {
    num_attention_heads: 32,
    num_key_value_heads: 8,  // 4:1 GQA ratio
    ..Default::default()
};
```

### Larger Intermediate Size

More capacity in FFN:

```rust
let config = MistralConfig {
    hidden_size: 4096,
    intermediate_size: 14336,  // 3.5x hidden (vs 2.67x in LLaMA)
    ..Default::default()
};
```

## Mistral Variants

### Mistral 7B

- Base model (2023)
- 8K sliding window
- Strong general performance

### Mistral 7B Instruct

- Instruction-tuned
- Chat-optimized
- Same architecture

### Mixtral 8x7B

MoE version. See [Mixtral documentation](../moe/mixtral.md).

## Loading from Ollama

```rust
use unillm::ollama::OllamaRegistry;

// Base model
let path = OllamaRegistry::pull("mistral:7b")?;

// Instruct
let path = OllamaRegistry::pull("mistral:7b-instruct")?;

// Quantized
let path = OllamaRegistry::pull("mistral:7b-q4_0")?;
```

## Generation Examples

### Chat Format

```rust
let prompt = "<s>[INST] What is machine learning? [/INST]";

let config = GenerationConfig {
    max_new_tokens: 256,
    temperature: 0.7,
    stop_sequences: vec!["</s>".to_string()],
    ..Default::default()
};

let response = model.generate(prompt, &config)?;
```

### Multi-turn Conversation

```rust
let prompt = r#"<s>[INST] Hi! [/INST] Hello! How can I help?</s>
[INST] What's 2+2? [/INST]"#;

let response = model.generate(prompt, &config)?;
```

### System Prompts (Mistral Instruct v0.2+)

```rust
let prompt = r#"<s>[INST] <<SYS>>
You are a helpful coding assistant.
<</SYS>>

Write a Python hello world [/INST]"#;

let response = model.generate(prompt, &config)?;
```

## Memory Requirements

| Format | Memory |
|--------|--------|
| F32 | 28 GB |
| F16 | 14 GB |
| Q8_0 | 7 GB |
| Q4_K_M | 4 GB |

## Performance

Mistral 7B outperforms LLaMA 2 13B on most benchmarks:

| Benchmark | Mistral 7B | LLaMA 2 7B | LLaMA 2 13B |
|-----------|------------|------------|-------------|
| MMLU | 60.1 | 45.3 | 54.8 |
| HellaSwag | 81.3 | 77.2 | 80.7 |
| Arc-C | 55.5 | 45.9 | 49.4 |
| HumanEval | 30.5 | 12.8 | 18.3 |

## Use Cases

### Ideal For

- **General chat** - Strong instruction following
- **Code assistance** - Good at coding tasks
- **Long documents** - Sliding window handles well
- **Production** - Good quality/size ratio

### Comparison with Alternatives

| Use Case | Best Choice |
|----------|-------------|
| Smallest size | Phi-3 Mini |
| Best 7B quality | Mistral 7B |
| Longer context | Mistral 7B (32K) |
| MoE efficiency | Mixtral 8x7B |

## Best Practices

1. **Use Instruct version** for chat/assistant use
2. **Use proper prompt format** - [INST] tags matter
3. **Leverage sliding window** for long documents
4. **Consider Mixtral** for higher quality needs

## References

- [Mistral 7B Paper](https://arxiv.org/abs/2310.06825)
- [Mistral AI](https://mistral.ai/)
- [Mistral GitHub](https://github.com/mistralai)
