# Gemma

Gemma is Google's family of lightweight, open-weight language models built from the same research as Gemini models.

## Overview

| Property | Value |
|----------|-------|
| **Architecture** | Decoder-only Transformer |
| **Parameters** | 2B, 7B (v1), 2B, 9B, 27B (v2) |
| **Context Length** | 8192 tokens |
| **Attention** | Multi-Query Attention |
| **Position Encoding** | RoPE |
| **Activation** | GELU |
| **Normalization** | RMSNorm |

## Quick Start

```rust
use unillm::models_v2::gemma::{GemmaModelV2, GemmaConfig};
use unillm::weight_loader_core::WeightLoader;
use unillm::{Model, GenerationConfig};

// Load model
let weights = WeightLoader::from_gguf("gemma-2b.gguf")?;
let config = GemmaConfig::from_gguf_metadata(weights.metadata())?;
let model = GemmaModelV2::from_weights(config, weights)?;

// Generate
let response = model.generate(
    "Write a haiku about programming:",
    &GenerationConfig::default(),
)?;
```

## Configuration

```rust
model_config!(GemmaConfig {
    vocab_size: usize = 256000,
    hidden_size: usize = 2048,
    intermediate_size: usize = 16384,
    num_hidden_layers: usize = 18,
    num_attention_heads: usize = 8,
    num_key_value_heads: usize = 1,
    head_dim: usize = 256,
    max_position_embeddings: usize = 8192,
    rope_theta: f32 = 10000.0,
    rms_norm_eps: f32 = 1e-6,
});
```

### Model Sizes

| Variant | hidden_size | num_layers | num_heads | kv_heads |
|---------|-------------|------------|-----------|----------|
| Gemma 2B | 2048 | 18 | 8 | 1 |
| Gemma 7B | 3072 | 28 | 16 | 16 |
| Gemma 2 2B | 2304 | 26 | 8 | 4 |
| Gemma 2 9B | 3584 | 42 | 16 | 8 |
| Gemma 2 27B | 4608 | 46 | 32 | 16 |

## Features

### Multi-Query Attention

Gemma 2B uses MQA for efficiency:

```rust
let config = GemmaConfig {
    num_attention_heads: 8,
    num_key_value_heads: 1,  // Single KV head
    ..Default::default()
};
```

### Large Head Dimension

Uses larger head dimension for better representation:

```rust
let config = GemmaConfig {
    head_dim: 256,  // Larger than typical 64 or 128
    ..Default::default()
};
```

### Extended Vocabulary

Large vocabulary for multilingual support:

```rust
let config = GemmaConfig {
    vocab_size: 256000,  // Very large vocabulary
    ..Default::default()
};
```

## Gemma Versions

### Gemma 1

- Initial release (2024)
- Sizes: 2B, 7B
- 8K context
- Strong for size

### Gemma 2

- Improved architecture
- Sizes: 2B, 9B, 27B
- Better instruction following
- Sliding window attention option

## Loading from Ollama

```rust
use unillm::ollama::OllamaRegistry;

// Gemma
let path = OllamaRegistry::pull("gemma:2b")?;

// Gemma 2
let path = OllamaRegistry::pull("gemma2:9b")?;

// Quantized
let path = OllamaRegistry::pull("gemma2:9b-q4_0")?;
```

## Generation Examples

### Basic Generation

```rust
let config = GenerationConfig {
    max_new_tokens: 256,
    temperature: 0.7,
    top_p: 0.95,
    ..Default::default()
};

let response = model.generate("Explain photosynthesis:", &config)?;
```

### Instruction Format

```rust
let prompt = "<start_of_turn>user
What is the capital of France?<end_of_turn>
<start_of_turn>model
";

let config = GenerationConfig {
    stop_sequences: vec!["<end_of_turn>".to_string()],
    ..Default::default()
};

let response = model.generate(prompt, &config)?;
```

## Memory Requirements

| Model | F32 | F16 | Q8_0 | Q4_K_M |
|-------|-----|-----|------|--------|
| 2B | 8 GB | 4 GB | 2 GB | 1.2 GB |
| 7B | 28 GB | 14 GB | 7 GB | 4 GB |
| 9B | 36 GB | 18 GB | 9 GB | 5 GB |
| 27B | 108 GB | 54 GB | 27 GB | 15 GB |

## Use Cases

### Ideal For

- **Edge deployment** - Small model size
- **Quick inference** - Fast on consumer hardware
- **Mobile applications** - 2B fits in memory
- **Learning/experimentation** - Good starter model

### Comparison

| Use Case | Recommended |
|----------|-------------|
| Smallest footprint | Gemma 2B |
| Best quality/size | Gemma 2 9B |
| Multi-turn chat | Gemma 2 27B |

## Best Practices

1. **Use Gemma 2** for better instruction following
2. **Use 2B variant** for resource-constrained environments
3. **Apply proper chat formatting** for conversational use
4. **Consider quantization** for deployment

## References

- [Gemma Technical Report](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf)
- [Gemma 2 Technical Report](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf)
- [Google Gemma](https://ai.google.dev/gemma)
