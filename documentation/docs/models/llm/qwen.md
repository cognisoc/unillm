# Qwen

Qwen (Tongyi Qianwen) is Alibaba's family of large language models, known for strong multilingual capabilities and competitive performance.

## Overview

| Property | Value |
|----------|-------|
| **Architecture** | Decoder-only Transformer |
| **Parameters** | 0.5B, 1.8B, 4B, 7B, 14B, 72B |
| **Context Length** | 8K-128K (version dependent) |
| **Attention** | Grouped Query Attention |
| **Position Encoding** | RoPE with NTK-aware scaling |
| **Activation** | SwiGLU |
| **Normalization** | RMSNorm |

## Quick Start

```rust
use unillm::models_v2::qwen::{QwenModelV2, QwenConfig};
use unillm::weight_loader_core::WeightLoader;
use unillm::{Model, GenerationConfig};

// Load model
let weights = WeightLoader::from_gguf("qwen-7b.gguf")?;
let config = QwenConfig::from_gguf_metadata(weights.metadata())?;
let model = QwenModelV2::from_weights(config, weights)?;

// Generate
let response = model.generate(
    "Explain machine learning:",
    &GenerationConfig::default(),
)?;
```

## Configuration

```rust
model_config!(QwenConfig {
    vocab_size: usize = 151936,
    hidden_size: usize = 4096,
    intermediate_size: usize = 11008,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 32,
    num_key_value_heads: usize = 32,
    max_position_embeddings: usize = 8192,
    rope_theta: f32 = 10000.0,
    rms_norm_eps: f32 = 1e-6,
    use_sliding_window: bool = false,
    sliding_window: usize = 32768,
});
```

### Model Sizes

| Variant | hidden_size | num_layers | num_heads | kv_heads |
|---------|-------------|------------|-----------|----------|
| Qwen 0.5B | 1024 | 24 | 16 | 16 |
| Qwen 1.8B | 2048 | 24 | 16 | 16 |
| Qwen 4B | 2560 | 40 | 20 | 20 |
| Qwen 7B | 4096 | 32 | 32 | 32 |
| Qwen 14B | 5120 | 40 | 40 | 40 |
| Qwen 72B | 8192 | 80 | 64 | 8 |

## Features

### Extended Vocabulary

Qwen uses a large vocabulary optimized for multilingual support:

```rust
let config = QwenConfig {
    vocab_size: 151936,  // Large vocab for multilingual
    ..Default::default()
};
```

### NTK-Aware RoPE Scaling

Dynamic position interpolation for extended context:

```rust
let config = QwenConfig {
    rope_theta: 1000000.0,  // High theta for long context
    max_position_embeddings: 32768,
    ..Default::default()
};
```

### Sliding Window Attention

Optional sliding window for efficiency:

```rust
let config = QwenConfig {
    use_sliding_window: true,
    sliding_window: 32768,
    ..Default::default()
};
```

## Qwen Versions

### Qwen 1.5

- Improved base model
- Better instruction following
- Extended context to 32K

### Qwen 2

- Latest release (2024)
- Sizes: 0.5B to 72B
- 128K context for larger models
- Strong coding abilities

### Qwen 2.5

- Most recent version
- Improved reasoning
- Better multilingual support

## Loading from Ollama

```rust
use unillm::ollama::OllamaRegistry;

// Qwen 2
let path = OllamaRegistry::pull("qwen2:7b")?;

// Qwen 2.5
let path = OllamaRegistry::pull("qwen2.5:7b")?;

// Quantized
let path = OllamaRegistry::pull("qwen2:7b-q4_0")?;
```

## Generation Examples

### Chat Format

```rust
let prompt = r#"<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
"#;

let config = GenerationConfig {
    max_new_tokens: 256,
    temperature: 0.7,
    stop_sequences: vec!["<|im_end|>".to_string()],
    ..Default::default()
};

let response = model.generate(prompt, &config)?;
```

### Multilingual

```rust
// Chinese
let response = model.generate("用中文解释人工智能", &config)?;

// Japanese
let response = model.generate("人工知能について説明してください", &config)?;
```

## Memory Requirements

| Model | F32 | F16 | Q8_0 | Q4_K_M |
|-------|-----|-----|------|--------|
| 0.5B | 2 GB | 1 GB | 0.5 GB | 0.3 GB |
| 7B | 28 GB | 14 GB | 7 GB | 4 GB |
| 14B | 56 GB | 28 GB | 14 GB | 8 GB |
| 72B | 288 GB | 144 GB | 72 GB | 40 GB |

## Specialized Variants

### Qwen-VL

Vision-language variant. See [Qwen2-VL documentation](../vision-language/qwen2-vl.md).

### CodeQwen

Code-specialized variant with strong programming abilities.

### Qwen-Audio

Audio understanding variant.

## Best Practices

1. **Use ChatML format** for chat applications
2. **Enable long context** with NTK scaling for documents
3. **Consider Qwen 2.5** for latest improvements
4. **Use quantized models** for deployment

## References

- [Qwen Technical Report](https://arxiv.org/abs/2309.16609)
- [Qwen 2 Technical Report](https://arxiv.org/abs/2407.10671)
- [Qwen GitHub](https://github.com/QwenLM/Qwen)
