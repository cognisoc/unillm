# LLaMA

LLaMA (Large Language Model Meta AI) is Meta's family of open-weight language models, known for strong performance and extensive community support.

## Overview

| Property | Value |
|----------|-------|
| **Architecture** | Decoder-only Transformer |
| **Parameters** | 7B, 13B, 30B, 65B (v1), 7B-70B (v2/v3) |
| **Context Length** | 2K-4K (v1/v2), up to 128K (v3) |
| **Attention** | Multi-head (v1), Grouped Query (v2+) |
| **Position Encoding** | RoPE (Rotary Position Embedding) |
| **Activation** | SwiGLU |
| **Normalization** | RMSNorm |

## Quick Start

```rust
use unillm::models_v2::llama::{LlamaModelV2, LlamaConfig};
use unillm::weight_loader_core::WeightLoader;
use unillm::{Model, GenerationConfig};

// Load model
let weights = WeightLoader::from_gguf("llama-7b.gguf")?;
let config = LlamaConfig::from_gguf_metadata(weights.metadata())?;
let model = LlamaModelV2::from_weights(config, weights)?;

// Generate
let response = model.generate(
    "Explain quantum computing:",
    &GenerationConfig::default(),
)?;
println!("{}", response);
```

## Configuration

```rust
model_config!(LlamaConfig {
    vocab_size: usize = 32000,
    hidden_size: usize = 4096,
    intermediate_size: usize = 11008,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 32,
    num_key_value_heads: usize = 32,
    max_position_embeddings: usize = 2048,
    rope_theta: f32 = 10000.0,
    rope_scaling: Option<String> = None,
    rms_norm_eps: f32 = 1e-6,
    pad_token_id: i64 = 0,
    bos_token_id: i64 = 1,
    eos_token_id: i64 = 2,
});
```

### Model Sizes

| Variant | hidden_size | num_layers | num_heads | kv_heads |
|---------|-------------|------------|-----------|----------|
| LLaMA 7B | 4096 | 32 | 32 | 32 |
| LLaMA 13B | 5120 | 40 | 40 | 40 |
| LLaMA 2 7B | 4096 | 32 | 32 | 32 |
| LLaMA 2 13B | 5120 | 40 | 40 | 40 |
| LLaMA 2 70B | 8192 | 80 | 64 | 8 |
| LLaMA 3 8B | 4096 | 32 | 32 | 8 |
| LLaMA 3 70B | 8192 | 80 | 64 | 8 |

## Features

### Grouped Query Attention (GQA)

LLaMA 2 70B and LLaMA 3 use GQA for memory efficiency:

```rust
// GQA configuration
let config = LlamaConfig {
    num_attention_heads: 64,    // Query heads
    num_key_value_heads: 8,     // KV heads (8:1 ratio)
    ..Default::default()
};
```

### RoPE (Rotary Position Embedding)

Position information is encoded using rotary embeddings:

```rust
// Standard RoPE
let config = LlamaConfig {
    rope_theta: 10000.0,
    rope_scaling: None,
    ..Default::default()
};

// Extended context (LLaMA 3)
let config = LlamaConfig {
    rope_theta: 500000.0,  // Higher theta for longer context
    rope_scaling: Some("linear".to_string()),
    ..Default::default()
};
```

### SwiGLU Activation

The FFN uses SwiGLU (Swish-Gated Linear Unit):

```rust
// In MLP forward pass
let gate = ops_fn::linear(&hidden, &gate_proj, None)?;
let up = ops_fn::linear(&hidden, &up_proj, None)?;
let gate = ops_fn::silu(&gate)?;  // Swish activation
let hidden = ops_fn::mul(&gate, &up)?;
let output = ops_fn::linear(&hidden, &down_proj, None)?;
```

## Loading from Ollama

```rust
use unillm::ollama::OllamaRegistry;

// LLaMA 2
let path = OllamaRegistry::pull("llama2:7b")?;

// LLaMA 3
let path = OllamaRegistry::pull("llama3:8b")?;

// Quantized versions
let path = OllamaRegistry::pull("llama3:8b-q4_0")?;
```

## Generation Examples

### Chat Completion

```rust
let prompt = r#"<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Hello!<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

"#;

let config = GenerationConfig {
    max_new_tokens: 256,
    temperature: 0.7,
    top_p: 0.9,
    stop_sequences: vec!["<|eot_id|>".to_string()],
    ..Default::default()
};

let response = model.generate(prompt, &config)?;
```

### Code Generation

```rust
let config = GenerationConfig {
    temperature: 0.2,  // Lower for code
    max_new_tokens: 512,
    ..Default::default()
};

let response = model.generate(
    "Write a Python function to sort a list:",
    &config,
)?;
```

## Memory Requirements

| Model | F32 | F16 | Q8_0 | Q4_K_M |
|-------|-----|-----|------|--------|
| 7B | 28 GB | 14 GB | 7 GB | 4 GB |
| 13B | 52 GB | 26 GB | 13 GB | 7 GB |
| 70B | 280 GB | 140 GB | 70 GB | 40 GB |

## Variants

### LLaMA 1
- Original release (2023)
- Sizes: 7B, 13B, 30B, 65B
- Context: 2048 tokens

### LLaMA 2
- Improved training
- Sizes: 7B, 13B, 70B
- Context: 4096 tokens
- 70B uses GQA

### LLaMA 3
- Latest release (2024)
- Sizes: 8B, 70B
- Context: 8K standard, up to 128K
- All sizes use GQA

### CodeLlama
- Code-specialized LLaMA 2
- See [CodeLlama documentation](../code/codellama.md)

## Best Practices

1. **Use quantized models** for consumer hardware
2. **Match prompt format** to training (especially chat models)
3. **Use GQA models** (70B, LLaMA 3) for memory efficiency
4. **Set appropriate context length** to avoid memory issues

## References

- [LLaMA Paper](https://arxiv.org/abs/2302.13971)
- [LLaMA 2 Paper](https://arxiv.org/abs/2307.09288)
- [Meta AI LLaMA](https://ai.meta.com/llama/)
