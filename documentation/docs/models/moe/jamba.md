# Jamba

Jamba is AI21's hybrid architecture combining Transformer attention layers, Mamba state-space layers, and Mixture-of-Experts for efficient long-context processing.

## Overview

| Property | Value |
|----------|-------|
| **Architecture** | Hybrid Mamba-Transformer MoE |
| **Parameters** | 52B total, 12B active |
| **Context Length** | 256K tokens |
| **Layer Mix** | 1:7 Attention:Mamba ratio |
| **Experts** | 16 experts, top-2 active |
| **Position Encoding** | RoPE (attention layers) |

## Quick Start

```rust
use unillm::models_v2::jamba::{JambaModelV2, JambaConfig};
use unillm::weight_loader_core::WeightLoader;
use unillm::{Model, GenerationConfig};

// Load model
let weights = WeightLoader::from_gguf("jamba-v0.1.gguf")?;
let config = JambaConfig::from_gguf_metadata(weights.metadata())?;
let model = JambaModelV2::from_weights(config, weights)?;

// Generate with long context
let response = model.generate(
    &long_document,
    &GenerationConfig::default(),
)?;
```

## Configuration

```rust
model_config!(JambaConfig {
    vocab_size: usize = 65536,
    hidden_size: usize = 4096,
    intermediate_size: usize = 14336,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 32,
    num_key_value_heads: usize = 8,
    num_experts: usize = 16,
    num_experts_per_tok: usize = 2,
    mamba_d_state: usize = 16,
    mamba_d_conv: usize = 4,
    mamba_expand: usize = 2,
    attn_layer_period: usize = 8,
    attn_layer_offset: usize = 4,
    max_position_embeddings: usize = 262144,
    rms_norm_eps: f32 = 1e-6,
});
```

## Hybrid Architecture

### Layer Structure

Jamba interleaves Mamba and Attention layers:

```
Layer 0:  Mamba
Layer 1:  Mamba
Layer 2:  Mamba
Layer 3:  Mamba
Layer 4:  Attention + MoE  ← Every 8th layer
Layer 5:  Mamba
Layer 6:  Mamba
Layer 7:  Mamba
Layer 8:  Mamba
Layer 9:  Mamba
Layer 10: Mamba
Layer 11: Mamba
Layer 12: Attention + MoE  ← Every 8th layer
...
```

### Why Hybrid?

| Component | Strength | Weakness |
|-----------|----------|----------|
| Attention | Global context | O(n²) memory |
| Mamba | O(n) memory | Local patterns |
| MoE | Capacity | Complexity |

Jamba combines strengths:
- **Mamba** for efficient sequence processing
- **Attention** for global reasoning (every 8 layers)
- **MoE** for increased capacity at attention layers

## Mamba Layers

### State Space Model

Mamba layers use selective state spaces:

```rust
struct MambaLayer {
    conv1d: Tensor,      // 1D convolution
    in_proj: Tensor,     // Input projection
    out_proj: Tensor,    // Output projection
    dt_proj: Tensor,     // Time step projection
    A: Tensor,           // State matrix
    D: Tensor,           // Skip connection
}

fn forward_mamba(&self, hidden: &Tensor) -> Result<Tensor> {
    // Project input
    let xz = ops_fn::linear(hidden, &self.in_proj, None)?;
    let (x, z) = xz.split_at(hidden_size)?;

    // Convolution
    let x = ops_fn::conv1d(&x, &self.conv1d, None, 1, self.d_conv/2)?;
    let x = ops_fn::silu(&x)?;

    // Selective scan (SSM)
    let y = self.selective_scan(&x)?;

    // Gate and project
    let z = ops_fn::silu(&z)?;
    let output = ops_fn::mul(&y, &z)?;
    ops_fn::linear(&output, &self.out_proj, None)
}
```

### Benefits

- **Linear complexity** - O(n) vs O(n²) for attention
- **Constant memory** - Fixed state size regardless of sequence
- **Fast inference** - Recurrent computation

## Memory Efficiency

### Comparison at 256K Context

| Model | Memory for Context |
|-------|-------------------|
| Transformer (256K) | ~256 GB KV cache |
| Jamba (256K) | ~4 GB KV cache |

The 1:7 attention ratio dramatically reduces memory:
- Only 4 attention layers (every 8th)
- Mamba layers have constant memory
- MoE only at attention layers

### Active Parameters

```
Total Parameters:     52B
Active per token:     12B (23%)
Mamba state size:     O(1)
Attention KV cache:   O(n) but only 4 layers
```

## Loading from Ollama

```rust
use unillm::ollama::OllamaRegistry;

// Jamba v0.1
let path = OllamaRegistry::pull("jamba:latest")?;

// Quantized
let path = OllamaRegistry::pull("jamba:q4_0")?;
```

## Generation Examples

### Long Document Processing

```rust
let document = std::fs::read_to_string("long_document.txt")?;
let prompt = format!(
    "Summarize the following document:\n\n{}\n\nSummary:",
    document
);

let config = GenerationConfig {
    max_new_tokens: 512,
    temperature: 0.3,
    ..Default::default()
};

let summary = model.generate(&prompt, &config)?;
```

### RAG with Large Context

```rust
// Can fit many documents in 256K context
let context = documents.join("\n\n---\n\n");
let prompt = format!(
    "Context:\n{}\n\nQuestion: {}\nAnswer:",
    context, question
);

let answer = model.generate(&prompt, &config)?;
```

## Memory Requirements

| Format | Memory |
|--------|--------|
| F16 | ~100 GB |
| Q8_0 | ~55 GB |
| Q4_K_M | ~32 GB |
| Q3_K_M | ~25 GB |

## Performance

### Benchmarks

| Benchmark | Jamba | Mixtral 8x7B | LLaMA 2 70B |
|-----------|-------|--------------|-------------|
| MMLU | 67.4 | 70.6 | 68.9 |
| HellaSwag | 87.1 | 84.4 | 85.3 |
| WinoGrande | 82.5 | 81.2 | 83.7 |

### Throughput (256K context)

| Model | Tokens/sec |
|-------|------------|
| Jamba | 30-40 |
| Transformer (if fits) | 5-10 |

## Use Cases

### Ideal For

- **Very long documents** - 256K context native
- **RAG applications** - Many documents in context
- **Streaming** - Efficient incremental processing
- **Memory-constrained** - Less KV cache needed

### When to Consider

| Scenario | Recommendation |
|----------|----------------|
| <32K context | Use standard transformer |
| 32K-256K context | Jamba shines |
| Maximum quality | Use Mixtral/LLaMA 70B |
| Maximum context | Jamba (256K native) |

## Best Practices

1. **Leverage long context** - Jamba is designed for it
2. **Use quantization** - Q4 works well
3. **Batch efficiently** - Mamba has good batching
4. **Consider hybrid prompts** - Mix short and long inputs

## Advanced Topics

### Mamba State Caching

For multi-turn conversations:

```rust
// State can be cached and reused
let state = model.get_mamba_state()?;
// ... later
model.set_mamba_state(&state)?;
```

### Layer Selection

Different layer types for different tasks:

```rust
// Attention layers handle global reasoning
// Mamba layers handle local patterns
// MoE increases capacity at attention points
```

## References

- [Jamba Paper](https://arxiv.org/abs/2403.19887)
- [Mamba Paper](https://arxiv.org/abs/2312.00752)
- [AI21 Labs](https://www.ai21.com/)
