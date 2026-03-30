# Mixtral

Mixtral is Mistral AI's Mixture-of-Experts (MoE) model, achieving near GPT-4 performance with efficient sparse computation.

## Overview

| Property | Value |
|----------|-------|
| **Architecture** | Sparse MoE Transformer |
| **Parameters** | 46.7B total, 12.9B active |
| **Experts** | 8 experts per layer |
| **Active Experts** | 2 per token |
| **Context Length** | 32K tokens |
| **Attention** | Grouped Query (8 KV heads) |
| **Position Encoding** | RoPE |

## Quick Start

```rust
use unillm::models_v2::mixtral::{MixtralModelV2, MixtralConfig};
use unillm::weight_loader_core::WeightLoader;
use unillm::{Model, GenerationConfig};

// Load model
let weights = WeightLoader::from_gguf("mixtral-8x7b.gguf")?;
let config = MixtralConfig::from_gguf_metadata(weights.metadata())?;
let model = MixtralModelV2::from_weights(config, weights)?;

// Generate
let response = model.generate(
    "Explain how neural networks learn:",
    &GenerationConfig::default(),
)?;
```

## Configuration

```rust
model_config!(MixtralConfig {
    vocab_size: usize = 32000,
    hidden_size: usize = 4096,
    intermediate_size: usize = 14336,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 32,
    num_key_value_heads: usize = 8,
    num_experts: usize = 8,
    num_experts_per_tok: usize = 2,
    max_position_embeddings: usize = 32768,
    sliding_window: usize = 4096,
    rope_theta: f32 = 1000000.0,
    rms_norm_eps: f32 = 1e-5,
    router_aux_loss_coef: f32 = 0.02,
});
```

## How MoE Works

### Architecture

```
Input
  │
  ▼
┌─────────────────────────────────┐
│ Self-Attention (GQA)            │
└─────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────┐
│ Router Network                   │
│ ├─ Expert 0 ─┐                  │
│ ├─ Expert 1 ─┼─ Select Top-2    │
│ ├─ Expert 2 ─┘                  │
│ ├─ Expert 3                     │
│ ├─ Expert 4                     │
│ ├─ Expert 5                     │
│ ├─ Expert 6                     │
│ └─ Expert 7                     │
└─────────────────────────────────┘
  │
  ▼
Output
```

### Router

The router selects which experts process each token:

```rust
fn forward_moe(&self, hidden: &Tensor) -> Result<Tensor> {
    // Compute router logits
    let router_logits = ops_fn::linear(hidden, &self.gate, None)?;

    // Select top-k experts
    let (weights, indices) = ops_fn::topk(
        &router_logits,
        self.config.num_experts_per_tok,  // k=2
        -1
    )?;

    // Normalize weights
    let weights = ops_fn::softmax(&weights, -1)?;

    // Process through selected experts and combine
    // ...
}
```

### Benefits

- **Efficiency**: Only ~25% of parameters active per token
- **Capacity**: Total knowledge of 47B parameters
- **Speed**: Similar latency to 13B dense model
- **Quality**: Performance approaching GPT-4

## Loading from Ollama

```rust
use unillm::ollama::OllamaRegistry;

// Full model
let path = OllamaRegistry::pull("mixtral:8x7b")?;

// Instruct version
let path = OllamaRegistry::pull("mixtral:8x7b-instruct")?;

// Quantized (recommended)
let path = OllamaRegistry::pull("mixtral:8x7b-q4_0")?;
```

## Generation Examples

### Chat Format

```rust
let prompt = "<s>[INST] Explain quantum computing [/INST]";

let config = GenerationConfig {
    max_new_tokens: 512,
    temperature: 0.7,
    top_p: 0.9,
    ..Default::default()
};

let response = model.generate(prompt, &config)?;
```

### Multi-turn

```rust
let prompt = r#"<s>[INST] What is Python? [/INST] Python is a high-level programming language...</s>
[INST] How do I install it? [/INST]"#;

let response = model.generate(prompt, &config)?;
```

## Memory Requirements

| Format | VRAM | RAM |
|--------|------|-----|
| F16 | 90 GB | 90 GB |
| Q8_0 | 50 GB | 50 GB |
| Q6_K | 40 GB | 40 GB |
| Q4_K_M | 28 GB | 28 GB |
| Q3_K_M | 22 GB | 22 GB |

For consumer GPUs, use Q4_K_M or lower quantization.

## Performance

### Benchmarks

| Benchmark | Mixtral 8x7B | LLaMA 2 70B | GPT-3.5 |
|-----------|--------------|-------------|---------|
| MMLU | 70.6 | 68.9 | 70.0 |
| HellaSwag | 84.4 | 85.3 | 85.5 |
| Arc-C | 66.4 | 64.6 | 85.2 |
| HumanEval | 40.2 | 29.9 | 48.1 |

### Inference Speed

Mixtral is faster than dense models with similar quality:

| Model | Active Params | Speed* |
|-------|---------------|--------|
| Mixtral 8x7B | 12.9B | 30 tok/s |
| LLaMA 2 13B | 13B | 28 tok/s |
| LLaMA 2 70B | 70B | 8 tok/s |

*Approximate, hardware dependent

## Expert Analysis

### Expert Specialization

Experts tend to specialize in different domains:

| Expert | Tendency |
|--------|----------|
| 0 | General knowledge |
| 1 | Technical/scientific |
| 2 | Code and formatting |
| 3 | Creative writing |
| ... | ... |

### Load Balancing

The router includes auxiliary loss to prevent expert collapse:

```rust
let config = MixtralConfig {
    router_aux_loss_coef: 0.02,  // Encourages balanced routing
    ..Default::default()
};
```

## Use Cases

### Ideal For

- **Complex reasoning** - Benefits from expert diversity
- **Multi-domain tasks** - Experts cover different areas
- **Production** - Better quality/compute ratio
- **Long context** - 32K native context

### When to Use

| Scenario | Recommendation |
|----------|----------------|
| Limited VRAM (<16GB) | Use Mistral 7B Q4 |
| 24GB VRAM | Use Mixtral Q4 |
| 48GB+ VRAM | Use Mixtral Q8 or F16 |
| Maximum quality | Use Mixtral Instruct |

## Best Practices

1. **Use quantization** - Q4_K_M maintains good quality
2. **Proper prompting** - Use [INST] format for Instruct
3. **Balance load** - MoE benefits from diverse prompts
4. **Monitor memory** - All experts must fit in memory

## Variants

### Mixtral 8x7B Base

Pre-trained model for fine-tuning.

### Mixtral 8x7B Instruct

Instruction-tuned for chat and assistance.

### Mixtral 8x22B

Larger MoE variant (not yet supported).

## References

- [Mixtral Paper](https://arxiv.org/abs/2401.04088)
- [Mistral AI Blog](https://mistral.ai/news/mixtral-of-experts/)
- [MoE Survey](https://arxiv.org/abs/2209.01667)
