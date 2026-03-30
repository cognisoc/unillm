# DeepSeek-MoE

DeepSeek-MoE is a Mixture-of-Experts model with fine-grained expert segmentation and shared expert isolation for improved efficiency.

## Overview

| Property | Value |
|----------|-------|
| **Architecture** | Fine-grained MoE Transformer |
| **Parameters** | 16B total, ~2.8B active |
| **Experts** | 64 routed + 2 shared |
| **Active Experts** | 6 per token |
| **Context Length** | 4096 tokens |
| **Attention** | Grouped Query |
| **Position Encoding** | RoPE |

## Quick Start

```rust
use unillm::models_v2::deepseek_moe::{DeepSeekMoEModelV2, DeepSeekMoEConfig};
use unillm::weight_loader_core::WeightLoader;
use unillm::{Model, GenerationConfig};

// Load model
let weights = WeightLoader::from_gguf("deepseek-moe-16b.gguf")?;
let config = DeepSeekMoEConfig::from_gguf_metadata(weights.metadata())?;
let model = DeepSeekMoEModelV2::from_weights(config, weights)?;

// Generate
let response = model.generate(
    "Explain the concept of entropy:",
    &GenerationConfig::default(),
)?;
```

## Configuration

```rust
model_config!(DeepSeekMoEConfig {
    vocab_size: usize = 102400,
    hidden_size: usize = 2048,
    intermediate_size: usize = 1408,
    num_hidden_layers: usize = 28,
    num_attention_heads: usize = 16,
    num_key_value_heads: usize = 16,
    num_experts: usize = 64,
    num_shared_experts: usize = 2,
    num_experts_per_tok: usize = 6,
    moe_intermediate_size: usize = 1408,
    max_position_embeddings: usize = 4096,
    rope_theta: f32 = 10000.0,
    rms_norm_eps: f32 = 1e-6,
    first_k_dense_replace: usize = 1,
});
```

## Key Innovations

### Fine-Grained Experts

Instead of 8 large experts, DeepSeek-MoE uses 64 smaller experts:

```
Standard MoE:        DeepSeek-MoE:
8 Large Experts  →   64 Fine-Grained Experts
                     + 2 Shared Experts
```

Benefits:
- **More specialization** - Experts can focus on narrow domains
- **Better routing** - More options for token assignment
- **Efficiency** - Smaller experts are faster

### Shared Expert Isolation

Two experts are always active for every token:

```rust
fn forward_moe(&self, hidden: &Tensor) -> Result<Tensor> {
    // Shared experts always process
    let shared_out = self.forward_shared_experts(hidden)?;

    // Route to top-k routed experts
    let routed_out = self.forward_routed_experts(hidden)?;

    // Combine outputs
    ops_fn::add(&shared_out, &routed_out)
}
```

### First Layer Dense

The first layer uses dense FFN instead of MoE:

```rust
let config = DeepSeekMoEConfig {
    first_k_dense_replace: 1,  // First 1 layer(s) are dense
    ..Default::default()
};
```

## Architecture Details

### Expert Structure

```
Layer Input
    │
    ├──────────────────────────┐
    │                          │
    ▼                          ▼
┌─────────┐        ┌──────────────────┐
│ Shared  │        │ Router (Top-6)   │
│ Expert 1│        │ → 64 Experts     │
│ Expert 2│        └──────────────────┘
└─────────┘                │
    │                      │
    └──────────┬───────────┘
               │
               ▼
           Output
```

### Efficiency Analysis

| Model | Total Params | Active Params | Ratio |
|-------|--------------|---------------|-------|
| DeepSeek-MoE 16B | 16.4B | 2.8B | 17% |
| Mixtral 8x7B | 46.7B | 12.9B | 28% |
| Dense 7B | 7B | 7B | 100% |

## Generation Examples

### Basic Generation

```rust
let config = GenerationConfig {
    max_new_tokens: 256,
    temperature: 0.7,
    top_p: 0.9,
    ..Default::default()
};

let response = model.generate(
    "What are the applications of machine learning?",
    &config,
)?;
```

### Chat Format

```rust
let prompt = "User: What is deep learning?\nAssistant:";

let config = GenerationConfig {
    stop_sequences: vec!["\nUser:".to_string()],
    ..Default::default()
};

let response = model.generate(prompt, &config)?;
```

## Memory Requirements

| Format | Memory |
|--------|--------|
| F16 | 33 GB |
| Q8_0 | 17 GB |
| Q4_K_M | 10 GB |
| Q3_K_M | 8 GB |

## Performance

### Benchmarks

| Benchmark | DeepSeek-MoE 16B | LLaMA 2 7B | Mixtral 8x7B |
|-----------|------------------|------------|--------------|
| MMLU | 45.0 | 45.3 | 70.6 |
| HellaSwag | 77.1 | 77.2 | 84.4 |
| HumanEval | 26.8 | 12.8 | 40.2 |

### Efficiency

DeepSeek-MoE 16B matches 7B dense models while using:
- **40% less** compute per token
- **Similar** memory to 7B model during inference

## Use Cases

### Ideal For

- **Efficiency-focused** - Maximum quality per FLOP
- **Research** - Novel MoE architecture
- **Fine-tuning** - Good base for domain adaptation

### Comparison

| Priority | Best Choice |
|----------|-------------|
| Smallest active params | DeepSeek-MoE |
| Highest quality | Mixtral 8x7B |
| Simplest deployment | Dense 7B |

## Advanced Topics

### Expert Load Balancing

DeepSeek-MoE uses auxiliary losses for balanced routing:

```rust
// Load balance across experts
let aux_loss = compute_load_balance_loss(&router_probs)?;
```

### Capacity Factor

Controls maximum tokens per expert:

```rust
// Expert capacity = (tokens / num_experts) * capacity_factor
let capacity_factor = 1.25;  // 25% extra capacity
```

## Best Practices

1. **Use Q4 quantization** - Good balance of quality/memory
2. **Monitor expert usage** - Check for collapsed experts
3. **Batch efficiently** - MoE benefits from larger batches
4. **Consider dense** - For simpler deployments

## References

- [DeepSeek-MoE Paper](https://arxiv.org/abs/2401.06066)
- [DeepSeek AI](https://www.deepseek.com/)
- [MoE Survey](https://arxiv.org/abs/2209.01667)
