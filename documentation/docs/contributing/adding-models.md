# Adding New Models

This guide explains how to implement support for new model architectures in UniLLM.

## Overview

Adding a new model involves:

1. Understanding the model architecture
2. Creating the configuration
3. Implementing the model structure
4. Implementing the forward pass
5. Adding weight loading
6. Testing
7. Documentation

## Prerequisites

- Read the [Architecture Overview](../architecture/index.md)
- Understand the [Model Pattern](../architecture/model-pattern.md)
- Have access to model weights and reference implementation

## Step 1: Create Configuration

Use the `model_config!` macro:

```rust
// crates/runtime/src/models_v2/my_model.rs

use crate::model_config;
use super::traits::*;

model_config!(MyModelConfig {
    vocab_size: usize = 32000,
    hidden_size: usize = 4096,
    intermediate_size: usize = 11008,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 32,
    num_key_value_heads: usize = 32,
    max_position_embeddings: usize = 2048,
    rms_norm_eps: f32 = 1e-6,
    rope_theta: f32 = 10000.0,
    // Add model-specific fields
});
```

### Finding Configuration Values

- **HuggingFace** - Check `config.json` in model repo
- **GGUF** - Extract from metadata
- **Paper** - Reference the original paper

## Step 2: Define Model Structure

Create the model components:

```rust
// Attention mechanism
struct MyAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

// Feed-forward network
struct MyMLP {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
}

// Transformer layer
struct MyLayer {
    attention: MyAttention,
    mlp: MyMLP,
    input_norm: Tensor,
    post_attn_norm: Tensor,
}

// Main model
pub struct MyModelV2 {
    config: MyModelConfig,
    embed_tokens: Tensor,
    layers: Vec<MyLayer>,
    norm: Tensor,
    lm_head: Tensor,
    device: Device,
}
```

## Step 3: Implement Model Trait

```rust
impl Model for MyModelV2 {
    type Config = MyModelConfig;

    fn new(config: Self::Config) -> Result<Self> {
        // Create with dummy weights (for testing)
        let device = Device::CPU;

        let embed_tokens = ops_fn::zeros(
            &[config.vocab_size, config.hidden_size],
            DataType::Float32,
            &device,
        )?;

        let layers = (0..config.num_hidden_layers)
            .map(|_| Self::create_layer(&config, &device))
            .collect::<Result<Vec<_>>>()?;

        let norm = ops_fn::ones(
            &[config.hidden_size],
            DataType::Float32,
            &device,
        )?;

        let lm_head = ops_fn::zeros(
            &[config.vocab_size, config.hidden_size],
            DataType::Float32,
            &device,
        )?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            device,
        })
    }

    fn from_weights(config: Self::Config, weights: ModelWeights) -> Result<Self> {
        // Load actual weights
        // See "Weight Loading" section below
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        // See "Forward Pass" section below
    }

    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        // Text generation implementation
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn memory_requirements(&self) -> MemoryRequirements {
        // Calculate memory needs
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        // Move tensors to device
    }
}
```

## Step 4: Implement Forward Pass

### Main Forward

```rust
fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
    let input_ids = match inputs {
        ModelInputs::Text { input_ids, .. } => input_ids,
        _ => return Err(anyhow::anyhow!("Expected text input")),
    };

    // Token embeddings
    let mut hidden = ops_fn::embedding(input_ids, &self.embed_tokens)?;

    // Get sequence length for position encoding
    let seq_len = hidden.shape()[1];

    // Process through layers
    for layer in &self.layers {
        hidden = self.forward_layer(&hidden, layer, seq_len)?;
    }

    // Final normalization
    hidden = ops_fn::rms_norm(&hidden, &self.norm, self.config.rms_norm_eps)?;

    // Output projection
    let logits = ops_fn::linear(&hidden, &self.lm_head, None)?;

    Ok(ModelOutputs::Logits {
        logits,
        hidden_states: None,
    })
}
```

### Layer Forward

```rust
fn forward_layer(&self, hidden: &Tensor, layer: &MyLayer, seq_len: usize) -> Result<Tensor> {
    // Pre-attention normalization
    let normed = ops_fn::rms_norm(hidden, &layer.input_norm, self.config.rms_norm_eps)?;

    // Self-attention
    let attn_out = self.forward_attention(&normed, &layer.attention, seq_len)?;

    // Residual connection
    let hidden = ops_fn::add(hidden, &attn_out)?;

    // Post-attention normalization
    let normed = ops_fn::rms_norm(&hidden, &layer.post_attn_norm, self.config.rms_norm_eps)?;

    // MLP
    let mlp_out = self.forward_mlp(&normed, &layer.mlp)?;

    // Residual connection
    ops_fn::add(&hidden, &mlp_out)
}
```

### Attention

```rust
fn forward_attention(&self, hidden: &Tensor, attn: &MyAttention, seq_len: usize) -> Result<Tensor> {
    let (batch, seq, _) = (hidden.shape()[0], hidden.shape()[1], hidden.shape()[2]);

    // Project Q, K, V
    let q = ops_fn::linear(hidden, &attn.q_proj, None)?;
    let k = ops_fn::linear(hidden, &attn.k_proj, None)?;
    let v = ops_fn::linear(hidden, &attn.v_proj, None)?;

    // Reshape for multi-head attention
    let q = ops_fn::reshape(&q, &[batch, seq, attn.num_heads, attn.head_dim])?;
    let k = ops_fn::reshape(&k, &[batch, seq, attn.num_kv_heads, attn.head_dim])?;
    let v = ops_fn::reshape(&v, &[batch, seq, attn.num_kv_heads, attn.head_dim])?;

    // Transpose to [batch, heads, seq, dim]
    let q = ops_fn::transpose(&q, 1, 2)?;
    let k = ops_fn::transpose(&k, 1, 2)?;
    let v = ops_fn::transpose(&v, 1, 2)?;

    // Apply RoPE (if model uses it)
    let (q, k) = self.apply_rope(&q, &k, seq_len)?;

    // Handle GQA if num_kv_heads != num_heads
    let (k, v) = if attn.num_kv_heads != attn.num_heads {
        self.expand_kv(&k, &v, attn.num_heads / attn.num_kv_heads)?
    } else {
        (k, v)
    };

    // Create causal mask
    let mask = self.create_causal_mask(seq_len)?;

    // Scaled dot-product attention
    let attn_out = ops_fn::attention(&q, &k, &v, Some(&mask))?;

    // Reshape back
    let attn_out = ops_fn::transpose(&attn_out, 1, 2)?;
    let attn_out = ops_fn::reshape(&attn_out, &[batch, seq, attn.num_heads * attn.head_dim])?;

    // Output projection
    ops_fn::linear(&attn_out, &attn.o_proj, None)
}
```

### MLP

```rust
fn forward_mlp(&self, hidden: &Tensor, mlp: &MyMLP) -> Result<Tensor> {
    // SwiGLU activation
    let gate = ops_fn::linear(hidden, &mlp.gate_proj, None)?;
    let up = ops_fn::linear(hidden, &mlp.up_proj, None)?;

    let gate = ops_fn::silu(&gate)?;
    let hidden = ops_fn::mul(&gate, &up)?;

    ops_fn::linear(&hidden, &mlp.down_proj, None)
}
```

## Step 5: Weight Loading

Map weight names from source format:

```rust
fn from_weights(config: Self::Config, weights: ModelWeights) -> Result<Self> {
    let device = Device::CPU;

    // Load embeddings
    let embed_tokens = weights.require("model.embed_tokens.weight")?.clone();

    // Load layers
    let mut layers = Vec::new();
    for i in 0..config.num_hidden_layers {
        layers.push(MyLayer {
            attention: MyAttention {
                q_proj: weights.require(&format!(
                    "model.layers.{}.self_attn.q_proj.weight", i
                ))?.clone(),
                k_proj: weights.require(&format!(
                    "model.layers.{}.self_attn.k_proj.weight", i
                ))?.clone(),
                v_proj: weights.require(&format!(
                    "model.layers.{}.self_attn.v_proj.weight", i
                ))?.clone(),
                o_proj: weights.require(&format!(
                    "model.layers.{}.self_attn.o_proj.weight", i
                ))?.clone(),
                num_heads: config.num_attention_heads,
                num_kv_heads: config.num_key_value_heads,
                head_dim: config.hidden_size / config.num_attention_heads,
            },
            mlp: MyMLP {
                gate_proj: weights.require(&format!(
                    "model.layers.{}.mlp.gate_proj.weight", i
                ))?.clone(),
                up_proj: weights.require(&format!(
                    "model.layers.{}.mlp.up_proj.weight", i
                ))?.clone(),
                down_proj: weights.require(&format!(
                    "model.layers.{}.mlp.down_proj.weight", i
                ))?.clone(),
            },
            input_norm: weights.require(&format!(
                "model.layers.{}.input_layernorm.weight", i
            ))?.clone(),
            post_attn_norm: weights.require(&format!(
                "model.layers.{}.post_attention_layernorm.weight", i
            ))?.clone(),
        });
    }

    let norm = weights.require("model.norm.weight")?.clone();
    let lm_head = weights.require("lm_head.weight")?.clone();

    Ok(Self {
        config,
        embed_tokens,
        layers,
        norm,
        lm_head,
        device,
    })
}
```

## Step 6: Register the Model

Add to `models_v2/mod.rs`:

```rust
pub mod my_model;
pub use my_model::{MyModelV2, MyModelConfig};
```

## Step 7: Testing

Add tests in the model file:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config() {
        let config = MyModelConfig::default();
        assert_eq!(config.vocab_size(), 32000);
        assert_eq!(config.hidden_size(), 4096);
    }

    #[test]
    fn test_model_creation() {
        let config = MyModelConfig::default();
        let model = MyModelV2::new(config).unwrap();
        assert_eq!(model.config().num_layers(), 32);
    }

    #[test]
    fn test_forward_shape() {
        let config = MyModelConfig {
            num_hidden_layers: 2,  // Small for testing
            ..Default::default()
        };
        let model = MyModelV2::new(config).unwrap();

        let input_ids = ops_fn::zeros(
            &[1, 10],
            DataType::Int64,
            &Device::CPU,
        ).unwrap();

        let inputs = ModelInputs::text(input_ids);
        let outputs = model.forward(&inputs).unwrap();

        match outputs {
            ModelOutputs::Logits { logits, .. } => {
                assert_eq!(logits.shape()[0], 1);
                assert_eq!(logits.shape()[1], 10);
                assert_eq!(logits.shape()[2], 32000);
            }
            _ => panic!("Expected logits output"),
        }
    }
}
```

Run tests:

```bash
cargo test my_model --lib -p runtime
```

## Step 8: Documentation

Create documentation page in `documentation/docs/models/`:

```markdown
# My Model

Brief description of the model.

## Overview

| Property | Value |
|----------|-------|
| **Architecture** | ... |
| **Parameters** | ... |

## Quick Start

\`\`\`rust
use unillm::models_v2::my_model::{MyModelV2, MyModelConfig};
// ...
\`\`\`

## Configuration

...

## References

- [Paper](...)
- [HuggingFace](...)
```

## Checklist

- [ ] Configuration with `model_config!` macro
- [ ] Model structure (attention, MLP, layers)
- [ ] `Model` trait implementation
- [ ] Forward pass
- [ ] Weight loading
- [ ] Registration in `mod.rs`
- [ ] Unit tests
- [ ] Documentation

## Tips

1. **Start simple** - Get basic forward pass working first
2. **Reference implementations** - Look at existing models
3. **Test incrementally** - Test each component
4. **Use print debugging** - Check tensor shapes
5. **Compare outputs** - Verify against reference

## Common Issues

### Shape Mismatches

Check tensor dimensions at each step:

```rust
println!("hidden shape: {:?}", hidden.shape());
```

### Weight Name Mismatches

Print available weight names:

```rust
for name in weights.keys() {
    println!("{}", name);
}
```

### Numerical Differences

Small differences are normal due to floating-point precision. Large differences indicate bugs.

## Getting Help

- Look at similar models (e.g., `llama.rs` for LLaMA-style models)
- Check the [Architecture docs](../architecture/index.md)
- Ask in discussions
