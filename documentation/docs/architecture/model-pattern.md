# Model Implementation Pattern

All 47 models in UniLLM follow a consistent implementation pattern. This guide explains how models are structured and how to implement new ones.

## Standard Structure

Every model implementation follows this structure:

```rust
// crates/runtime/src/models_v2/my_model.rs

use anyhow::Result;
use crate::model_config;
use crate::tensor_core::{ops_fn, Tensor, Device, DataType};
use super::traits::*;

// ═══════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════

model_config!(MyModelConfig {
    vocab_size: usize = 32000,
    hidden_size: usize = 4096,
    intermediate_size: usize = 11008,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 32,
    num_key_value_heads: usize = 32,
    max_position_embeddings: usize = 2048,
    rms_norm_eps: f32 = 1e-6,
});

// ═══════════════════════════════════════════════════════════
// Subcomponents
// ═══════════════════════════════════════════════════════════

struct MyAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    num_heads: usize,
    head_dim: usize,
}

struct MyMLP {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
}

struct MyLayer {
    attention: MyAttention,
    mlp: MyMLP,
    input_norm: Tensor,
    post_attn_norm: Tensor,
}

// ═══════════════════════════════════════════════════════════
// Main Model
// ═══════════════════════════════════════════════════════════

pub struct MyModelV2 {
    config: MyModelConfig,
    embed_tokens: Tensor,
    layers: Vec<MyLayer>,
    norm: Tensor,
    lm_head: Tensor,
    device: Device,
}

// ═══════════════════════════════════════════════════════════
// Model Trait Implementation
// ═══════════════════════════════════════════════════════════

impl Model for MyModelV2 {
    type Config = MyModelConfig;

    fn new(config: Self::Config) -> Result<Self> {
        // Create with dummy/zero weights
    }

    fn from_weights(config: Self::Config, weights: ModelWeights) -> Result<Self> {
        // Load weights from container
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        // Run forward pass
    }

    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        // Text generation
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

## Configuration Pattern

### Using model_config! Macro

The macro automatically generates:

```rust
model_config!(LlamaConfig {
    vocab_size: usize = 32000,
    hidden_size: usize = 4096,
    // ...
});

// Generates:
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    // ...
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            // ...
        }
    }
}

impl ModelConfig for LlamaConfig {
    fn architecture(&self) -> &str { "llama" }
    fn vocab_size(&self) -> usize { self.vocab_size }
    fn hidden_size(&self) -> usize { self.hidden_size }
    fn num_layers(&self) -> usize { self.num_hidden_layers }
    fn validate(&self) -> Result<()> { /* validation */ }
}
```

### Loading from GGUF

```rust
impl LlamaConfig {
    pub fn from_gguf_metadata(meta: &WeightMetadata) -> Result<Self> {
        let gguf = meta.gguf_metadata.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No GGUF metadata"))?;

        Ok(Self {
            vocab_size: gguf.vocab_size.unwrap_or(32000),
            hidden_size: gguf.embedding_length.unwrap_or(4096),
            num_hidden_layers: gguf.num_layers.unwrap_or(32),
            ..Default::default()
        })
    }
}
```

## Forward Pass Pattern

### Decoder-Only Models (LLaMA, GPT)

```rust
fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
    let input_ids = match inputs {
        ModelInputs::Text { input_ids, .. } => input_ids,
        _ => return Err(anyhow::anyhow!("Expected text input")),
    };

    // 1. Token embeddings
    let mut hidden = ops_fn::embedding(input_ids, &self.embed_tokens)?;

    // 2. Transformer layers
    for layer in &self.layers {
        hidden = self.forward_layer(&hidden, layer)?;
    }

    // 3. Final normalization
    hidden = ops_fn::rms_norm(&hidden, &self.norm, self.config.rms_norm_eps)?;

    // 4. Output projection
    let logits = ops_fn::linear(&hidden, &self.lm_head, None)?;

    Ok(ModelOutputs::Logits {
        logits,
        hidden_states: None,
    })
}
```

### Layer Forward

```rust
fn forward_layer(&self, hidden: &Tensor, layer: &MyLayer) -> Result<Tensor> {
    // Pre-attention norm
    let normed = ops_fn::rms_norm(hidden, &layer.input_norm, self.config.rms_norm_eps)?;

    // Self-attention
    let attn_out = self.forward_attention(&normed, &layer.attention)?;

    // Residual connection
    let hidden = ops_fn::add(hidden, &attn_out)?;

    // Post-attention norm
    let normed = ops_fn::rms_norm(&hidden, &layer.post_attn_norm, self.config.rms_norm_eps)?;

    // MLP
    let mlp_out = self.forward_mlp(&normed, &layer.mlp)?;

    // Residual connection
    ops_fn::add(&hidden, &mlp_out)
}
```

### Attention Pattern

```rust
fn forward_attention(&self, hidden: &Tensor, attn: &MyAttention) -> Result<Tensor> {
    let (batch, seq_len, _) = hidden.shape3()?;

    // Project Q, K, V
    let q = ops_fn::linear(hidden, &attn.q_proj, None)?;
    let k = ops_fn::linear(hidden, &attn.k_proj, None)?;
    let v = ops_fn::linear(hidden, &attn.v_proj, None)?;

    // Reshape for attention
    let q = ops_fn::reshape(&q, &[batch, seq_len, attn.num_heads, attn.head_dim])?;
    let k = ops_fn::reshape(&k, &[batch, seq_len, attn.num_heads, attn.head_dim])?;
    let v = ops_fn::reshape(&v, &[batch, seq_len, attn.num_heads, attn.head_dim])?;

    // Transpose to [batch, heads, seq, dim]
    let q = ops_fn::transpose(&q, 1, 2)?;
    let k = ops_fn::transpose(&k, 1, 2)?;
    let v = ops_fn::transpose(&v, 1, 2)?;

    // Apply RoPE (for models that use it)
    let (q, k) = self.apply_rope(&q, &k, seq_len)?;

    // Scaled dot-product attention
    let attn_out = ops_fn::attention(&q, &k, &v, Some(&causal_mask))?;

    // Reshape back
    let attn_out = ops_fn::transpose(&attn_out, 1, 2)?;
    let attn_out = ops_fn::reshape(&attn_out, &[batch, seq_len, attn.num_heads * attn.head_dim])?;

    // Output projection
    ops_fn::linear(&attn_out, &attn.o_proj, None)
}
```

### MLP Pattern

```rust
fn forward_mlp(&self, hidden: &Tensor, mlp: &MyMLP) -> Result<Tensor> {
    // SwiGLU activation (LLaMA style)
    let gate = ops_fn::linear(hidden, &mlp.gate_proj, None)?;
    let up = ops_fn::linear(hidden, &mlp.up_proj, None)?;

    let gate = ops_fn::silu(&gate)?;
    let hidden = ops_fn::mul(&gate, &up)?;

    ops_fn::linear(&hidden, &mlp.down_proj, None)
}
```

## Model Variants

### MoE Models

```rust
struct MoELayer {
    gate: Tensor,  // [hidden_size, num_experts]
    experts: Vec<MLP>,
    num_experts_per_tok: usize,
}

fn forward_moe(&self, hidden: &Tensor, moe: &MoELayer) -> Result<Tensor> {
    // Route to experts
    let router_logits = ops_fn::linear(hidden, &moe.gate, None)?;
    let (weights, indices) = ops_fn::topk(&router_logits, moe.num_experts_per_tok, -1)?;
    let weights = ops_fn::softmax(&weights, -1)?;

    // Process through selected experts
    let mut output = ops_fn::zeros_like(hidden)?;
    for (expert_idx, expert) in moe.experts.iter().enumerate() {
        // ... route and aggregate
    }

    output
}
```

### Vision-Language Models

```rust
pub struct VisionLanguageModel {
    vision_encoder: VisionEncoder,
    projector: Tensor,
    language_model: LlamaModelV2,
}

fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
    match inputs {
        ModelInputs::Multimodal { input_ids, pixel_values, .. } => {
            // Encode images
            let image_features = self.vision_encoder.forward(pixel_values)?;
            let projected = ops_fn::linear(&image_features, &self.projector, None)?;

            // Merge with text embeddings
            let text_embeds = self.language_model.get_embeddings(input_ids)?;
            let merged = self.merge_embeddings(&text_embeds, &projected)?;

            // Forward through LLM
            self.language_model.forward_from_embeddings(&merged)
        }
        _ => self.language_model.forward(inputs),
    }
}
```

### Audio Models

```rust
pub struct AudioEncoder {
    conv_layers: Vec<ConvBlock>,
    transformer: Vec<TransformerLayer>,
}

fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
    let features = match inputs {
        ModelInputs::Audio { input_features, .. } => input_features,
        _ => return Err(anyhow::anyhow!("Expected audio input")),
    };

    // Convolutional feature extraction
    let mut hidden = features.clone();
    for conv in &self.conv_layers {
        hidden = self.forward_conv(&hidden, conv)?;
    }

    // Transformer encoding
    for layer in &self.transformer {
        hidden = self.forward_layer(&hidden, layer)?;
    }

    Ok(ModelOutputs::Embeddings {
        embeddings: hidden,
        pooled: None,
    })
}
```

## Weight Loading

### Weight Name Mapping

```rust
impl MyModelV2 {
    fn from_weights(config: MyModelConfig, weights: ModelWeights) -> Result<Self> {
        let embed_tokens = weights.require("model.embed_tokens.weight")?.clone();

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            layers.push(MyLayer {
                attention: MyAttention {
                    q_proj: weights.require(&format!("model.layers.{}.self_attn.q_proj.weight", i))?.clone(),
                    k_proj: weights.require(&format!("model.layers.{}.self_attn.k_proj.weight", i))?.clone(),
                    v_proj: weights.require(&format!("model.layers.{}.self_attn.v_proj.weight", i))?.clone(),
                    o_proj: weights.require(&format!("model.layers.{}.self_attn.o_proj.weight", i))?.clone(),
                    num_heads: config.num_attention_heads,
                    head_dim: config.hidden_size / config.num_attention_heads,
                },
                // ...
            });
        }

        Ok(Self {
            config,
            embed_tokens,
            layers,
            // ...
        })
    }
}
```

## Testing Pattern

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let config = MyModelConfig::default();
        let model = MyModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 32000);
    }

    #[test]
    fn test_forward_pass() {
        let config = MyModelConfig::default();
        let model = MyModelV2::new(config).unwrap();

        let input_ids = ops_fn::zeros(&[1, 10], DataType::Int64, &Device::CPU).unwrap();
        let inputs = ModelInputs::text(input_ids);

        let outputs = model.forward(&inputs).unwrap();
        match outputs {
            ModelOutputs::Logits { logits, .. } => {
                assert_eq!(logits.shape()[2], 32000);  // vocab_size
            }
            _ => panic!("Expected logits output"),
        }
    }
}
```

## Checklist for New Models

1. [ ] Create configuration with `model_config!` macro
2. [ ] Define subcomponents (Attention, MLP, Layer)
3. [ ] Implement main model struct
4. [ ] Implement `Model` trait
5. [ ] Add weight name mapping
6. [ ] Write unit tests
7. [ ] Add to `models_v2/mod.rs` exports
8. [ ] Update documentation
