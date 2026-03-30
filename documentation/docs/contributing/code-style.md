# Code Style Guide

This guide covers coding conventions and best practices for UniLLM development.

## General Principles

1. **Clarity over brevity** - Write readable, self-documenting code
2. **Consistency** - Follow existing patterns in the codebase
3. **Simplicity** - Avoid unnecessary complexity
4. **Testability** - Write code that's easy to test

## Rust Conventions

### Formatting

Use `rustfmt` with default settings:

```bash
cargo fmt
```

### Imports

Organize imports in groups:

```rust
// Standard library
use std::collections::HashMap;
use std::sync::Arc;

// External crates
use anyhow::Result;
use log::{debug, info};

// Internal crates (workspace)
use crate::tensor_core::{ops_fn, Tensor, Device, DataType};
use crate::model_core::{Model, ModelConfig, ModelInputs, ModelOutputs};

// Local modules
use super::traits::*;
```

### Naming Conventions

```rust
// Types: PascalCase
struct LlamaConfig { }
enum ModelInputs { }
trait TensorOps { }

// Functions/methods: snake_case
fn forward_attention(&self, hidden: &Tensor) -> Result<Tensor>
fn apply_rope(&self, q: &Tensor, k: &Tensor) -> (Tensor, Tensor)

// Constants: SCREAMING_SNAKE_CASE
const MAX_SEQUENCE_LENGTH: usize = 4096;
const DEFAULT_TEMPERATURE: f32 = 1.0;

// Modules: snake_case
mod tensor_core;
mod model_core;
mod models_v2;
```

### Documentation

Document public items:

```rust
/// A transformer layer with self-attention and feed-forward network.
///
/// # Example
///
/// ```rust
/// let layer = TransformerLayer::new(config)?;
/// let output = layer.forward(&input)?;
/// ```
pub struct TransformerLayer {
    /// Self-attention mechanism
    attention: Attention,
    /// Feed-forward network
    mlp: MLP,
}

/// Run the forward pass through this layer.
///
/// # Arguments
///
/// * `hidden` - Input tensor of shape `[batch, seq_len, hidden_size]`
/// * `mask` - Optional attention mask
///
/// # Returns
///
/// Output tensor of shape `[batch, seq_len, hidden_size]`
pub fn forward(&self, hidden: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    // ...
}
```

### Error Handling

Use `anyhow::Result` for errors:

```rust
use anyhow::{Result, anyhow, bail, Context};

// Return errors with context
fn load_weights(path: &str) -> Result<ModelWeights> {
    let file = File::open(path)
        .context("Failed to open weights file")?;
    // ...
}

// Create errors
fn validate_shape(tensor: &Tensor) -> Result<()> {
    if tensor.shape().len() != 3 {
        bail!("Expected 3D tensor, got {:?}", tensor.shape());
    }
    Ok(())
}

// Use anyhow! for ad-hoc errors
if config.hidden_size == 0 {
    return Err(anyhow!("hidden_size must be > 0"));
}
```

## UniLLM Patterns

### Using TensorCore

Always use `ops_fn` for tensor operations:

```rust
// Good
let result = ops_fn::matmul(&a, &b)?;
let normalized = ops_fn::layer_norm(&hidden, &weight, None, 1e-5)?;

// Bad - don't access backend directly
let result = candle_ops::matmul(&a.inner, &b.inner)?;
```

### Model Configuration

Use the `model_config!` macro:

```rust
// Good
model_config!(LlamaConfig {
    vocab_size: usize = 32000,
    hidden_size: usize = 4096,
});

// Bad - manual implementation
#[derive(Clone, Debug)]
struct LlamaConfig {
    vocab_size: usize,
    hidden_size: usize,
}
impl Default for LlamaConfig { /* ... */ }
impl ModelConfig for LlamaConfig { /* ... */ }
```

### Model Naming

Follow consistent naming:

```rust
// Model: {Name}ModelV2
pub struct LlamaModelV2 { }

// Config: {Name}Config
pub struct LlamaConfig { }

// Layer: {Name}Layer or just Layer if internal
struct LlamaLayer { }
struct Layer { }  // If only used internally

// Attention: {Name}Attention
struct LlamaAttention { }

// MLP: {Name}MLP
struct LlamaMLP { }
```

### Forward Pass Structure

Follow the established pattern:

```rust
fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
    // 1. Extract inputs
    let input_ids = match inputs {
        ModelInputs::Text { input_ids, .. } => input_ids,
        _ => bail!("Expected text input"),
    };

    // 2. Embeddings
    let mut hidden = ops_fn::embedding(input_ids, &self.embed_tokens)?;

    // 3. Process layers
    for layer in &self.layers {
        hidden = self.forward_layer(&hidden, layer)?;
    }

    // 4. Final normalization
    hidden = ops_fn::rms_norm(&hidden, &self.norm, self.config.rms_norm_eps)?;

    // 5. Output projection
    let logits = ops_fn::linear(&hidden, &self.lm_head, None)?;

    // 6. Return outputs
    Ok(ModelOutputs::Logits {
        logits,
        hidden_states: None,
    })
}
```

## Code Organization

### File Structure

```rust
// crates/runtime/src/models_v2/llama.rs

// 1. Imports
use anyhow::Result;
use crate::model_config;
use crate::tensor_core::{ops_fn, Tensor, Device, DataType};
use super::traits::*;

// 2. Configuration
model_config!(LlamaConfig { /* ... */ });

// 3. Internal structures (private)
struct LlamaAttention { /* ... */ }
struct LlamaMLP { /* ... */ }
struct LlamaLayer { /* ... */ }

// 4. Main model (public)
pub struct LlamaModelV2 { /* ... */ }

// 5. Implementation
impl LlamaModelV2 {
    // Helper methods
    fn forward_layer(&self, ...) -> Result<Tensor> { }
    fn forward_attention(&self, ...) -> Result<Tensor> { }
    fn forward_mlp(&self, ...) -> Result<Tensor> { }
}

// 6. Trait implementation
impl Model for LlamaModelV2 {
    // Required methods
}

// 7. Tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config() { }

    #[test]
    fn test_forward() { }
}
```

### Module Exports

In `mod.rs`, export only what's needed:

```rust
// crates/runtime/src/models_v2/mod.rs

pub mod llama;
pub mod qwen;
pub mod gemma;
// ...

// Re-export main types
pub use llama::{LlamaModelV2, LlamaConfig};
pub use qwen::{QwenModelV2, QwenConfig};
pub use gemma::{GemmaModelV2, GemmaConfig};
```

## Testing

### Test Structure

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_descriptive_name() {
        // Arrange
        let config = LlamaConfig::default();
        let model = LlamaModelV2::new(config).unwrap();

        // Act
        let input = create_test_input();
        let output = model.forward(&input);

        // Assert
        assert!(output.is_ok());
    }

    #[test]
    fn test_handles_error_case() {
        let result = some_fallible_operation();
        assert!(result.is_err());
    }
}
```

### Test Naming

```rust
#[test]
fn test_forward_pass_with_valid_input() { }

#[test]
fn test_config_validation_rejects_zero_vocab() { }

#[test]
fn test_attention_mask_applied_correctly() { }
```

## Comments

### When to Comment

```rust
// Good - explains WHY
// Apply RoPE scaling for extended context (beyond 2K tokens)
let scaled_theta = self.config.rope_theta * (seq_len as f32 / 2048.0);

// Bad - states the obvious
// Loop through layers
for layer in &self.layers {

// Good - documents complex logic
// GQA: expand KV heads to match query heads
// num_heads=32, num_kv_heads=8 -> repeat each KV head 4 times
let expanded_k = self.expand_kv(&k, self.config.num_heads / self.config.num_kv_heads)?;
```

### TODO Comments

```rust
// TODO: Implement KV caching for efficient generation
// TODO(username): Fix shape mismatch for batch_size > 1
```

## Performance

### Avoid Unnecessary Allocations

```rust
// Good - reuse buffer
let mut hidden = initial_hidden;
for layer in &self.layers {
    hidden = layer.forward(&hidden)?;
}

// Bad - unnecessary intermediate allocations
let outputs: Vec<Tensor> = self.layers
    .iter()
    .fold(vec![initial_hidden], |mut acc, layer| {
        acc.push(layer.forward(acc.last().unwrap()).unwrap());
        acc
    });
```

### Use References

```rust
// Good - borrow when possible
fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs>
fn apply_mask(&self, tensor: &Tensor, mask: &Tensor) -> Result<Tensor>

// Bad - unnecessary clone
fn forward(&self, inputs: ModelInputs) -> Result<ModelOutputs>
fn apply_mask(&self, tensor: Tensor, mask: Tensor) -> Result<Tensor>
```

## Linting

Run clippy and fix warnings:

```bash
cargo clippy -- -D warnings
```

Common clippy fixes:

```rust
// Use `?` instead of `unwrap` in fallible functions
let value = operation()?;  // Not: operation().unwrap()

// Use `if let` for single pattern matching
if let Some(value) = optional {
    // use value
}

// Use `is_empty()` instead of `len() == 0`
if vec.is_empty() {
    // ...
}
```

## Git Commits

### Commit Messages

```
Add LLaMA model implementation

- Implement LlamaModelV2 with full forward pass
- Add configuration via model_config! macro
- Include RoPE and GQA support
- Add unit tests for forward pass
```

### Keep Commits Focused

- One logical change per commit
- Separate refactoring from feature changes
- Include tests with the feature
