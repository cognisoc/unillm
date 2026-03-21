# UniLLM Developer Guide

Welcome to UniLLM development! This guide will help you understand the architecture, contribute effectively, and extend the system.

## Architecture Overview

UniLLM is built on three core abstraction layers that provide clean separation of concerns:

### 1. TensorCore (`tensor_core.rs`)
**Unified tensor operations and device management**

```rust
// Device-agnostic tensor operations
let tensor = ops_fn::zeros(&[2, 3], DataType::Float32, &Device::CPU)?;
let result = ops_fn::matmul(&tensor_a, &tensor_b)?;

// Automatic device management
tensor.to_device(&Device::CUDA(0))?;
```

**Key Components:**
- `Tensor` - Universal tensor type with device abstraction
- `Device` - CPU, CUDA, Metal backend enumeration
- `TensorOps` - Trait defining all tensor operations
- `ops_fn` - Functional interface to tensor operations

### 2. ModelCore (`model_core.rs`)
**Clean model interfaces and configuration**

```rust
// Universal Model trait
pub trait Model: Send + Sync {
    type Config: ModelConfig;

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs>;
    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String>;
    // ... other methods
}

// Configuration macro with automatic trait implementation
model_config!(LlamaConfig {
    vocab_size: usize = 32000,
    hidden_size: usize = 4096,
    num_hidden_layers: usize = 32,
});
```

**Key Components:**
- `Model` trait - Universal interface for all models
- `ModelConfig` trait - Consistent configuration system
- `ModelInputs`/`ModelOutputs` - Unified I/O types
- `model_config!` macro - Automatic trait implementations

### 3. WeightLoaderCore (`weight_loader_core.rs`)
**Format-agnostic weight loading and management**

```rust
// Load weights from any format
let weights = WeightLoader::from_safetensors("model.safetensors")?;
let model = LlamaModelV2::from_weights(config, weights)?;

// Automatic format detection
let weights = WeightLoader::auto_detect("model_file")?;
```

## Adding New Models

Thanks to the solid abstractions, adding new models is straightforward:

### Step 1: Define Configuration

```rust
// In models_v2/your_model.rs
use crate::model_config;
use super::traits::*;

model_config!(YourModelConfig {
    vocab_size: usize = 50000,
    hidden_size: usize = 1024,
    num_hidden_layers: usize = 24,
    num_attention_heads: usize = 16,
    // ... model-specific fields
});
```

### Step 2: Implement Model Structure

```rust
pub struct YourModelV2 {
    config: YourModelConfig,
    device: Device,
    embed_tokens: Tensor,
    layers: Vec<YourModelLayer>,
    norm: Tensor,
    lm_head: Tensor,
}

pub struct YourModelLayer {
    self_attn: YourAttention,
    mlp: YourMLP,
    input_layernorm: Tensor,
    post_attention_layernorm: Tensor,
}
```

### Step 3: Implement Model Trait

```rust
impl Model for YourModelV2 {
    type Config = YourModelConfig;

    fn new(config: YourModelConfig) -> Result<Self> {
        // Initialize model with zero tensors
        let device = Device::CPU;
        // ... create layers
        Ok(Self { config, device, /* ... */ })
    }

    fn from_weights(config: YourModelConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;
        // Load weights into model
        if let Some(w) = weights.get("embed_tokens.weight") {
            model.embed_tokens = w.clone();
        }
        // ... load other weights
        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        let input_ids = match inputs {
            ModelInputs::Text { input_ids, .. } => input_ids,
            _ => return Err(anyhow::anyhow!("Expected text input")),
        };

        // Forward pass implementation
        let mut hidden_states = ops_fn::embedding(input_ids, &self.embed_tokens)?;

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }

        let logits = ops_fn::matmul(&hidden_states, &self.lm_head)?;

        Ok(ModelOutputs::Logits { logits, hidden_states: Some(hidden_states) })
    }

    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        // Basic generation (can be made more sophisticated)
        Ok(format!("Generated from: {}", prompt))
    }

    fn config(&self) -> &Self::Config { &self.config }

    fn memory_requirements(&self) -> MemoryRequirements {
        let param_size = self.config.vocab_size * self.config.hidden_size * 4;
        MemoryRequirements {
            gpu_memory: param_size,
            cpu_memory: param_size / 4,
            kv_cache_memory: self.config.hidden_size * 1000 * 4, // rough estimate
            peak_memory: param_size * 2,
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        self.embed_tokens = self.embed_tokens.to_device(device)?;
        // ... move other tensors
        Ok(())
    }
}
```

### Step 4: Export Model

```rust
// In models_v2/mod.rs
pub mod your_model;
pub use your_model::YourModelV2;
```

## Development Workflow

### Setting Up Environment

```bash
# Clone and build
git clone https://github.com/your-org/unillm.git
cd unillm

# Verify everything compiles
cargo check

# Run tests
cargo test
```

### Project Structure

```
crates/runtime/src/
├── lib.rs                    # Module exports and runtime
├── tensor_core.rs            # Tensor operations
├── model_core.rs             # Model abstractions
├── weight_loader_core.rs     # 💾 Weight loading
├── models_v2/                # 📁 All model implementations
│   ├── mod.rs               # Model exports
│   ├── llama.rs            # Llama family
│   ├── qwen.rs             # Qwen family
│   ├── gemma.rs            # Gemma family
│   └── ...                 # 15+ other families
├── inference.rs              # Inference pipeline
├── tokenizer.rs              # Tokenization
├── sampler.rs                # 🎲 Sampling methods
└── bin/                      # 🔨 Example binaries
```

### Testing Your Changes

```bash
# Test specific components
cargo test -p runtime tensor_core
cargo test -p runtime model_core
cargo test -p runtime models_v2::your_model

# Integration tests
cargo test -p runtime inference

# Full workspace test
cargo test
```

### Common Development Tasks

#### Adding New Tensor Operations

```rust
// In tensor_core.rs TensorOps trait
fn your_operation(&self, input: &Tensor, params: &[f32]) -> Result<Tensor>;

// In CpuTensorOpsImpl
fn your_operation(&self, input: &Tensor, params: &[f32]) -> Result<Tensor> {
    // CPU implementation
    // Return new Tensor with computed values
}

// In ops_fn module
pub fn your_operation(input: &Tensor, params: &[f32]) -> Result<Tensor> {
    input.ops().your_operation(input, params)
}
```

#### Adding New Configuration Fields

```rust
// The model_config! macro automatically handles new fields
model_config!(YourConfig {
    existing_field: usize = 1000,
    new_field: f32 = 0.1,        // Just add new field with default
    optional_field: Option<String> = None,
});

// Automatic trait implementations are regenerated
```

#### Implementing Custom Layers

```rust
pub struct YourCustomLayer {
    weight: Tensor,
    bias: Option<Tensor>,
    config: YourModelConfig,
}

impl YourCustomLayer {
    fn new(config: &YourModelConfig, device: &Device) -> Result<Self> {
        let weight = ops_fn::zeros(&[config.hidden_size, config.hidden_size],
                                   DataType::Float32, device)?;
        Ok(Self { weight, bias: None, config: config.clone() })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let output = ops_fn::matmul(input, &self.weight)?;
        if let Some(ref bias) = self.bias {
            ops_fn::add(&output, bias)
        } else {
            Ok(output)
        }
    }

    fn load_weights(&mut self, weights: &ModelWeights, prefix: &str) -> Result<()> {
        if let Some(w) = weights.get(&format!("{}.weight", prefix)) {
            self.weight = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.bias", prefix)) {
            self.bias = Some(w.clone());
        }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.weight = self.weight.to_device(device)?;
        if let Some(ref mut bias) = self.bias {
            *bias = bias.to_device(device)?;
        }
        Ok(())
    }
}
```

## Testing Patterns

### Unit Testing Models

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let config = YourModelConfig::default();
        let model = YourModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size, 50000);
    }

    #[test]
    fn test_forward_pass() {
        let config = YourModelConfig { hidden_size: 64, ..Default::default() };
        let model = YourModelV2::new(config).unwrap();

        let input_tensor = ops_fn::zeros(&[1, 10], DataType::Int64, &Device::CPU).unwrap();
        let inputs = ModelInputs::Text {
            input_ids: input_tensor,
            attention_mask: None,
            position_ids: None
        };

        let outputs = model.forward(&inputs).unwrap();
        match outputs {
            ModelOutputs::Logits { logits, .. } => {
                assert_eq!(logits.shape(), &[1, 10, config.vocab_size]);
            },
            _ => panic!("Expected logits output"),
        }
    }
}
```

### Integration Testing

```rust
#[test]
fn test_end_to_end_generation() {
    let config = YourModelConfig::default();
    let model = YourModelV2::new(config).unwrap();
    let gen_config = GenerationConfig::default();

    let output = model.generate("Hello world", &gen_config).unwrap();
    assert!(!output.is_empty());
}
```

## Performance Guidelines

### Memory Management

```rust
// Good: Reuse tensors when possible
fn efficient_computation(input: &Tensor) -> Result<Tensor> {
    let mut result = input.clone();
    ops_fn::relu_inplace(&mut result)?;  // In-place when possible
    Ok(result)
}

// Avoid: Creating unnecessary intermediate tensors
fn inefficient_computation(input: &Tensor) -> Result<Tensor> {
    let temp1 = ops_fn::relu(input)?;
    let temp2 = ops_fn::relu(&temp1)?;  // temp1 now unused
    ops_fn::relu(&temp2)
}
```

### Device Management

```rust
// Good: Check device compatibility
fn smart_operation(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.device() != b.device() {
        let b = b.to_device(a.device())?;
        ops_fn::matmul(a, &b)
    } else {
        ops_fn::matmul(a, b)
    }
}
```

## 🐛 Common Issues and Solutions

### Compilation Errors

**Issue**: Model doesn't implement required traits
```rust
error[E0277]: `YourModelV2` doesn't implement `Model`
```
**Solution**: Make sure you implement all Model trait methods

**Issue**: Configuration macro errors
```rust
error: no field `hidden_size` on type `&YourModelConfig`
```
**Solution**: The `model_config!` macro looks for `hidden_size` or `d_model` fields. Make sure one exists.

### Runtime Errors

**Issue**: Tensor shape mismatches
```rust
thread 'main' panicked at 'assertion failed: shapes compatible'
```
**Solution**: Check tensor shapes before operations:

```rust
fn safe_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(anyhow::anyhow!("Expected 2D tensors"));
    }

    if a_shape[1] != b_shape[0] {
        return Err(anyhow::anyhow!(
            "Incompatible shapes: {:?} x {:?}", a_shape, b_shape
        ));
    }

    ops_fn::matmul(a, b)
}
```

## Best Practices

### 1. Follow the Abstraction Layers
- Use `ops_fn` for tensor operations (don't call TensorOps directly)
- Use the `Model` trait for all model implementations
- Use `model_config!` for configuration structs

### 2. Error Handling
```rust
// Good: Descriptive errors with context
fn load_layer(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
    let prefix = format!("model.layers.{}", layer_idx);
    if let Some(w) = weights.get(&format!("{}.weight", prefix)) {
        self.weight = w.clone();
    } else {
        return Err(anyhow::anyhow!("Missing weight: {}.weight", prefix));
    }
    Ok(())
}
```

### 3. Configuration Patterns
```rust
// Good: Provide sensible defaults
model_config!(ModelConfig {
    vocab_size: usize = 32000,
    hidden_size: usize = 4096,
    intermediate_size: usize = 11008,  // derived: hidden_size * 8/3
    rope_theta: f32 = 10000.0,
    layer_norm_epsilon: f32 = 1e-6,
    // Optional fields
    rope_scaling: Option<String> = None,
});
```

### 4. Testing
- Test model creation with different configurations
- Test forward pass with various input shapes
- Test weight loading and device transfer
- Include performance tests for critical paths

## 🤝 Contributing

1. **Pick an area**: Models, tensor ops, inference, or infrastructure
2. **Start small**: Begin with tests or small improvements
3. **Follow patterns**: Look at existing implementations
4. **Document**: Add clear comments and update docs
5. **Test thoroughly**: Include unit and integration tests

## 📚 Additional Resources

- [Rust Book](https://doc.rust-lang.org/book/) - Rust fundamentals
- [API Reference](api_reference.md) - Detailed API docs
- [Architecture](../ARCHITECTURE.md) - System design deep-dive

---

**Happy coding! 🦀** The solid abstractions make UniLLM a pleasure to work with.