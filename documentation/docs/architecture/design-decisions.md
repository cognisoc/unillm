# Design Decisions

This document explains the key architectural decisions made in UniLLM and their rationale.

## Functional Tensor Operations

**Decision:** Use a functional interface (`ops_fn::*`) rather than methods on Tensor.

**Rationale:**

```rust
// Chosen approach: Functional
let result = ops_fn::matmul(&a, &b)?;
let normed = ops_fn::layer_norm(&hidden, &weight, None, 1e-6)?;

// Alternative: Method-based
let result = a.matmul(&b)?;
let normed = hidden.layer_norm(&weight, None, 1e-6)?;
```

Benefits of functional approach:
- **Explicit operations** - Clear what operation is being performed
- **Backend flexibility** - Easy to swap implementations
- **Consistency** - Same pattern for all operations
- **Testing** - Operations can be tested independently

## Single Model Trait

**Decision:** All models implement a single `Model` trait rather than specialized traits per model type.

**Rationale:**

```rust
// Chosen approach: Universal trait
pub trait Model {
    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs>;
    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String>;
}

// Alternative: Specialized traits
pub trait LanguageModel { fn generate(&self, ...) -> ...; }
pub trait VisionModel { fn encode_image(&self, ...) -> ...; }
pub trait AudioModel { fn transcribe(&self, ...) -> ...; }
```

Benefits:
- **Simplicity** - One interface to learn
- **Composability** - Models can be used interchangeably
- **Extensibility** - Easy to add new model types
- **Enum-based I/O** - `ModelInputs` and `ModelOutputs` handle type differences

## model_config! Macro

**Decision:** Use a macro to generate configuration structs and trait implementations.

**Rationale:**

```rust
// With macro (current)
model_config!(LlamaConfig {
    vocab_size: usize = 32000,
    hidden_size: usize = 4096,
});

// Without macro (alternative)
#[derive(Clone, Debug)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
}

impl Default for LlamaConfig { /* ... */ }
impl ModelConfig for LlamaConfig { /* ... */ }
```

Benefits:
- **Reduced boilerplate** - 47 models × ~50 lines saved = significant reduction
- **Consistency** - All configs follow same pattern
- **Defaults inline** - Easy to see default values
- **Automatic validation** - Macro can generate validation

## Enum-Based I/O

**Decision:** Use enums for `ModelInputs` and `ModelOutputs` rather than generics.

**Rationale:**

```rust
// Chosen approach: Enums
pub enum ModelInputs {
    Text { input_ids: Tensor, ... },
    Image { pixel_values: Tensor, ... },
    Multimodal { ... },
    Audio { ... },
}

// Alternative: Generics
pub trait ModelInput { fn to_tensor(&self) -> Tensor; }
fn forward<I: ModelInput>(&self, input: I) -> Result<O>;
```

Benefits:
- **Explicit types** - Clear what each model accepts
- **Runtime flexibility** - Can switch input types dynamically
- **Pattern matching** - Easy to handle different cases
- **Serialization** - Simpler to serialize/deserialize

## Automatic Dequantization

**Decision:** GGUF quantized weights are automatically dequantized to F32 during loading.

**Rationale:**

```rust
// Current behavior
let weights = WeightLoader::from_gguf("model-Q4_K_M.gguf")?;
// All weights are F32, ready for any backend

// Alternative: Keep quantized
let weights = WeightLoader::from_gguf("model-Q4_K_M.gguf")?;
// Weights stay quantized, need special ops
```

Benefits:
- **Backend compatibility** - All backends support F32
- **Simpler models** - No quantization-aware code paths
- **Correctness** - Same numerical results as reference

Trade-offs:
- **Memory** - Uses more memory than quantized
- **Future** - Will add quantized inference as optimization

## Device as Enum

**Decision:** Use an enum for device representation rather than trait objects.

**Rationale:**

```rust
// Chosen approach: Enum
pub enum Device {
    CPU,
    CUDA(usize),
    Metal(usize),
}

// Alternative: Trait object
pub trait Device: Send + Sync { ... }
pub struct CpuDevice;
pub struct CudaDevice(usize);
```

Benefits:
- **Simple** - Easy to understand and use
- **Pattern matching** - Clean device-specific logic
- **No vtable** - Slightly more efficient
- **Copy** - Can be passed by value

## Candle as Backend

**Decision:** Use Candle as the primary tensor backend.

**Rationale:**

Evaluated alternatives:
- **PyTorch (via tch-rs)** - Heavy dependency, C++ complexity
- **ONNX Runtime** - Graph-based, less flexible
- **Custom** - Too much work for same functionality
- **Candle** - Pure Rust, active development, good API

Benefits of Candle:
- **Pure Rust** - No C++ dependencies
- **HuggingFace** - Same maintainers as transformers
- **CUDA/Metal** - GPU support built-in
- **Quantization** - GGUF support included
- **Active** - Regular updates and improvements

## Error Handling with anyhow

**Decision:** Use `anyhow::Result` for error handling rather than custom error types.

**Rationale:**

```rust
// Chosen approach: anyhow
use anyhow::Result;
fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs>;

// Alternative: Custom errors
#[derive(Error)]
enum ModelError { ... }
fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs, ModelError>;
```

Benefits:
- **Simplicity** - Less boilerplate
- **Composability** - Different error types work together
- **Context** - Easy to add error context
- **Development speed** - Can refine error types later

## Layer-by-Layer Construction

**Decision:** Models are constructed layer by layer from weights, not from a graph.

**Rationale:**

```rust
// Chosen approach: Manual construction
let layers: Vec<Layer> = (0..config.num_layers)
    .map(|i| Layer::from_weights(&weights, i))
    .collect()?;

// Alternative: Graph-based
let graph = Graph::from_onnx("model.onnx")?;
let model = Model::from_graph(graph)?;
```

Benefits:
- **Flexibility** - Full control over model structure
- **Debugging** - Easy to inspect intermediate states
- **Optimization** - Can apply architecture-specific optimizations
- **Understanding** - Clear how model works

## No Training Support

**Decision:** UniLLM is inference-only; no gradient computation or training.

**Rationale:**

- **Focus** - Inference and training have different requirements
- **Performance** - No backward pass overhead
- **Simplicity** - Much simpler codebase
- **Use case** - Target audience wants fast inference

## Ollama Integration

**Decision:** Integrate with Ollama registry for model downloads.

**Rationale:**

```rust
// Easy model access
let path = OllamaRegistry::pull("llama2:7b")?;
let weights = WeightLoader::from_gguf(&path)?;
```

Benefits:
- **Ecosystem** - Leverage existing model library
- **Convenience** - No manual download needed
- **Caching** - Built-in model caching
- **Compatibility** - GGUF format already supported

## Consistent Naming

**Decision:** Follow consistent naming conventions across all code.

**Patterns:**
- Models: `{Name}ModelV2` (e.g., `LlamaModelV2`)
- Configs: `{Name}Config` (e.g., `LlamaConfig`)
- Layers: `{Name}Layer` (e.g., `LlamaLayer`)
- Attention: `{Name}Attention` (e.g., `LlamaAttention`)
- MLP: `{Name}MLP` (e.g., `LlamaMLP`)

Benefits:
- **Predictability** - Know names without looking
- **Search** - Easy to find related code
- **Refactoring** - Consistent patterns to update

## Test Structure

**Decision:** Tests live in `#[cfg(test)]` modules within source files.

**Rationale:**

```rust
// In model.rs
pub struct MyModel { ... }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward() { ... }
}
```

Benefits:
- **Proximity** - Tests next to implementation
- **Access** - Can test private functions
- **Compilation** - Only compile tests when needed
- **Discovery** - Easy to find tests for code
