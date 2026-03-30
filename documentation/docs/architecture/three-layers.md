# Three-Layer Architecture

UniLLM's architecture is built on three core abstraction layers that work together to provide a consistent, extensible framework for model inference.

## TensorCore

**File:** `crates/runtime/src/tensor_core.rs`

TensorCore is the foundation layer that provides device-agnostic tensor operations.

### Design Principles

1. **Functional Interface** - All operations through `ops_fn::*` functions
2. **Device Abstraction** - Same code runs on any hardware
3. **Zero-Cost Wrapping** - Minimal overhead over raw backend operations
4. **Explicit Devices** - No hidden device transfers

### Key Components

#### Tensor

The universal tensor type that wraps backend-specific implementations:

```rust
pub struct Tensor {
    inner: TensorInner,
    device: Device,
}

impl Tensor {
    pub fn shape(&self) -> &[usize];
    pub fn dtype(&self) -> DataType;
    pub fn device(&self) -> &Device;
    pub fn to_device(&self, device: &Device) -> Result<Tensor>;
}
```

#### Device

Hardware abstraction for CPU and GPU:

```rust
pub enum Device {
    CPU,
    CUDA(usize),
    Metal(usize),
}

impl Device {
    pub fn auto() -> Device {
        #[cfg(feature = "cuda")]
        if cuda_available() {
            return Device::CUDA(0);
        }
        #[cfg(feature = "metal")]
        if metal_available() {
            return Device::Metal(0);
        }
        Device::CPU
    }
}
```

#### ops_fn Module

Functional interface for all tensor operations:

```rust
pub mod ops_fn {
    // Creation
    pub fn zeros(shape: &[usize], dtype: DataType, device: &Device) -> Result<Tensor>;
    pub fn ones(shape: &[usize], dtype: DataType, device: &Device) -> Result<Tensor>;

    // Math
    pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor>;
    pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor>;

    // Neural Network
    pub fn embedding(indices: &Tensor, weight: &Tensor) -> Result<Tensor>;
    pub fn layer_norm(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>, eps: f32) -> Result<Tensor>;
    pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor>;
}
```

### Backend Abstraction

The `TensorOps` trait allows pluggable backends:

```rust
pub trait TensorOps: Send + Sync {
    fn zeros(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<Tensor>;
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
    // ... all operations
}

// Implementations
pub struct CpuTensorOpsImpl;   // Candle CPU
pub struct CudaTensorOpsImpl;  // Candle CUDA
pub struct MetalTensorOpsImpl; // Candle Metal
```

## ModelCore

**File:** `crates/runtime/src/model_core.rs`

ModelCore provides the universal model interface and configuration system.

### Design Principles

1. **Trait-Based Interface** - All models implement `Model` trait
2. **Automatic Configuration** - `model_config!` macro reduces boilerplate
3. **Unified I/O** - Consistent input/output types across all models
4. **Composability** - Models can be combined (e.g., vision encoder + LLM)

### Key Components

#### Model Trait

The universal interface for all models:

```rust
pub trait Model: Send + Sync {
    type Config: ModelConfig;

    /// Create model with configuration
    fn new(config: Self::Config) -> Result<Self> where Self: Sized;

    /// Create model with pre-loaded weights
    fn from_weights(config: Self::Config, weights: ModelWeights) -> Result<Self> where Self: Sized;

    /// Run forward pass
    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs>;

    /// High-level text generation
    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String>;

    /// Get configuration
    fn config(&self) -> &Self::Config;

    /// Memory requirements
    fn memory_requirements(&self) -> MemoryRequirements;

    /// Move to device
    fn to_device(&mut self, device: &Device) -> Result<()>;
}
```

#### ModelConfig Trait

Interface for model configurations:

```rust
pub trait ModelConfig: Clone + Send + Sync + std::fmt::Debug {
    fn architecture(&self) -> &str;
    fn vocab_size(&self) -> usize;
    fn hidden_size(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn validate(&self) -> Result<()>;
}
```

#### model_config! Macro

Automatically generates configuration structs:

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
    rms_norm_eps: f32 = 1e-6,
});
```

This generates:
- Struct definition with all fields
- `Default` implementation with specified values
- `Clone`, `Debug` implementations
- `ModelConfig` trait implementation
- Accessor methods for all fields

#### ModelInputs / ModelOutputs

Unified input/output types:

```rust
pub enum ModelInputs {
    Text { input_ids: Tensor, attention_mask: Option<Tensor>, position_ids: Option<Tensor> },
    Image { pixel_values: Tensor, image_mask: Option<Tensor> },
    Multimodal { input_ids: Tensor, pixel_values: Option<Tensor>, ... },
    Audio { input_features: Tensor, attention_mask: Option<Tensor> },
}

pub enum ModelOutputs {
    Logits { logits: Tensor, hidden_states: Option<Tensor> },
    Embeddings { embeddings: Tensor, pooled: Option<Tensor> },
    Multimodal { text_logits: Option<Tensor>, image_features: Option<Tensor>, ... },
}
```

## WeightLoaderCore

**File:** `crates/runtime/src/weight_loader_core.rs`

WeightLoaderCore provides format-agnostic weight loading.

### Design Principles

1. **Format Agnostic** - Single interface for all formats
2. **Auto Detection** - Infer format from file extension
3. **Metadata Extraction** - Get configuration from weight files
4. **Streaming Loading** - Memory-efficient for large models

### Key Components

#### WeightLoader

Main entry point for loading weights:

```rust
pub struct WeightLoader;

impl WeightLoader {
    pub fn from_gguf(path: &str) -> Result<ModelWeights>;
    pub fn from_safetensors(path: &str) -> Result<ModelWeights>;
    pub fn from_pytorch(path: &str) -> Result<ModelWeights>;
    pub fn auto_detect(path: &str) -> Result<ModelWeights>;
}
```

#### ModelWeights

Container for loaded weights:

```rust
pub struct ModelWeights {
    tensors: HashMap<String, Tensor>,
    metadata: WeightMetadata,
}

impl ModelWeights {
    pub fn get(&self, name: &str) -> Option<&Tensor>;
    pub fn require(&self, name: &str) -> Result<&Tensor>;
    pub fn keys(&self) -> Vec<&str>;
    pub fn metadata(&self) -> &WeightMetadata;
}
```

#### GGUF Support

Handles quantized GGUF files:

```rust
// Supported quantization types
Q4_0, Q4_1, Q4_K_S, Q4_K_M,
Q5_0, Q5_1, Q5_K_S, Q5_K_M,
Q6_K, Q8_0, F16, F32

// Automatic dequantization during loading
let weights = WeightLoader::from_gguf("model-Q4_K_M.gguf")?;
// Weights are dequantized to F32
```

## Layer Interaction

The three layers work together:

```rust
// 1. Load weights (Layer 3)
let weights = WeightLoader::from_gguf("model.gguf")?;

// 2. Create model (Layer 2)
let config = LlamaConfig::from_gguf_metadata(weights.metadata())?;
let model = LlamaModelV2::from_weights(config, weights)?;

// 3. Run inference (Layer 1 used internally)
let inputs = ModelInputs::text(input_ids);
let outputs = model.forward(&inputs)?;  // Uses ops_fn internally
```

## Extension Points

### Adding a New Backend

Implement `TensorOps` trait:

```rust
pub struct MyBackendOps;

impl TensorOps for MyBackendOps {
    fn zeros(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<Tensor> {
        // Your implementation
    }
    // ... all operations
}
```

### Adding a New Model

Implement `Model` trait with `model_config!`:

```rust
model_config!(MyModelConfig {
    vocab_size: usize = 32000,
    // ...
});

pub struct MyModel { /* ... */ }

impl Model for MyModel {
    type Config = MyModelConfig;
    // ... implement methods
}
```

### Adding a New Weight Format

Extend `WeightLoader`:

```rust
impl WeightLoader {
    pub fn from_my_format(path: &str) -> Result<ModelWeights> {
        // Parse format
        // Extract tensors
        // Build metadata
        Ok(ModelWeights { tensors, metadata })
    }
}
```
