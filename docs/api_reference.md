# UniLLM API Reference

Complete API documentation for UniLLM's clean abstraction layers and model implementations.

## Core Abstractions

### TensorCore (`tensor_core.rs`)

#### `Tensor` - Universal Tensor Type

```rust
pub struct Tensor {
    // Device-agnostic tensor with automatic memory management
}

impl Tensor {
    // Shape and metadata
    pub fn shape(&self) -> &[usize];
    pub fn device(&self) -> &Device;
    pub fn dtype(&self) -> DataType;
    pub fn numel(&self) -> usize;

    // Device operations
    pub fn to_device(&self, device: &Device) -> Result<Tensor>;
    pub fn is_on_device(&self, device: &Device) -> bool;

    // Data access (when needed)
    pub fn to_vec<T: Copy>(&self) -> Result<Vec<T>>;

    // Operations interface
    pub fn ops(&self) -> &dyn TensorOps;
}
```

#### `Device` - Hardware Abstraction

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum Device {
    CPU,
    CUDA(usize),  // GPU index
    Metal(usize), // GPU index
}

impl Device {
    pub fn auto() -> Device;  // Automatic best device selection
    pub fn is_gpu(&self) -> bool;
    pub fn index(&self) -> Option<usize>;
}
```

#### `TensorOps` - Operation Interface

```rust
pub trait TensorOps: Send + Sync {
    // Creation
    fn zeros(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<Tensor>;
    fn ones(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<Tensor>;
    fn rand(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<Tensor>;

    // Basic math
    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

    // Neural network operations
    fn layer_norm(&self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>, eps: f32) -> Result<Tensor>;
    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor>;
    fn embedding(&self, indices: &Tensor, weight: &Tensor) -> Result<Tensor>;
    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor>;

    // Activation functions
    fn relu(&self, input: &Tensor) -> Result<Tensor>;
    fn silu(&self, input: &Tensor) -> Result<Tensor>;
    fn gelu(&self, input: &Tensor) -> Result<Tensor>;
    fn softmax(&self, input: &Tensor, dim: isize) -> Result<Tensor>;

    // Shape operations
    fn reshape(&self, input: &Tensor, shape: &[usize]) -> Result<Tensor>;
    fn transpose(&self, input: &Tensor, dim0: usize, dim1: usize) -> Result<Tensor>;
    fn concat(&self, tensors: &[&Tensor], dim: usize) -> Result<Tensor>;
    fn slice(&self, input: &Tensor, ranges: &[(usize, usize)]) -> Result<Tensor>;
}
```

#### `ops_fn` - Functional Interface

```rust
pub mod ops_fn {
    // Tensor creation
    pub fn zeros(shape: &[usize], dtype: DataType, device: &Device) -> Result<Tensor>;
    pub fn ones(shape: &[usize], dtype: DataType, device: &Device) -> Result<Tensor>;
    pub fn rand(shape: &[usize], dtype: DataType, device: &Device) -> Result<Tensor>;

    // Math operations
    pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor>;
    pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor>;
    pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor>;
    pub fn scale(input: &Tensor, factor: f32) -> Result<Tensor>;

    // Neural network layers
    pub fn linear(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor>;
    pub fn layer_norm(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>, eps: f32) -> Result<Tensor>;
    pub fn rms_norm(input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor>;
    pub fn embedding(indices: &Tensor, weight: &Tensor) -> Result<Tensor>;
    pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor>;

    // Activations
    pub fn relu(input: &Tensor) -> Result<Tensor>;
    pub fn silu(input: &Tensor) -> Result<Tensor>;
    pub fn gelu(input: &Tensor) -> Result<Tensor>;
    pub fn softmax(input: &Tensor, dim: isize) -> Result<Tensor>;

    // Shape operations
    pub fn reshape(input: &Tensor, shape: &[usize]) -> Result<Tensor>;
    pub fn transpose(input: &Tensor, dim0: usize, dim1: usize) -> Result<Tensor>;
    pub fn concat(tensors: &[&Tensor], dim: usize) -> Result<Tensor>;
}
```

### ModelCore (`model_core.rs`)

#### `Model` - Universal Model Interface

```rust
pub trait Model: Send + Sync {
    type Config: ModelConfig;

    // Model lifecycle
    fn new(config: Self::Config) -> Result<Self> where Self: Sized;
    fn from_weights(config: Self::Config, weights: ModelWeights) -> Result<Self> where Self: Sized;

    // Inference
    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs>;
    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String>;

    // Configuration and metadata
    fn config(&self) -> &Self::Config;
    fn memory_requirements(&self) -> MemoryRequirements;
    fn to_device(&mut self, device: &Device) -> Result<()>;
}
```

#### `ModelConfig` - Configuration Interface

```rust
pub trait ModelConfig: Clone + Send + Sync + std::fmt::Debug {
    fn architecture(&self) -> &str;
    fn vocab_size(&self) -> usize;
    fn hidden_size(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn validate(&self) -> Result<()>;
}
```

#### `model_config!` - Configuration Macro

```rust
// Automatically generates ModelConfig implementation
model_config!(YourConfig {
    vocab_size: usize = 32000,
    hidden_size: usize = 4096,
    num_hidden_layers: usize = 32,  // or n_layer: usize
    // ... other fields with defaults
});

// Expands to:
impl Default for YourConfig { /* ... */ }
impl ModelConfig for YourConfig {
    fn architecture(&self) -> &str { "YourConfig" }
    fn vocab_size(&self) -> usize { self.vocab_size }
    fn hidden_size(&self) -> usize {
        // Tries hidden_size, then d_model
        self.hidden_size  // or self.d_model
    }
    fn num_layers(&self) -> usize {
        // Tries num_hidden_layers, then n_layer
        self.num_hidden_layers  // or self.n_layer
    }
    fn validate(&self) -> Result<()> { /* validation logic */ }
}
```

#### Input/Output Types

```rust
#[derive(Debug, Clone)]
pub enum ModelInputs {
    Text {
        input_ids: Tensor,
        attention_mask: Option<Tensor>,
        position_ids: Option<Tensor>,
    },
    Image {
        pixel_values: Tensor,
        image_mask: Option<Tensor>,
    },
    Multimodal {
        input_ids: Tensor,
        pixel_values: Option<Tensor>,
        attention_mask: Option<Tensor>,
        image_mask: Option<Tensor>,
    },
    Audio {
        input_features: Tensor,
        attention_mask: Option<Tensor>,
    },
}

#[derive(Debug, Clone)]
pub enum ModelOutputs {
    Logits {
        logits: Tensor,
        hidden_states: Option<Tensor>,
    },
    Embeddings {
        embeddings: Tensor,
        pooled: Option<Tensor>,
    },
    Multimodal {
        text_logits: Option<Tensor>,
        image_features: Option<Tensor>,
        cross_attention: Option<Tensor>,
    },
}
```

#### Generation Configuration

```rust
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub do_sample: bool,
    pub repetition_penalty: f32,
    pub stop_sequences: Vec<String>,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
}
```

### WeightLoaderCore (`weight_loader_core.rs`)

#### `WeightLoader` - Format-Agnostic Loading

```rust
pub struct WeightLoader;

impl WeightLoader {
    // Format-specific loaders
    pub fn from_safetensors<P: AsRef<Path>>(path: P) -> Result<ModelWeights>;
    pub fn from_gguf<P: AsRef<Path>>(path: P) -> Result<ModelWeights>;
    pub fn from_pytorch<P: AsRef<Path>>(path: P) -> Result<ModelWeights>;

    // Automatic format detection
    pub fn auto_detect<P: AsRef<Path>>(path: P) -> Result<ModelWeights>;

    // Remote loading (planned)
    pub fn from_hf_hub(repo_id: &str, filename: &str) -> Result<ModelWeights>;
}
```

#### `ModelWeights` - Weight Container

```rust
pub struct ModelWeights {
    // Internal storage abstracted away
}

impl ModelWeights {
    pub fn get(&self, key: &str) -> Option<&Tensor>;
    pub fn keys(&self) -> impl Iterator<Item = &str>;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;

    // Metadata
    pub fn metadata(&self) -> &WeightMetadata;
    pub fn format(&self) -> WeightFormat;

    // Device management
    pub fn to_device(&mut self, device: &Device) -> Result<()>;
}
```

## Model Implementations

### Supported Model Families

All models implement the `Model` trait with consistent patterns:

```rust
// Available in models_v2::
pub use llama::LlamaModelV2;      // Llama, Llama2, Code Llama, etc.
pub use qwen::QwenModelV2;        // Qwen, Qwen1.5, Qwen2, QwenVL
pub use gemma::GemmaModelV2;      // Gemma-2B, Gemma-7B
pub use phi::PhiModelV2;          // Phi-1, Phi-1.5, Phi-2, Phi-3
pub use deepseek::DeepSeekModelV2;// DeepSeek-Coder, DeepSeek-Chat
pub use yi::YiModelV2;            // Yi-6B, Yi-34B, Yi-Chat
pub use baichuan::BaichuanModelV2;// Baichuan-7B, Baichuan-13B
pub use internlm::InternLMModelV2;// InternLM, InternLM2
pub use chatglm::ChatGLMModelV2;  // ChatGLM-6B, ChatGLM2-6B, ChatGLM3
pub use falcon::FalconModelV2;    // Falcon-7B, Falcon-40B
pub use bert::BertModelV2;        // BERT, RoBERTa, etc.
pub use t5::T5ModelV2;            // T5, UL2, Flan-T5
pub use whisper::WhisperModelV2;  // Whisper speech recognition
pub use clip::ClipModelV2;        // CLIP vision-language
pub use llava::LlavaModelV2;      // LLaVA multimodal
pub use mamba::MambaModelV2;      // Mamba state-space models
pub use minicpm::MiniCPMModelV2;  // MiniCPM efficient models
```

### Common Usage Pattern

All models follow the same interface:

```rust
use unillm::models_v2::llama::{LlamaModelV2, LlamaConfig};
use unillm::{Model, ModelInputs, ModelOutputs, GenerationConfig, WeightLoader};

// 1. Create configuration
let config = LlamaConfig {
    vocab_size: 32000,
    hidden_size: 4096,
    num_hidden_layers: 32,
    num_attention_heads: 32,
    ..Default::default()
};

// 2. Load model with weights
let weights = WeightLoader::from_safetensors("model.safetensors")?;
let model = LlamaModelV2::from_weights(config, weights)?;

// 3. Prepare inputs
let input_tensor = ops_fn::zeros(&[1, 10], DataType::Int64, &Device::CPU)?;
let inputs = ModelInputs::Text {
    input_ids: input_tensor,
    attention_mask: None,
    position_ids: None,
};

// 4. Run inference
let outputs = model.forward(&inputs)?;
match outputs {
    ModelOutputs::Logits { logits, .. } => {
        // Process logits
        println!("Output shape: {:?}", logits.shape());
    },
    _ => unreachable!(),
}

// 5. Generate text
let gen_config = GenerationConfig::default();
let response = model.generate("Hello world", &gen_config)?;
```

## Inference Pipeline (`inference.rs`)

### `InferencePipeline` - End-to-End Generation

```rust
pub struct InferencePipeline {
    // Internal model, tokenizer, and sampler
}

impl InferencePipeline {
    pub fn new(model: impl Model, tokenizer: Tokenizer) -> Self;

    // High-level generation
    pub fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String>;

    // Low-level token generation
    pub fn generate_tokens(&self, input_tokens: &[u32], config: &GenerationConfig) -> Result<Vec<u32>>;

    // Configuration access
    pub fn model_config(&self) -> &dyn ModelConfig;
    pub fn tokenizer(&self) -> &Tokenizer;
}
```

### `InferencePipelineBuilder` - Builder Pattern

```rust
pub struct InferencePipelineBuilder;

impl InferencePipelineBuilder {
    pub fn new() -> Self;
    pub fn with_model_config(self, config: impl ModelConfig) -> Self;
    pub fn with_tokenizer(self, tokenizer: Tokenizer) -> Self;
    pub fn build(self) -> Result<InferencePipeline>;
}

// Usage:
let pipeline = InferencePipelineBuilder::new()
    .with_model_config(config)
    .with_tokenizer(tokenizer)
    .build()?;
```

### `Sampler` - Text Generation Strategies

```rust
pub struct Sampler;

impl Sampler {
    pub fn new() -> Self;

    // Sampling strategies
    pub fn sample_greedy(&self, logits: &[f32]) -> Result<u32>;
    pub fn sample_temperature(&self, logits: &[f32], temperature: f32) -> Result<u32>;
    pub fn sample_top_p(&self, logits: &[f32], top_p: f32, temperature: f32) -> Result<u32>;
    pub fn sample_top_k(&self, logits: &[f32], top_k: usize, temperature: f32) -> Result<u32>;
}
```

## Tokenization (`tokenizer.rs`)

### `Tokenizer` - Text Processing

```rust
pub struct Tokenizer {
    // Internal tokenizer state
}

impl Tokenizer {
    pub fn new() -> Self;

    // Basic operations
    pub fn encode(&self, text: &str) -> Vec<u32>;
    pub fn decode(&self, tokens: &[u32]) -> String;

    // Configuration
    pub fn vocab_size(&self) -> usize;
    pub fn eos_token_id(&self) -> u32;
    pub fn bos_token_id(&self) -> u32;
    pub fn pad_token_id(&self) -> u32;

    // Advanced features (planned)
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<u32>>;
    pub fn decode_batch(&self, token_batches: &[&[u32]]) -> Vec<String>;
}
```

## Sampling (`sampler.rs`)

### Sampling Strategies

```rust
pub enum SamplingStrategy {
    Greedy,
    Temperature { temperature: f32 },
    TopP { top_p: f32, temperature: f32 },
    TopK { top_k: usize, temperature: f32 },
    Nucleus { top_p: f32, top_k: Option<usize>, temperature: f32 },
}

pub struct AdvancedSampler {
    strategy: SamplingStrategy,
}

impl AdvancedSampler {
    pub fn new(strategy: SamplingStrategy) -> Self;
    pub fn sample(&self, logits: &[f32]) -> Result<u32>;
    pub fn sample_batch(&self, logits_batch: &[&[f32]]) -> Result<Vec<u32>>;
}
```

## Types and Utilities (`types.rs`)

### Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Model initialization failed: {0}")]
    InitializationFailed(String),

    #[error("Computation failed: {0}")]
    ComputationFailed(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Device error: {0}")]
    DeviceError(String),

    #[error("Weight loading error: {0}")]
    WeightLoadingError(String),
}

pub type ModelResult<T> = std::result::Result<T, ModelError>;
```

### Memory Management

```rust
#[derive(Debug, Clone)]
pub struct MemoryRequirements {
    pub gpu_memory: usize,        // GPU memory in bytes
    pub cpu_memory: usize,        // CPU memory in bytes
    pub kv_cache_memory: usize,   // KV cache memory in bytes
    pub peak_memory: usize,       // Peak memory usage in bytes
}
```

## Testing and Validation

### Test Utilities

```rust
pub mod test_utils {
    // Model testing helpers
    pub fn create_dummy_config<C: ModelConfig + Default>() -> C;
    pub fn create_test_tensor(shape: &[usize]) -> Tensor;
    pub fn assert_tensor_shape(tensor: &Tensor, expected: &[usize]);

    // Performance testing
    pub fn benchmark_model_forward<M: Model>(model: &M, inputs: &ModelInputs) -> Duration;
    pub fn benchmark_generation<M: Model>(model: &M, prompt: &str) -> Duration;
}
```

## Performance Monitoring (`simple_observability.rs`)

### Metrics Collection

```rust
pub struct InferenceMetrics {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub total_tokens: usize,
    pub inference_time_ms: f64,
    pub tokens_per_second: f64,
    pub memory_usage_mb: f64,
}

pub struct MetricsCollector {
    // Internal state
}

impl MetricsCollector {
    pub fn new() -> Self;
    pub fn record_inference(&mut self, metrics: InferenceMetrics);
    pub fn get_stats(&self) -> InferenceStats;
    pub fn reset(&mut self);
}
```

---

## Example Usage

### Complete End-to-End Example

```rust
use unillm::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Set up model configuration
    let config = LlamaConfig {
        vocab_size: 32000,
        hidden_size: 4096,
        num_hidden_layers: 32,
        num_attention_heads: 32,
        intermediate_size: 11008,
        max_position_embeddings: 2048,
        ..Default::default()
    };

    // 2. Load model weights
    let weights = WeightLoader::from_safetensors("llama-7b-hf/model.safetensors")?;
    let mut model = LlamaModelV2::from_weights(config, weights)?;

    // 3. Move to GPU if available
    let device = Device::auto();
    model.to_device(&device)?;

    // 4. Create tokenizer
    let tokenizer = Tokenizer::new();

    // 5. Build inference pipeline
    let pipeline = InferencePipeline::new(model, tokenizer);

    // 6. Configure generation
    let gen_config = GenerationConfig {
        max_new_tokens: 100,
        temperature: 0.7,
        top_p: 0.9,
        do_sample: true,
        ..Default::default()
    };

    // 7. Generate text
    let response = pipeline.generate("The future of AI is", &gen_config)?;
    println!("Generated: {}", response);

    Ok(())
}
```

This API design emphasizes clean abstractions, consistent patterns, and ease of use while providing the flexibility to extend and optimize for different use cases.