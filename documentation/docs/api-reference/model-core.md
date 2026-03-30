# ModelCore API

The ModelCore module provides the universal model interface and configuration system.

## Model Trait

The universal interface implemented by all 47 model architectures.

```rust
pub trait Model: Send + Sync {
    type Config: ModelConfig;

    /// Create model with default/zero weights
    fn new(config: Self::Config) -> Result<Self> where Self: Sized;

    /// Create model with loaded weights
    fn from_weights(config: Self::Config, weights: ModelWeights) -> Result<Self> where Self: Sized;

    /// Run forward pass
    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs>;

    /// Generate text from prompt
    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String>;

    /// Get model configuration
    fn config(&self) -> &Self::Config;

    /// Get memory requirements
    fn memory_requirements(&self) -> MemoryRequirements;

    /// Move model to device
    fn to_device(&mut self, device: &Device) -> Result<()>;
}
```

### Usage

```rust
use unillm::models_v2::llama::{LlamaModelV2, LlamaConfig};
use unillm::Model;

// Create model
let model = LlamaModelV2::new(LlamaConfig::default())?;

// Run inference
let outputs = model.forward(&inputs)?;

// Generate text
let response = model.generate("Hello", &GenerationConfig::default())?;

// Check configuration
println!("Hidden size: {}", model.config().hidden_size());
```

## ModelConfig Trait

The configuration interface for all models.

```rust
pub trait ModelConfig: Clone + Send + Sync + std::fmt::Debug {
    /// Get architecture name
    fn architecture(&self) -> &str;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;

    /// Get hidden dimension
    fn hidden_size(&self) -> usize;

    /// Get number of layers
    fn num_layers(&self) -> usize;

    /// Validate configuration
    fn validate(&self) -> Result<()>;
}
```

## model_config! Macro

Automatically generates configuration structs with trait implementations.

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
    pad_token_id: i64 = 0,
    bos_token_id: i64 = 1,
    eos_token_id: i64 = 2,
});
```

This generates:

- `Default` implementation
- `Clone` implementation
- `ModelConfig` implementation
- Accessor methods for all fields

## ModelInputs

Unified input types for all models.

```rust
#[derive(Debug, Clone)]
pub enum ModelInputs {
    /// Text-only inputs (language models)
    Text {
        input_ids: Tensor,
        attention_mask: Option<Tensor>,
        position_ids: Option<Tensor>,
    },

    /// Image-only inputs (vision models)
    Image {
        pixel_values: Tensor,
        image_mask: Option<Tensor>,
    },

    /// Multimodal inputs (vision-language models)
    Multimodal {
        input_ids: Tensor,
        pixel_values: Option<Tensor>,
        attention_mask: Option<Tensor>,
        image_mask: Option<Tensor>,
    },

    /// Audio inputs (speech models)
    Audio {
        input_features: Tensor,
        attention_mask: Option<Tensor>,
    },
}
```

### Creating Inputs

```rust
// Text input
let text_input = ModelInputs::Text {
    input_ids: ops_fn::zeros(&[1, 10], DataType::Int64, &Device::CPU)?,
    attention_mask: None,
    position_ids: None,
};

// Multimodal input
let multimodal_input = ModelInputs::Multimodal {
    input_ids: token_tensor,
    pixel_values: Some(image_tensor),
    attention_mask: None,
    image_mask: None,
};

// Helper constructors
let text_input = ModelInputs::text(input_ids);
let audio_input = ModelInputs::audio(features);
```

### Input Methods

```rust
impl ModelInputs {
    /// Get batch size
    pub fn batch_size(&self) -> usize;

    /// Get sequence length (for text inputs)
    pub fn sequence_length(&self) -> Option<usize>;
}
```

## ModelOutputs

Unified output types for all models.

```rust
#[derive(Debug, Clone)]
pub enum ModelOutputs {
    /// Logits output (language models)
    Logits {
        logits: Tensor,
        hidden_states: Option<Tensor>,
    },

    /// Embeddings output (encoder models)
    Embeddings {
        embeddings: Tensor,
        pooled: Option<Tensor>,
    },

    /// Multimodal output (vision-language models)
    Multimodal {
        text_logits: Option<Tensor>,
        image_features: Option<Tensor>,
        cross_attention: Option<Tensor>,
    },
}
```

### Processing Outputs

```rust
let outputs = model.forward(&inputs)?;

match outputs {
    ModelOutputs::Logits { logits, hidden_states } => {
        // logits shape: [batch, seq_len, vocab_size]
        println!("Logits shape: {:?}", logits.shape());

        if let Some(hidden) = hidden_states {
            println!("Hidden states shape: {:?}", hidden.shape());
        }
    }
    ModelOutputs::Embeddings { embeddings, pooled } => {
        // embeddings shape: [batch, seq_len, hidden_size]
        println!("Embeddings shape: {:?}", embeddings.shape());
    }
    ModelOutputs::Multimodal { text_logits, image_features, .. } => {
        // Process multimodal outputs
    }
}
```

## GenerationConfig

Configuration for text generation.

```rust
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum new tokens to generate
    pub max_new_tokens: usize,

    /// Sampling temperature
    pub temperature: f32,

    /// Top-p (nucleus) sampling
    pub top_p: f32,

    /// Top-k sampling (None = disabled)
    pub top_k: Option<usize>,

    /// Enable sampling (vs greedy)
    pub do_sample: bool,

    /// Repetition penalty
    pub repetition_penalty: f32,

    /// Stop strings
    pub stop_sequences: Vec<String>,

    /// End of sequence token
    pub eos_token_id: u32,

    /// Padding token
    pub pad_token_id: u32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 100,
            temperature: 1.0,
            top_p: 1.0,
            top_k: None,
            do_sample: false,
            repetition_penalty: 1.0,
            stop_sequences: vec![],
            eos_token_id: 2,
            pad_token_id: 0,
        }
    }
}
```

## MemoryRequirements

Memory usage information for models.

```rust
#[derive(Debug, Clone)]
pub struct MemoryRequirements {
    /// GPU memory in bytes
    pub gpu_memory: usize,

    /// CPU memory in bytes
    pub cpu_memory: usize,

    /// KV cache memory in bytes
    pub kv_cache_memory: usize,

    /// Peak memory usage in bytes
    pub peak_memory: usize,
}
```

### Usage

```rust
let requirements = model.memory_requirements();

println!("GPU Memory: {} MB", requirements.gpu_memory / 1_000_000);
println!("KV Cache: {} MB", requirements.kv_cache_memory / 1_000_000);
```

## Available Models

All models implement the `Model` trait:

```rust
// Core LLMs
pub use llama::LlamaModelV2;
pub use qwen::QwenModelV2;
pub use gemma::GemmaModelV2;
pub use phi::PhiModelV2;
pub use mistral::MistralModelV2;

// MoE Models
pub use mixtral::MixtralModelV2;
pub use deepseek_moe::DeepSeekMoEModelV2;
pub use jamba::JambaModelV2;

// Vision-Language
pub use llava::LlavaModelV2;
pub use clip::ClipModelV2;
pub use qwen2_vl::Qwen2VLModelV2;

// Audio
pub use whisper::WhisperModelV2;
pub use wav2vec2::Wav2Vec2ModelV2;

// ... 35 more models
```
