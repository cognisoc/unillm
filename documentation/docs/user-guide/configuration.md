# Configuration

This guide covers configuration options for models and generation in UniLLM.

## Model Configuration

### Using the `model_config!` Macro

UniLLM uses the `model_config!` macro to define model configurations:

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
    // ... additional fields
});
```

### Common Configuration Fields

| Field | Description | Typical Values |
|-------|-------------|----------------|
| `vocab_size` | Vocabulary size | 32000 - 128000 |
| `hidden_size` | Hidden dimension | 768 - 8192 |
| `intermediate_size` | FFN intermediate size | 2048 - 28672 |
| `num_hidden_layers` | Number of transformer layers | 12 - 80 |
| `num_attention_heads` | Number of attention heads | 12 - 64 |
| `num_key_value_heads` | KV heads (for GQA) | 1 - 64 |
| `max_position_embeddings` | Maximum sequence length | 2048 - 128000 |

### Creating Custom Configurations

```rust
// Use default values
let config = LlamaConfig::default();

// Override specific fields
let config = LlamaConfig {
    hidden_size: 2048,
    num_hidden_layers: 16,
    ..Default::default()
};

// Load from GGUF metadata
let gguf_config = weights.metadata().gguf_config();
let config = LlamaConfig::from_gguf_config(&gguf_config);
```

## Generation Configuration

### GenerationConfig Fields

```rust
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate
    pub max_new_tokens: usize,

    /// Sampling temperature (0.0 = greedy, higher = more random)
    pub temperature: f32,

    /// Nucleus sampling probability threshold
    pub top_p: f32,

    /// Top-k sampling (None = disabled)
    pub top_k: Option<usize>,

    /// Whether to use sampling (false = greedy decoding)
    pub do_sample: bool,

    /// Penalty for repeating tokens
    pub repetition_penalty: f32,

    /// Strings that stop generation
    pub stop_sequences: Vec<String>,

    /// End of sequence token ID
    pub eos_token_id: u32,

    /// Padding token ID
    pub pad_token_id: u32,
}
```

### Preset Configurations

```rust
// Deterministic (for code, facts)
let deterministic = GenerationConfig {
    do_sample: false,
    temperature: 0.0,
    ..Default::default()
};

// Creative (for stories, brainstorming)
let creative = GenerationConfig {
    do_sample: true,
    temperature: 1.0,
    top_p: 0.95,
    repetition_penalty: 1.2,
    ..Default::default()
};

// Balanced (general use)
let balanced = GenerationConfig {
    do_sample: true,
    temperature: 0.7,
    top_p: 0.9,
    top_k: Some(50),
    repetition_penalty: 1.1,
    ..Default::default()
};

// Code generation
let code = GenerationConfig {
    do_sample: true,
    temperature: 0.2,
    top_p: 0.95,
    max_new_tokens: 512,
    stop_sequences: vec!["```".to_string()],
    ..Default::default()
};
```

## Device Configuration

### Device Enum

```rust
pub enum Device {
    CPU,
    CUDA(usize),  // GPU index
    Metal(usize), // Apple GPU index
}
```

### Device Selection

```rust
use unillm::tensor_core::Device;

// Automatic best device
let device = Device::auto();

// Specific device
let device = Device::CPU;
let device = Device::CUDA(0);  // First NVIDIA GPU
let device = Device::Metal(0); // First Apple GPU

// Check device type
if device.is_gpu() {
    println!("Using GPU {}", device.index().unwrap());
}
```

## Model-Specific Configurations

### LLaMA Configuration

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
    rope_scaling: Option<String> = None,
    rms_norm_eps: f32 = 1e-6,
});
```

### Mixtral (MoE) Configuration

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
    // ...
});
```

### Vision-Language Configuration

```rust
model_config!(LlavaConfig {
    // Text model config
    vocab_size: usize = 32000,
    hidden_size: usize = 4096,
    // Vision config
    image_size: usize = 336,
    patch_size: usize = 14,
    vision_hidden_size: usize = 1024,
    // Projector config
    projector_hidden_size: usize = 4096,
    // ...
});
```

## Environment Variables

UniLLM respects several environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `UNILLM_CACHE_DIR` | Model cache directory | `~/.cache/unillm` |
| `UNILLM_LOG_LEVEL` | Logging verbosity | `info` |
| `CUDA_VISIBLE_DEVICES` | GPU visibility | All GPUs |

```bash
# Set cache directory
export UNILLM_CACHE_DIR=/path/to/cache

# Enable debug logging
export UNILLM_LOG_LEVEL=debug

# Use only GPU 0 and 1
export CUDA_VISIBLE_DEVICES=0,1
```

## Configuration Validation

Configurations are validated when creating models:

```rust
let config = LlamaConfig {
    vocab_size: 0,  // Invalid!
    ..Default::default()
};

// This will return an error
let result = LlamaModelV2::new(config);
assert!(result.is_err());
```

The `ModelConfig` trait provides validation:

```rust
impl ModelConfig for LlamaConfig {
    fn validate(&self) -> Result<()> {
        if self.vocab_size() == 0 {
            return Err(anyhow::anyhow!("vocab_size must be > 0"));
        }
        if self.hidden_size() == 0 {
            return Err(anyhow::anyhow!("hidden_size must be > 0"));
        }
        Ok(())
    }
}
```

## Next Steps

- Explore the [Model Catalog](../models/index.md) for model-specific configurations
- See [API Reference](../api-reference/index.md) for complete API documentation
