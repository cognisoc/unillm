# Inference Pipeline API

The inference module provides high-level inference pipeline and sampling utilities.

## InferencePipeline

High-level interface for running model inference.

```rust
pub struct InferencePipeline<M: Model> {
    model: M,
    tokenizer: Tokenizer,
    device: Device,
}

impl<M: Model> InferencePipeline<M> {
    /// Create new pipeline
    pub fn new(model: M, tokenizer: Tokenizer, device: Device) -> Self;

    /// Generate text from prompt
    pub fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String>;

    /// Generate with streaming callback
    pub fn generate_streaming<F>(&self, prompt: &str, config: &GenerationConfig, callback: F) -> Result<String>
    where
        F: FnMut(&str);

    /// Run forward pass with raw inputs
    pub fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs>;

    /// Get next token logits
    pub fn get_logits(&self, input_ids: &Tensor) -> Result<Tensor>;
}
```

### Basic Usage

```rust
use unillm::inference::InferencePipeline;
use unillm::models_v2::llama::{LlamaModelV2, LlamaConfig};

// Create model and tokenizer
let model = LlamaModelV2::from_weights(config, weights)?;
let tokenizer = Tokenizer::from_gguf("model.gguf")?;

// Create pipeline
let pipeline = InferencePipeline::new(model, tokenizer, Device::auto());

// Generate text
let response = pipeline.generate("Hello, world!", &GenerationConfig::default())?;
```

### Streaming Generation

```rust
let config = GenerationConfig {
    max_new_tokens: 100,
    ..Default::default()
};

pipeline.generate_streaming("Once upon a time", &config, |token| {
    print!("{}", token);
    std::io::stdout().flush().unwrap();
})?;
```

## Sampler

Token sampling strategies for text generation.

```rust
pub struct Sampler {
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    repetition_penalty: f32,
}

impl Sampler {
    /// Create new sampler with config
    pub fn new(config: &GenerationConfig) -> Self;

    /// Sample next token from logits
    pub fn sample(&self, logits: &Tensor, past_tokens: &[u32]) -> Result<u32>;

    /// Apply temperature scaling
    pub fn apply_temperature(&self, logits: &Tensor) -> Result<Tensor>;

    /// Apply top-k filtering
    pub fn apply_top_k(&self, logits: &Tensor) -> Result<Tensor>;

    /// Apply top-p (nucleus) filtering
    pub fn apply_top_p(&self, logits: &Tensor) -> Result<Tensor>;

    /// Apply repetition penalty
    pub fn apply_repetition_penalty(&self, logits: &Tensor, past_tokens: &[u32]) -> Result<Tensor>;
}
```

### Sampling Strategies

```rust
use unillm::sampler::Sampler;

// Greedy sampling (temperature = 0)
let greedy = Sampler::new(&GenerationConfig {
    do_sample: false,
    ..Default::default()
});

// Temperature sampling
let temp_sampler = Sampler::new(&GenerationConfig {
    do_sample: true,
    temperature: 0.7,
    ..Default::default()
});

// Top-p (nucleus) sampling
let nucleus_sampler = Sampler::new(&GenerationConfig {
    do_sample: true,
    temperature: 0.8,
    top_p: 0.9,
    ..Default::default()
});

// Combined sampling
let combined = Sampler::new(&GenerationConfig {
    do_sample: true,
    temperature: 0.7,
    top_p: 0.9,
    top_k: Some(50),
    repetition_penalty: 1.1,
    ..Default::default()
});
```

### Manual Sampling

```rust
let logits = model.get_logits(&input_ids)?;
let sampler = Sampler::new(&gen_config);

// Apply all transformations
let scaled = sampler.apply_temperature(&logits)?;
let filtered = sampler.apply_top_k(&scaled)?;
let nucleus = sampler.apply_top_p(&filtered)?;
let penalized = sampler.apply_repetition_penalty(&nucleus, &past_tokens)?;

// Sample token
let next_token = sampler.sample(&penalized, &past_tokens)?;
```

## Tokenizer

Text tokenization and detokenization.

```rust
pub struct Tokenizer {
    // Internal implementation
}

impl Tokenizer {
    /// Load tokenizer from GGUF file
    pub fn from_gguf(path: &str) -> Result<Self>;

    /// Load tokenizer from HuggingFace tokenizers JSON
    pub fn from_hf_tokenizers(path: &str) -> Result<Self>;

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>>;

    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32]) -> Result<String>;

    /// Decode single token
    pub fn decode_token(&self, id: u32) -> Result<String>;

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize;

    /// Get special tokens
    pub fn special_tokens(&self) -> &SpecialTokens;
}
```

### Tokenization

```rust
use unillm::tokenizer::Tokenizer;

let tokenizer = Tokenizer::from_gguf("model.gguf")?;

// Encode text
let ids = tokenizer.encode("Hello, world!")?;
println!("Token IDs: {:?}", ids);

// Decode back to text
let text = tokenizer.decode(&ids)?;
println!("Decoded: {}", text);

// Single token decode
for id in &ids {
    let token = tokenizer.decode_token(*id)?;
    println!("{}: {:?}", id, token);
}
```

### Special Tokens

```rust
pub struct SpecialTokens {
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub unk_token_id: u32,
}

let special = tokenizer.special_tokens();
println!("BOS: {}", special.bos_token_id);
println!("EOS: {}", special.eos_token_id);
```

## GenerationConfig

Configuration for text generation.

```rust
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum new tokens to generate
    pub max_new_tokens: usize,

    /// Sampling temperature (0.0 = greedy)
    pub temperature: f32,

    /// Nucleus sampling threshold
    pub top_p: f32,

    /// Top-k sampling (None = disabled)
    pub top_k: Option<usize>,

    /// Enable sampling vs greedy
    pub do_sample: bool,

    /// Repetition penalty
    pub repetition_penalty: f32,

    /// Stop sequences
    pub stop_sequences: Vec<String>,

    /// End of sequence token ID
    pub eos_token_id: u32,

    /// Padding token ID
    pub pad_token_id: u32,
}
```

### Preset Configurations

```rust
impl GenerationConfig {
    /// Greedy decoding (deterministic)
    pub fn greedy() -> Self {
        Self {
            do_sample: false,
            temperature: 0.0,
            ..Default::default()
        }
    }

    /// Creative writing
    pub fn creative() -> Self {
        Self {
            do_sample: true,
            temperature: 1.0,
            top_p: 0.95,
            repetition_penalty: 1.2,
            ..Default::default()
        }
    }

    /// Balanced (general purpose)
    pub fn balanced() -> Self {
        Self {
            do_sample: true,
            temperature: 0.7,
            top_p: 0.9,
            top_k: Some(50),
            repetition_penalty: 1.1,
            ..Default::default()
        }
    }

    /// Code generation
    pub fn code() -> Self {
        Self {
            do_sample: true,
            temperature: 0.2,
            top_p: 0.95,
            max_new_tokens: 512,
            ..Default::default()
        }
    }
}
```

## KV Cache

Key-value cache for efficient autoregressive generation.

```rust
pub struct KVCache {
    // Internal implementation
}

impl KVCache {
    /// Create new cache
    pub fn new(num_layers: usize, max_seq_len: usize, num_heads: usize, head_dim: usize) -> Self;

    /// Get cached key/value for a layer
    pub fn get(&self, layer: usize) -> Option<(&Tensor, &Tensor)>;

    /// Update cache with new key/value
    pub fn update(&mut self, layer: usize, key: Tensor, value: Tensor);

    /// Get current sequence length
    pub fn seq_len(&self) -> usize;

    /// Clear cache
    pub fn clear(&mut self);
}
```

### Using KV Cache

```rust
let mut cache = KVCache::new(32, 2048, 32, 128);

// First token - no cache
let outputs = model.forward_with_cache(&inputs, &mut cache)?;

// Subsequent tokens - use cache
for _ in 0..max_tokens {
    let outputs = model.forward_with_cache(&next_input, &mut cache)?;
    // Cache automatically updated
}
```

## Batch Inference

Running inference on multiple inputs.

```rust
// Create batched inputs
let batch_size = 4;
let prompts = vec![
    "Hello",
    "How are you",
    "What is AI",
    "Tell me a story",
];

// Tokenize all prompts
let batch_ids: Vec<Vec<u32>> = prompts.iter()
    .map(|p| tokenizer.encode(p))
    .collect::<Result<_>>()?;

// Pad to same length
let max_len = batch_ids.iter().map(|ids| ids.len()).max().unwrap();
let padded = pad_sequences(&batch_ids, max_len, tokenizer.special_tokens().pad_token_id);

// Create batched tensor
let input_tensor = Tensor::from_slice(&padded, &[batch_size, max_len])?;

// Run batched forward pass
let outputs = model.forward(&ModelInputs::text(input_tensor))?;
```

## Error Handling

```rust
use anyhow::Result;

fn generate_with_fallback(pipeline: &InferencePipeline<impl Model>, prompt: &str) -> Result<String> {
    // Try with preferred settings
    let config = GenerationConfig::balanced();

    match pipeline.generate(prompt, &config) {
        Ok(response) => Ok(response),
        Err(e) => {
            eprintln!("Generation failed: {}", e);
            // Fallback to greedy
            pipeline.generate(prompt, &GenerationConfig::greedy())
        }
    }
}
```

## Examples

### Complete Generation Pipeline

```rust
use unillm::prelude::*;

fn main() -> Result<()> {
    // Load model
    let weights = WeightLoader::from_gguf("model.gguf")?;
    let config = LlamaConfig::from_gguf_metadata(weights.metadata())?;
    let model = LlamaModelV2::from_weights(config, weights)?;

    // Load tokenizer
    let tokenizer = Tokenizer::from_gguf("model.gguf")?;

    // Create pipeline
    let pipeline = InferencePipeline::new(model, tokenizer, Device::auto());

    // Generate
    let response = pipeline.generate(
        "Explain quantum computing in simple terms:",
        &GenerationConfig::balanced(),
    )?;

    println!("{}", response);
    Ok(())
}
```

### Chat Interface

```rust
fn chat_loop(pipeline: &InferencePipeline<impl Model>) -> Result<()> {
    let mut history = String::new();

    loop {
        print!("User: ");
        let input = read_line()?;

        if input == "quit" {
            break;
        }

        history.push_str(&format!("User: {}\nAssistant: ", input));

        let response = pipeline.generate(
            &history,
            &GenerationConfig {
                max_new_tokens: 256,
                stop_sequences: vec!["User:".to_string()],
                ..GenerationConfig::balanced()
            },
        )?;

        println!("Assistant: {}", response);
        history.push_str(&format!("{}\n", response));
    }

    Ok(())
}
```
