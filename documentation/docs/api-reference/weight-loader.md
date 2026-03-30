# WeightLoader API

The WeightLoaderCore module provides format-agnostic model weight loading.

## WeightLoader

The main entry point for loading model weights from various formats.

```rust
pub struct WeightLoader {
    // Internal implementation
}

impl WeightLoader {
    /// Load weights from GGUF format
    pub fn from_gguf(path: &str) -> Result<ModelWeights>;

    /// Load weights from SafeTensors format
    pub fn from_safetensors(path: &str) -> Result<ModelWeights>;

    /// Load weights from PyTorch format
    pub fn from_pytorch(path: &str) -> Result<ModelWeights>;

    /// Auto-detect format and load
    pub fn auto_detect(path: &str) -> Result<ModelWeights>;
}
```

### Usage

```rust
use unillm::weight_loader_core::WeightLoader;

// Auto-detect format
let weights = WeightLoader::auto_detect("model.gguf")?;

// Explicit format
let weights = WeightLoader::from_gguf("model.gguf")?;
let weights = WeightLoader::from_safetensors("model.safetensors")?;
```

## ModelWeights

Container for loaded model weights.

```rust
pub struct ModelWeights {
    tensors: HashMap<String, Tensor>,
    metadata: WeightMetadata,
}

impl ModelWeights {
    /// Get a tensor by name
    pub fn get(&self, name: &str) -> Option<&Tensor>;

    /// Get tensor, returning error if not found
    pub fn require(&self, name: &str) -> Result<&Tensor>;

    /// List all tensor names
    pub fn keys(&self) -> Vec<&str>;

    /// Get metadata
    pub fn metadata(&self) -> &WeightMetadata;

    /// Get total number of tensors
    pub fn len(&self) -> usize;

    /// Check if empty
    pub fn is_empty(&self) -> bool;
}
```

### Accessing Weights

```rust
let weights = WeightLoader::from_gguf("model.gguf")?;

// Get optional tensor
if let Some(tensor) = weights.get("model.embed_tokens.weight") {
    println!("Embedding shape: {:?}", tensor.shape());
}

// Get required tensor (returns error if missing)
let embed = weights.require("model.embed_tokens.weight")?;

// List all tensors
for name in weights.keys() {
    println!("Tensor: {}", name);
}
```

## WeightMetadata

Metadata extracted from weight files.

```rust
pub struct WeightMetadata {
    /// Original format
    pub format: WeightFormat,

    /// Model architecture (if available)
    pub architecture: Option<String>,

    /// Quantization type (for GGUF)
    pub quantization: Option<String>,

    /// Original data types
    pub dtypes: HashMap<String, DataType>,

    /// GGUF-specific metadata
    pub gguf_metadata: Option<GgufMetadata>,
}
```

### Accessing Metadata

```rust
let weights = WeightLoader::from_gguf("model.gguf")?;
let metadata = weights.metadata();

println!("Format: {:?}", metadata.format);

if let Some(arch) = &metadata.architecture {
    println!("Architecture: {}", arch);
}

if let Some(quant) = &metadata.quantization {
    println!("Quantization: {}", quant);
}
```

## WeightFormat

Supported weight file formats.

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightFormat {
    /// GGUF format (llama.cpp compatible)
    GGUF,

    /// SafeTensors format (HuggingFace)
    SafeTensors,

    /// PyTorch checkpoint
    PyTorch,
}
```

## GgufMetadata

GGUF-specific metadata.

```rust
pub struct GgufMetadata {
    /// Model name
    pub name: Option<String>,

    /// Model description
    pub description: Option<String>,

    /// Vocabulary size
    pub vocab_size: Option<usize>,

    /// Context length
    pub context_length: Option<usize>,

    /// Embedding length
    pub embedding_length: Option<usize>,

    /// Number of layers
    pub num_layers: Option<usize>,

    /// Quantization type
    pub quantization_type: Option<String>,

    /// Raw key-value pairs
    pub raw: HashMap<String, GgufValue>,
}
```

### Extracting Configuration from GGUF

```rust
let weights = WeightLoader::from_gguf("model.gguf")?;

if let Some(gguf) = &weights.metadata().gguf_metadata {
    let config = LlamaConfig {
        vocab_size: gguf.vocab_size.unwrap_or(32000),
        hidden_size: gguf.embedding_length.unwrap_or(4096),
        num_hidden_layers: gguf.num_layers.unwrap_or(32),
        ..Default::default()
    };
}
```

## GGUF Loading

### Supported Quantization Types

| Type | Bits | Description |
|------|------|-------------|
| F32 | 32 | Full precision |
| F16 | 16 | Half precision |
| Q8_0 | 8 | 8-bit quantization |
| Q6_K | 6 | 6-bit k-quant |
| Q5_K_M | 5 | 5-bit k-quant medium |
| Q4_K_M | 4 | 4-bit k-quant medium |
| Q4_0 | 4 | 4-bit quantization |
| Q3_K_M | 3 | 3-bit k-quant medium |
| Q2_K | 2 | 2-bit k-quant |

### Loading Quantized Models

```rust
// GGUF files are automatically dequantized to F32
let weights = WeightLoader::from_gguf("model-Q4_K_M.gguf")?;

// Weights are ready to use
let tensor = weights.require("model.layers.0.self_attn.q_proj.weight")?;
assert_eq!(tensor.dtype(), DataType::Float32);
```

## SafeTensors Loading

### Loading Multiple Shards

```rust
// Single file
let weights = WeightLoader::from_safetensors("model.safetensors")?;

// Multiple shards (auto-detected)
let weights = WeightLoader::from_safetensors("model-00001-of-00003.safetensors")?;
// Automatically loads all shards
```

### Index File Support

```rust
// Load from index file
let weights = WeightLoader::from_safetensors("model.safetensors.index.json")?;
// Automatically loads all referenced shards
```

## Weight Name Mapping

Different formats use different naming conventions. UniLLM normalizes names.

```rust
// HuggingFace style
"model.layers.0.self_attn.q_proj.weight"

// GGUF style (automatically mapped)
"blk.0.attn_q.weight"

// Both map to the same internal tensor
```

## Creating Models from Weights

```rust
use unillm::models_v2::llama::{LlamaModelV2, LlamaConfig};
use unillm::Model;

// Load weights
let weights = WeightLoader::from_gguf("llama-7b.gguf")?;

// Extract config from metadata
let config = LlamaConfig::from_gguf_metadata(weights.metadata())?;

// Create model with weights
let model = LlamaModelV2::from_weights(config, weights)?;

// Ready for inference
let response = model.generate("Hello", &GenerationConfig::default())?;
```

## Ollama Integration

Load models from Ollama registry.

```rust
use unillm::ollama::OllamaRegistry;

// Download and cache model
let model_path = OllamaRegistry::pull("llama2:7b")?;

// Load weights
let weights = WeightLoader::from_gguf(&model_path)?;
```

### Listing Cached Models

```rust
let cached = OllamaRegistry::list_cached()?;

for model in cached {
    println!("{}: {} MB", model.name, model.size / 1_000_000);
}
```

## Error Handling

```rust
use anyhow::Result;

fn load_model(path: &str) -> Result<ModelWeights> {
    let weights = WeightLoader::auto_detect(path)?;

    // Check for required tensors
    weights.require("model.embed_tokens.weight")?;
    weights.require("lm_head.weight")?;

    Ok(weights)
}
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `File not found` | Path doesn't exist | Check file path |
| `Unknown format` | Unrecognized file extension | Use explicit loader |
| `Missing tensor` | Required weight not in file | Check model compatibility |
| `Quantization unsupported` | GGUF quant type not supported | Use different quantization |

## Examples

### Loading and Inspecting Weights

```rust
let weights = WeightLoader::from_gguf("model.gguf")?;

println!("Total tensors: {}", weights.len());
println!("Format: {:?}", weights.metadata().format);

// Print all tensor shapes
for name in weights.keys() {
    let tensor = weights.get(name).unwrap();
    println!("{}: {:?}", name, tensor.shape());
}
```

### Converting Between Formats

```rust
// Load from one format
let weights = WeightLoader::from_gguf("model.gguf")?;

// Save to another (future feature)
// WeightSaver::to_safetensors(&weights, "model.safetensors")?;
```
