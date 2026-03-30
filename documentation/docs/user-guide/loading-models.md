# Loading Models

UniLLM supports loading models from multiple formats with a unified interface.

## Supported Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| **GGUF** | `.gguf` | Quantized format used by llama.cpp, Ollama |
| **SafeTensors** | `.safetensors` | HuggingFace's safe serialization format |
| **PyTorch** | `.bin`, `.pt` | PyTorch checkpoint format |

## Using WeightLoader

The `WeightLoader` provides format-agnostic loading:

### Auto-Detection

```rust
use unillm::weight_loader_core::WeightLoader;

// Automatically detect format from file extension
let weights = WeightLoader::auto_detect("path/to/model")?;
```

### Format-Specific Loading

```rust
// Load GGUF (Ollama/llama.cpp format)
let weights = WeightLoader::from_gguf("model.gguf")?;

// Load SafeTensors (HuggingFace format)
let weights = WeightLoader::from_safetensors("model.safetensors")?;

// Load PyTorch checkpoint
let weights = WeightLoader::from_pytorch("model.bin")?;
```

## Loading from Ollama

The easiest way to get models is through the Ollama integration:

```rust
use unillm::ollama::OllamaRegistry;

// Download and cache model
let registry = OllamaRegistry::new();
let model_path = registry.pull("tinyllama")?;

// Load the downloaded model
let weights = WeightLoader::from_gguf(&model_path)?;
```

### Available Ollama Models

```bash
# List some popular models
tinyllama      # 1.1B parameters, ~600MB
llama2:7b      # 7B parameters, ~4GB (Q4)
mistral:7b     # 7B parameters, ~4GB (Q4)
mixtral:8x7b   # 47B parameters, ~26GB (Q4)
```

## Creating Models with Weights

Once weights are loaded, create a model instance:

```rust
use unillm::models_v2::llama::{LlamaModelV2, LlamaConfig};

// 1. Define configuration
let config = LlamaConfig {
    vocab_size: 32000,
    hidden_size: 4096,
    num_hidden_layers: 32,
    num_attention_heads: 32,
    ..Default::default()
};

// 2. Load weights
let weights = WeightLoader::from_gguf("llama-7b.gguf")?;

// 3. Create model with weights
let model = LlamaModelV2::from_weights(config, weights)?;
```

## Weight Format Details

### GGUF Format

GGUF files contain:

- Model weights (quantized)
- Tokenizer vocabulary
- Model configuration metadata

```rust
// GGUF provides configuration automatically
let gguf_config = weights.metadata().gguf_config();
let config = LlamaConfig::from_gguf_config(&gguf_config);
```

### SafeTensors Format

SafeTensors files are memory-mapped for efficiency:

```rust
// SafeTensors supports lazy loading
let weights = WeightLoader::from_safetensors("model.safetensors")?;

// Weights are loaded on-demand when accessed
let embed_weight = weights.get("model.embed_tokens.weight")?;
```

## Working with ModelWeights

The `ModelWeights` container provides access to loaded tensors:

```rust
// Get a specific weight
if let Some(weight) = weights.get("model.layers.0.self_attn.q_proj.weight") {
    println!("Shape: {:?}", weight.shape());
}

// Iterate over all weights
for key in weights.keys() {
    println!("Weight: {}", key);
}

// Check weight count
println!("Total weights: {}", weights.len());
```

## Device Transfer

Move loaded weights to a specific device:

```rust
use unillm::tensor_core::Device;

// Load weights on CPU first
let mut weights = WeightLoader::from_gguf("model.gguf")?;

// Transfer to GPU
weights.to_device(&Device::CUDA(0))?;

// Create model on GPU
let model = LlamaModelV2::from_weights(config, weights)?;
```

## Memory Considerations

!!! tip "Memory Usage"
    - GGUF Q4 models use ~0.5 bytes per parameter
    - SafeTensors F16 models use ~2 bytes per parameter
    - Full F32 models use ~4 bytes per parameter

| Model Size | GGUF Q4 | SafeTensors F16 | Full F32 |
|------------|---------|-----------------|----------|
| 7B | ~3.5GB | ~14GB | ~28GB |
| 13B | ~6.5GB | ~26GB | ~52GB |
| 70B | ~35GB | ~140GB | ~280GB |

## Error Handling

```rust
use anyhow::Result;

fn load_model() -> Result<LlamaModelV2> {
    let weights = WeightLoader::from_gguf("model.gguf")
        .map_err(|e| anyhow::anyhow!("Failed to load weights: {}", e))?;

    let config = LlamaConfig::default();
    let model = LlamaModelV2::from_weights(config, weights)?;

    Ok(model)
}
```

## Next Steps

- Learn about [Running Inference](inference.md) with loaded models
- Explore [Configuration Options](configuration.md) for fine-tuning
