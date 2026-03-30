# Your First Model

This tutorial walks you through running your first model with UniLLM.

## Using the Ollama Integration

The easiest way to get started is using the built-in Ollama integration, which automatically downloads and manages models.

### Step 1: Run the Ollama Test

```bash
cargo run --bin test_ollama -p runtime
```

This will:

1. Download **TinyLlama** (~600MB) from the Ollama registry
2. Load the GGUF weights
3. Run a sample inference

Expected output:

```
Downloading tinyllama...
Loading model weights...
Running inference...
Prompt: "The quick brown fox"
Generated: "jumps over the lazy dog..."
```

### Step 2: Try Different Models

```bash
# Use a different model
cargo run --bin test_ollama -p runtime -- --model llama2:7b

# List cached models
cargo run --bin test_ollama -p runtime -- --list-cached
```

## Using the API Directly

For more control, use the API directly in your Rust code.

### Basic Example

```rust
use unillm::models_v2::llama::{LlamaModelV2, LlamaConfig};
use unillm::{Model, ModelInputs, GenerationConfig};
use unillm::tensor_core::{ops_fn, DataType, Device};

fn main() -> anyhow::Result<()> {
    // 1. Create model configuration
    let config = LlamaConfig {
        vocab_size: 32000,
        hidden_size: 4096,
        num_hidden_layers: 32,
        num_attention_heads: 32,
        ..Default::default()
    };

    // 2. Create the model
    let model = LlamaModelV2::new(config)?;

    // 3. Configure generation
    let gen_config = GenerationConfig {
        max_new_tokens: 50,
        temperature: 0.7,
        do_sample: true,
        ..Default::default()
    };

    // 4. Generate text
    let response = model.generate("Hello, world!", &gen_config)?;
    println!("Generated: {}", response);

    Ok(())
}
```

### Loading Pre-trained Weights

```rust
use unillm::weight_loader_core::WeightLoader;

// Load from SafeTensors
let weights = WeightLoader::from_safetensors("model.safetensors")?;

// Load from GGUF (Ollama format)
let weights = WeightLoader::from_gguf("model.gguf")?;

// Auto-detect format
let weights = WeightLoader::auto_detect("model_file")?;

// Create model with weights
let model = LlamaModelV2::from_weights(config, weights)?;
```

## Understanding the Output

When you run inference, UniLLM:

1. **Tokenizes** the input text into token IDs
2. **Embeds** the tokens into continuous vectors
3. **Runs** the forward pass through transformer layers
4. **Samples** the next token from output logits
5. **Decodes** tokens back to text

```
Input: "Hello"
  ↓ Tokenize
[1, 15496]  (token IDs)
  ↓ Embed
[batch, seq, hidden_size]  (embeddings)
  ↓ Forward Pass
[batch, seq, vocab_size]  (logits)
  ↓ Sample
[12345]  (next token)
  ↓ Decode
"Hello world"
```

## Model Memory Requirements

Different models have different memory requirements:

| Model | Parameters | RAM Required |
|-------|------------|--------------|
| TinyLlama | 1.1B | ~2GB |
| LLaMA-7B | 7B | ~14GB |
| LLaMA-13B | 13B | ~26GB |
| Mixtral-8x7B | 47B | ~94GB |

!!! tip "Quantized Models"
    GGUF models are quantized, reducing memory requirements significantly. A Q4 quantized 7B model uses ~4GB instead of ~14GB.

## Next Steps

Now that you've run your first model:

- Learn about [Loading Models](../user-guide/loading-models.md) in detail
- Explore the [Model Catalog](../models/index.md) for all supported architectures
- Understand [Configuration Options](../user-guide/configuration.md)
