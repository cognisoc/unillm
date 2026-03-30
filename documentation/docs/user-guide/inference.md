# Running Inference

This guide covers how to run inference with UniLLM models.

## Forward Pass

The `forward()` method runs a single forward pass through the model:

```rust
use unillm::models_v2::llama::{LlamaModelV2, LlamaConfig};
use unillm::{Model, ModelInputs, ModelOutputs};
use unillm::tensor_core::{ops_fn, DataType, Device};

// Create model
let model = LlamaModelV2::new(LlamaConfig::default())?;

// Prepare input tensor (token IDs)
let input_ids = ops_fn::zeros(&[1, 10], DataType::Int64, &Device::CPU)?;

// Create input
let inputs = ModelInputs::Text {
    input_ids,
    attention_mask: None,
    position_ids: None,
};

// Run forward pass
let outputs = model.forward(&inputs)?;

// Process outputs
match outputs {
    ModelOutputs::Logits { logits, hidden_states } => {
        println!("Logits shape: {:?}", logits.shape());
        // logits: [batch_size, seq_len, vocab_size]
    }
    _ => {}
}
```

## Input Types

### Text Input

For language models:

```rust
let inputs = ModelInputs::Text {
    input_ids: token_tensor,           // Required: [batch, seq_len]
    attention_mask: Some(mask_tensor), // Optional: [batch, seq_len]
    position_ids: Some(pos_tensor),    // Optional: [batch, seq_len]
};
```

### Image Input

For vision models:

```rust
let inputs = ModelInputs::Image {
    pixel_values: image_tensor,      // Required: [batch, channels, height, width]
    image_mask: Some(mask_tensor),   // Optional
};
```

### Multimodal Input

For vision-language models:

```rust
let inputs = ModelInputs::Multimodal {
    input_ids: token_tensor,
    pixel_values: Some(image_tensor),
    attention_mask: Some(mask_tensor),
    image_mask: None,
};
```

### Audio Input

For speech models:

```rust
let inputs = ModelInputs::Audio {
    input_features: audio_tensor,    // [batch, features, time]
    attention_mask: Some(mask),
};
```

## Output Types

### Logits Output

Most language models return logits:

```rust
match outputs {
    ModelOutputs::Logits { logits, hidden_states } => {
        // logits: [batch, seq_len, vocab_size]
        // Get last token logits for generation
        let last_logits = logits.slice(/* last position */)?;

        // Optional: hidden states from all layers
        if let Some(hidden) = hidden_states {
            println!("Hidden shape: {:?}", hidden.shape());
        }
    }
    _ => {}
}
```

### Embeddings Output

Encoder models return embeddings:

```rust
match outputs {
    ModelOutputs::Embeddings { embeddings, pooled } => {
        // embeddings: [batch, seq_len, hidden_size]
        // pooled: Optional [batch, hidden_size]
    }
    _ => {}
}
```

## Batch Processing

Process multiple inputs at once:

```rust
// Create batched input
let batch_size = 4;
let seq_len = 128;

let input_ids = ops_fn::zeros(
    &[batch_size, seq_len],
    DataType::Int64,
    &Device::CPU
)?;

let inputs = ModelInputs::Text {
    input_ids,
    attention_mask: None,
    position_ids: None,
};

// Forward pass processes all batches
let outputs = model.forward(&inputs)?;
// Output shape: [4, 128, vocab_size]
```

## Device Management

Move models between devices:

```rust
use unillm::tensor_core::Device;

// Create model on CPU
let mut model = LlamaModelV2::new(config)?;

// Move to GPU
model.to_device(&Device::CUDA(0))?;

// Create input on same device
let input_ids = ops_fn::zeros(&[1, 10], DataType::Int64, &Device::CUDA(0))?;

// Run inference on GPU
let outputs = model.forward(&ModelInputs::Text { input_ids, .. })?;
```

## Memory Requirements

Check model memory requirements:

```rust
let requirements = model.memory_requirements();

println!("GPU Memory: {} bytes", requirements.gpu_memory);
println!("CPU Memory: {} bytes", requirements.cpu_memory);
println!("KV Cache: {} bytes", requirements.kv_cache_memory);
println!("Peak Memory: {} bytes", requirements.peak_memory);
```

## Performance Tips

!!! tip "Optimization Tips"

    1. **Use batching** - Process multiple inputs together
    2. **Use GPU** - Move model to CUDA/Metal device
    3. **Use quantized models** - GGUF Q4 is 4x smaller than F16
    4. **Reuse tensors** - Avoid allocating new tensors each iteration

### Example: Efficient Inference Loop

```rust
// Pre-allocate tensors
let mut input_tensor = ops_fn::zeros(&[1, max_seq_len], DataType::Int64, &device)?;

for tokens in token_batches {
    // Reuse tensor, just update data
    // ... update input_tensor with new tokens

    let outputs = model.forward(&ModelInputs::Text {
        input_ids: input_tensor.clone(),
        attention_mask: None,
        position_ids: None,
    })?;

    // Process outputs
}
```

## Error Handling

```rust
use anyhow::Result;

fn run_inference(model: &LlamaModelV2, tokens: &[u32]) -> Result<Vec<f32>> {
    let input_ids = /* create tensor from tokens */;

    let outputs = model.forward(&ModelInputs::Text {
        input_ids,
        attention_mask: None,
        position_ids: None,
    })?;

    match outputs {
        ModelOutputs::Logits { logits, .. } => {
            Ok(logits.to_vec()?)
        }
        _ => Err(anyhow::anyhow!("Unexpected output type"))
    }
}
```

## Next Steps

- Learn about [Text Generation](generation.md) for autoregressive generation
- Explore [Configuration Options](configuration.md)
