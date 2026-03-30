# CLIP

CLIP (Contrastive Language-Image Pre-training) is OpenAI's model for learning visual concepts from natural language supervision.

## Overview

| Property | Value |
|----------|-------|
| **Architecture** | Dual Encoder (Vision + Text) |
| **Vision Encoder** | ViT (Vision Transformer) |
| **Text Encoder** | Transformer |
| **Parameters** | 400M - 1B |
| **Image Size** | 224x224, 336x336 |
| **Text Context** | 77 tokens |
| **Training** | Contrastive learning |

## Quick Start

```rust
use unillm::models_v2::clip::{ClipModelV2, ClipConfig};
use unillm::weight_loader_core::WeightLoader;
use unillm::{Model, ModelInputs};

// Load model
let weights = WeightLoader::from_safetensors("clip-vit-large.safetensors")?;
let config = ClipConfig::default();
let model = ClipModelV2::from_weights(config, weights)?;

// Encode image
let image_input = ModelInputs::Image {
    pixel_values: image_tensor,
    image_mask: None,
};
let image_features = model.encode_image(&image_input)?;

// Encode text
let text_input = ModelInputs::Text {
    input_ids: text_tokens,
    attention_mask: None,
    position_ids: None,
};
let text_features = model.encode_text(&text_input)?;

// Compute similarity
let similarity = ops_fn::matmul(&image_features, &text_features.t()?)?;
```

## Configuration

```rust
model_config!(ClipConfig {
    // Vision encoder
    vision_hidden_size: usize = 1024,
    vision_intermediate_size: usize = 4096,
    vision_num_layers: usize = 24,
    vision_num_heads: usize = 16,
    vision_patch_size: usize = 14,
    image_size: usize = 224,

    // Text encoder
    text_hidden_size: usize = 512,
    text_intermediate_size: usize = 2048,
    text_num_layers: usize = 12,
    text_num_heads: usize = 8,
    text_vocab_size: usize = 49408,
    text_max_length: usize = 77,

    // Projection
    projection_dim: usize = 512,
});
```

### Model Variants

| Variant | Vision | Text | Proj Dim |
|---------|--------|------|----------|
| ViT-B/32 | ViT-Base, patch 32 | 63M | 512 |
| ViT-B/16 | ViT-Base, patch 16 | 63M | 512 |
| ViT-L/14 | ViT-Large, patch 14 | 63M | 768 |
| ViT-L/14@336 | ViT-Large, 336px | 63M | 768 |

## How CLIP Works

### Contrastive Learning

CLIP is trained to match images with their text descriptions:

```
Image Encoder ─────┐
                   │
                   ├──> Cosine Similarity Matrix
                   │
Text Encoder  ─────┘

Diagonal = Matching pairs (maximize)
Off-diagonal = Non-matching (minimize)
```

### Embedding Space

Both modalities are projected to a shared embedding space:

```rust
// Image: [batch, 3, 224, 224] → [batch, 512]
let image_embed = model.encode_image(&image)?;

// Text: [batch, 77] → [batch, 512]
let text_embed = model.encode_text(&text)?;

// Compare in shared space
let similarity = cosine_similarity(&image_embed, &text_embed)?;
```

## Use Cases

### Zero-Shot Classification

Classify images without training:

```rust
// Define class labels
let labels = vec![
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a bird",
];

// Encode labels
let text_features = model.encode_text_batch(&labels)?;

// Encode image
let image_features = model.encode_image(&image)?;

// Get predictions
let logits = ops_fn::matmul(&image_features, &text_features.t()?)?;
let probs = ops_fn::softmax(&logits, -1)?;
```

### Image Search

Find images matching a text query:

```rust
// Pre-compute image embeddings
let image_embeddings: Vec<Tensor> = images.iter()
    .map(|img| model.encode_image(img))
    .collect()?;

// Search
let query = "a sunset over the ocean";
let query_embed = model.encode_text(&query)?;

// Find most similar
let similarities: Vec<f32> = image_embeddings.iter()
    .map(|img_embed| cosine_similarity(img_embed, &query_embed))
    .collect();
```

### Image-Text Matching

Check if image matches description:

```rust
let image_embed = model.encode_image(&image)?;
let text_embed = model.encode_text("a red car on a highway")?;

let score = cosine_similarity(&image_embed, &text_embed)?;
println!("Match score: {:.2}", score);  // 0.0 to 1.0
```

## Vision Encoder

### Architecture

```rust
struct VisionEncoder {
    patch_embedding: Tensor,  // [patch_size², hidden]
    class_embedding: Tensor,  // [1, hidden]
    position_embedding: Tensor,  // [num_patches + 1, hidden]
    layers: Vec<VisionLayer>,
    ln_post: Tensor,
    projection: Tensor,  // [hidden, proj_dim]
}
```

### Forward Pass

```rust
fn encode_image(&self, pixel_values: &Tensor) -> Result<Tensor> {
    // Patch embedding
    let patches = self.patchify(pixel_values)?;

    // Add class token
    let hidden = self.add_class_token(&patches)?;

    // Add position embeddings
    let hidden = ops_fn::add(&hidden, &self.position_embedding)?;

    // Transformer layers
    for layer in &self.layers {
        hidden = self.forward_layer(&hidden, layer)?;
    }

    // Extract class token, project
    let class_token = hidden.slice(&[(0, 1)])?;
    let projected = ops_fn::linear(&class_token, &self.projection, None)?;

    // Normalize
    ops_fn::normalize(&projected, 2, 1e-6)
}
```

## Text Encoder

### Architecture

```rust
struct TextEncoder {
    token_embedding: Tensor,  // [vocab_size, hidden]
    position_embedding: Tensor,  // [max_len, hidden]
    layers: Vec<TextLayer>,
    ln_final: Tensor,
    projection: Tensor,  // [hidden, proj_dim]
}
```

### Forward Pass

```rust
fn encode_text(&self, input_ids: &Tensor) -> Result<Tensor> {
    // Token + position embeddings
    let hidden = ops_fn::embedding(input_ids, &self.token_embedding)?;
    let hidden = ops_fn::add(&hidden, &self.position_embedding)?;

    // Transformer layers with causal mask
    for layer in &self.layers {
        hidden = self.forward_layer(&hidden, layer)?;
    }

    // Take EOT token embedding
    let eot_token = self.get_eot_position(input_ids)?;
    let hidden = hidden.gather(&eot_token)?;

    // Project and normalize
    let projected = ops_fn::linear(&hidden, &self.projection, None)?;
    ops_fn::normalize(&projected, 2, 1e-6)
}
```

## Memory Requirements

| Variant | F32 | F16 |
|---------|-----|-----|
| ViT-B/32 | 600 MB | 300 MB |
| ViT-B/16 | 600 MB | 300 MB |
| ViT-L/14 | 1.6 GB | 800 MB |
| ViT-L/14@336 | 1.6 GB | 800 MB |

## Best Practices

1. **Use appropriate variant** - ViT-L/14 for quality, ViT-B/32 for speed
2. **Normalize embeddings** - Required for cosine similarity
3. **Batch processing** - Process many images/texts together
4. **Prompt engineering** - "a photo of a {class}" works well

## Integration with VLMs

CLIP is often used as the vision encoder in VLMs:

- **LLaVA** - CLIP vision + LLaMA text
- **OpenCLIP** - Open-source CLIP variants
- **SigLIP** - Sigmoid loss variant

## References

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
