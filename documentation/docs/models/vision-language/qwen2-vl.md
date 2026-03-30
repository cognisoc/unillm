# Qwen2-VL

Qwen2-VL is Alibaba's vision-language model with native multimodal architecture, supporting images, videos, and text in a unified framework.

## Overview

| Property | Value |
|----------|-------|
| **Architecture** | Native Multimodal Transformer |
| **Parameters** | 2B, 7B, 72B |
| **Context Length** | 32K tokens |
| **Image Resolution** | Dynamic (up to 16K pixels) |
| **Video Support** | Yes (frame extraction) |
| **Position Encoding** | M-RoPE (Multimodal RoPE) |

## Quick Start

```rust
use unillm::models_v2::qwen2_vl::{Qwen2VLModelV2, Qwen2VLConfig};
use unillm::weight_loader_core::WeightLoader;
use unillm::{Model, ModelInputs, GenerationConfig};

// Load model
let weights = WeightLoader::from_gguf("qwen2-vl-7b.gguf")?;
let config = Qwen2VLConfig::from_gguf_metadata(weights.metadata())?;
let model = Qwen2VLModelV2::from_weights(config, weights)?;

// Generate response
let inputs = ModelInputs::Multimodal {
    input_ids: tokens,
    pixel_values: Some(image),
    attention_mask: None,
    image_mask: None,
};

let response = model.generate_from_inputs(&inputs, &GenerationConfig::default())?;
```

## Configuration

```rust
model_config!(Qwen2VLConfig {
    // Vision encoder
    vision_hidden_size: usize = 1280,
    vision_intermediate_size: usize = 5120,
    vision_num_layers: usize = 32,
    vision_num_heads: usize = 16,
    vision_patch_size: usize = 14,
    temporal_patch_size: usize = 2,
    spatial_merge_size: usize = 2,

    // Language model
    vocab_size: usize = 151936,
    hidden_size: usize = 3584,
    intermediate_size: usize = 18944,
    num_hidden_layers: usize = 28,
    num_attention_heads: usize = 28,
    num_key_value_heads: usize = 4,
    max_position_embeddings: usize = 32768,
    rope_theta: f32 = 1000000.0,
    rms_norm_eps: f32 = 1e-6,

    // Multimodal
    image_token_id: usize = 151655,
    video_token_id: usize = 151656,
});
```

### Model Sizes

| Variant | Vision Layers | LLM Layers | Total Params |
|---------|---------------|------------|--------------|
| Qwen2-VL 2B | 32 | 28 | 2.2B |
| Qwen2-VL 7B | 32 | 28 | 8.3B |
| Qwen2-VL 72B | 32 | 80 | 73B |

## Key Features

### Dynamic Resolution

Unlike fixed-resolution models, Qwen2-VL handles any image size:

```rust
// Process images at native resolution
let small_image = load_image("small.jpg")?;  // 224x224
let large_image = load_image("large.jpg")?;  // 1920x1080

// Both work without resizing
let response1 = model.process(&small_image, "Describe this")?;
let response2 = model.process(&large_image, "Describe this")?;
```

### M-RoPE (Multimodal RoPE)

Unified position encoding for text, images, and video:

```rust
struct MRoPE {
    temporal_rope: RoPE,  // For video frames
    height_rope: RoPE,    // For image height
    width_rope: RoPE,     // For image width
}

fn compute_mrope(&self, positions: &Positions3D) -> Tensor {
    let temporal = self.temporal_rope.forward(&positions.t)?;
    let height = self.height_rope.forward(&positions.h)?;
    let width = self.width_rope.forward(&positions.w)?;
    concat(&[temporal, height, width])
}
```

### Native Video Understanding

Process videos directly:

```rust
// Extract frames at specified FPS
let video_frames = extract_frames("video.mp4", fps: 2.0)?;

let inputs = ModelInputs::Multimodal {
    input_ids: tokens,
    pixel_values: Some(video_frames),
    ..Default::default()
};

let response = model.generate(&"Describe what happens in this video", &config)?;
```

## Architecture

### Vision Encoder

```
Image/Video
    │
    ▼
┌─────────────────┐
│ Patch Embedding │  14x14 patches + temporal
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ ViT Layers (32) │  With 3D position encoding
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Spatial Merge   │  2x2 spatial downsampling
└─────────────────┘
    │
    ▼
Vision Tokens
```

### Multimodal Fusion

```rust
fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
    // Encode vision
    let vision_features = self.vision_encoder.forward(&inputs.pixel_values)?;

    // Merge spatially
    let merged = self.spatial_merge(&vision_features)?;

    // Get text embeddings
    let text_embeds = self.get_text_embeddings(&inputs.input_ids)?;

    // Insert vision tokens at image positions
    let hidden = self.merge_multimodal(&text_embeds, &merged, &inputs.image_mask)?;

    // Forward through LLM
    self.language_model.forward_from_embeddings(&hidden)
}
```

## Generation Examples

### Image Understanding

```rust
let prompt = "<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>
What is in this image?<|im_end|>
<|im_start|>assistant
";

let config = GenerationConfig {
    max_new_tokens: 256,
    temperature: 0.7,
    stop_sequences: vec!["<|im_end|>".to_string()],
    ..Default::default()
};

let response = model.generate_multimodal(&prompt, &image, &config)?;
```

### Video Analysis

```rust
let prompt = "<|im_start|>user
<|vision_start|><|video_pad|><|vision_end|>
Summarize what happens in this video.<|im_end|>
<|im_start|>assistant
";

let frames = extract_frames("video.mp4", 2.0)?;  // 2 FPS
let response = model.generate_multimodal(&prompt, &frames, &config)?;
```

### Document OCR

```rust
let prompt = "<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>
Extract all text from this document.<|im_end|>
<|im_start|>assistant
";

let response = model.generate_multimodal(&prompt, &document_image, &config)?;
```

### Multi-Image Comparison

```rust
let prompt = "<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>
<|vision_start|><|image_pad|><|vision_end|>
Compare these two images. What are the differences?<|im_end|>
<|im_start|>assistant
";

let images = concat_images(&[image1, image2])?;
let response = model.generate_multimodal(&prompt, &images, &config)?;
```

## Memory Requirements

| Variant | F16 | Q8_0 | Q4_K_M |
|---------|-----|------|--------|
| 2B | 5 GB | 3 GB | 2 GB |
| 7B | 17 GB | 9 GB | 5 GB |
| 72B | 145 GB | 75 GB | 42 GB |

Note: Vision encoder adds ~1-2 GB depending on image resolution.

## Performance

### Benchmarks

| Benchmark | Qwen2-VL 7B | LLaVA 1.6 | GPT-4V |
|-----------|-------------|-----------|--------|
| VQAv2 | 83.0 | 79.8 | N/A |
| DocVQA | 94.5 | 78.2 | 87.2 |
| ChartQA | 83.0 | 67.3 | 78.1 |
| TextVQA | 84.3 | 72.1 | N/A |

### Strengths

- **OCR**: Excellent text recognition
- **Document**: Strong document understanding
- **Charts**: Good chart/graph reading
- **Video**: Native video support

## Loading from Ollama

```rust
use unillm::ollama::OllamaRegistry;

// Qwen2-VL
let path = OllamaRegistry::pull("qwen2-vl:7b")?;

// Quantized
let path = OllamaRegistry::pull("qwen2-vl:7b-q4_0")?;
```

## Best Practices

1. **Use native resolution** - Don't resize images unnecessarily
2. **Proper formatting** - Use vision tokens correctly
3. **Video FPS** - 1-4 FPS usually sufficient
4. **Batch frames** - Process video frames together

## Use Cases

### Ideal For

- **Document OCR** - Best-in-class text extraction
- **Chart analysis** - Strong graph understanding
- **Video QA** - Native video support
- **Multilingual** - Good non-English support

### Comparison

| Task | Best Choice |
|------|-------------|
| Fastest | Qwen2-VL 2B |
| Best quality | Qwen2-VL 72B |
| Document/OCR | Qwen2-VL (any) |
| General VQA | Qwen2-VL 7B |

## References

- [Qwen2-VL Technical Report](https://arxiv.org/abs/2409.12191)
- [Qwen GitHub](https://github.com/QwenLM/Qwen2-VL)
- [Qwen HuggingFace](https://huggingface.co/Qwen)
