# LLaVA

LLaVA (Large Language-and-Vision Assistant) is a multimodal model that combines a vision encoder with a large language model for visual understanding and conversation.

## Overview

| Property | Value |
|----------|-------|
| **Architecture** | Vision Encoder + Projector + LLM |
| **Vision Encoder** | CLIP ViT-L/14 |
| **Language Model** | LLaMA / Vicuna |
| **Parameters** | 7B, 13B |
| **Image Size** | 336x336 |
| **Context Length** | 4096 tokens |

## Quick Start

```rust
use unillm::models_v2::llava::{LlavaModelV2, LlavaConfig};
use unillm::weight_loader_core::WeightLoader;
use unillm::{Model, ModelInputs, GenerationConfig};

// Load model
let weights = WeightLoader::from_gguf("llava-v1.5-7b.gguf")?;
let config = LlavaConfig::from_gguf_metadata(weights.metadata())?;
let model = LlavaModelV2::from_weights(config, weights)?;

// Prepare multimodal input
let inputs = ModelInputs::Multimodal {
    input_ids: text_tokens,
    pixel_values: Some(image_tensor),
    attention_mask: None,
    image_mask: None,
};

// Generate response
let response = model.generate_from_inputs(&inputs, &GenerationConfig::default())?;
```

## Configuration

```rust
model_config!(LlavaConfig {
    // Vision encoder
    vision_hidden_size: usize = 1024,
    vision_intermediate_size: usize = 4096,
    vision_num_layers: usize = 24,
    vision_num_heads: usize = 16,
    vision_patch_size: usize = 14,
    image_size: usize = 336,

    // Projector
    projector_hidden_size: usize = 4096,
    projector_type: String = "mlp2x_gelu".to_string(),

    // Language model
    vocab_size: usize = 32000,
    hidden_size: usize = 4096,
    intermediate_size: usize = 11008,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 32,
    num_key_value_heads: usize = 32,
    max_position_embeddings: usize = 4096,
    rms_norm_eps: f32 = 1e-5,
});
```

## Architecture

### Components

```
Image ─────────────────────────────────────────────────┐
       │                                               │
       ▼                                               │
┌─────────────────┐                                    │
│ Vision Encoder  │  CLIP ViT-L/14                     │
│ (frozen/tuned)  │  224x224 or 336x336               │
└─────────────────┘                                    │
       │                                               │
       ▼                                               │
┌─────────────────┐                                    │
│ Projector       │  MLP: vision_dim → llm_dim        │
│ (trained)       │                                    │
└─────────────────┘                                    │
       │                                               │
       ▼                                               │
┌─────────────────────────────────────────────────────┐
│                    LLM Decoder                       │
│  [image tokens] + [text tokens] → response          │
│  LLaMA / Vicuna / Mistral                           │
└─────────────────────────────────────────────────────┘
```

### Projector Types

```rust
// MLP with GELU (LLaVA 1.5)
let projector = MLP {
    layers: vec![
        Linear(vision_dim, llm_dim),
        GELU,
        Linear(llm_dim, llm_dim),
    ]
};

// Simple linear (LLaVA 1.0)
let projector = Linear(vision_dim, llm_dim);
```

## LLaVA Versions

### LLaVA 1.0

- Initial release
- Linear projector
- CLIP ViT-L/14
- LLaMA 7B/13B

### LLaVA 1.5

- MLP projector (better)
- Higher resolution (336x336)
- Improved instruction following
- More training data

### LLaVA 1.6 (LLaVA-NeXT)

- Multiple image resolutions
- Dynamic aspect ratio
- Improved OCR
- Mistral/Qwen backbones

## Generation Examples

### Visual Q&A

```rust
let prompt = "<image>\nWhat is shown in this image?";

let inputs = ModelInputs::Multimodal {
    input_ids: tokenizer.encode(&prompt)?,
    pixel_values: Some(image),
    attention_mask: None,
    image_mask: None,
};

let config = GenerationConfig {
    max_new_tokens: 256,
    temperature: 0.7,
    ..Default::default()
};

let answer = model.generate_from_inputs(&inputs, &config)?;
```

### Detailed Description

```rust
let prompt = "<image>\nDescribe this image in detail, including colors, objects, and their positions.";

let response = model.generate_from_inputs(&inputs, &config)?;
```

### OCR and Document Understanding

```rust
let prompt = "<image>\nRead and transcribe all text visible in this image.";

let response = model.generate_from_inputs(&inputs, &config)?;
```

### Conversation

```rust
let conversation = vec![
    "<image>",
    "USER: What do you see in this image?",
    "ASSISTANT: I see a cat sitting on a windowsill.",
    "USER: What color is the cat?",
    "ASSISTANT:",
];

let prompt = conversation.join("\n");
let response = model.generate(&prompt, &config)?;
```

## Image Processing

### Preprocessing

```rust
fn preprocess_image(image: &Image, target_size: usize) -> Result<Tensor> {
    // Resize to target size
    let resized = image.resize(target_size, target_size)?;

    // Normalize with CLIP stats
    let mean = [0.48145466, 0.4578275, 0.40821073];
    let std = [0.26862954, 0.26130258, 0.27577711];

    let tensor = resized.to_tensor()?;
    let normalized = normalize(&tensor, &mean, &std)?;

    Ok(normalized)
}
```

### Multiple Images (LLaVA-NeXT)

```rust
let prompt = "<image>\n<image>\nCompare these two images.";

let inputs = ModelInputs::Multimodal {
    input_ids: tokenizer.encode(&prompt)?,
    pixel_values: Some(concat_images(&[image1, image2])?),
    ..
};
```

## Memory Requirements

| Variant | F16 | Q8_0 | Q4_K_M |
|---------|-----|------|--------|
| LLaVA 7B | 15 GB | 8 GB | 5 GB |
| LLaVA 13B | 27 GB | 14 GB | 8 GB |

Note: Vision encoder adds ~1 GB on top.

## Performance Tips

1. **Batch images** - Process multiple images together
2. **Cache vision features** - Reuse for same image, different prompts
3. **Quantize LLM** - Vision encoder can stay F16
4. **Resolution trade-off** - 224px faster, 336px better quality

## Use Cases

### Ideal For

- **Visual Q&A** - Answer questions about images
- **Image description** - Generate detailed captions
- **Document OCR** - Read text from images
- **Visual reasoning** - Compare, analyze, explain

### Comparison

| Task | LLaVA 1.5 | GPT-4V | Qwen2-VL |
|------|-----------|--------|----------|
| General VQA | Good | Excellent | Very Good |
| OCR | Good | Excellent | Excellent |
| Reasoning | Good | Excellent | Very Good |
| Speed | Fast | Slow (API) | Medium |

## Best Practices

1. **Use appropriate resolution** - Match training (336px for 1.5)
2. **Clear prompts** - Be specific about what you want
3. **Image placeholder** - Use `<image>` where image goes
4. **Chat format** - Use proper USER/ASSISTANT format

## References

- [LLaVA Paper](https://arxiv.org/abs/2304.08485)
- [LLaVA 1.5 Paper](https://arxiv.org/abs/2310.03744)
- [LLaVA GitHub](https://github.com/haotian-liu/LLaVA)
