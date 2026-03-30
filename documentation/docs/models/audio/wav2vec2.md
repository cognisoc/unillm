# Wav2Vec2

Wav2Vec2 is Meta's self-supervised speech representation model, learning from unlabeled audio for speech recognition and other tasks.

## Overview

| Property | Value |
|----------|-------|
| **Architecture** | CNN Feature Extractor + Transformer |
| **Parameters** | 95M - 1B |
| **Audio Input** | 16kHz raw waveform |
| **Training** | Self-supervised (contrastive) |
| **Tasks** | ASR, Speaker ID, Emotion |
| **Languages** | 53+ (XLS-R) |

## Quick Start

```rust
use unillm::models_v2::wav2vec2::{Wav2Vec2ModelV2, Wav2Vec2Config};
use unillm::weight_loader_core::WeightLoader;
use unillm::{Model, ModelInputs};

// Load model
let weights = WeightLoader::from_safetensors("wav2vec2-base.safetensors")?;
let config = Wav2Vec2Config::default();
let model = Wav2Vec2ModelV2::from_weights(config, weights)?;

// Extract features
let audio = load_audio("speech.wav")?;
let inputs = ModelInputs::Audio {
    input_features: audio.to_tensor()?,
    attention_mask: Some(create_attention_mask(&audio)?),
};

let outputs = model.forward(&inputs)?;
// outputs.embeddings: [batch, seq_len, hidden_size]
```

## Configuration

```rust
model_config!(Wav2Vec2Config {
    // Feature extractor (CNN)
    conv_dim: Vec<usize> = vec![512, 512, 512, 512, 512, 512, 512],
    conv_stride: Vec<usize> = vec![5, 2, 2, 2, 2, 2, 2],
    conv_kernel: Vec<usize> = vec![10, 3, 3, 3, 3, 2, 2],

    // Transformer
    hidden_size: usize = 768,
    intermediate_size: usize = 3072,
    num_hidden_layers: usize = 12,
    num_attention_heads: usize = 12,

    // Training
    num_codevector_groups: usize = 2,
    num_codevectors_per_group: usize = 320,

    // General
    vocab_size: usize = 32,  // CTC vocabulary
    hidden_dropout: f32 = 0.1,
    attention_dropout: f32 = 0.1,
    feat_extract_norm: String = "group".to_string(),
});
```

### Model Sizes

| Variant | Layers | Hidden | Params |
|---------|--------|--------|--------|
| Base | 12 | 768 | 95M |
| Large | 24 | 1024 | 317M |
| XLS-R 300M | 24 | 1024 | 317M |
| XLS-R 1B | 48 | 1280 | 965M |
| XLS-R 2B | 48 | 1920 | 2B |

## Architecture

### Feature Extractor

7-layer CNN that converts raw audio to latent representations:

```
Raw Audio (16kHz)
    │
    ▼
┌─────────────────┐
│ Conv1D (k=10)   │  512 → 512, stride 5
├─────────────────┤
│ Conv1D (k=3)    │  512 → 512, stride 2
├─────────────────┤
│ Conv1D (k=3)    │  512 → 512, stride 2
├─────────────────┤
│ ... 4 more      │
└─────────────────┘
    │
    ▼
Latent Representations
[batch, seq_len/320, 512]
```

### Transformer Encoder

```rust
struct Wav2Vec2Encoder {
    feature_projection: Linear,  // 512 → 768
    layers: Vec<TransformerLayer>,
    layer_norm: LayerNorm,
}

fn forward(&self, features: &Tensor) -> Result<Tensor> {
    // Project features
    let hidden = ops_fn::linear(features, &self.feature_projection, None)?;

    // Add positional encoding
    let hidden = self.add_positional_encoding(&hidden)?;

    // Transformer layers
    for layer in &self.layers {
        hidden = self.forward_layer(&hidden, layer)?;
    }

    ops_fn::layer_norm(&hidden, &self.layer_norm, None, 1e-5)
}
```

## Self-Supervised Learning

### Contrastive Task

Wav2Vec2 is trained by:
1. Masking spans of latent representations
2. Predicting the correct quantized target

```
┌─────────────────────────────────────────┐
│ Raw Audio                               │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ CNN Feature Extractor                    │
└─────────────────────────────────────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌───────┐   ┌───────────┐
│ Mask  │   │ Quantizer │
└───────┘   └───────────┘
    │             │
    ▼             │
┌───────────┐     │
│Transformer│     │
└───────────┘     │
    │             │
    └──────┬──────┘
           │
           ▼
    Contrastive Loss
```

## Speech Recognition

### With CTC Head

```rust
// Fine-tuned model with CTC head
let model = Wav2Vec2ForCTC::from_weights(config, weights)?;

let audio = load_audio("speech.wav")?;
let inputs = ModelInputs::Audio {
    input_features: audio.to_tensor()?,
    attention_mask: None,
};

// Get logits
let outputs = model.forward(&inputs)?;
let logits = outputs.logits;  // [batch, seq_len, vocab_size]

// Decode with CTC
let transcription = ctc_decode(&logits, &tokenizer)?;
```

### CTC Decoding

```rust
fn ctc_decode(logits: &Tensor, tokenizer: &Tokenizer) -> Result<String> {
    // Get argmax predictions
    let predictions = ops_fn::argmax(logits, -1)?;

    // Collapse repeated tokens and remove blanks
    let mut decoded = Vec::new();
    let mut prev_token = None;

    for token_id in predictions.to_vec::<i64>()? {
        if token_id != BLANK_TOKEN && Some(token_id) != prev_token {
            decoded.push(token_id);
        }
        prev_token = Some(token_id);
    }

    tokenizer.decode(&decoded)
}
```

## Feature Extraction

### For Downstream Tasks

```rust
// Extract representations for classification, etc.
let outputs = model.forward(&inputs)?;

match outputs {
    ModelOutputs::Embeddings { embeddings, pooled } => {
        // embeddings: [batch, seq_len, hidden_size]
        // Use for sequence tasks

        // Mean pooling for classification
        let pooled = embeddings.mean(1)?;

        // Classification head
        let logits = classifier.forward(&pooled)?;
    }
    _ => unreachable!(),
}
```

### Speaker Identification

```rust
// Fine-tuned for speaker ID
let speaker_model = Wav2Vec2ForSpeakerID::from_weights(config, weights)?;

let embeddings = speaker_model.extract_speaker_embedding(&audio)?;
// embeddings: [batch, embedding_dim]

// Compare with enrolled speakers
let similarities = cosine_similarity(&embeddings, &enrolled_embeddings)?;
```

## Memory Requirements

| Variant | F32 | F16 |
|---------|-----|-----|
| Base | 380 MB | 190 MB |
| Large | 1.3 GB | 650 MB |
| XLS-R 1B | 3.9 GB | 2.0 GB |

## Performance

### Word Error Rate (ASR)

| Model | LibriSpeech clean | LibriSpeech other |
|-------|-------------------|-------------------|
| Base | 3.4% | 8.0% |
| Large | 2.1% | 4.8% |
| XLS-R 1B | 1.9% | 4.0% |

## Use Cases

### Ideal For

- **Speech recognition** - High-quality ASR
- **Speaker identification** - Voice biometrics
- **Emotion recognition** - Sentiment from speech
- **Language identification** - Detect spoken language
- **Low-resource languages** - Transfer learning

### Comparison

| Task | Wav2Vec2 | Whisper |
|------|----------|---------|
| Real-time ASR | Better | Good |
| Multi-lingual | Good (XLS-R) | Excellent |
| Zero-shot | Needs fine-tune | Yes |
| Feature extraction | Excellent | Limited |

## Related Models

### HuBERT

Similar architecture but with different pre-training:

```rust
use unillm::models_v2::hubert::{HuBertModelV2, HuBertConfig};

let model = HuBertModelV2::from_weights(config, weights)?;
// Same interface as Wav2Vec2
```

### XLS-R

Multilingual Wav2Vec2:

```rust
// XLS-R for 128 languages
let config = Wav2Vec2Config {
    hidden_size: 1280,
    num_hidden_layers: 48,
    ..Default::default()
};
```

## Best Practices

1. **Normalize audio** - Mean/variance normalization
2. **Use attention mask** - For variable-length audio
3. **Fine-tune for task** - Pre-trained features need adaptation
4. **Consider XLS-R** - For multilingual applications

## References

- [Wav2Vec 2.0 Paper](https://arxiv.org/abs/2006.11477)
- [XLS-R Paper](https://arxiv.org/abs/2111.09296)
- [HuggingFace Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2)
