# Whisper

Whisper is OpenAI's automatic speech recognition (ASR) model, capable of multilingual transcription and translation.

## Overview

| Property | Value |
|----------|-------|
| **Architecture** | Encoder-Decoder Transformer |
| **Parameters** | 39M - 1.5B |
| **Audio Input** | 30 seconds, 16kHz mono |
| **Languages** | 99+ languages |
| **Tasks** | Transcription, Translation |
| **Feature Extraction** | Log-Mel Spectrogram |

## Quick Start

```rust
use unillm::models_v2::whisper::{WhisperModelV2, WhisperConfig};
use unillm::weight_loader_core::WeightLoader;
use unillm::{Model, ModelInputs, GenerationConfig};

// Load model
let weights = WeightLoader::from_gguf("whisper-large-v3.gguf")?;
let config = WhisperConfig::from_gguf_metadata(weights.metadata())?;
let model = WhisperModelV2::from_weights(config, weights)?;

// Transcribe audio
let audio = load_audio("speech.wav")?;
let features = extract_mel_features(&audio)?;

let inputs = ModelInputs::Audio {
    input_features: features,
    attention_mask: None,
};

let transcription = model.transcribe(&inputs, &TranscribeConfig::default())?;
println!("{}", transcription);
```

## Configuration

```rust
model_config!(WhisperConfig {
    // Encoder
    encoder_hidden_size: usize = 1280,
    encoder_intermediate_size: usize = 5120,
    encoder_num_layers: usize = 32,
    encoder_num_heads: usize = 20,

    // Decoder
    decoder_hidden_size: usize = 1280,
    decoder_intermediate_size: usize = 5120,
    decoder_num_layers: usize = 32,
    decoder_num_heads: usize = 20,

    // Audio
    num_mel_bins: usize = 128,
    max_source_positions: usize = 1500,

    // Text
    vocab_size: usize = 51865,
    max_target_positions: usize = 448,

    // Special tokens
    bos_token_id: usize = 50258,
    eos_token_id: usize = 50257,
    pad_token_id: usize = 50257,
});
```

### Model Sizes

| Variant | Encoder Layers | Decoder Layers | Parameters |
|---------|----------------|----------------|------------|
| Tiny | 4 | 4 | 39M |
| Base | 6 | 6 | 74M |
| Small | 12 | 12 | 244M |
| Medium | 24 | 24 | 769M |
| Large | 32 | 32 | 1.5B |
| Large-v2 | 32 | 32 | 1.5B |
| Large-v3 | 32 | 32 | 1.5B |

## Architecture

### Encoder

Processes audio into contextual representations:

```
Audio (30s, 16kHz)
       │
       ▼
┌─────────────────┐
│ Mel Spectrogram │  128 bins, 3000 frames
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ Conv Layers     │  2x Conv1D
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ Transformer     │  Encoder layers
│ Encoder         │  + Sinusoidal positions
└─────────────────┘
       │
       ▼
Encoder Hidden States
```

### Decoder

Autoregressive text generation:

```
Encoder Hidden States
       │
       ▼
┌─────────────────────────────────────┐
│ Transformer Decoder                  │
│ ├─ Self-Attention (causal)          │
│ ├─ Cross-Attention (to encoder)     │
│ └─ FFN                              │
└─────────────────────────────────────┘
       │
       ▼
Text Tokens (transcription)
```

## Audio Processing

### Feature Extraction

```rust
fn extract_mel_features(audio: &AudioData) -> Result<Tensor> {
    // Ensure 16kHz sample rate
    let resampled = resample_to_16khz(audio)?;

    // Pad/trim to 30 seconds
    let audio = pad_or_trim(&resampled, 480000)?;  // 30s * 16000Hz

    // Compute log-mel spectrogram
    let mel = compute_log_mel_spectrogram(
        &audio,
        n_fft: 400,
        hop_length: 160,
        n_mels: 128,
    )?;

    Ok(mel)
}
```

### Chunk Processing

For audio longer than 30 seconds:

```rust
fn transcribe_long_audio(model: &WhisperModel, audio: &AudioData) -> Result<String> {
    let chunk_size = 30 * 16000;  // 30 seconds
    let stride = 28 * 16000;      // 2 second overlap

    let mut transcription = String::new();

    for chunk in audio.chunks_with_stride(chunk_size, stride) {
        let features = extract_mel_features(&chunk)?;
        let text = model.transcribe(&features)?;
        transcription.push_str(&text);
    }

    Ok(transcription)
}
```

## Transcription

### Basic Transcription

```rust
let config = TranscribeConfig {
    language: Some("en"),
    task: Task::Transcribe,
    ..Default::default()
};

let text = model.transcribe(&audio_features, &config)?;
```

### With Timestamps

```rust
let config = TranscribeConfig {
    return_timestamps: true,
    ..Default::default()
};

let result = model.transcribe_with_timestamps(&audio_features, &config)?;

for segment in result.segments {
    println!("[{:.2}s - {:.2}s] {}", segment.start, segment.end, segment.text);
}
```

### Translation

```rust
let config = TranscribeConfig {
    language: Some("fr"),      // Source language
    task: Task::Translate,     // Translate to English
    ..Default::default()
};

let english_text = model.transcribe(&french_audio, &config)?;
```

## Language Detection

```rust
let detected = model.detect_language(&audio_features)?;

println!("Language: {} (confidence: {:.2})",
    detected.language,
    detected.probability
);
```

## Generation Config

```rust
pub struct TranscribeConfig {
    /// Target language (None = auto-detect)
    pub language: Option<&'static str>,

    /// Transcribe or translate
    pub task: Task,

    /// Return word/segment timestamps
    pub return_timestamps: bool,

    /// Beam search width
    pub beam_size: usize,

    /// Sampling temperature
    pub temperature: f32,

    /// Compression ratio threshold
    pub compression_ratio_threshold: f32,

    /// Log probability threshold
    pub logprob_threshold: f32,

    /// No speech probability threshold
    pub no_speech_threshold: f32,
}

impl Default for TranscribeConfig {
    fn default() -> Self {
        Self {
            language: None,
            task: Task::Transcribe,
            return_timestamps: false,
            beam_size: 5,
            temperature: 0.0,
            compression_ratio_threshold: 2.4,
            logprob_threshold: -1.0,
            no_speech_threshold: 0.6,
        }
    }
}
```

## Memory Requirements

| Variant | F32 | F16 | Q8_0 |
|---------|-----|-----|------|
| Tiny | 156 MB | 78 MB | 40 MB |
| Base | 296 MB | 148 MB | 74 MB |
| Small | 976 MB | 488 MB | 244 MB |
| Medium | 3.1 GB | 1.5 GB | 770 MB |
| Large | 6.2 GB | 3.1 GB | 1.5 GB |

## Performance

### Word Error Rate (WER)

| Variant | Fleurs (en) | Fleurs (multi) |
|---------|-------------|----------------|
| Tiny | 8.7% | 17.6% |
| Base | 6.7% | 14.5% |
| Small | 5.0% | 11.3% |
| Medium | 4.3% | 9.9% |
| Large-v3 | 3.5% | 8.4% |

### Speed

| Variant | RTF (CPU) | RTF (GPU) |
|---------|-----------|-----------|
| Tiny | 1.2x | 0.1x |
| Base | 2.5x | 0.15x |
| Small | 6x | 0.3x |
| Large | 30x | 1.0x |

*RTF = Real-Time Factor (1x = real-time)*

## Use Cases

### Ideal For

- **Transcription** - General speech to text
- **Translation** - Speech in any language to English
- **Subtitles** - Generate timestamped captions
- **Voice notes** - Quick audio transcription

### When to Use Each Size

| Use Case | Recommended |
|----------|-------------|
| Real-time on edge | Tiny/Base |
| Balanced | Small |
| High accuracy | Medium |
| Maximum accuracy | Large-v3 |

## Best Practices

1. **Use appropriate model size** - Tiny for speed, Large for accuracy
2. **Handle long audio** - Chunk with overlap
3. **Specify language** - Faster than auto-detect
4. **Use VAD** - Skip silent segments

## Loading from Ollama

```rust
use unillm::ollama::OllamaRegistry;

// Whisper models (check availability)
let path = OllamaRegistry::pull("whisper:large")?;
```

## References

- [Whisper Paper](https://arxiv.org/abs/2212.04356)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Whisper.cpp](https://github.com/ggerganov/whisper.cpp)
