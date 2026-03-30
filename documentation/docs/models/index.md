# Model Catalog

UniLLM supports 47 model architectures across 10 categories. This catalog provides details on each supported model.

## Overview

| Category | Count | Description |
|----------|-------|-------------|
| [Core LLMs](#core-llms) | 5 | Modern decoder-only language models |
| [GPT Family](#gpt-family) | 4 | Original GPT architecture variants |
| [Code Models](#code-models) | 3 | Code generation specialists |
| [MoE Models](#moe-models) | 6 | Mixture-of-Experts architectures |
| [RWKV Family](#rwkv-family) | 3 | Linear attention / RNN models |
| [Embedding Models](#embedding-models) | 4 | Encoder-only models for embeddings |
| [Vision-Language](#vision-language) | 8 | Multimodal vision-language models |
| [Multimodal](#multimodal) | 3 | Multi-input multimodal models |
| [Audio](#audio) | 4 | Speech and audio processing |
| [Specialized](#specialized) | 7 | Encoder-decoder and unique architectures |

## Core LLMs

Modern decoder-only transformer architectures.

| Model | Parameters | Context | Key Features |
|-------|------------|---------|--------------|
| [LLaMA](llm/llama.md) | 7B-70B | 4K-128K | RoPE, GQA, SwiGLU |
| [Qwen](llm/qwen.md) | 0.5B-72B | 8K-128K | Strong multilingual |
| [Gemma](llm/gemma.md) | 2B-7B | 8K | Google's open model |
| [Phi](llm/phi.md) | 1.3B-3B | 2K-128K | Efficient, high quality |
| [Mistral](llm/mistral.md) | 7B | 8K-32K | Sliding window attention |

## GPT Family

Original GPT-style decoder architectures.

| Model | Parameters | Context | Key Features |
|-------|------------|---------|--------------|
| GPT-2 | 124M-1.5B | 1K | Original architecture |
| GPT-J | 6B | 2K | Open-source GPT |
| GPT-NeoX | 20B | 2K | Rotary embeddings |
| OPT | 125M-175B | 2K | Meta's open model |

## Code Models

Specialized for code generation and understanding.

| Model | Parameters | Context | Key Features |
|-------|------------|---------|--------------|
| StarCoder | 1B-15B | 8K | Multi-language code |
| CodeLlama | 7B-34B | 16K | Code-specialized LLaMA |
| CodeGen | 350M-16B | 2K | Salesforce code model |

## MoE Models

Mixture-of-Experts architectures for efficiency at scale.

| Model | Parameters | Context | Key Features |
|-------|------------|---------|--------------|
| [Mixtral](moe/mixtral.md) | 8x7B | 32K | Top-2 gating |
| [DeepSeek-MoE](moe/deepseek-moe.md) | 16B | 4K | Fine-grained experts |
| Dbrx | 132B | 32K | Databricks MoE |
| Grok | 314B | 8K | xAI MoE |
| Arctic | 480B | 4K | Snowflake MoE |
| [Jamba](moe/jamba.md) | 52B | 256K | Mamba + Attention hybrid |

## RWKV Family

Linear attention and RNN-based models.

| Model | Parameters | Context | Key Features |
|-------|------------|---------|--------------|
| RWKV-4 | 169M-14B | Unlimited | Time-mixing mechanism |
| RWKV-6 | 1.6B-14B | Unlimited | Improved architecture |
| RecurrentGemma | 2B | 8K | Griffin architecture |

## Embedding Models

Encoder-only models for text embeddings.

| Model | Parameters | Context | Key Features |
|-------|------------|---------|--------------|
| BERT | 110M-340M | 512 | Bidirectional encoder |
| RoBERTa | 125M-355M | 512 | Optimized BERT |
| XLM-RoBERTa | 270M-550M | 512 | Multilingual |
| MPNet | 110M | 512 | Permuted language modeling |

## Vision-Language

Models that understand both images and text.

| Model | Parameters | Context | Key Features |
|-------|------------|---------|--------------|
| [CLIP](vision-language/clip.md) | 400M | 77 | Contrastive learning |
| [LLaVA](vision-language/llava.md) | 7B-13B | 4K | Visual instruction tuning |
| [Qwen2-VL](vision-language/qwen2-vl.md) | 2B-72B | 32K | Native multimodal |
| Phi-3-Vision | 4B | 128K | Efficient VLM |
| InternVL | 1B-40B | 4K | Strong vision encoder |
| CogVLM | 17B | 4K | Cognitive VLM |
| Idefics | 9B-80B | 4K | Open Flamingo |
| Florence | 230M-770M | N/A | Microsoft vision |

## Multimodal

Advanced multimodal architectures.

| Model | Parameters | Context | Key Features |
|-------|------------|---------|--------------|
| Flamingo | 3B-80B | 4K | Few-shot multimodal |
| BLIP-2 | 2.7B-12B | 512 | Q-Former architecture |
| PaLI | 3B-17B | N/A | Pathways multimodal |

## Audio

Speech and audio processing models.

| Model | Parameters | Context | Key Features |
|-------|------------|---------|--------------|
| [Whisper](audio/whisper.md) | 39M-1.5B | 30s | Speech recognition |
| [Wav2Vec2](audio/wav2vec2.md) | 95M-1B | N/A | Self-supervised ASR |
| HuBERT | 95M-1B | N/A | Hidden unit BERT |
| Encodec | 15M-80M | N/A | Neural audio codec |

## Specialized

Unique architectures and encoder-decoder models.

| Model | Parameters | Context | Key Features |
|-------|------------|---------|--------------|
| T5 | 60M-11B | 512 | Text-to-text |
| BART | 140M-400M | 1K | Denoising autoencoder |
| Falcon | 7B-180B | 2K | FlashAttention optimized |
| Bloom | 560M-176B | 2K | Multilingual (46 languages) |
| StableLM | 1.6B-7B | 4K | Stability AI |
| MusicGen | 300M-3.3B | N/A | Music generation |

## Quick Start

### Loading a Model

```rust
use unillm::models_v2::llama::{LlamaModelV2, LlamaConfig};
use unillm::weight_loader_core::WeightLoader;
use unillm::Model;

// Load from GGUF
let weights = WeightLoader::from_gguf("llama-7b.gguf")?;
let config = LlamaConfig::from_gguf_metadata(weights.metadata())?;
let model = LlamaModelV2::from_weights(config, weights)?;

// Generate text
let response = model.generate("Hello", &GenerationConfig::default())?;
```

### Loading from Ollama

```rust
use unillm::ollama::OllamaRegistry;

// Download and load
let path = OllamaRegistry::pull("llama2:7b")?;
let weights = WeightLoader::from_gguf(&path)?;
```

## Supported Formats

All models support:

| Format | Extension | Quantization | Notes |
|--------|-----------|--------------|-------|
| GGUF | `.gguf` | Q2-Q8, F16, F32 | Recommended |
| SafeTensors | `.safetensors` | F16, F32 | HuggingFace format |
| PyTorch | `.bin`, `.pt` | F16, F32 | Legacy support |

## Model Selection Guide

### By Use Case

| Use Case | Recommended Models |
|----------|-------------------|
| General chat | LLaMA 3, Qwen 2, Mistral |
| Code generation | CodeLlama, StarCoder |
| Creative writing | Mixtral, LLaMA 3 70B |
| Embeddings | BERT, MPNet |
| Image understanding | LLaVA, Qwen2-VL |
| Speech transcription | Whisper |

### By Hardware

| Hardware | Recommended Models |
|----------|-------------------|
| Consumer GPU (8GB) | Phi-3, Gemma 2B, Q4 quantized 7B |
| Gaming GPU (16GB) | 7B models, Mixtral Q4 |
| Pro GPU (24GB+) | 13B-34B models |
| Multi-GPU | 70B+ models |
| CPU only | Phi-2, Q4 quantized 7B |
