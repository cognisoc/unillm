# UniLLM

**A modular LLM inference runtime written in Rust.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CI](https://github.com/Skelf-Research/unillm/actions/workflows/ci.yml/badge.svg)](https://github.com/Skelf-Research/unillm/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org)

UniLLM provides a unified, type-safe interface for running large language models across 47 architectures. It is built around three composable abstractions -- TensorCore, ModelCore, and WeightLoaderCore -- that let you load weights in any format, run inference on any device, and add new model architectures with minimal boilerplate.

---

## Quick Start

```bash
git clone https://github.com/Skelf-Research/unillm.git
cd unillm
cargo check            # verify compilation
cargo test --workspace # run all tests
```

### Run inference with a model from Ollama

```bash
# Downloads TinyLlama and runs a basic generation
cargo run --bin test_ollama -p runtime

# List locally cached models
cargo run --bin test_ollama -p runtime -- --list-cached

# Use a different model
cargo run --bin test_ollama -p runtime -- --model llama2:7b
```

## Supported Models

UniLLM implements 47 model architectures across 10 categories:

| Category | Models |
|---|---|
| Core LLMs | LLaMA, Qwen, Gemma, Phi, DeepSeek, Mistral, Mixtral |
| GPT Family | GPT-2, GPT-J, GPT-NeoX, OPT, BLOOM, MPT |
| Code | StarCoder, CodeLlama |
| MoE | DeepSeek-MoE, DBRX, Grok, Arctic, Jamba |
| RWKV / Linear Attention | RWKV-4, RWKV-6, RecurrentGemma |
| Vision-Language | Qwen2-VL, Phi-3-Vision, InternVL, CogVLM, Idefics, Florence, LLaVA, CLIP |
| Audio / Speech | Wav2Vec2, HuBERT, MusicGen, Encodec, Whisper |
| Encoder | BERT, T5 |
| Specialized | Mamba, MiniCPM, OLMo, Granite |
| Additional | Yi, Falcon, Baichuan, InternLM, ChatGLM |

All models share the same `Model` trait and are configured through the `model_config!` macro.

## Architecture

UniLLM is organized into three layers:

1. **TensorCore** -- Device-agnostic tensor operations (CPU, CUDA, Metal). All ops go through `ops_fn::operation()`.
2. **ModelCore** -- Universal `Model` trait with `forward()` and `generate()`. Configuration via `model_config!` macro.
3. **WeightLoaderCore** -- Format-agnostic weight loading for SafeTensors, GGUF, and PyTorch files.

### Adding a model

```rust
model_config!(MyModelConfig {
    vocab_size: usize = 32000,
    hidden_size: usize = 4096,
    num_hidden_layers: usize = 32,
});

impl Model for MyModel {
    type Config = MyModelConfig;

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        // model-specific forward pass
    }
}
```

## Project Structure

```
crates/
  runtime/       Core inference runtime (tensor ops, model trait, weight loading, 47 models)
  inference/     High-level inference engine and batching
  kv/            Hybrid KV cache (RadixAttention + PagedAttention)
  scheduler/     Request scheduling with continuous batching
docs/            Architecture docs, API reference, developer guide
```

## Development

```bash
cargo check                       # type-check the workspace
cargo test --workspace            # run all tests
cargo test --lib -p runtime       # test the runtime crate
cargo clippy --workspace          # lint
cargo fmt --all                   # format
cargo build --release             # optimized build
```

See [docs/developer_guide.md](docs/developer_guide.md) for a full development setup guide.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) -- Three-layer system design
- [API Reference](docs/api_reference.md) -- Detailed API docs
- [Developer Guide](docs/developer_guide.md) -- Getting started with development

## License

Apache-2.0 -- see [LICENSE](LICENSE) for details.
