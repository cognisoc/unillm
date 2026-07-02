# unillm

**A modular LLM inference runtime in Rust — 47 architectures behind one `Model` trait.**

[![crates.io](https://img.shields.io/crates/v/unillm-runtime.svg)](https://crates.io/crates/unillm-runtime)
[![docs.rs](https://img.shields.io/docsrs/unillm-runtime)](https://docs.rs/unillm-runtime)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/cognisoc/unillm/actions/workflows/ci.yml/badge.svg)](https://github.com/cognisoc/unillm/actions/workflows/ci.yml)

**[Website](https://unillm.cognisoc.com)** · **[Docs](https://docs.cognisoc.com/unillm/)** · **[crates.io](https://crates.io/crates/unillm-runtime)** · **[API docs](https://docs.rs/unillm-runtime)**

---

## Why unillm

unillm is a unified, type-safe inference runtime for large language models. Every one of its 47 supported
architectures implements the same `Model` trait, so a single code path drives them all — no per-model glue, no
framework lock-in. Weights load format-agnostically from SafeTensors, GGUF, or PyTorch checkpoints; tensor ops run
device-agnostically across CPU, CUDA, and Metal. The three-layer design keeps concerns cleanly separated, and adding
a new architecture is a matter of a config macro and a `forward()` pass — not a fork.

- **Modular** — three composable layers (tensor ops, model trait, weight loading) you can extend independently.
- **Type-safe** — one `Model` trait and a `model_config!` macro give every architecture a consistent, checked API.
- **Format-agnostic** — SafeTensors, GGUF, and PyTorch weights load through one loader.
- **Built to scale** — a hybrid KV cache (RadixAttention + PagedAttention) and continuous batching in the box.

## Install

Add the runtime to your project:

```bash
cargo add unillm-runtime
```

Or build from source:

```bash
git clone https://github.com/cognisoc/unillm.git
cd unillm
cargo check            # verify compilation
cargo test --workspace # run all tests
```

## Quick start

```bash
# Generate text (downloads TinyLlama on first run, ~600MB)
cargo run --bin unillm -p unillm-runtime -- generate --prompt "Explain gravity"

# Use a different model
cargo run --bin unillm -p unillm-runtime -- generate --model llama2:7b --prompt "Hello"

# List cached models
cargo run --bin unillm -p unillm-runtime -- models
```

## Architecture

unillm is organized into three composable layers:

1. **TensorCore** — Device-agnostic tensor operations (CPU, CUDA, Metal). All ops go through `ops_fn::operation()`.
2. **ModelCore** — Universal `Model` trait with `forward()` and `generate()`. Configuration via the `model_config!` macro.
3. **WeightLoaderCore** — Format-agnostic weight loading for SafeTensors, GGUF, and PyTorch files.

On top of these, the workspace layers a high-level inference engine, a hybrid KV cache
(RadixAttention + PagedAttention), and a request scheduler with continuous batching.

```
crates/
  runtime/       Core inference runtime (tensor ops, model trait, weight loading, 47 models)
  inference/     High-level inference engine and batching
  kv/            Hybrid KV cache (RadixAttention + PagedAttention)
  scheduler/     Request scheduling with continuous batching
docs/            Architecture docs, API reference, developer guide
```

## Supported models

unillm implements 47 model architectures across 10 categories:

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

## Adding a model

A new architecture is a config plus a forward pass — no changes to the core:

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

## Development

```bash
cargo check                        # type-check the workspace
cargo test --workspace             # run all tests
cargo test --lib -p unillm-runtime # test the runtime crate
cargo clippy --workspace           # lint
cargo fmt --all                    # format
cargo build --release              # optimized build
```

See [docs/developer_guide.md](docs/developer_guide.md) for a full development setup guide.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — Three-layer system design
- [API Reference](docs/api_reference.md) — Detailed API docs
- [Developer Guide](docs/developer_guide.md) — Getting started with development
- [Roadmap](docs/ROADMAP.md) — Where unillm is headed

## Status

unillm runs inference today across all 47 architectures on CPU, CUDA, and Metal, with the hybrid KV cache and
continuous-batching scheduler in place. Active work is focused on broadening quantization support, deepening the
vision-language and audio paths, and performance tuning. See the [roadmap](docs/ROADMAP.md) for what's next, and
[CONTRIBUTING.md](CONTRIBUTING.md) if you'd like to add an architecture.

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Part of the Cognisoc stack

**[Cognisoc](https://www.cognisoc.com)** builds open-source LLM inference for every language and every device — *LLM inference, everywhere.* This project is one of six:

| Project | Language | What it does |
|---|---|---|
| [mullama](https://github.com/cognisoc/mullama) | Python · Node · Go · PHP · Rust · C | Local LLM runtime & server, drop-in Ollama alternative |
| unillm **(this project)** | Rust | Modular inference runtime, 47 architectures |
| [llamafu](https://github.com/cognisoc/llamafu) | Dart / Flutter | On-device inference for mobile apps |
| [llmdot](https://github.com/cognisoc/llmdot) | C# / .NET | Local GGUF inference for the .NET ecosystem |
| [cllm](https://github.com/cognisoc/cllm) | C | Bare-metal unikernel — boots straight into inference |
| [zigllm](https://github.com/cognisoc/zigllm) | Zig | Learn LLMs by building one, from tensors to text |

🌐 [cognisoc.com](https://www.cognisoc.com) · 📚 [docs.cognisoc.com](https://docs.cognisoc.com) · 🐙 [github.com/cognisoc](https://github.com/cognisoc)

---
