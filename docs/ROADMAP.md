# UniLLM Roadmap

This document is an honest assessment of where UniLLM is today, what works, what doesn't, and where we're headed.

## Current Status (v0.1.0)

### What works

- **LLaMA inference end-to-end** -- Download a model from Ollama, load GGUF weights, tokenize, run forward pass, generate text. The `unillm generate` CLI does this.
- **Weight loading** -- GGUF (with dequantization for Q4_0, Q8_0, etc.), SafeTensors, and PyTorch formats via `UnifiedWeightLoader`.
- **Tokenization** -- GGUF-based tokenizer with byte-level fallback, HuggingFace tokenizer support.
- **Sampling** -- Greedy, temperature, and top-p (nucleus) sampling.
- **47 model architectures defined** -- All implement the `Model` trait with `forward()` and `generate()` methods using the `model_config!` macro.
- **KV cache** -- Hybrid RadixAttention + PagedAttention cache with adaptive tiering policy.
- **SIMD kernels** -- AVX2, AVX-512, and NEON implementations for quantized matmul, RMSNorm, RoPE, SwiGLU.
- **Scheduler** -- Request scheduling with continuous batching, chunked prefill, admission control.
- **Tests** -- 201 tests pass across all workspace crates.

### What doesn't work yet

- **GPU acceleration** -- All inference runs on CPU. The tensor abstraction supports CUDA and Metal via Candle feature flags, but no GPU-specific optimization has been done.
- **Most model architectures are untested with real weights** -- Only LLaMA has been validated end-to-end with actual GGUF models. The other 46 architectures have correct forward pass implementations (verified with unit tests using dummy tensors) but have not been tested with real model weights.
- **No HTTP server** -- There is no API server. Inference is CLI-only or programmatic.
- **No streaming generation** -- Tokens are generated sequentially but not streamed to clients (the CLI does print tokens as they're produced).
- **Advanced sampling** -- No repetition penalty, beam search, or min-p sampling.
- **Quantized inference without dequantization** -- GGUF weights are currently dequantized to f32 before inference. Running quantized matmul directly on Q4/Q8 data is not yet implemented.
- **KV cache integration with inference** -- The KV cache system is implemented and tested but not yet wired into the generation loop for token reuse.

### Known limitations

- The inference pipeline is hardcoded to LLaMA. Other model architectures need config-level integration to be selectable from the CLI.
- Generation speed on CPU is slow for large models. TinyLlama (~1B params) is usable; 7B+ models will be very slow without GPU.
- No model quantization tooling -- you need pre-quantized GGUF files.

## Near-term priorities

### 1. Validate more model architectures
Test Qwen, Phi, Gemma, Mistral, and DeepSeek with real GGUF weights from Ollama. Fix any config or shape mismatches that surface.

### 2. GPU acceleration
Enable CUDA and Metal backends via Candle feature flags. Profile and optimize the hot path (attention, matmul).

### 3. Wire KV cache into generation
Connect the existing `KVCache` / `LayerKVCache` into the autoregressive generation loop to avoid recomputing K/V for previous tokens.

### 4. HTTP API server
Add an OpenAI-compatible API endpoint (`/v1/chat/completions`) using the existing axum dependency. Support streaming via SSE.

### 5. Cargo publish readiness
Rename crates to `unillm-*` namespace, add proper metadata, ensure all crate Cargo.toml files are publish-ready.

## Long-term vision

- **Production inference server** with continuous batching, request queuing, and health monitoring
- **Distributed inference** across multiple GPUs / nodes
- **Speculative decoding** for faster generation
- **Direct quantized inference** -- run Q4_K, Q5_K matmul without dequantization
- **More model families** -- validate and optimize all 47 architectures
- **ONNX / TensorRT export** for deployment

## Architecture

UniLLM is built on three composable layers:

1. **TensorCore** -- Device-agnostic tensor operations
2. **ModelCore** -- Universal `Model` trait and `model_config!` macro
3. **WeightLoaderCore** -- Format-agnostic weight loading

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

## Contributing

High-impact areas for contribution:

- **Model validation** -- Pick a model architecture, download real weights, test inference, fix issues
- **GPU backends** -- CUDA and Metal kernel optimization
- **Benchmarking** -- Compare against llama.cpp and other runtimes
- **Documentation** -- Improve API docs, add examples

See [developer_guide.md](developer_guide.md) for development setup.
