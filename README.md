# UniLLM

UniLLM is a unikernel-based LLM inference engine supporting both NVIDIA (CUDA) and AMD (ROCm/HIP) GPUs. It aims to provide performance competitive with vLLM and TensorRT-LLM while maintaining the benefits of a unikernel architecture including low latency, fast cold starts, and small footprint.

## Features

- KVM-based virtualization with VFIO passthrough
- Dual backend support (CUDA/HIP)
- FlashAttention-2/3 support
- Paged KV cache allocator
- Continuous batching with chunked prefill
- Low-precision inference (INT8/INT4/FP8)
- Speculative decoding
- Multi-GPU tensor parallelism
- Production hardening features

## Architecture

The project follows a modular architecture with the following crates:

```
/crates
  /hypervisor        # KVM/VFIO
  /hal               # PCIe/IOMMU/MSI-X
  /gpu-backend
    /cuda            # cuBLASLt, FA2/FA3, CUDA Graphs
    /hip             # hipBLASLt/rocBLAS, Triton/FA2 HIP, HIP Graphs
  /kernels           # bindings/build.rs for CUDA & HIP variants
  /kv                # paged-KV allocator
  /scheduler         # continuous batching + prefill chunking
  /runtime           # model graph, loader, sampler
  /tokenizer         # Rust tokenizers
  /telemetry         # tracing + counters
/app                 # main + bench harness
```

## Build

To build UniLLM, you need Rust and either CUDA or ROCm development libraries installed.

For NVIDIA GPUs:
```bash
cargo build --features cuda
```

For AMD GPUs:
```bash
cargo build --features hip
```

Additional features can be enabled:
- `fa3`: Enable FlashAttention-3 support (NVIDIA Hopper only)
- `fp8`: Enable FP8 low-precision inference
- `int8`: Enable INT8 low-precision inference
- `int4`: Enable INT4 low-precision inference

## Performance Targets

| Scenario | After M1 | After M2 | After M3 (INTx/FP8) | After M4 | After M5 |
|---------|----------|----------|-------------------|----------|----------|
| NVIDIA single-stream p50 | +5–15% vs vLLM | +5–10% | +10–25% (FA-3/FP8) | +20–40% | +20–40% |
| NVIDIA throughput 8–32c | -10–20% | -0–15% | -0–15% | -0–10% | -5–20% |
| AMD (MI300X) single-stream p50 | parity to +10% vs vLLM ROCm | +5–10% | +10–25% (FP8) | +20–35% | +20–35% |
| AMD (MI300X) throughput 8–32c | -10–15% | parity to -10% | parity to -10% | parity to -5% | -5–15% |

## Documentation

- [Todo and Progress Tracker](docs/todo_and_progress.md)
- [Technical Specifications](specs.md)

## License

MIT