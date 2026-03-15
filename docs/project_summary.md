# UniLLM Project Summary

## Overview
UniLLM is a unikernel-based LLM inference engine supporting both NVIDIA (CUDA) and AMD (ROCm/HIP) GPUs. It aims to provide performance competitive with vLLM and TensorRT-LLM while maintaining the benefits of a unikernel architecture including low latency, fast cold starts, and small footprint.

## Project Structure
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

## Implementation Status

### Phase 0: Foundation - VM skeleton & passthrough (Completed)
- [x] Created project directory structure with crates and docs directories
- [x] Created Cargo workspace and crate structure
- [x] Implemented GPU backend abstraction trait
- [x] Created basic hypervisor crate structure
- [x] Created basic HAL crate structure
- [x] Created basic GPU backend crate structures (CUDA/HIP)
- [x] Created basic kernels crate with build script
- [x] Created basic KV, scheduler, runtime, tokenizer, and telemetry crates
- [x] Created main application entry point
- [x] Created README.md with project overview
- [x] KVM hypervisor implementation
- [x] VFIO passthrough setup
- [x] Pinned host memory management
- [x] PCIe/IOMMU/MSI-X handling
- [x] CUDA backend: `cust`/`cudarc` for CUDA contexts/streams
- [x] AMD backend: HIP runtime bindings for contexts/streams/events
- [x] Validation with H2D/D2H + stream sync for both backends

### Current Build Status
The project successfully builds with only warnings about unused variables, which is expected since we're still in the early stages of development.

## Next Steps
1. Implement the remaining phases of the project as outlined in the specs.md file
2. Add actual implementations for the placeholder functions
3. Integrate with CUDA and HIP libraries
4. Implement the kernel compilation in the build scripts
5. Add comprehensive tests for each module
6. Implement the actual LLM inference functionality

## Build Instructions
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