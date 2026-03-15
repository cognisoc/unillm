# UniLLM Project - Todo and Progress Tracker

## Project Overview
UniLLM is a unikernel-based LLM inference engine supporting both NVIDIA (CUDA) and AMD (ROCm/HIP) GPUs with the following key features:
- KVM-based virtualization with VFIO passthrough
- Dual backend support (CUDA/HIP)
- Performance targets competitive with vLLM and TensorRT-LLM
- Support for advanced features like FlashAttention, paged KV, continuous batching, low-precision inference, and speculative decoding

## Phased Implementation Plan

### Phase 0: Foundation - VM skeleton & passthrough
**Goal**: Establish KVM boot, VFIO passthrough, PCIe/IOMMU/MSI-X, pinned host memory; implement GPU backend abstraction

#### Milestone 0 Tasks
- [x] Create project directory structure with crates and docs directories
- [x] Create Cargo workspace and crate structure
- [x] Implement GPU backend abstraction trait
- [x] Create basic hypervisor crate structure
- [x] Create basic HAL crate structure
- [x] Create basic GPU backend crate structures (CUDA/HIP)
- [x] Create basic kernels crate with build script
- [x] Create basic KV, scheduler, runtime, tokenizer, and telemetry crates
- [x] Create main application entry point
- [x] Create README.md with project overview
- [x] KVM hypervisor implementation
- [x] VFIO passthrough setup
- [x] Pinned host memory management
- [x] PCIe/IOMMU/MSI-X handling
- [x] Rust FFI islands for GPU interaction
- [x] CUDA backend: Real CUDA runtime bindings for contexts/streams
- [x] AMD backend: Real HIP runtime bindings for contexts/streams/events
- [x] Validation with H2D/D2H + stream sync for both backends
- [x] ROCm stack validation for supported GPUs (MI300/MI200 + RDNA3)

### Phase 1: First Token Fast Path
**Goal**: Implement FA-2/FA-3 + graph-captured decode (single stream)

#### Milestone 1 Tasks
- [ ] Model loader implementation
- [ ] Rust tokenizer integration
- [ ] Greedy sampler
- [ ] Graph-captured steady-state decode loop
- [ ] CUDA backend: FA-2/FA-3 with runtime SM check, cuBLASLt GEMMs, CUDA Graphs capture
- [ ] AMD backend: hipBLASLt/rocBLAS GEMMs, FlashAttention-2 HIP or Triton attention, HIP Graphs capture
- [ ] Performance validation (NVIDIA: +5-15% vs vLLM; AMD: parity to +10% vs vLLM ROCm)

### Phase 2: Batching and Paged KV
**Goal**: Implement paged KV allocator and continuous batching with chunked prefill

#### Milestone 2 Tasks
- [ ] vLLM-style paged KV allocator
- [ ] In-flight batching with 1-3 ms window
- [ ] Chunked prefill implementation
- [ ] H2D/compute overlap via multiple streams
- [ ] Performance validation (1.5-2.5x improvement vs M1 under 8-32 concurrency)

### Phase 3: Low-Precision Support
**Goal**: Implement INT8/INT4 and FP8 support for both NVIDIA and AMD

#### Milestone 3 Tasks
- [ ] INT8/INT4 decode paths (GPTQ/AWQ/etc.)
- [ ] Dequant fusion implementation
- [ ] NVIDIA: FP8 Hopper path with fused dequant + FA-2/3 integration
- [ ] AMD: FP8 on MI300X via hipBLASLt/ComposableKernel/rocWMMA
- [ ] Performance validation (VRAM relief up to 50-75%; long-context decode +10-40% vs M2)

### Phase 4: Speculative Decoding
**Goal**: Implement speculative decoding with accept/draft pipeline

#### Milestone 4 Tasks
- [ ] Two-engine pipeline (draft + target)
- [ ] Batching-aware speculation window
- [ ] Recurrent Drafter variant implementation
- [ ] Graph capture around accept path
- [ ] Performance validation (-15 to -35% latency/token on longer outputs)

### Phase 5: Multi-GPU Support
**Goal**: Implement tensor parallelism with NUMA discipline

#### Milestone 5 Tasks
- [ ] NCCL/RCCL ring implementation
- [ ] Graph-captured multi-GPU decode
- [ ] Per-GPU pinned buffers
- [ ] Performance validation (2x GPUs: 1.7-1.9x; 4x: 3.2-3.6x decode-heavy)

### Phase 6: Production Hardening
**Goal**: Implement production-ready features and optimizations

#### Milestone 6 Tasks
- [ ] Driver reset health checks (CUDA & ROCm)
- [ ] Determinism improvements
- [ ] Crash-safe snapshots
- [ ] Profiling support (Nsight Systems / rocProfiler/rocTracer)
- [ ] AMD MI300X acceptance/perf checklists implementation

## Repository Structure
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

## Build Configuration
- Features: `--features cuda`, `--features hip`, plus `fa3`, `fp8`, `int8`, `int4`
- Runtime GPU detection with automatic path selection

## Performance Targets
| Scenario | After M1 | After M2 | After M3 (INTx/FP8) | After M4 | After M5 |
|---------|----------|----------|-------------------|----------|----------|
| NVIDIA single-stream p50 | +5–15% vs vLLM | +5–10% | +10–25% (FA-3/FP8) | +20–40% | +20–40% |
| NVIDIA throughput 8–32c | -10–20% | -0–15% | -0–15% | -0–10% | -5–20% |
| AMD (MI300X) single-stream p50 | parity to +10% vs vLLM ROCm | +5–10% | +10–25% (FP8) | +20–35% | +20–35% |
| AMD (MI300X) throughput 8–32c | -10–15% | parity to -10% | parity to -10% | parity to -5% | -5–15% |

## Current Status
- [x] Phase 0: Actually Completed (8/8 tasks completed with real implementations)
- [ ] Phase 1: Not started (7 tasks)
- [ ] Phase 2: Not started (5 tasks)
- [ ] Phase 3: Not started (5 tasks)
- [ ] Phase 4: Not started (5 tasks)
- [ ] Phase 5: Not started (4 tasks)
- [ ] Phase 6: Not started (5 tasks)

## Implementation Summary
We've successfully implemented all the foundational components for Phase 0 with actual working code rather than just stubs:

1. **KVM initialization** - Real implementation using libc bindings with VM setup, VCPU creation, and run data mapping
2. **VFIO passthrough** - Actual VFIO container/group management with device enablement and DMA mapping
3. **PCIe/IOMMU/MSI-X handling** - Proper hardware abstraction layer with configuration space access and BAR management
4. **Pinned host memory management** - Real memory allocation with mmap/mlock and proper cleanup
5. **CUDA context and stream management** - Real CUDA runtime bindings with device management and error handling
6. **HIP context, stream, and event management** - Real HIP runtime bindings with device management and error handling
7. **H2D/D2H transfers** - Actual transfer implementations with error handling and data validation
8. **Build system and validation** - Proper CUDA/HIP build dependencies and comprehensive validation tests

## Competitive Strategy Update

Based on comprehensive analysis of vLLM and SGLang codebases, UniLLM's roadmap has been updated to incorporate the best innovations from both frameworks while leveraging our unikernel advantages:

### Key Competitive Differentiators
- **Hybrid Memory Management**: Combine vLLM's PagedAttention with SGLang's RadixAttention
- **Intelligent Scheduling**: Cache-aware adaptive policies beyond simple FCFS
- **Zero-Overhead Design**: Rust + unikernel eliminates Python/OS bottlenecks
- **Enterprise-First**: Built-in observability, fault tolerance, and security

### Updated Performance Targets
| Metric | Target vs vLLM | Target vs SGLang | Competitive Advantage |
|--------|---------------|------------------|----------------------|
| **Single-stream Latency** | +20-40% | +15-30% | Unikernel overhead elimination |
| **Batch Throughput** | +10-30% | +10-20% | Hybrid caching + intelligent scheduling |
| **Memory Efficiency** | +30-50% | +20-35% | Multi-tier cache hierarchy |
| **Multi-GPU Scaling** | +5-15% | +5-10% | Custom communication optimization |

## Documentation Strategy

Comprehensive documentation is being maintained as we implement:

- **[Competitive Analysis](competitive_analysis.md)**: Detailed vLLM vs SGLang technical analysis
- **[Implementation Roadmap](implementation_roadmap.md)**: 4-phase technical implementation plan
- **[Implementation Docs](implementation/)**: Component-by-component implementation tracking

## Next Steps - Revised Implementation Plan

### Phase 1: Foundation & Core Performance (Months 1-3)
**NEW FOCUS**: Hybrid innovations combining best of vLLM + SGLang

1. **Advanced Memory Management** (`crates/kv/`):
   - Hybrid RadixAttention + PagedAttention caching
   - Multi-tier cache hierarchy (L1: radix, L2: paged, L3: compressed)
   - Reference counting and intelligent eviction

2. **Zero-Overhead Scheduler** (`crates/scheduler/`):
   - Cache-aware batch formation
   - Adaptive policy engine (FCFS → LPM → ML-optimized)
   - Sub-millisecond scheduling decisions

3. **High-Performance Kernels** (`crates/kernels/`):
   - Template-based kernel generation
   - CUDA/HIP graph capture and optimization
   - Fused operations (attention+scaling, quantization+GEMM)

### Phase 2-4: Advanced Features & Enterprise Scale
Building on the solid Phase 1 foundation with:
- Advanced quantization framework
- Speculative decoding engine
- Multi-GPU distributed scaling
- Production monitoring and fault tolerance

Each phase builds upon the previous ones, with continuous benchmarking against vLLM and SGLang to ensure we meet our performance targets.