Absolutely—here’s the **Rust + KVM (VFIO passthrough)** plan, now **dual-track for NVIDIA *and* AMD**. I’ve kept the same milestones and folded in what changes for ROCm/HIP (MI300/MI250 and RDNA3 prosumer), plus realistic perf expectations vs **vLLM**/**TensorRT-LLM** on NVIDIA and **vLLM ROCm** on AMD.

---

# Core idea

Keep your host/unikernel the same; swap the GPU layer behind a thin **backend trait**:

```rust
trait GpuBackend {
  fn device_init(&mut self) -> Result<()>;
  fn alloc(&mut self, bytes: usize, pinned: bool) -> DevicePtr;
  fn h2d_async(&self, dst: DevicePtr, src: *const u8, n: usize, stream: Stream);
  fn d2h_async(&self, dst: *mut u8, src: DevicePtr, n: usize, stream: Stream);
  fn launch_graph(&self, graph: &DecodeGraph, stream: Stream);
  fn gemm(&self, /* cuBLASLt | hipBLASLt */);
  fn flash_attention(&self, /* FA2/FA3 CUDA | Triton/FA2 HIP */);
  // ...
}
```

* **CUDA backend** → cuBLASLt, FA-2/FA-3, CUDA Graphs.
* **ROCm backend** → hipBLASLt/rocBLAS, FA-2 or Triton attention, **HIP Graphs**. ([ROCm Documentation][1])

---

# Milestones (what’s new for AMD)

## M0 — VM skeleton & passthrough

**Common:** KVM boot, VFIO passthrough, PCIe/IOMMU/MSI-X, pinned host memory; Rust FFI islands.

**NVIDIA:** `cust`/`cudarc` for CUDA contexts/streams.
**AMD:** **HIP runtime** bindings (or C FFI to `libamdhip64.so`) for contexts/streams/events. Validate with H2D/D2H + stream sync.

* ROCm stack overview & supported GPUs (MI300/MI200 + RDNA3 prosumer). ([AMD][2], [docs.vllm.ai][3])

---

## M1 — First token fast path: FA-2/FA-3 + graph-captured decode (single stream)

**What stays the same:** model loader, Rust tokenizer, greedy sampler, graph-captured steady-state decode loop.

**NVIDIA path**

* **FA-2** broadly; **FA-3 on Hopper (H100)** with runtime SM check; cuBLASLt GEMMs; **CUDA Graphs** capture. ([ROCm Documentation][4])

**AMD path**

* GEMMs via **hipBLASLt/rocBLAS**; attention via one of:

  * **FlashAttention-2 HIP** (where available), or
  * **Triton on ROCm** kernels, or
  * **xFormers memory-efficient attention** on ROCm.
* Capture decode loop with **HIP Graphs**. ([Rocm Blog][5], [GitHub][6], [ROCm Documentation][7])

**Perf targets**

* NVIDIA: single-stream p50 **+5–15% vs vLLM**; **parity to −10%** vs TensorRT-LLM (FP16/BF16).
* AMD (MI300X): single-stream p50 **parity to small win vs vLLM ROCm**, assuming graph capture + fast attention. AMD docs and vLLM MI300X posts show strong ROCm baselines. ([ROCm Documentation][4], [vLLM Blog][8])

---

## M2 — **Paged KV** + **Continuous batching** (+ chunked prefill & overlap)

**Common:** vLLM-style **paged KV** allocator; **in-flight batching** with 1–3 ms window; chunked prefill; overlap H2D/compute via multiple streams.

* AMD guidance explicitly highlights **paged attention** benefits; vLLM ROCm ships it. ([ROCm Documentation][4], [docs.vllm.ai][3])

**Perf targets**

* Both vendors: **+1.5× … +2.5×** vs M1 under 8–32 conc.
* vs vLLM (NVIDIA): **parity to −15%** depending on paging/admission tightness.
* vs vLLM ROCm (MI300X): **parity plausible** if your paged-KV indirection matches their kernels. ([ROCm Documentation][4])

---

## M3 — **Low-precision**: INT8/INT4 (both) + **FP8** (H100/MI300X)

**Common:** INT8/INT4 decode paths (GPTQ/AWQ/etc.), dequant fusions where possible.

**NVIDIA:** FP8 Hopper path; aim for fused dequant + FA-2/3 integration.
**AMD:** **FP8 on MI300X**—vLLM supports FP8 KV cache selection on MI300X; target hipBLASLt/ComposableKernel/rocWMMA paths. Report separate FP8 numbers (weights and/or KV). ([vLLM Blog][8], [AMD Instinct Documentation][9])

**Perf targets**

* VRAM relief up to **\~50–75%**; long-context decode **+10–40%** vs M2.
* NVIDIA vs TensorRT-LLM FP8: still **−10 … −20%** unless you reuse equivalent fusions.
* AMD vs vLLM ROCm (FP8 enabled): **parity ±10%** depending on kernels. ([vLLM Blog][8])

---

## M4 — **Speculative decoding** (Accept/Draft)

Two-engine pipeline (draft + target), batching-aware speculation window; Recurrent Drafter variant ok.

* Works on both CUDA and HIP backends; keep graph capture around the accept path.

**Perf targets**

* **−15 … −35%** latency/token on longer outputs on both vendors. (vLLM and AMD guides recommend it for MI300X throughput/latency.) ([Rocm Blog][10])

---

## M5 — Multi-GPU (tensor parallel) + NUMA discipline

* **NVIDIA:** NCCL rings; graph-captured multi-GPU decode; per-GPU pinned buffers.
* **AMD:** **RCCL** (ROCm’s NCCL) rings; same orchestration.

**Perf targets**

* 2× GPUs: **1.7–1.9×**; 4×: **3.2–3.6×** decode-heavy.
* Expect **−5 … −20%** vs the most fused vendor stacks unless your comm/compute overlap matches theirs.

---

## M6 — Production hardening

* Health checks for driver resets (both CUDA & ROCm), determinism, crash-safe snapshots, Nsight Systems (CUDA) / rocProfiler/rocTracer equivalents (ROCm).
* AMD MI300X acceptance/perf checklists exist; follow them. ([AMD Instinct Documentation][9], [ROCm Documentation][11])

---

## NVIDIA vs AMD: which attention kernels?

* **NVIDIA:** FA-2 everywhere, **FA-3 on Hopper**; CUDA Graphs. ([ROCm Documentation][4])
* **AMD:** Options (pick by maturity on your target):

  1. **Triton on ROCm** kernels (actively supported by AMD) for attention/GEMM-adjacent ops. ([Rocm Blog][5])
  2. **FlashAttention-2 HIP** builds when available. ([GitHub][12])
  3. **xFormers memory-efficient attention** on ROCm as a fallback. ([GitHub][6])
* **Graph capture**: use **HIP Graphs** API on ROCm; mirror your CUDA path. ([ROCm Documentation][7])

---

## Quantization notes (AMD)

* FP8 is **first-class on MI300X**; vLLM exposes KV FP8 selection to extend context/deployability—mirror that behavior. ([vLLM Blog][8])
* INT8/INT4: use hipBLASLt/Composable Kernel paths; AMD docs indicate these libraries are the perf-oriented route. ([AMD Instinct Documentation][9])

---

## Updated perf table (both vendors)

| Scenario                           |                        After M1 |           After M2 |    After M3 (INTx/FP8) |      After M4 | After M5 |
| ---------------------------------- | ------------------------------: | -----------------: | ---------------------: | ------------: | -------: |
| **NVIDIA single-stream p50**       |              **+5–15% vs vLLM** |             +5–10% | **+10–25%** (FA-3/FP8) |   **+20–40%** |  +20–40% |
| **NVIDIA throughput 8–32c**        |                         −10–20% |         **−0–15%** |                 −0–15% |        −0–10% |   −5–20% |
| **AMD (MI300X) single-stream p50** | **parity to +10% vs vLLM ROCm** |             +5–10% |      **+10–25%** (FP8) |   **+20–35%** |  +20–35% |
| **AMD (MI300X) throughput 8–32c**  |                         −10–15% | **parity to −10%** |         parity to −10% | parity to −5% |   −5–15% |

*Notes:*

* These compare your unikernel to **vLLM (CUDA)** on NVIDIA and **vLLM (ROCm)** on AMD. TRT-LLM remains the NVIDIA FP8 leader; you’ll generally trail **10–20%** unless you replicate its most aggressive fusions. ([Rocm Blog][10])

---

## Repo layout (unchanged, now with `cuda` and `hip` backends)

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

**Build toggles**: `--features cuda`, `--features hip`, plus `fa3`, `fp8`, `int8`, `int4`. Detect GPU and pick the fastest viable path at runtime.

---

## Bench & sanity checklist (AMD add-ons)

* Follow AMD’s MI300X **acceptance/perf bench guides**; lock clocks, validate hipBLASLt throughput, and verify **HIP Graphs** speedup. ([AMD Instinct Documentation][9], [ROCm Documentation][11])
* Include **vLLM ROCm** as a baseline; AMD and vLLM both document setup and best practices. ([docs.vllm.ai][3], [Rocm Blog][13])
* If using Triton kernels on ROCm, use AMD’s “Developing Triton Kernels on AMD GPUs” guidance to validate tile sizes and occupancy. ([Rocm Blog][5])

---

### Bottom line

This dual-backend plan keeps your **unikernel advantages** (latency, cold start, footprint) while aligning with **best-in-class** GPU algorithms on both vendors: paged-KV, in-flight batching, FlashAttention-class kernels, graph-captured decode, and low-precision (FP8/INTx). On NVIDIA you’ll chase TRT-LLM on FP8; on AMD MI300X you should be **at or near vLLM ROCm** once M2–M3 land.

If you want, I can generate a tiny **workspace scaffold** with the `GpuBackend` trait, `cuda/hip` crates, and `build.rs` stubs for nvcc/hipcc so you can start coding immediately.

[1]: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/model-acceleration-libraries.html?utm_source=chatgpt.com "Model acceleration libraries - ROCm Documentation - AMD"
[2]: https://www.amd.com/en/products/software/rocm.html?utm_source=chatgpt.com "AMD ROCm™ Software"
[3]: https://docs.vllm.ai/en/v0.6.5/getting_started/amd-installation.html?utm_source=chatgpt.com "Installation with ROCm - vLLM"
[4]: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/llm-inference-frameworks.html?utm_source=chatgpt.com "LLM inference frameworks - ROCm Documentation - AMD"
[5]: https://rocm.blogs.amd.com/artificial-intelligence/triton/README.html?utm_source=chatgpt.com "Developing Triton Kernels on AMD GPUs — ROCm Blogs"
[6]: https://github.com/facebookresearch/xformers?utm_source=chatgpt.com "facebookresearch/xformers: Hackable and optimized ..."
[7]: https://rocm.docs.amd.com/projects/HIP/en/docs-develop/how-to/hip_runtime_api/hipgraph.html?utm_source=chatgpt.com "HIP graphs — HIP 7.1.0 Documentation"
[8]: https://blog.vllm.ai/2024/10/23/vllm-serving-amd.html?utm_source=chatgpt.com "Serving LLMs on AMD MI300X: Best Practices - vLLM Blog"
[9]: https://instinct.docs.amd.com/projects/system-acceptance/en/latest/mi300x/performance-bench.html?utm_source=chatgpt.com "Performance benchmarking - Instinct™ Documentation - AMD"
[10]: https://rocm.blogs.amd.com/artificial-intelligence/LLM_Inference/README.html?utm_source=chatgpt.com "Best practices for competitive inference optimization on AMD ..."
[11]: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html?utm_source=chatgpt.com "AMD Instinct MI300X workload optimization"
[12]: https://github.com/Dao-AILab/flash-attention/issues/707?utm_source=chatgpt.com "add support for AMD / ROCm / HIP · Issue #707"
[13]: https://rocm.blogs.amd.com/artificial-intelligence/triton_server_vllm/README.html?utm_source=chatgpt.com "Triton Inference Server with vLLM on AMD GPUs — ROCm Blogs"

