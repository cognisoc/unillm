//! AVX-512 Optimized Kernels
//!
//! Provides SIMD-accelerated operations for x86_64 processors with AVX-512.
//! Targets Intel Skylake-X (2017+) and AMD Zen 4 (2022+).
//!
//! Key optimizations:
//! - Process 16 f32 values per iteration (512-bit registers)
//! - Use AVX-512BW for byte operations
//! - Use AVX-512VL for mixed-width operations

use crate::simd::quant::QuantizedTensor;
use crate::simd::{CpuFeatures, SimdBackend};

/// AVX-512 backend
pub struct Avx512Backend;

impl Avx512Backend {
    pub fn new() -> Self {
        Self
    }

    /// Check if full AVX-512 is available
    #[cfg(target_arch = "x86_64")]
    pub fn is_supported() -> bool {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vl")
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn is_supported() -> bool {
        false
    }
}

impl Default for Avx512Backend {
    fn default() -> Self {
        Self::new()
    }
}

// For now, AVX-512 backend delegates to AVX2
// TODO: Implement native AVX-512 kernels for 2x throughput

impl SimdBackend for Avx512Backend {
    fn name(&self) -> &'static str {
        "AVX-512"
    }

    fn required_features(&self) -> CpuFeatures {
        CpuFeatures {
            avx: true,
            avx2: true,
            avx512f: true,
            avx512bw: true,
            avx512vl: true,
            fma: true,
            ..Default::default()
        }
    }

    fn q4_gemv(&self, weight: &QuantizedTensor, x: &[f32], out: &mut [f32]) {
        // Delegate to AVX2 for now
        super::avx2::Avx2Backend.q4_gemv(weight, x, out)
    }

    fn q4_gemm(
        &self,
        weight: &QuantizedTensor,
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) {
        super::avx2::Avx2Backend.q4_gemm(weight, x, out, m, k, n)
    }

    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
        super::avx2::Avx2Backend.rms_norm(x, weight, eps, out)
    }

    fn rms_norm_inplace(&self, x: &mut [f32], weight: &[f32], eps: f32) {
        super::avx2::Avx2Backend.rms_norm_inplace(x, weight, eps)
    }

    fn fused_swiglu(&self, gate: &[f32], up: &[f32], out: &mut [f32]) {
        super::avx2::Avx2Backend.fused_swiglu(gate, up, out)
    }

    fn apply_rope(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        cos: &[f32],
        sin: &[f32],
        head_dim: usize,
        num_heads: usize,
    ) {
        super::avx2::Avx2Backend.apply_rope(q, k, cos, sin, head_dim, num_heads)
    }

    fn softmax(&self, x: &[f32], out: &mut [f32]) {
        super::avx2::Avx2Backend.softmax(x, out)
    }

    fn softmax_inplace(&self, x: &mut [f32]) {
        super::avx2::Avx2Backend.softmax_inplace(x)
    }

    fn silu(&self, x: &[f32], out: &mut [f32]) {
        super::avx2::Avx2Backend.silu(x, out)
    }

    fn mul(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        super::avx2::Avx2Backend.mul(a, b, out)
    }

    fn add(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        super::avx2::Avx2Backend.add(a, b, out)
    }

    fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        super::avx2::Avx2Backend.dot(a, b)
    }
}
