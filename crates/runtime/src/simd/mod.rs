//! Native SIMD Kernels for Quantized Inference
//!
//! This module provides optimized SIMD implementations for quantized tensor operations,
//! achieving llama.cpp-competitive performance by:
//! - Keeping weights in Q4 format (no full dequantization)
//! - Dequantizing on-the-fly during computation into SIMD registers
//! - Fusing operations to minimize memory bandwidth
//!
//! Architecture support:
//! - AVX2 (Intel Haswell+, AMD Zen+)
//! - AVX-512 (Intel Skylake-X+, AMD Zen4+)
//! - ARM NEON (Apple M1+, ARM servers)
//! - Scalar fallback for all platforms

pub mod arch;
pub mod matmul;
pub mod ops;
pub mod quant;

use std::sync::OnceLock;

use crate::simd::quant::QuantizedTensor;

/// Global SIMD backend instance, initialized once at runtime
static SIMD_BACKEND: OnceLock<Box<dyn SimdBackend>> = OnceLock::new();

/// CPU feature flags detected at runtime
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuFeatures {
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vl: bool,
    pub fma: bool,
    pub neon: bool,
    pub sve: bool,
}

impl CpuFeatures {
    /// Detect CPU features at runtime
    #[cfg(target_arch = "x86_64")]
    pub fn detect() -> Self {
        Self {
            avx: std::arch::is_x86_feature_detected!("avx"),
            avx2: std::arch::is_x86_feature_detected!("avx2"),
            avx512f: std::arch::is_x86_feature_detected!("avx512f"),
            avx512bw: std::arch::is_x86_feature_detected!("avx512bw"),
            avx512vl: std::arch::is_x86_feature_detected!("avx512vl"),
            fma: std::arch::is_x86_feature_detected!("fma"),
            neon: false,
            sve: false,
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub fn detect() -> Self {
        Self {
            avx: false,
            avx2: false,
            avx512f: false,
            avx512bw: false,
            avx512vl: false,
            fma: false,
            neon: std::arch::is_aarch64_feature_detected!("neon"),
            sve: std::arch::is_aarch64_feature_detected!("sve"),
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub fn detect() -> Self {
        Self::default()
    }

    /// Check if AVX2 with FMA is available (best x86 baseline)
    pub fn has_avx2_fma(&self) -> bool {
        self.avx2 && self.fma
    }

    /// Check if full AVX-512 is available
    pub fn has_avx512(&self) -> bool {
        self.avx512f && self.avx512bw && self.avx512vl
    }

    /// Get the best available SIMD level as a string
    pub fn best_level(&self) -> &'static str {
        if self.has_avx512() {
            "AVX-512"
        } else if self.has_avx2_fma() {
            "AVX2+FMA"
        } else if self.neon {
            "NEON"
        } else {
            "Scalar"
        }
    }
}

/// Trait for SIMD backend implementations
///
/// Each architecture (AVX2, AVX-512, NEON, Scalar) implements this trait
/// to provide optimized kernels for quantized inference.
pub trait SimdBackend: Send + Sync {
    /// Name of this backend (e.g., "AVX2", "AVX-512", "NEON", "Scalar")
    fn name(&self) -> &'static str;

    /// CPU features this backend requires
    fn required_features(&self) -> CpuFeatures;

    // ============================================================
    // Quantized Matrix Operations
    // ============================================================

    /// Quantized matrix-vector multiply (decode phase - single token)
    ///
    /// Computes: out = weight @ x
    /// - weight: [out_features, in_features] in Q4 format
    /// - x: [in_features] in f32
    /// - out: [out_features] in f32
    ///
    /// This is the critical hot path for autoregressive decoding.
    fn q4_gemv(
        &self,
        weight: &QuantizedTensor,
        x: &[f32],
        out: &mut [f32],
    );

    /// Quantized matrix-matrix multiply (prefill phase - multiple tokens)
    ///
    /// Computes: out = x @ weight^T
    /// - weight: [out_features, in_features] in Q4 format
    /// - x: [m, in_features] in f32 (m = sequence length)
    /// - out: [m, out_features] in f32
    ///
    /// Used during prompt processing (prefill).
    fn q4_gemm(
        &self,
        weight: &QuantizedTensor,
        x: &[f32],
        out: &mut [f32],
        m: usize,  // rows in x (sequence length)
        k: usize,  // cols in x / cols in weight (hidden dim)
        n: usize,  // rows in weight (output dim)
    );

    // ============================================================
    // Fused Operations (minimize memory bandwidth)
    // ============================================================

    /// RMS Normalization: out = x * weight / rms(x)
    ///
    /// Single fused operation instead of separate variance + normalize.
    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]);

    /// In-place RMS Normalization
    fn rms_norm_inplace(&self, x: &mut [f32], weight: &[f32], eps: f32);

    /// Fused SwiGLU: out = silu(gate) * up
    ///
    /// Combines SiLU activation with element-wise multiply.
    /// Used in LLaMA/Mistral/Qwen MLP layers.
    fn fused_swiglu(&self, gate: &[f32], up: &[f32], out: &mut [f32]);

    /// Apply Rotary Position Embedding to Q and K
    ///
    /// Modifies q and k in-place by applying rotation based on position.
    fn apply_rope(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        cos: &[f32],
        sin: &[f32],
        head_dim: usize,
        num_heads: usize,
    );

    /// Softmax: out = softmax(x)
    ///
    /// Numerically stable implementation with SIMD.
    fn softmax(&self, x: &[f32], out: &mut [f32]);

    /// In-place softmax
    fn softmax_inplace(&self, x: &mut [f32]);

    // ============================================================
    // Element-wise Operations
    // ============================================================

    /// SiLU activation: out = x * sigmoid(x)
    fn silu(&self, x: &[f32], out: &mut [f32]);

    /// Element-wise multiply: out = a * b
    fn mul(&self, a: &[f32], b: &[f32], out: &mut [f32]);

    /// Element-wise add: out = a + b
    fn add(&self, a: &[f32], b: &[f32], out: &mut [f32]);

    /// Dot product: sum(a * b)
    fn dot(&self, a: &[f32], b: &[f32]) -> f32;
}

/// Initialize the global SIMD backend based on detected CPU features
pub fn init_simd_backend() -> &'static dyn SimdBackend {
    SIMD_BACKEND.get_or_init(|| {
        let features = CpuFeatures::detect();

        #[cfg(target_arch = "x86_64")]
        {
            if features.has_avx512() {
                tracing::info!("SIMD backend: AVX-512");
                return Box::new(arch::avx512::Avx512Backend::new());
            }
            if features.has_avx2_fma() {
                tracing::info!("SIMD backend: AVX2+FMA");
                return Box::new(arch::avx2::Avx2Backend::new());
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if features.neon {
                tracing::info!("SIMD backend: NEON");
                return Box::new(arch::neon::NeonBackend::new());
            }
        }

        tracing::info!("SIMD backend: Scalar (fallback)");
        Box::new(arch::scalar::ScalarBackend::new())
    }).as_ref()
}

/// Get the current SIMD backend (initializes if needed)
pub fn get_simd_backend() -> &'static dyn SimdBackend {
    init_simd_backend()
}

/// Check if SIMD is available and initialized
pub fn is_simd_available() -> bool {
    SIMD_BACKEND.get().is_some()
}

/// Get detected CPU features
pub fn cpu_features() -> CpuFeatures {
    CpuFeatures::detect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_feature_detection() {
        let features = CpuFeatures::detect();
        println!("CPU Features: {:?}", features);
        println!("Best SIMD level: {}", features.best_level());

        // At minimum, we should be able to detect something
        #[cfg(target_arch = "x86_64")]
        {
            // Most modern x86_64 CPUs have at least AVX
            // But don't assert, as CI might run on older hardware
            println!("AVX: {}, AVX2: {}, FMA: {}", features.avx, features.avx2, features.fma);
        }

        #[cfg(target_arch = "aarch64")]
        {
            // All AArch64 has NEON
            assert!(features.neon, "NEON should be available on AArch64");
        }
    }

    #[test]
    fn test_simd_backend_init() {
        let backend = get_simd_backend();
        println!("Initialized backend: {}", backend.name());

        // Backend name should match detected features
        let features = CpuFeatures::detect();
        let name = backend.name();

        // Verify consistency
        if features.has_avx512() {
            assert!(name.contains("512") || name == "Scalar");
        } else if features.has_avx2_fma() {
            assert!(name.contains("AVX2") || name == "Scalar");
        }
    }
}
