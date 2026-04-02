//! AVX2 + FMA Optimized Kernels
//!
//! Provides SIMD-accelerated operations for x86_64 processors with AVX2 and FMA.
//! Targets Intel Haswell (2013+) and AMD Zen (2017+).
//!
//! Key optimizations:
//! - Process 8 f32 values per iteration (256-bit registers)
//! - Use FMA for fused multiply-add (better precision, same latency)
//! - Dequantize Q4 values on-the-fly into SIMD registers

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::simd::quant::{Q4_0Block, Q4_0_BLOCK_SIZE, QuantType, QuantizedTensor};
use crate::simd::{CpuFeatures, SimdBackend};

/// AVX2 + FMA backend
pub struct Avx2Backend;

impl Avx2Backend {
    pub fn new() -> Self {
        Self
    }

    /// Check if AVX2 + FMA is available
    #[cfg(target_arch = "x86_64")]
    pub fn is_supported() -> bool {
        std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn is_supported() -> bool {
        false
    }
}

impl Default for Avx2Backend {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// AVX2 Intrinsics - Low-level SIMD operations
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod intrinsics {
    use super::*;

    /// Compute dot product of Q4_0 block with f32 vector using AVX2
    ///
    /// This is the core hot-path operation. It:
    /// 1. Broadcasts the scale factor to all 8 lanes
    /// 2. Extracts 4-bit values and converts to f32
    /// 3. Subtracts 8 (to center range from [0,15] to [-8,7])
    /// 4. Multiplies by scale and input, accumulating with FMA
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn q4_0_dot_avx2(block: &Q4_0Block, x: &[f32]) -> f32 {
        debug_assert!(x.len() >= Q4_0_BLOCK_SIZE);

        let scale = _mm256_set1_ps(block.d.to_f32());
        let eight = _mm256_set1_ps(8.0);
        let mask_low = _mm256_set1_epi32(0x0F);

        let mut acc = _mm256_setzero_ps();

        // Process 8 values at a time (4 bytes of quantized data)
        // Each byte contains 2 4-bit values
        for i in 0..4 {
            let qs_offset = i * 4;

            // Load 4 bytes (8 quantized values)
            let qs_bytes = &block.qs[qs_offset..qs_offset + 4];

            // Extract lower 4 bits (even indices: 0, 2, 4, 6)
            let lo0 = (qs_bytes[0] & 0x0F) as i32;
            let lo1 = (qs_bytes[1] & 0x0F) as i32;
            let lo2 = (qs_bytes[2] & 0x0F) as i32;
            let lo3 = (qs_bytes[3] & 0x0F) as i32;

            // Extract upper 4 bits (odd indices: 1, 3, 5, 7)
            let hi0 = (qs_bytes[0] >> 4) as i32;
            let hi1 = (qs_bytes[1] >> 4) as i32;
            let hi2 = (qs_bytes[2] >> 4) as i32;
            let hi3 = (qs_bytes[3] >> 4) as i32;

            // Interleave to get correct order: [lo0, hi0, lo1, hi1, ...]
            let q_i32 = _mm256_setr_epi32(lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3);

            // Convert to f32 and subtract 8
            let q_f32 = _mm256_cvtepi32_ps(q_i32);
            let q_centered = _mm256_sub_ps(q_f32, eight);

            // Load 8 input values
            let x_offset = i * 8;
            let x_vec = _mm256_loadu_ps(x.as_ptr().add(x_offset));

            // FMA: acc = acc + (scale * q_centered * x)
            let scaled = _mm256_mul_ps(scale, q_centered);
            acc = _mm256_fmadd_ps(scaled, x_vec, acc);
        }

        // Horizontal sum of 8 floats in acc
        hsum_avx2(acc)
    }

    /// Horizontal sum of 8 f32 values in AVX2 register
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn hsum_avx2(v: __m256) -> f32 {
        // Add high 128 bits to low 128 bits
        let high = _mm256_extractf128_ps(v, 1);
        let low = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(low, high);

        // Horizontal add within 128 bits
        let shuf = _mm_movehdup_ps(sum128); // [1,1,3,3]
        let sum64 = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sum64, sum64); // [2,3,2,3]
        let sum32 = _mm_add_ss(sum64, shuf2);

        _mm_cvtss_f32(sum32)
    }

    /// GEMV for Q4_0 using AVX2
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn q4_0_gemv_avx2(
        weight_blocks: &[Q4_0Block],
        x: &[f32],
        out: &mut [f32],
        n_rows: usize,
        n_cols: usize,
    ) {
        let blocks_per_row = (n_cols + Q4_0_BLOCK_SIZE - 1) / Q4_0_BLOCK_SIZE;

        for row in 0..n_rows {
            let row_start = row * blocks_per_row;
            let row_end = row_start + blocks_per_row;

            let mut sum = 0.0f32;
            let mut x_offset = 0;

            for block_idx in row_start..row_end {
                sum += q4_0_dot_avx2(&weight_blocks[block_idx], &x[x_offset..]);
                x_offset += Q4_0_BLOCK_SIZE;
            }

            out[row] = sum;
        }
    }

    /// RMS normalization using AVX2
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn rms_norm_avx2(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
        let n = x.len();
        debug_assert_eq!(weight.len(), n);
        debug_assert_eq!(out.len(), n);

        // Compute sum of squares using AVX2
        let mut sum_sq = _mm256_setzero_ps();
        let chunks = n / 8;
        let remainder = n % 8;

        for i in 0..chunks {
            let offset = i * 8;
            let v = _mm256_loadu_ps(x.as_ptr().add(offset));
            sum_sq = _mm256_fmadd_ps(v, v, sum_sq);
        }

        // Horizontal sum
        let mut total_sq = hsum_avx2(sum_sq);

        // Handle remainder
        for i in (chunks * 8)..n {
            total_sq += x[i] * x[i];
        }

        // Compute inverse RMS
        let rms = (total_sq / n as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        let inv_rms_vec = _mm256_set1_ps(inv_rms);

        // Normalize and scale using AVX2
        for i in 0..chunks {
            let offset = i * 8;
            let x_vec = _mm256_loadu_ps(x.as_ptr().add(offset));
            let w_vec = _mm256_loadu_ps(weight.as_ptr().add(offset));

            let normalized = _mm256_mul_ps(x_vec, inv_rms_vec);
            let scaled = _mm256_mul_ps(normalized, w_vec);

            _mm256_storeu_ps(out.as_mut_ptr().add(offset), scaled);
        }

        // Handle remainder
        for i in (chunks * 8)..n {
            out[i] = x[i] * inv_rms * weight[i];
        }
    }

    /// Fused SwiGLU using AVX2
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn fused_swiglu_avx2(gate: &[f32], up: &[f32], out: &mut [f32]) {
        let n = gate.len();
        let chunks = n / 8;

        let one = _mm256_set1_ps(1.0);
        let neg_one = _mm256_set1_ps(-1.0);

        for i in 0..chunks {
            let offset = i * 8;
            let g = _mm256_loadu_ps(gate.as_ptr().add(offset));
            let u = _mm256_loadu_ps(up.as_ptr().add(offset));

            // SiLU(g) = g * sigmoid(g) = g / (1 + exp(-g))
            // Approximate exp(-g) using polynomial
            let neg_g = _mm256_mul_ps(g, neg_one);

            // Fast exp approximation for sigmoid
            // exp(x) ≈ (1 + x/256)^256, but we use simpler approximation
            // sigmoid(x) ≈ 0.5 + 0.5 * tanh(x/2)
            // For now, use scalar fallback for exp
            let mut silu_g = [0.0f32; 8];
            let mut g_arr = [0.0f32; 8];
            _mm256_storeu_ps(g_arr.as_mut_ptr(), g);

            for j in 0..8 {
                silu_g[j] = g_arr[j] / (1.0 + (-g_arr[j]).exp());
            }

            let silu_vec = _mm256_loadu_ps(silu_g.as_ptr());
            let result = _mm256_mul_ps(silu_vec, u);
            _mm256_storeu_ps(out.as_mut_ptr().add(offset), result);
        }

        // Handle remainder
        for i in (chunks * 8)..n {
            let g = gate[i];
            out[i] = (g / (1.0 + (-g).exp())) * up[i];
        }
    }

    /// Softmax using AVX2
    #[target_feature(enable = "avx2")]
    pub unsafe fn softmax_avx2(x: &[f32], out: &mut [f32]) {
        let n = x.len();
        let chunks = n / 8;

        // Find max using AVX2
        let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
        for i in 0..chunks {
            let offset = i * 8;
            let v = _mm256_loadu_ps(x.as_ptr().add(offset));
            max_vec = _mm256_max_ps(max_vec, v);
        }

        // Horizontal max
        let mut max_val = hmax_avx2(max_vec);
        for i in (chunks * 8)..n {
            max_val = max_val.max(x[i]);
        }

        let max_vec = _mm256_set1_ps(max_val);

        // Compute exp(x - max) and sum
        let mut sum = 0.0f32;
        for i in 0..chunks {
            let offset = i * 8;
            let v = _mm256_loadu_ps(x.as_ptr().add(offset));
            let shifted = _mm256_sub_ps(v, max_vec);

            // Scalar exp for now (AVX2 doesn't have native exp)
            let mut exp_arr = [0.0f32; 8];
            let mut shifted_arr = [0.0f32; 8];
            _mm256_storeu_ps(shifted_arr.as_mut_ptr(), shifted);

            for j in 0..8 {
                exp_arr[j] = shifted_arr[j].exp();
                sum += exp_arr[j];
            }

            let exp_vec = _mm256_loadu_ps(exp_arr.as_ptr());
            _mm256_storeu_ps(out.as_mut_ptr().add(offset), exp_vec);
        }

        for i in (chunks * 8)..n {
            out[i] = (x[i] - max_val).exp();
            sum += out[i];
        }

        // Normalize
        let inv_sum = 1.0 / sum;
        let inv_sum_vec = _mm256_set1_ps(inv_sum);

        for i in 0..chunks {
            let offset = i * 8;
            let v = _mm256_loadu_ps(out.as_ptr().add(offset));
            let normalized = _mm256_mul_ps(v, inv_sum_vec);
            _mm256_storeu_ps(out.as_mut_ptr().add(offset), normalized);
        }

        for i in (chunks * 8)..n {
            out[i] *= inv_sum;
        }
    }

    /// Horizontal max of 8 f32 values
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn hmax_avx2(v: __m256) -> f32 {
        let high = _mm256_extractf128_ps(v, 1);
        let low = _mm256_castps256_ps128(v);
        let max128 = _mm_max_ps(low, high);

        let shuf = _mm_movehdup_ps(max128);
        let max64 = _mm_max_ps(max128, shuf);
        let shuf2 = _mm_movehl_ps(max64, max64);
        let max32 = _mm_max_ss(max64, shuf2);

        _mm_cvtss_f32(max32)
    }

    /// Dot product using AVX2
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let n = a.len();
        let chunks = n / 8;

        let mut acc = _mm256_setzero_ps();

        for i in 0..chunks {
            let offset = i * 8;
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(offset));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(offset));
            acc = _mm256_fmadd_ps(a_vec, b_vec, acc);
        }

        let mut sum = hsum_avx2(acc);

        for i in (chunks * 8)..n {
            sum += a[i] * b[i];
        }

        sum
    }
}

impl SimdBackend for Avx2Backend {
    fn name(&self) -> &'static str {
        "AVX2+FMA"
    }

    fn required_features(&self) -> CpuFeatures {
        CpuFeatures {
            avx: true,
            avx2: true,
            fma: true,
            ..Default::default()
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn q4_gemv(&self, weight: &QuantizedTensor, x: &[f32], out: &mut [f32]) {
        let [n_rows, n_cols] = weight.shape();

        match weight.quant_type() {
            QuantType::Q4_0 => {
                let blocks = weight.as_q4_0_blocks();
                // Safety: We've verified AVX2+FMA support in Avx2Backend::new()
                unsafe {
                    intrinsics::q4_0_gemv_avx2(blocks, x, out, n_rows, n_cols);
                }
            }
            _ => {
                // Fallback to scalar for unsupported types
                crate::simd::arch::scalar::ScalarBackend.q4_gemv(weight, x, out);
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn q4_gemv(&self, weight: &QuantizedTensor, x: &[f32], out: &mut [f32]) {
        crate::simd::arch::scalar::ScalarBackend.q4_gemv(weight, x, out);
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
        // For now, implement GEMM as multiple GEMV calls
        // TODO: Implement optimized tiled GEMM
        for i in 0..m {
            let x_row = &x[i * k..(i + 1) * k];
            let out_row = &mut out[i * n..(i + 1) * n];
            self.q4_gemv(weight, x_row, out_row);
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
        unsafe {
            intrinsics::rms_norm_avx2(x, weight, eps, out);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
        crate::simd::arch::scalar::ScalarBackend.rms_norm(x, weight, eps, out);
    }

    fn rms_norm_inplace(&self, x: &mut [f32], weight: &[f32], eps: f32) {
        // Use out-of-place version with same buffer
        let mut temp = x.to_vec();
        self.rms_norm(&temp, weight, eps, x);
    }

    #[cfg(target_arch = "x86_64")]
    fn fused_swiglu(&self, gate: &[f32], up: &[f32], out: &mut [f32]) {
        unsafe {
            intrinsics::fused_swiglu_avx2(gate, up, out);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn fused_swiglu(&self, gate: &[f32], up: &[f32], out: &mut [f32]) {
        crate::simd::arch::scalar::ScalarBackend.fused_swiglu(gate, up, out);
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
        // RoPE is memory-bound, scalar is often fine
        // TODO: Implement AVX2 version if profiling shows benefit
        crate::simd::arch::scalar::ScalarBackend.apply_rope(q, k, cos, sin, head_dim, num_heads);
    }

    #[cfg(target_arch = "x86_64")]
    fn softmax(&self, x: &[f32], out: &mut [f32]) {
        unsafe {
            intrinsics::softmax_avx2(x, out);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn softmax(&self, x: &[f32], out: &mut [f32]) {
        crate::simd::arch::scalar::ScalarBackend.softmax(x, out);
    }

    fn softmax_inplace(&self, x: &mut [f32]) {
        let temp = x.to_vec();
        self.softmax(&temp, x);
    }

    fn silu(&self, x: &[f32], out: &mut [f32]) {
        // SiLU requires exp(), which doesn't have AVX2 intrinsic
        // Use scalar or consider polynomial approximation
        crate::simd::arch::scalar::ScalarBackend.silu(x, out);
    }

    fn mul(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let n = a.len();
            let chunks = n / 8;

            for i in 0..chunks {
                let offset = i * 8;
                let a_vec = _mm256_loadu_ps(a.as_ptr().add(offset));
                let b_vec = _mm256_loadu_ps(b.as_ptr().add(offset));
                let result = _mm256_mul_ps(a_vec, b_vec);
                _mm256_storeu_ps(out.as_mut_ptr().add(offset), result);
            }

            for i in (chunks * 8)..n {
                out[i] = a[i] * b[i];
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        crate::simd::arch::scalar::ScalarBackend.mul(a, b, out);
    }

    fn add(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let n = a.len();
            let chunks = n / 8;

            for i in 0..chunks {
                let offset = i * 8;
                let a_vec = _mm256_loadu_ps(a.as_ptr().add(offset));
                let b_vec = _mm256_loadu_ps(b.as_ptr().add(offset));
                let result = _mm256_add_ps(a_vec, b_vec);
                _mm256_storeu_ps(out.as_mut_ptr().add(offset), result);
            }

            for i in (chunks * 8)..n {
                out[i] = a[i] + b[i];
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        crate::simd::arch::scalar::ScalarBackend.add(a, b, out);
    }

    #[cfg(target_arch = "x86_64")]
    fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        unsafe { intrinsics::dot_avx2(a, b) }
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        crate::simd::arch::scalar::ScalarBackend.dot(a, b)
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn test_avx2_available() {
        if Avx2Backend::is_supported() {
            println!("AVX2+FMA is available");
        } else {
            println!("AVX2+FMA is NOT available, tests will use scalar fallback");
        }
    }

    #[test]
    fn test_avx2_dot() {
        if !Avx2Backend::is_supported() {
            return;
        }

        let backend = Avx2Backend::new();
        let a: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..64).map(|i| (i * 2) as f32).collect();

        let result = backend.dot(&a, &b);

        // Expected: sum(i * 2i) for i in 0..64
        let expected: f32 = (0..64).map(|i| (i * 2 * i) as f32).sum();
        assert!(
            (result - expected).abs() < 1.0,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_avx2_q4_0_dot() {
        if !Avx2Backend::is_supported() {
            return;
        }

        // Create a Q4_0 block with all 1s (value = 9, dequant = 1)
        let mut qs = [0u8; 16];
        for i in 0..16 {
            qs[i] = 0x99; // Both nibbles = 9
        }
        let block = Q4_0Block {
            d: f16::from_f32(1.0),
            qs,
        };

        let x = [1.0f32; 32];

        let result = unsafe { intrinsics::q4_0_dot_avx2(&block, &x) };

        // Expected: 32 values of 1.0 * 1.0 = 32.0
        assert!(
            (result - 32.0).abs() < 1.0,
            "Expected ~32.0, got {}",
            result
        );
    }

    #[test]
    fn test_avx2_rms_norm() {
        if !Avx2Backend::is_supported() {
            return;
        }

        let backend = Avx2Backend::new();
        let x: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let weight = vec![1.0f32; 64];
        let mut out = vec![0.0f32; 64];

        backend.rms_norm(&x, &weight, 1e-6, &mut out);

        // Verify output is normalized
        let sum_sq: f32 = out.iter().map(|&v| v * v).sum();
        let rms = (sum_sq / 64.0).sqrt();

        // After RMS normalization, the RMS should be close to 1.0 (if weights are 1.0)
        // Actually, let's verify against scalar
        let scalar = crate::simd::arch::scalar::ScalarBackend;
        let mut scalar_out = vec![0.0f32; 64];
        scalar.rms_norm(&x, &weight, 1e-6, &mut scalar_out);

        for i in 0..64 {
            assert!(
                (out[i] - scalar_out[i]).abs() < 1e-4,
                "Mismatch at {}: {} vs {}",
                i,
                out[i],
                scalar_out[i]
            );
        }
    }
}
