//! ARM NEON Optimized Kernels
//!
//! Provides SIMD-accelerated operations for AArch64 processors with NEON.
//! Targets Apple M1+ and ARM server processors.
//!
//! Key optimizations:
//! - Process 4 f32 values per iteration (128-bit registers)
//! - Use NEON intrinsics for vectorized operations
//! - Optimized for ARM's memory hierarchy

use crate::simd::quant::{QuantizedTensor, QuantType, Q4_0Block, Q4_KBlock, Q4_0_BLOCK_SIZE, Q4_K_BLOCK_SIZE};
use crate::simd::{CpuFeatures, SimdBackend};

/// ARM NEON backend
pub struct NeonBackend;

impl NeonBackend {
    pub fn new() -> Self {
        Self
    }

    /// Check if NEON is available
    #[cfg(target_arch = "aarch64")]
    pub fn is_supported() -> bool {
        // NEON is mandatory on AArch64
        true
    }

    #[cfg(not(target_arch = "aarch64"))]
    pub fn is_supported() -> bool {
        false
    }
}

impl Default for NeonBackend {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// NEON Implementation
// ============================================================================

#[cfg(target_arch = "aarch64")]
mod neon_impl {
    use super::*;
    use std::arch::aarch64::*;

    /// Q4_0 dot product with NEON intrinsics
    ///
    /// Processes 32 values per block, 4 at a time using NEON vectors.
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn q4_0_dot_neon(block: &Q4_0Block, x: &[f32]) -> f32 {
        let scale = vdupq_n_f32(block.d.to_f32());
        let zero_point = vdupq_n_f32(8.0);
        let mut acc = vdupq_n_f32(0.0);

        // Process 8 bytes at a time (16 values, 4 vectors of 4 floats each)
        for chunk in 0..2 {
            let base = chunk * 8;

            // Process 4 bytes (8 values) per iteration
            for i in 0..4 {
                let byte = block.qs[base + i * 2];
                let byte2 = block.qs[base + i * 2 + 1];

                // Extract nibbles
                let lo0 = (byte & 0x0F) as f32;
                let hi0 = (byte >> 4) as f32;
                let lo1 = (byte2 & 0x0F) as f32;
                let hi1 = (byte2 >> 4) as f32;

                // Create vector of dequantized values
                let vals = [lo0, hi0, lo1, hi1];
                let q_vec = vld1q_f32(vals.as_ptr());
                let dequant = vmulq_f32(scale, vsubq_f32(q_vec, zero_point));

                // Load input values
                let x_offset = chunk * 16 + i * 4;
                let x_vec = vld1q_f32(x.as_ptr().add(x_offset));

                // Fused multiply-add
                acc = vfmaq_f32(acc, dequant, x_vec);
            }
        }

        // Horizontal sum
        vaddvq_f32(acc)
    }

    /// Q4_0 vector dot product (multiple blocks)
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn q4_0_vec_dot_neon(blocks: &[Q4_0Block], x: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        let mut offset = 0;

        for block in blocks {
            sum += q4_0_dot_neon(block, &x[offset..]);
            offset += Q4_0_BLOCK_SIZE;
        }

        sum
    }

    /// Q4_K dot product with NEON intrinsics
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn q4_k_dot_neon(block: &Q4_KBlock, x: &[f32]) -> f32 {
        let d = block.d.to_f32();
        let dmin = block.dmin.to_f32();
        let mut acc = vdupq_n_f32(0.0);

        // Process 8 sub-blocks of 32 values each
        for sb in 0..8 {
            let scale = d * block.get_scale(sb) as f32;
            let min = dmin * block.get_min(sb) as f32;

            let scale_vec = vdupq_n_f32(scale);
            let min_vec = vdupq_n_f32(min);

            let qs_offset = sb * 16;
            let x_offset = sb * 32;

            // Process 16 bytes (32 values) in 8 iterations of 4 values
            for i in 0..8 {
                let byte0 = block.qs[qs_offset + i * 2];
                let byte1 = block.qs[qs_offset + i * 2 + 1];

                // Extract nibbles
                let lo0 = (byte0 & 0x0F) as f32;
                let hi0 = (byte0 >> 4) as f32;
                let lo1 = (byte1 & 0x0F) as f32;
                let hi1 = (byte1 >> 4) as f32;

                let vals = [lo0, hi0, lo1, hi1];
                let q_vec = vld1q_f32(vals.as_ptr());

                // Dequantize: scale * val - min
                let dequant = vsubq_f32(vmulq_f32(scale_vec, q_vec), min_vec);

                // Load input
                let x_vec = vld1q_f32(x.as_ptr().add(x_offset + i * 4));

                // Accumulate
                acc = vfmaq_f32(acc, dequant, x_vec);
            }
        }

        vaddvq_f32(acc)
    }

    /// Q4_K vector dot product (multiple blocks)
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn q4_k_vec_dot_neon(blocks: &[Q4_KBlock], x: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        let mut offset = 0;

        for block in blocks {
            sum += q4_k_dot_neon(block, &x[offset..]);
            offset += Q4_K_BLOCK_SIZE;
        }

        sum
    }

    /// RMS normalization with NEON
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn rms_norm_neon(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
        let n = x.len();
        let chunks = n / 4;
        let remainder = n % 4;

        // Compute sum of squares
        let mut sum_sq = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(x.as_ptr().add(i * 4));
            sum_sq = vfmaq_f32(sum_sq, v, v);
        }
        let mut total_sq = vaddvq_f32(sum_sq);

        // Handle remainder
        for i in (chunks * 4)..n {
            total_sq += x[i] * x[i];
        }

        // RMS = sqrt(mean(x^2) + eps)
        let rms = (total_sq / n as f32 + eps).sqrt();
        let rms_inv = 1.0 / rms;
        let rms_inv_vec = vdupq_n_f32(rms_inv);

        // Normalize and scale
        for i in 0..chunks {
            let x_vec = vld1q_f32(x.as_ptr().add(i * 4));
            let w_vec = vld1q_f32(weight.as_ptr().add(i * 4));
            let normalized = vmulq_f32(x_vec, rms_inv_vec);
            let result = vmulq_f32(normalized, w_vec);
            vst1q_f32(out.as_mut_ptr().add(i * 4), result);
        }

        // Handle remainder
        for i in (chunks * 4)..n {
            out[i] = x[i] * rms_inv * weight[i];
        }
    }

    /// SiLU activation with NEON
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn silu_neon(x: &[f32], out: &mut [f32]) {
        let n = x.len();
        let chunks = n / 4;

        let one = vdupq_n_f32(1.0);

        for i in 0..chunks {
            let v = vld1q_f32(x.as_ptr().add(i * 4));

            // Compute sigmoid: 1 / (1 + exp(-x))
            // Using approximation for performance
            let neg_v = vnegq_f32(v);

            // exp(-x) approximation using polynomial
            // For better accuracy, use a lookup table or more terms
            let exp_neg = exp_approx_neon(neg_v);
            let sigmoid = vdivq_f32(one, vaddq_f32(one, exp_neg));

            // silu = x * sigmoid(x)
            let result = vmulq_f32(v, sigmoid);
            vst1q_f32(out.as_mut_ptr().add(i * 4), result);
        }

        // Handle remainder with scalar
        for i in (chunks * 4)..n {
            let v = x[i];
            out[i] = v * (1.0 / (1.0 + (-v).exp()));
        }
    }

    /// Fast exponential approximation for NEON
    /// Uses polynomial approximation valid for reasonable ranges
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn exp_approx_neon(x: float32x4_t) -> float32x4_t {
        // Clamp to prevent overflow
        let max_val = vdupq_n_f32(88.0);
        let min_val = vdupq_n_f32(-88.0);
        let x = vminq_f32(vmaxq_f32(x, min_val), max_val);

        // exp(x) ≈ (1 + x/256)^256 for small x
        // Using 2^n * exp(x - n*ln2) decomposition
        let log2e = vdupq_n_f32(1.4426950408889634);
        let ln2 = vdupq_n_f32(0.6931471805599453);

        // x * log2(e) to get the power of 2
        let z = vmulq_f32(x, log2e);

        // Integer part (floor)
        let n = vcvtq_s32_f32(z);
        let nf = vcvtq_f32_s32(n);

        // Fractional part: r = x - n * ln(2)
        let r = vsubq_f32(x, vmulq_f32(nf, ln2));

        // Polynomial approximation for exp(r) where r in [-0.5*ln2, 0.5*ln2]
        // exp(r) ≈ 1 + r + r^2/2 + r^3/6 + r^4/24
        let c1 = vdupq_n_f32(1.0);
        let c2 = vdupq_n_f32(0.5);
        let c3 = vdupq_n_f32(0.16666666666666666);
        let c4 = vdupq_n_f32(0.041666666666666664);

        let r2 = vmulq_f32(r, r);
        let r3 = vmulq_f32(r2, r);
        let r4 = vmulq_f32(r2, r2);

        let mut exp_r = c1;
        exp_r = vaddq_f32(exp_r, r);
        exp_r = vaddq_f32(exp_r, vmulq_f32(c2, r2));
        exp_r = vaddq_f32(exp_r, vmulq_f32(c3, r3));
        exp_r = vaddq_f32(exp_r, vmulq_f32(c4, r4));

        // Scale by 2^n (using integer bit manipulation)
        // 2^n = reinterpret((n + 127) << 23) for f32
        let bias = vdupq_n_s32(127);
        let exp_bits = vshlq_n_s32(vaddq_s32(n, bias), 23);
        let scale = vreinterpretq_f32_s32(exp_bits);

        vmulq_f32(exp_r, scale)
    }

    /// Fused SwiGLU with NEON: out = silu(gate) * up
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn fused_swiglu_neon(gate: &[f32], up: &[f32], out: &mut [f32]) {
        let n = gate.len();
        let chunks = n / 4;
        let one = vdupq_n_f32(1.0);

        for i in 0..chunks {
            let g = vld1q_f32(gate.as_ptr().add(i * 4));
            let u = vld1q_f32(up.as_ptr().add(i * 4));

            // silu(gate) = gate * sigmoid(gate)
            let neg_g = vnegq_f32(g);
            let exp_neg = exp_approx_neon(neg_g);
            let sigmoid = vdivq_f32(one, vaddq_f32(one, exp_neg));
            let silu_g = vmulq_f32(g, sigmoid);

            // silu(gate) * up
            let result = vmulq_f32(silu_g, u);
            vst1q_f32(out.as_mut_ptr().add(i * 4), result);
        }

        // Handle remainder
        for i in (chunks * 4)..n {
            let g = gate[i];
            let silu_g = g * (1.0 / (1.0 + (-g).exp()));
            out[i] = silu_g * up[i];
        }
    }

    /// Softmax with NEON
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn softmax_neon(x: &[f32], out: &mut [f32]) {
        let n = x.len();
        if n == 0 { return; }

        // Find max
        let chunks = n / 4;
        let mut max_vec = vdupq_n_f32(f32::NEG_INFINITY);
        for i in 0..chunks {
            let v = vld1q_f32(x.as_ptr().add(i * 4));
            max_vec = vmaxq_f32(max_vec, v);
        }
        let mut max_val = vmaxvq_f32(max_vec);
        for i in (chunks * 4)..n {
            max_val = max_val.max(x[i]);
        }

        // Compute exp(x - max) and sum
        let max_broadcast = vdupq_n_f32(max_val);
        let mut sum_vec = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let v = vld1q_f32(x.as_ptr().add(i * 4));
            let shifted = vsubq_f32(v, max_broadcast);
            let exp_val = exp_approx_neon(shifted);
            vst1q_f32(out.as_mut_ptr().add(i * 4), exp_val);
            sum_vec = vaddq_f32(sum_vec, exp_val);
        }

        let mut sum = vaddvq_f32(sum_vec);
        for i in (chunks * 4)..n {
            let exp_val = (x[i] - max_val).exp();
            out[i] = exp_val;
            sum += exp_val;
        }

        // Normalize
        let inv_sum = vdupq_n_f32(1.0 / sum);
        for i in 0..chunks {
            let v = vld1q_f32(out.as_ptr().add(i * 4));
            let normalized = vmulq_f32(v, inv_sum);
            vst1q_f32(out.as_mut_ptr().add(i * 4), normalized);
        }
        for i in (chunks * 4)..n {
            out[i] /= sum;
        }
    }

    /// Element-wise multiply with NEON
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn mul_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len();
        let chunks = n / 4;

        for i in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            let result = vmulq_f32(va, vb);
            vst1q_f32(out.as_mut_ptr().add(i * 4), result);
        }

        for i in (chunks * 4)..n {
            out[i] = a[i] * b[i];
        }
    }

    /// Element-wise add with NEON
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn add_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len();
        let chunks = n / 4;

        for i in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            let result = vaddq_f32(va, vb);
            vst1q_f32(out.as_mut_ptr().add(i * 4), result);
        }

        for i in (chunks * 4)..n {
            out[i] = a[i] + b[i];
        }
    }

    /// Dot product with NEON
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / 4;
        let mut acc = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            acc = vfmaq_f32(acc, va, vb);
        }

        let mut sum = vaddvq_f32(acc);
        for i in (chunks * 4)..n {
            sum += a[i] * b[i];
        }

        sum
    }

    /// Apply RoPE with NEON
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn apply_rope_neon(
        q: &mut [f32],
        k: &mut [f32],
        cos: &[f32],
        sin: &[f32],
        head_dim: usize,
        num_heads: usize,
    ) {
        let half_dim = head_dim / 2;

        for head in 0..num_heads {
            let offset = head * head_dim;
            let chunks = half_dim / 4;

            for i in 0..chunks {
                let base = i * 4;

                // Load Q values
                let q0 = vld1q_f32(q.as_ptr().add(offset + base));
                let q1 = vld1q_f32(q.as_ptr().add(offset + base + half_dim));

                // Load K values
                let k0 = vld1q_f32(k.as_ptr().add(offset + base));
                let k1 = vld1q_f32(k.as_ptr().add(offset + base + half_dim));

                // Load cos/sin
                let c = vld1q_f32(cos.as_ptr().add(base));
                let s = vld1q_f32(sin.as_ptr().add(base));

                // Apply rotation to Q
                // q0' = q0 * cos - q1 * sin
                // q1' = q0 * sin + q1 * cos
                let q0_new = vsubq_f32(vmulq_f32(q0, c), vmulq_f32(q1, s));
                let q1_new = vaddq_f32(vmulq_f32(q0, s), vmulq_f32(q1, c));

                vst1q_f32(q.as_mut_ptr().add(offset + base), q0_new);
                vst1q_f32(q.as_mut_ptr().add(offset + base + half_dim), q1_new);

                // Apply rotation to K
                let k0_new = vsubq_f32(vmulq_f32(k0, c), vmulq_f32(k1, s));
                let k1_new = vaddq_f32(vmulq_f32(k0, s), vmulq_f32(k1, c));

                vst1q_f32(k.as_mut_ptr().add(offset + base), k0_new);
                vst1q_f32(k.as_mut_ptr().add(offset + base + half_dim), k1_new);
            }

            // Handle remainder
            for i in (chunks * 4)..half_dim {
                let q0 = q[offset + i];
                let q1 = q[offset + i + half_dim];
                let k0 = k[offset + i];
                let k1 = k[offset + i + half_dim];
                let c = cos[i];
                let s = sin[i];

                q[offset + i] = q0 * c - q1 * s;
                q[offset + i + half_dim] = q0 * s + q1 * c;
                k[offset + i] = k0 * c - k1 * s;
                k[offset + i + half_dim] = k0 * s + k1 * c;
            }
        }
    }
}

// ============================================================================
// SimdBackend Implementation
// ============================================================================

impl SimdBackend for NeonBackend {
    fn name(&self) -> &'static str {
        "NEON"
    }

    fn required_features(&self) -> CpuFeatures {
        CpuFeatures {
            neon: true,
            ..Default::default()
        }
    }

    fn q4_gemv(&self, weight: &QuantizedTensor, x: &[f32], out: &mut [f32]) {
        #[cfg(target_arch = "aarch64")]
        {
            let n_rows = weight.rows();

            match weight.quant_type() {
                QuantType::Q4_0 => {
                    for row in 0..n_rows {
                        let blocks = weight.q4_0_row(row);
                        out[row] = unsafe { neon_impl::q4_0_vec_dot_neon(blocks, x) };
                    }
                }
                QuantType::Q4_K => {
                    for row in 0..n_rows {
                        let blocks = weight.q4_k_row(row);
                        out[row] = unsafe { neon_impl::q4_k_vec_dot_neon(blocks, x) };
                    }
                }
                _ => super::scalar::ScalarBackend.q4_gemv(weight, x, out),
            }
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            super::scalar::ScalarBackend.q4_gemv(weight, x, out)
        }
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
        #[cfg(target_arch = "aarch64")]
        {
            // For GEMM, use row-wise GEMV for each input row
            match weight.quant_type() {
                QuantType::Q4_0 | QuantType::Q4_K => {
                    for i in 0..m {
                        let x_row = &x[i * k..(i + 1) * k];
                        let out_row = &mut out[i * n..(i + 1) * n];
                        self.q4_gemv(weight, x_row, out_row);
                    }
                }
                _ => super::scalar::ScalarBackend.q4_gemm(weight, x, out, m, k, n),
            }
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            super::scalar::ScalarBackend.q4_gemm(weight, x, out, m, k, n)
        }
    }

    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
        #[cfg(target_arch = "aarch64")]
        unsafe { neon_impl::rms_norm_neon(x, weight, eps, out) }

        #[cfg(not(target_arch = "aarch64"))]
        super::scalar::ScalarBackend.rms_norm(x, weight, eps, out)
    }

    fn rms_norm_inplace(&self, x: &mut [f32], weight: &[f32], eps: f32) {
        #[cfg(target_arch = "aarch64")]
        {
            let mut temp = vec![0.0f32; x.len()];
            unsafe { neon_impl::rms_norm_neon(x, weight, eps, &mut temp) };
            x.copy_from_slice(&temp);
        }

        #[cfg(not(target_arch = "aarch64"))]
        super::scalar::ScalarBackend.rms_norm_inplace(x, weight, eps)
    }

    fn fused_swiglu(&self, gate: &[f32], up: &[f32], out: &mut [f32]) {
        #[cfg(target_arch = "aarch64")]
        unsafe { neon_impl::fused_swiglu_neon(gate, up, out) }

        #[cfg(not(target_arch = "aarch64"))]
        super::scalar::ScalarBackend.fused_swiglu(gate, up, out)
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
        #[cfg(target_arch = "aarch64")]
        unsafe { neon_impl::apply_rope_neon(q, k, cos, sin, head_dim, num_heads) }

        #[cfg(not(target_arch = "aarch64"))]
        super::scalar::ScalarBackend.apply_rope(q, k, cos, sin, head_dim, num_heads)
    }

    fn softmax(&self, x: &[f32], out: &mut [f32]) {
        #[cfg(target_arch = "aarch64")]
        unsafe { neon_impl::softmax_neon(x, out) }

        #[cfg(not(target_arch = "aarch64"))]
        super::scalar::ScalarBackend.softmax(x, out)
    }

    fn softmax_inplace(&self, x: &mut [f32]) {
        #[cfg(target_arch = "aarch64")]
        {
            let mut temp = vec![0.0f32; x.len()];
            unsafe { neon_impl::softmax_neon(x, &mut temp) };
            x.copy_from_slice(&temp);
        }

        #[cfg(not(target_arch = "aarch64"))]
        super::scalar::ScalarBackend.softmax_inplace(x)
    }

    fn silu(&self, x: &[f32], out: &mut [f32]) {
        #[cfg(target_arch = "aarch64")]
        unsafe { neon_impl::silu_neon(x, out) }

        #[cfg(not(target_arch = "aarch64"))]
        super::scalar::ScalarBackend.silu(x, out)
    }

    fn mul(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        #[cfg(target_arch = "aarch64")]
        unsafe { neon_impl::mul_neon(a, b, out) }

        #[cfg(not(target_arch = "aarch64"))]
        super::scalar::ScalarBackend.mul(a, b, out)
    }

    fn add(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        #[cfg(target_arch = "aarch64")]
        unsafe { neon_impl::add_neon(a, b, out) }

        #[cfg(not(target_arch = "aarch64"))]
        super::scalar::ScalarBackend.add(a, b, out)
    }

    fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        #[cfg(target_arch = "aarch64")]
        unsafe { neon_impl::dot_neon(a, b) }

        #[cfg(not(target_arch = "aarch64"))]
        super::scalar::ScalarBackend.dot(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn test_neon_backend_creation() {
        let backend = NeonBackend::new();
        assert_eq!(backend.name(), "NEON");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_dot_product() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let backend = NeonBackend::new();
        let result = backend.dot(&a, &b);

        assert!((result - 36.0).abs() < 1e-5, "Expected 36.0, got {}", result);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_rms_norm() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0f32, 1.0, 1.0, 1.0];
        let mut out = vec![0.0f32; 4];

        let backend = NeonBackend::new();
        backend.rms_norm(&x, &weight, 1e-6, &mut out);

        // Check output is normalized
        let sum_sq: f32 = out.iter().map(|x| x * x).sum();
        assert!(sum_sq > 0.0);
    }
}
