//! Scalar (Non-SIMD) Reference Implementation
//!
//! This provides a portable fallback that works on all platforms.
//! Used for correctness testing and on systems without SIMD support.

use crate::simd::quant::{q4_0_gemv_scalar, q4_k_gemv_scalar, QuantType, QuantizedTensor};
use crate::simd::{CpuFeatures, SimdBackend};

/// Scalar backend - portable reference implementation
pub struct ScalarBackend;

impl ScalarBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ScalarBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl SimdBackend for ScalarBackend {
    fn name(&self) -> &'static str {
        "Scalar"
    }

    fn required_features(&self) -> CpuFeatures {
        CpuFeatures::default() // No special features required
    }

    fn q4_gemv(&self, weight: &QuantizedTensor, x: &[f32], out: &mut [f32]) {
        let [n_rows, n_cols] = weight.shape();

        match weight.quant_type() {
            QuantType::Q4_0 => {
                let blocks = weight.as_q4_0_blocks();
                q4_0_gemv_scalar(blocks, x, out, n_rows, n_cols);
            }
            QuantType::Q4_K => {
                let blocks = weight.as_q4_k_blocks();
                q4_k_gemv_scalar(blocks, x, out, n_rows, n_cols);
            }
            _ => unimplemented!("Scalar GEMV for {:?}", weight.quant_type()),
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
        match weight.quant_type() {
            QuantType::Q4_0 => {
                let blocks = weight.as_q4_0_blocks();
                crate::simd::quant::q4_0_gemm_scalar(blocks, x, out, m, k, n);
            }
            QuantType::Q4_K => {
                let blocks = weight.as_q4_k_blocks();
                crate::simd::quant::q4_k_gemm_scalar(blocks, x, out, m, k, n);
            }
            _ => unimplemented!("Scalar GEMM for {:?}", weight.quant_type()),
        }
    }

    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
        let n = x.len();
        debug_assert_eq!(weight.len(), n);
        debug_assert_eq!(out.len(), n);

        // Compute sum of squares
        let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
        let rms = (sum_sq / n as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        // Normalize and scale
        for i in 0..n {
            out[i] = x[i] * inv_rms * weight[i];
        }
    }

    fn rms_norm_inplace(&self, x: &mut [f32], weight: &[f32], eps: f32) {
        let n = x.len();
        debug_assert_eq!(weight.len(), n);

        let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
        let rms = (sum_sq / n as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        for i in 0..n {
            x[i] = x[i] * inv_rms * weight[i];
        }
    }

    fn fused_swiglu(&self, gate: &[f32], up: &[f32], out: &mut [f32]) {
        debug_assert_eq!(gate.len(), up.len());
        debug_assert_eq!(gate.len(), out.len());

        for i in 0..gate.len() {
            // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
            let g = gate[i];
            let silu_g = g / (1.0 + (-g).exp());
            out[i] = silu_g * up[i];
        }
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
        debug_assert_eq!(q.len(), num_heads * head_dim);
        debug_assert_eq!(k.len(), num_heads * head_dim);
        debug_assert!(cos.len() >= head_dim / 2);
        debug_assert!(sin.len() >= head_dim / 2);

        let half_dim = head_dim / 2;

        for head in 0..num_heads {
            let offset = head * head_dim;

            // Apply rotation to Q
            for i in 0..half_dim {
                let q0 = q[offset + i];
                let q1 = q[offset + i + half_dim];
                let c = cos[i];
                let s = sin[i];

                q[offset + i] = q0 * c - q1 * s;
                q[offset + i + half_dim] = q0 * s + q1 * c;
            }

            // Apply rotation to K
            for i in 0..half_dim {
                let k0 = k[offset + i];
                let k1 = k[offset + i + half_dim];
                let c = cos[i];
                let s = sin[i];

                k[offset + i] = k0 * c - k1 * s;
                k[offset + i + half_dim] = k0 * s + k1 * c;
            }
        }
    }

    fn softmax(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), out.len());

        // Find max for numerical stability
        let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Compute exp(x - max) and sum
        let mut sum = 0.0f32;
        for i in 0..x.len() {
            out[i] = (x[i] - max_val).exp();
            sum += out[i];
        }

        // Normalize
        let inv_sum = 1.0 / sum;
        for v in out.iter_mut() {
            *v *= inv_sum;
        }
    }

    fn softmax_inplace(&self, x: &mut [f32]) {
        let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let mut sum = 0.0f32;
        for v in x.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }

        let inv_sum = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv_sum;
        }
    }

    fn silu(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), out.len());

        for i in 0..x.len() {
            let v = x[i];
            out[i] = v / (1.0 + (-v).exp());
        }
    }

    fn mul(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), out.len());

        for i in 0..a.len() {
            out[i] = a[i] * b[i];
        }
    }

    fn add(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), out.len());

        for i in 0..a.len() {
            out[i] = a[i] + b[i];
        }
    }

    fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_rms_norm() {
        let backend = ScalarBackend::new();

        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0; 4];

        backend.rms_norm(&x, &weight, 1e-6, &mut out);

        // RMS = sqrt((1 + 4 + 9 + 16) / 4) = sqrt(7.5) ≈ 2.739
        let rms = (30.0f32 / 4.0).sqrt();
        let expected: Vec<f32> = x.iter().map(|&v| v / rms).collect();

        for (o, e) in out.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-5, "Expected {}, got {}", e, o);
        }
    }

    #[test]
    fn test_scalar_softmax() {
        let backend = ScalarBackend::new();

        let x = vec![1.0, 2.0, 3.0];
        let mut out = vec![0.0; 3];

        backend.softmax(&x, &mut out);

        // Sum should be 1.0
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Values should be monotonically increasing
        assert!(out[0] < out[1]);
        assert!(out[1] < out[2]);
    }

    #[test]
    fn test_scalar_fused_swiglu() {
        let backend = ScalarBackend::new();

        let gate = vec![0.0, 1.0, 2.0];
        let up = vec![1.0, 1.0, 1.0];
        let mut out = vec![0.0; 3];

        backend.fused_swiglu(&gate, &up, &mut out);

        // SiLU(0) * 1 = 0
        assert!((out[0] - 0.0).abs() < 1e-5);

        // SiLU(1) ≈ 0.731
        assert!((out[1] - 0.731).abs() < 0.01);

        // SiLU(2) ≈ 1.762
        assert!((out[2] - 1.762).abs() < 0.01);
    }

    #[test]
    fn test_scalar_silu() {
        let backend = ScalarBackend::new();

        let x = vec![0.0, 1.0, -1.0];
        let mut out = vec![0.0; 3];

        backend.silu(&x, &mut out);

        // SiLU(0) = 0
        assert!((out[0] - 0.0).abs() < 1e-5);

        // SiLU(1) ≈ 0.731
        assert!((out[1] - 0.731).abs() < 0.01);

        // SiLU(-1) ≈ -0.269
        assert!((out[2] - (-0.269)).abs() < 0.01);
    }
}
