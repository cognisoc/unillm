//! RMS Normalization Operations
//!
//! RMSNorm is used extensively in LLaMA-style models as a replacement for LayerNorm.
//! Formula: out = x * weight / sqrt(mean(x^2) + eps)

use crate::simd::get_simd_backend;

/// Compute RMS normalization using the best available SIMD backend
///
/// out[i] = x[i] * weight[i] / rms(x)
/// where rms(x) = sqrt(mean(x^2) + eps)
#[inline]
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    get_simd_backend().rms_norm(x, weight, eps, out);
}

/// In-place RMS normalization
#[inline]
pub fn rms_norm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    get_simd_backend().rms_norm_inplace(x, weight, eps);
}

/// Compute RMS value only (without normalization)
pub fn compute_rms(x: &[f32], eps: f32) -> f32 {
    let n = x.len();
    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    (sum_sq / n as f32 + eps).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0; 4];

        rms_norm(&x, &weight, 1e-6, &mut out);

        // Verify output has reasonable values
        assert!(out.iter().all(|&v| v.is_finite()));
        assert!(out.iter().any(|&v| v != 0.0));
    }
}
