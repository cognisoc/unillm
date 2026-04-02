//! SwiGLU Activation Operations
//!
//! SwiGLU is used in LLaMA/Mistral/Qwen MLP layers.
//! Formula: out = silu(gate) * up = (gate * sigmoid(gate)) * up

use crate::simd::get_simd_backend;

/// Compute fused SwiGLU: out = silu(gate) * up
///
/// This fuses SiLU activation with element-wise multiplication,
/// reducing memory bandwidth by avoiding intermediate tensor.
#[inline]
pub fn fused_swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    get_simd_backend().fused_swiglu(gate, up, out);
}

/// Compute SiLU (Swish) activation: out = x * sigmoid(x)
#[inline]
pub fn silu(x: &[f32], out: &mut [f32]) {
    get_simd_backend().silu(x, out);
}

/// In-place SiLU
pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_swiglu() {
        let gate = vec![0.0, 1.0, 2.0, -1.0];
        let up = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0; 4];

        fused_swiglu(&gate, &up, &mut out);

        // silu(0) = 0
        assert!((out[0] - 0.0).abs() < 1e-5);

        // silu(1) ≈ 0.731
        assert!((out[1] - 0.731).abs() < 0.01);

        // silu(2) ≈ 1.762
        assert!((out[2] - 1.762).abs() < 0.01);

        // silu(-1) ≈ -0.269
        assert!((out[3] - (-0.269)).abs() < 0.01);
    }

    #[test]
    fn test_silu() {
        let x = vec![0.0, 1.0, -1.0];
        let mut out = vec![0.0; 3];

        silu(&x, &mut out);

        assert!((out[0] - 0.0).abs() < 1e-5);
        assert!((out[1] - 0.731).abs() < 0.01);
        assert!((out[2] - (-0.269)).abs() < 0.01);
    }
}
