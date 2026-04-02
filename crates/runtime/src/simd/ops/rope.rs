//! Rotary Position Embedding (RoPE) Operations
//!
//! RoPE applies rotation to Q and K vectors based on position,
//! enabling the model to understand relative positions.

use crate::simd::get_simd_backend;

/// Apply RoPE to Q and K tensors in-place
///
/// q, k: [num_heads * head_dim] flattened tensors
/// cos, sin: [head_dim / 2] precomputed rotation values
#[inline]
pub fn apply_rope(
    q: &mut [f32],
    k: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    head_dim: usize,
    num_heads: usize,
) {
    get_simd_backend().apply_rope(q, k, cos, sin, head_dim, num_heads);
}

/// Apply RoPE to Q only
pub fn apply_rope_q(
    q: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    head_dim: usize,
    num_heads: usize,
) {
    let half_dim = head_dim / 2;

    for head in 0..num_heads {
        let offset = head * head_dim;

        for i in 0..half_dim {
            let q0 = q[offset + i];
            let q1 = q[offset + i + half_dim];
            let c = cos[i];
            let s = sin[i];

            q[offset + i] = q0 * c - q1 * s;
            q[offset + i + half_dim] = q0 * s + q1 * c;
        }
    }
}

/// Precompute cos/sin values for RoPE
///
/// Returns (cos, sin) arrays of shape [max_seq_len, head_dim / 2]
pub fn precompute_rope_cache(
    head_dim: usize,
    max_seq_len: usize,
    base: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half_dim = head_dim / 2;
    let mut cos_cache = vec![0.0f32; max_seq_len * half_dim];
    let mut sin_cache = vec![0.0f32; max_seq_len * half_dim];

    // Compute inverse frequencies
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / base.powf(2.0 * i as f32 / head_dim as f32))
        .collect();

    for pos in 0..max_seq_len {
        let pos_f = pos as f32;
        for (i, &freq) in inv_freq.iter().enumerate() {
            let angle = pos_f * freq;
            cos_cache[pos * half_dim + i] = angle.cos();
            sin_cache[pos * half_dim + i] = angle.sin();
        }
    }

    (cos_cache, sin_cache)
}

/// Get cos/sin values for a specific position from precomputed cache
pub fn get_rope_at_position<'a>(
    cos_cache: &'a [f32],
    sin_cache: &'a [f32],
    position: usize,
    head_dim: usize,
) -> (&'a [f32], &'a [f32]) {
    let half_dim = head_dim / 2;
    let offset = position * half_dim;
    (
        &cos_cache[offset..offset + half_dim],
        &sin_cache[offset..offset + half_dim],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precompute_rope_cache() {
        let head_dim = 64;
        let max_seq_len = 128;
        let base = 10000.0;

        let (cos_cache, sin_cache) = precompute_rope_cache(head_dim, max_seq_len, base);

        assert_eq!(cos_cache.len(), max_seq_len * head_dim / 2);
        assert_eq!(sin_cache.len(), max_seq_len * head_dim / 2);

        // At position 0, cos should be 1.0 and sin should be 0.0
        let (cos, sin) = get_rope_at_position(&cos_cache, &sin_cache, 0, head_dim);
        for &c in cos {
            assert!((c - 1.0).abs() < 1e-5);
        }
        for &s in sin {
            assert!(s.abs() < 1e-5);
        }
    }

    #[test]
    fn test_apply_rope() {
        let head_dim = 4;
        let num_heads = 2;

        // Initialize Q and K with simple values
        let mut q = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let mut k = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];

        // cos=1, sin=0 (no rotation at position 0)
        let cos = vec![1.0, 1.0];
        let sin = vec![0.0, 0.0];

        apply_rope(&mut q, &mut k, &cos, &sin, head_dim, num_heads);

        // With cos=1, sin=0, values should be unchanged
        assert!((q[0] - 1.0).abs() < 1e-5);
        assert!((q[1] - 0.0).abs() < 1e-5);
    }
}
