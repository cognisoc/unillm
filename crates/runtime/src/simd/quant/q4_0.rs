//! Q4_0 Quantization Kernels
//!
//! Optimized dequantization and dot product operations for Q4_0 format.

use super::block::{Q4_0Block, Q4_0_BLOCK_SIZE};

/// Compute dot product of Q4_0 block with f32 vector (scalar reference)
///
/// This is the core operation for quantized inference:
/// result = sum(dequant(block) * x)
///
/// Optimized versions in arch/ modules use SIMD to process multiple values.
#[inline]
pub fn q4_0_dot_scalar(block: &Q4_0Block, x: &[f32]) -> f32 {
    debug_assert!(x.len() >= Q4_0_BLOCK_SIZE);

    let scale = block.d.to_f32();
    let mut sum = 0.0f32;

    for i in 0..16 {
        let byte = block.qs[i];
        // Lower 4 bits
        let v0 = (byte & 0x0F) as i8 - 8;
        // Upper 4 bits
        let v1 = (byte >> 4) as i8 - 8;

        sum += (v0 as f32) * x[i * 2];
        sum += (v1 as f32) * x[i * 2 + 1];
    }

    sum * scale
}

/// Compute dot product of multiple Q4_0 blocks with f32 vector
///
/// Used for GEMV: one row of weight matrix (multiple blocks) dot input vector.
#[inline]
pub fn q4_0_vec_dot_scalar(blocks: &[Q4_0Block], x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut offset = 0;

    for block in blocks {
        sum += q4_0_dot_scalar(block, &x[offset..]);
        offset += Q4_0_BLOCK_SIZE;
    }

    sum
}

/// Dequantize Q4_0 blocks and multiply by vector, accumulating into output
///
/// Used for GEMV: computes one output element.
/// out[row] = sum_over_k(weight[row, k] * x[k])
#[inline]
pub fn q4_0_gemv_row_scalar(blocks: &[Q4_0Block], x: &[f32], out: &mut f32) {
    *out = q4_0_vec_dot_scalar(blocks, x);
}

/// Full GEMV using Q4_0 weights (scalar reference)
///
/// Computes: out = weight @ x
/// - weight: [n_rows, n_cols] stored as Q4_0 blocks
/// - x: [n_cols] f32 input vector
/// - out: [n_rows] f32 output vector
pub fn q4_0_gemv_scalar(
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
        let row_blocks = &weight_blocks[row_start..row_end];

        out[row] = q4_0_vec_dot_scalar(row_blocks, x);
    }
}

/// Full GEMM using Q4_0 weights (scalar reference)
///
/// Computes: out = x @ weight^T
/// - weight: [n, k] stored as Q4_0 blocks (n = output dim, k = hidden dim)
/// - x: [m, k] f32 input matrix (m = sequence length)
/// - out: [m, n] f32 output matrix
pub fn q4_0_gemm_scalar(
    weight_blocks: &[Q4_0Block],
    x: &[f32],
    out: &mut [f32],
    m: usize, // rows in x (seq len)
    k: usize, // cols in x / cols in weight (hidden)
    n: usize, // rows in weight (output dim)
) {
    let blocks_per_row = (k + Q4_0_BLOCK_SIZE - 1) / Q4_0_BLOCK_SIZE;

    for i in 0..m {
        let x_row = &x[i * k..(i + 1) * k];

        for j in 0..n {
            let w_start = j * blocks_per_row;
            let w_end = w_start + blocks_per_row;
            let w_blocks = &weight_blocks[w_start..w_end];

            out[i * n + j] = q4_0_vec_dot_scalar(w_blocks, x_row);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    fn make_test_block(scale: f32, values: &[i8; 32]) -> Q4_0Block {
        let mut qs = [0u8; 16];
        for i in 0..16 {
            let v0 = (values[i * 2] + 8) as u8;
            let v1 = (values[i * 2 + 1] + 8) as u8;
            qs[i] = v0 | (v1 << 4);
        }
        Q4_0Block {
            d: f16::from_f32(scale),
            qs,
        }
    }

    #[test]
    fn test_q4_0_dot_zeros() {
        let block = Q4_0Block {
            d: f16::from_f32(1.0),
            qs: [0x88; 16], // All values = 8 → dequant = 0
        };
        let x = [1.0f32; 32];

        let result = q4_0_dot_scalar(&block, &x);
        assert!((result - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_q4_0_dot_ones() {
        // Values = 9 → dequant = 1.0 with scale 1.0
        let mut qs = [0u8; 16];
        for i in 0..16 {
            qs[i] = 0x99; // Both nibbles = 9 → value = 1
        }
        let block = Q4_0Block {
            d: f16::from_f32(1.0),
            qs,
        };
        let x = [1.0f32; 32];

        let result = q4_0_dot_scalar(&block, &x);
        assert!((result - 32.0).abs() < 1e-4, "Expected 32.0, got {}", result);
    }

    #[test]
    fn test_q4_0_gemv() {
        // 2x64 weight matrix (2 rows, 64 cols = 2 blocks per row)
        let blocks: Vec<Q4_0Block> = (0..4)
            .map(|_| Q4_0Block {
                d: f16::from_f32(0.1),
                qs: [0x99; 16], // All 1s
            })
            .collect();

        let x = vec![1.0f32; 64];
        let mut out = vec![0.0f32; 2];

        q4_0_gemv_scalar(&blocks, &x, &mut out, 2, 64);

        // Each row: 64 values of 0.1 * 1 * 1 = 6.4
        for &v in &out {
            assert!((v - 6.4).abs() < 0.1, "Expected ~6.4, got {}", v);
        }
    }
}
