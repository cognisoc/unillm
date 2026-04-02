//! Q4_K Quantization Kernels
//!
//! Optimized operations for Q4_K_M format (llama.cpp's high-quality 4-bit).
//! Q4_K uses per-super-block and per-sub-block scales for better accuracy.

use super::block::{Q4_KBlock, Q4_K_BLOCK_SIZE};

/// Compute dot product of Q4_K block with f32 vector (scalar reference)
#[inline]
pub fn q4_k_dot_scalar(block: &Q4_KBlock, x: &[f32]) -> f32 {
    debug_assert!(x.len() >= Q4_K_BLOCK_SIZE);

    let d = block.d.to_f32();
    let dmin = block.dmin.to_f32();
    let mut sum = 0.0f32;

    // Process 8 sub-blocks of 32 values each
    for sb in 0..8 {
        let scale = d * block.get_scale(sb) as f32;
        let min = dmin * block.get_min(sb) as f32;
        let qs_offset = sb * 16;
        let x_offset = sb * 32;

        for i in 0..16 {
            let byte = block.qs[qs_offset + i];
            let v0 = (byte & 0x0F) as f32;
            let v1 = (byte >> 4) as f32;

            let x0 = x[x_offset + i * 2];
            let x1 = x[x_offset + i * 2 + 1];

            sum += (scale * v0 - min) * x0;
            sum += (scale * v1 - min) * x1;
        }
    }

    sum
}

/// Compute dot product of multiple Q4_K blocks with f32 vector
#[inline]
pub fn q4_k_vec_dot_scalar(blocks: &[Q4_KBlock], x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut offset = 0;

    for block in blocks {
        sum += q4_k_dot_scalar(block, &x[offset..]);
        offset += Q4_K_BLOCK_SIZE;
    }

    sum
}

/// Full GEMV using Q4_K weights (scalar reference)
pub fn q4_k_gemv_scalar(
    weight_blocks: &[Q4_KBlock],
    x: &[f32],
    out: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) {
    let blocks_per_row = (n_cols + Q4_K_BLOCK_SIZE - 1) / Q4_K_BLOCK_SIZE;

    for row in 0..n_rows {
        let row_start = row * blocks_per_row;
        let row_end = row_start + blocks_per_row;
        let row_blocks = &weight_blocks[row_start..row_end];

        out[row] = q4_k_vec_dot_scalar(row_blocks, x);
    }
}

/// Full GEMM using Q4_K weights (scalar reference)
pub fn q4_k_gemm_scalar(
    weight_blocks: &[Q4_KBlock],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    let blocks_per_row = (k + Q4_K_BLOCK_SIZE - 1) / Q4_K_BLOCK_SIZE;

    for i in 0..m {
        let x_row = &x[i * k..(i + 1) * k];

        for j in 0..n {
            let w_start = j * blocks_per_row;
            let w_end = w_start + blocks_per_row;
            let w_blocks = &weight_blocks[w_start..w_end];

            out[i * n + j] = q4_k_vec_dot_scalar(w_blocks, x_row);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn test_q4_k_block_size() {
        assert_eq!(Q4_KBlock::SIZE_BYTES, 144);
        assert_eq!(Q4_KBlock::VALUES_PER_BLOCK, 256);
    }
}
