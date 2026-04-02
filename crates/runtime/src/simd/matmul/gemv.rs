//! GEMV (General Matrix-Vector multiply) Operations
//!
//! GEMV is the critical hot path for autoregressive decoding,
//! where we process one token at a time.
//!
//! Computes: out = weight @ x
//! - weight: [out_features, in_features] quantized
//! - x: [in_features] f32
//! - out: [out_features] f32

use crate::simd::get_simd_backend;
use crate::simd::quant::QuantizedTensor;

/// Quantized GEMV using best available SIMD backend
///
/// This is the primary hot path for token generation (decode phase).
/// Each output element is a dot product of one weight row with the input.
#[inline]
pub fn q4_gemv(weight: &QuantizedTensor, x: &[f32], out: &mut [f32]) {
    get_simd_backend().q4_gemv(weight, x, out);
}

/// Quantized GEMV with specified output range
///
/// Useful for computing only a subset of outputs (e.g., partial vocabulary).
pub fn q4_gemv_partial(
    weight: &QuantizedTensor,
    x: &[f32],
    out: &mut [f32],
    start_row: usize,
    num_rows: usize,
) {
    // For now, compute all then slice
    // TODO: Optimize to only compute needed rows
    let mut full_out = vec![0.0f32; weight.rows()];
    q4_gemv(weight, x, &mut full_out);
    out[..num_rows].copy_from_slice(&full_out[start_row..start_row + num_rows]);
}

/// Parallel GEMV using rayon for large matrices
///
/// Splits rows across threads for better throughput on multi-core systems.
#[cfg(feature = "simd")]
pub fn q4_gemv_parallel(weight: &QuantizedTensor, x: &[f32], out: &mut [f32]) {
    use rayon::prelude::*;

    let n_rows = weight.rows();
    let backend = get_simd_backend();

    // Only parallelize if worth the overhead
    if n_rows < 256 {
        backend.q4_gemv(weight, x, out);
        return;
    }

    // Split into chunks and process in parallel
    // Each thread processes a contiguous range of rows
    let chunk_size = (n_rows + rayon::current_num_threads() - 1) / rayon::current_num_threads();

    out.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start_row = chunk_idx * chunk_size;
            let end_row = (start_row + out_chunk.len()).min(n_rows);

            // Compute this chunk's rows
            for (local_idx, row) in (start_row..end_row).enumerate() {
                // Get the row's blocks and compute dot product
                match weight.quant_type() {
                    crate::simd::quant::QuantType::Q4_0 => {
                        let blocks = weight.q4_0_row(row);
                        out_chunk[local_idx] = crate::simd::quant::q4_0_vec_dot_scalar(blocks, x);
                    }
                    crate::simd::quant::QuantType::Q4_K => {
                        let blocks = weight.q4_k_row(row);
                        out_chunk[local_idx] = crate::simd::quant::q4_k_vec_dot_scalar(blocks, x);
                    }
                    _ => unimplemented!(),
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::quant::{Q4_0Block, QuantizedTensor};
    use half::f16;

    #[test]
    fn test_q4_gemv_basic() {
        // Create a 2x64 weight matrix (2 output features, 64 input features)
        // Each row needs 2 Q4_0 blocks (64 values / 32 per block)
        let num_blocks = 4; // 2 rows * 2 blocks per row
        let blocks: Vec<Q4_0Block> = (0..num_blocks)
            .map(|_| Q4_0Block {
                d: f16::from_f32(0.1),
                qs: [0x99; 16], // All values = 9 -> dequant = 1
            })
            .collect();

        let weight = QuantizedTensor::from_q4_0_blocks(&blocks, 2, 64);

        let x = vec![1.0f32; 64];
        let mut out = vec![0.0f32; 2];

        q4_gemv(&weight, &x, &mut out);

        // Each output: 64 values of 0.1 * 1 * 1 = 6.4
        for &v in &out {
            assert!((v - 6.4).abs() < 0.5, "Expected ~6.4, got {}", v);
        }
    }
}
