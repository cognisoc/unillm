//! GEMM (General Matrix-Matrix multiply) Operations
//!
//! GEMM is used during the prefill phase when processing the entire prompt.
//! Optimized for cache efficiency with tiled computation.
//!
//! Computes: out = x @ weight^T
//! - weight: [n, k] quantized (n = output dim, k = hidden dim)
//! - x: [m, k] f32 (m = sequence length)
//! - out: [m, n] f32

use crate::simd::get_simd_backend;
use crate::simd::quant::{QuantizedTensor, QuantType, Q4_0_BLOCK_SIZE, Q4_K_BLOCK_SIZE};

// ============================================================================
// Tile Sizes (tuned for typical cache hierarchies)
// ============================================================================

/// Outer tile size for L2 cache efficiency (256KB typical)
const TILE_M: usize = 64;  // Output rows (sequence positions)
const TILE_N: usize = 64;  // Output cols (output features)
const TILE_K: usize = 256; // Hidden dimension (should be multiple of block size)

/// Threshold for switching to parallel implementation
const PARALLEL_THRESHOLD_M: usize = 8;
const PARALLEL_THRESHOLD_N: usize = 64;

/// Wrapper for raw pointer to make it Send + Sync
/// Safety: Only safe when used to write to disjoint memory regions from parallel threads
#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

// ============================================================================
// Main GEMM Entry Points
// ============================================================================

/// Quantized GEMM using best available method
///
/// Automatically selects between tiled, parallel, or simple implementation
/// based on problem size.
#[inline]
pub fn q4_gemm(
    weight: &QuantizedTensor,
    x: &[f32],
    out: &mut [f32],
    m: usize, // rows in x (sequence length)
    k: usize, // cols in x / cols in weight (hidden dim)
    n: usize, // rows in weight (output dim)
) {
    // Choose implementation based on problem size
    if m >= PARALLEL_THRESHOLD_M && n >= PARALLEL_THRESHOLD_N {
        // Large enough for parallel processing
        q4_gemm_tiled_parallel(weight, x, out, m, k, n);
    } else if m >= 4 || n >= 64 {
        // Medium size: tiled but serial
        q4_gemm_tiled(weight, x, out, m, k, n);
    } else {
        // Small: use backend's default implementation
        get_simd_backend().q4_gemm(weight, x, out, m, k, n);
    }
}

/// Tiled GEMM for cache efficiency (serial)
///
/// Uses three-level tiling over M, N, K dimensions to maximize cache utilization.
pub fn q4_gemm_tiled(
    weight: &QuantizedTensor,
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    // Initialize output to zero
    out.fill(0.0);

    // Three-level tiling: outer (L2), then K (accumulation), then inner
    // This keeps weight blocks in L2 cache during K-loop
    for i_tile in (0..m).step_by(TILE_M) {
        let i_end = (i_tile + TILE_M).min(m);

        for j_tile in (0..n).step_by(TILE_N) {
            let j_end = (j_tile + TILE_N).min(n);

            // Tile over K dimension for cache-optimal weight access
            for k_tile in (0..k).step_by(TILE_K) {
                let k_end = (k_tile + TILE_K).min(k);

                // Process this tile
                process_tile(
                    weight, x, out,
                    i_tile, i_end,
                    j_tile, j_end,
                    k_tile, k_end,
                    n, k
                );
            }
        }
    }
}

/// Parallel tiled GEMM using rayon
///
/// Parallelizes over M tiles (output rows) for multi-core utilization.
/// Each thread processes independent output rows, no synchronization needed.
pub fn q4_gemm_tiled_parallel(
    weight: &QuantizedTensor,
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    use rayon::prelude::*;

    // Initialize output to zero
    out.fill(0.0);

    // Collect M tile ranges
    let m_tiles: Vec<(usize, usize)> = (0..m)
        .step_by(TILE_M)
        .map(|i| (i, (i + TILE_M).min(m)))
        .collect();

    // Get raw pointer for parallel access
    // Safety: Each thread writes to disjoint rows, so no data races occur
    let out_ptr = SendPtr(out.as_mut_ptr());

    // Process M tiles in parallel
    // Each thread handles a contiguous range of output rows
    m_tiles.par_iter().for_each(move |&(i_tile, i_end)| {
        // Compute output for rows [i_tile, i_end)
        let local_out = out_ptr;  // Copy the SendPtr (it's Copy)
        for j_tile in (0..n).step_by(TILE_N) {
            let j_end = (j_tile + TILE_N).min(n);

            for k_tile in (0..k).step_by(TILE_K) {
                let k_end = (k_tile + TILE_K).min(k);

                // Process tile - need unsafe for parallel write to out
                // This is safe because each thread writes to disjoint rows
                unsafe {
                    process_tile_unsafe(
                        weight, x, local_out.0,
                        i_tile, i_end,
                        j_tile, j_end,
                        k_tile, k_end,
                        n, k
                    );
                }
            }
        }
    });
}

// ============================================================================
// Tile Processing
// ============================================================================

/// Process a single tile of the GEMM (serial version)
#[inline]
fn process_tile(
    weight: &QuantizedTensor,
    x: &[f32],
    out: &mut [f32],
    i_start: usize, i_end: usize, // M range (output rows)
    j_start: usize, j_end: usize, // N range (output cols)
    k_start: usize, k_end: usize, // K range (accumulation)
    n: usize, k: usize,
) {
    match weight.quant_type() {
        QuantType::Q4_0 => {
            process_tile_q4_0(weight, x, out, i_start, i_end, j_start, j_end, k_start, k_end, n, k);
        }
        QuantType::Q4_K => {
            process_tile_q4_k(weight, x, out, i_start, i_end, j_start, j_end, k_start, k_end, n, k);
        }
        _ => unimplemented!("GEMM for {:?}", weight.quant_type()),
    }
}

/// Process a single tile (unsafe parallel version)
#[inline]
unsafe fn process_tile_unsafe(
    weight: &QuantizedTensor,
    x: &[f32],
    out: *mut f32,
    i_start: usize, i_end: usize,
    j_start: usize, j_end: usize,
    k_start: usize, k_end: usize,
    n: usize, k: usize,
) {
    match weight.quant_type() {
        QuantType::Q4_0 => {
            process_tile_q4_0_unsafe(weight, x, out, i_start, i_end, j_start, j_end, k_start, k_end, n, k);
        }
        QuantType::Q4_K => {
            process_tile_q4_k_unsafe(weight, x, out, i_start, i_end, j_start, j_end, k_start, k_end, n, k);
        }
        _ => unimplemented!(),
    }
}

/// Q4_0 tile processing
#[inline]
fn process_tile_q4_0(
    weight: &QuantizedTensor,
    x: &[f32],
    out: &mut [f32],
    i_start: usize, i_end: usize,
    j_start: usize, j_end: usize,
    k_start: usize, k_end: usize,
    n: usize, k: usize,
) {
    use crate::simd::quant::q4_0_vec_dot_scalar;

    let blocks_per_row = (k + Q4_0_BLOCK_SIZE - 1) / Q4_0_BLOCK_SIZE;
    let block_k_start = k_start / Q4_0_BLOCK_SIZE;
    let block_k_end = ((k_end + Q4_0_BLOCK_SIZE - 1) / Q4_0_BLOCK_SIZE).min(blocks_per_row);
    let num_blocks = block_k_end - block_k_start;

    if num_blocks == 0 { return; }

    for i in i_start..i_end {
        // Extract input slice for this row (only the K range we're processing)
        let x_k_start = k_start.min(k);
        let x_k_end = k_end.min(k);
        let x_row = &x[i * k + x_k_start..i * k + x_k_end];

        for j in j_start..j_end {
            // Get weight blocks for this output row, limited to K range
            let all_blocks = weight.as_q4_0_blocks();
            let w_row_start = j * blocks_per_row + block_k_start;
            let w_row_end = (j * blocks_per_row + block_k_end).min(all_blocks.len());

            if w_row_start >= all_blocks.len() { continue; }

            let w_blocks = &all_blocks[w_row_start..w_row_end];

            // Compute partial dot product and accumulate
            let partial = q4_0_vec_dot_scalar(w_blocks, x_row);
            out[i * n + j] += partial;
        }
    }
}

/// Q4_0 tile processing (unsafe for parallel)
#[inline]
unsafe fn process_tile_q4_0_unsafe(
    weight: &QuantizedTensor,
    x: &[f32],
    out: *mut f32,
    i_start: usize, i_end: usize,
    j_start: usize, j_end: usize,
    k_start: usize, k_end: usize,
    n: usize, k: usize,
) {
    use crate::simd::quant::q4_0_vec_dot_scalar;

    let blocks_per_row = (k + Q4_0_BLOCK_SIZE - 1) / Q4_0_BLOCK_SIZE;
    let block_k_start = k_start / Q4_0_BLOCK_SIZE;
    let block_k_end = ((k_end + Q4_0_BLOCK_SIZE - 1) / Q4_0_BLOCK_SIZE).min(blocks_per_row);
    let num_blocks = block_k_end - block_k_start;

    if num_blocks == 0 { return; }

    for i in i_start..i_end {
        let x_k_start = k_start.min(k);
        let x_k_end = k_end.min(k);
        let x_row = &x[i * k + x_k_start..i * k + x_k_end];

        for j in j_start..j_end {
            let all_blocks = weight.as_q4_0_blocks();
            let w_row_start = j * blocks_per_row + block_k_start;
            let w_row_end = (j * blocks_per_row + block_k_end).min(all_blocks.len());

            if w_row_start >= all_blocks.len() { continue; }

            let w_blocks = &all_blocks[w_row_start..w_row_end];
            let partial = q4_0_vec_dot_scalar(w_blocks, x_row);

            // Atomic-like addition (safe because each thread writes to disjoint rows)
            let ptr = out.add(i * n + j);
            *ptr += partial;
        }
    }
}

/// Q4_K tile processing
#[inline]
fn process_tile_q4_k(
    weight: &QuantizedTensor,
    x: &[f32],
    out: &mut [f32],
    i_start: usize, i_end: usize,
    j_start: usize, j_end: usize,
    k_start: usize, k_end: usize,
    n: usize, k: usize,
) {
    use crate::simd::quant::q4_k_vec_dot_scalar;

    let blocks_per_row = (k + Q4_K_BLOCK_SIZE - 1) / Q4_K_BLOCK_SIZE;
    let block_k_start = k_start / Q4_K_BLOCK_SIZE;
    let block_k_end = ((k_end + Q4_K_BLOCK_SIZE - 1) / Q4_K_BLOCK_SIZE).min(blocks_per_row);
    let num_blocks = block_k_end - block_k_start;

    if num_blocks == 0 { return; }

    for i in i_start..i_end {
        let x_k_start = k_start.min(k);
        let x_k_end = k_end.min(k);
        let x_row = &x[i * k + x_k_start..i * k + x_k_end];

        for j in j_start..j_end {
            let all_blocks = weight.as_q4_k_blocks();
            let w_row_start = j * blocks_per_row + block_k_start;
            let w_row_end = (j * blocks_per_row + block_k_end).min(all_blocks.len());

            if w_row_start >= all_blocks.len() { continue; }

            let w_blocks = &all_blocks[w_row_start..w_row_end];
            let partial = q4_k_vec_dot_scalar(w_blocks, x_row);
            out[i * n + j] += partial;
        }
    }
}

/// Q4_K tile processing (unsafe for parallel)
#[inline]
unsafe fn process_tile_q4_k_unsafe(
    weight: &QuantizedTensor,
    x: &[f32],
    out: *mut f32,
    i_start: usize, i_end: usize,
    j_start: usize, j_end: usize,
    k_start: usize, k_end: usize,
    n: usize, k: usize,
) {
    use crate::simd::quant::q4_k_vec_dot_scalar;

    let blocks_per_row = (k + Q4_K_BLOCK_SIZE - 1) / Q4_K_BLOCK_SIZE;
    let block_k_start = k_start / Q4_K_BLOCK_SIZE;
    let block_k_end = ((k_end + Q4_K_BLOCK_SIZE - 1) / Q4_K_BLOCK_SIZE).min(blocks_per_row);
    let num_blocks = block_k_end - block_k_start;

    if num_blocks == 0 { return; }

    for i in i_start..i_end {
        let x_k_start = k_start.min(k);
        let x_k_end = k_end.min(k);
        let x_row = &x[i * k + x_k_start..i * k + x_k_end];

        for j in j_start..j_end {
            let all_blocks = weight.as_q4_k_blocks();
            let w_row_start = j * blocks_per_row + block_k_start;
            let w_row_end = (j * blocks_per_row + block_k_end).min(all_blocks.len());

            if w_row_start >= all_blocks.len() { continue; }

            let w_blocks = &all_blocks[w_row_start..w_row_end];
            let partial = q4_k_vec_dot_scalar(w_blocks, x_row);

            let ptr = out.add(i * n + j);
            *ptr += partial;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::quant::{Q4_0Block, QuantizedTensor};
    use half::f16;

    #[test]
    fn test_q4_gemm_basic() {
        // weight: 2x64 (2 output features, 64 input features)
        // x: 3x64 (3 tokens, 64 hidden)
        // out: 3x2

        let num_blocks = 4; // 2 rows * 2 blocks per row
        let blocks: Vec<Q4_0Block> = (0..num_blocks)
            .map(|_| Q4_0Block {
                d: f16::from_f32(0.1),
                qs: [0x99; 16], // All 1s
            })
            .collect();

        let weight = QuantizedTensor::from_q4_0_blocks(&blocks, 2, 64);

        let x = vec![1.0f32; 3 * 64]; // 3 tokens
        let mut out = vec![0.0f32; 3 * 2];

        q4_gemm(&weight, &x, &mut out, 3, 64, 2);

        // Each output element: 64 * 0.1 * 1 * 1 = 6.4
        for &v in &out {
            assert!((v - 6.4).abs() < 0.5, "Expected ~6.4, got {}", v);
        }
    }

    #[test]
    fn test_q4_gemm_tiled() {
        // Larger test to exercise tiling
        let rows = 4;
        let cols = 128; // 4 blocks per row
        let num_blocks = rows * 4;

        let blocks: Vec<Q4_0Block> = (0..num_blocks)
            .map(|_| Q4_0Block {
                d: f16::from_f32(0.1),
                qs: [0x99; 16],
            })
            .collect();

        let weight = QuantizedTensor::from_q4_0_blocks(&blocks, rows, cols);

        let m = 8; // 8 sequence positions
        let x = vec![1.0f32; m * cols];
        let mut out = vec![0.0f32; m * rows];

        q4_gemm_tiled(&weight, &x, &mut out, m, cols, rows);

        // Each output: 128 * 0.1 * 1 = 12.8
        for &v in &out {
            assert!((v - 12.8).abs() < 1.0, "Expected ~12.8, got {}", v);
        }
    }
}
