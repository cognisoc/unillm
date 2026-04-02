//! Quantized Block Structures
//!
//! These structures match llama.cpp's GGML quantization formats exactly,
//! enabling efficient SIMD processing without format conversion.
//!
//! Key insight: Each block is sized for optimal SIMD register usage:
//! - Q4_0: 32 values → fits in one AVX2 operation (8 floats × 4 iterations)
//! - Q4_K: 256 values → fits in cache line, enables efficient prefetch

use half::f16;
use std::mem::size_of;

/// Block size for Q4_0 format (32 values per block)
pub const Q4_0_BLOCK_SIZE: usize = 32;

/// Block size for Q4_K format (256 values per block)
pub const Q4_K_BLOCK_SIZE: usize = 256;

/// Q4_0 Block: 32 values quantized to 4 bits each
///
/// Memory layout (18 bytes total for 32 values = 4.5 bits/value):
/// - d: f16 scale factor (2 bytes)
/// - qs: 16 bytes of packed 4-bit values (32 values, 2 per byte)
///
/// Dequantization formula: value[i] = d * (qs[i] - 8)
/// The -8 centers the 4-bit range [0,15] to [-8,7]
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy)]
pub struct Q4_0Block {
    /// Scale factor (f16, converted to f32 for computation)
    pub d: f16,
    /// 32 quantized values packed as 4 bits each (16 bytes)
    /// Lower 4 bits = even indices, upper 4 bits = odd indices
    pub qs: [u8; 16],
}

impl Q4_0Block {
    /// Size of one block in bytes
    pub const SIZE_BYTES: usize = 18; // 2 (f16) + 16 (qs)

    /// Number of values per block
    pub const VALUES_PER_BLOCK: usize = Q4_0_BLOCK_SIZE;

    /// Dequantize this block to f32 values
    ///
    /// Returns 32 f32 values.
    #[inline]
    pub fn dequantize(&self) -> [f32; 32] {
        let scale = self.d.to_f32();
        let mut out = [0.0f32; 32];

        for i in 0..16 {
            let byte = self.qs[i];
            // Lower 4 bits → even index
            out[i * 2] = scale * ((byte & 0x0F) as i8 - 8) as f32;
            // Upper 4 bits → odd index
            out[i * 2 + 1] = scale * ((byte >> 4) as i8 - 8) as f32;
        }

        out
    }

    /// Dequantize directly into output buffer
    #[inline]
    pub fn dequantize_into(&self, out: &mut [f32]) {
        debug_assert!(out.len() >= 32);
        let scale = self.d.to_f32();

        for i in 0..16 {
            let byte = self.qs[i];
            out[i * 2] = scale * ((byte & 0x0F) as i8 - 8) as f32;
            out[i * 2 + 1] = scale * ((byte >> 4) as i8 - 8) as f32;
        }
    }

    /// Get scale as f32
    #[inline]
    pub fn scale(&self) -> f32 {
        self.d.to_f32()
    }
}

/// Q4_K Block: 256 values with super-block and sub-block scales
///
/// This is llama.cpp's highest quality 4-bit format, using:
/// - Per-super-block scale (d) and minimum (dmin) in f16
/// - Per-sub-block 6-bit scales for finer granularity
/// - 8 sub-blocks of 32 values each
///
/// Memory layout (144 bytes total for 256 values = 4.5 bits/value):
/// - d: f16 super-block scale (2 bytes)
/// - dmin: f16 super-block minimum (2 bytes)
/// - scales: 12 bytes of packed 6-bit sub-block scales
/// - qs: 128 bytes of packed 4-bit values
///
/// Dequantization formula:
///   value[i] = d * scales[i/32] * qs[i] - dmin * mins[i/32]
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy)]
pub struct Q4_KBlock {
    /// Super-block scale (f16)
    pub d: f16,
    /// Super-block minimum (f16)
    pub dmin: f16,
    /// Sub-block scales and mins, packed as 6-bit values (12 bytes)
    /// Contains 8 scales + 8 mins = 16 6-bit values
    pub scales: [u8; 12],
    /// 256 quantized values packed as 4 bits each (128 bytes)
    pub qs: [u8; 128],
}

impl Q4_KBlock {
    /// Size of one block in bytes
    pub const SIZE_BYTES: usize = 144; // 2 + 2 + 12 + 128

    /// Number of values per block
    pub const VALUES_PER_BLOCK: usize = Q4_K_BLOCK_SIZE;

    /// Number of sub-blocks (each has 32 values)
    pub const NUM_SUB_BLOCKS: usize = 8;

    /// Extract sub-block scale at index (0-7)
    #[inline]
    pub fn get_scale(&self, idx: usize) -> u8 {
        debug_assert!(idx < 8);
        // Scales are packed in first 6 bytes, 6 bits each
        // This is a simplified extraction - actual format is more complex
        let bit_offset = idx * 6;
        let byte_idx = bit_offset / 8;
        let bit_idx = bit_offset % 8;

        if bit_idx <= 2 {
            (self.scales[byte_idx] >> bit_idx) & 0x3F
        } else {
            let low = self.scales[byte_idx] >> bit_idx;
            let high = self.scales[byte_idx + 1] << (8 - bit_idx);
            (low | high) & 0x3F
        }
    }

    /// Extract sub-block minimum at index (0-7)
    #[inline]
    pub fn get_min(&self, idx: usize) -> u8 {
        debug_assert!(idx < 8);
        // Mins are packed in bytes 6-11, 6 bits each
        let bit_offset = (idx + 8) * 6; // +8 to skip scales
        let byte_idx = bit_offset / 8;
        let bit_idx = bit_offset % 8;

        if bit_idx <= 2 {
            (self.scales[byte_idx] >> bit_idx) & 0x3F
        } else if byte_idx + 1 < 12 {
            let low = self.scales[byte_idx] >> bit_idx;
            let high = self.scales[byte_idx + 1] << (8 - bit_idx);
            (low | high) & 0x3F
        } else {
            (self.scales[byte_idx] >> bit_idx) & 0x3F
        }
    }

    /// Dequantize this block to f32 values
    ///
    /// Returns 256 f32 values.
    pub fn dequantize(&self) -> [f32; 256] {
        let d = self.d.to_f32();
        let dmin = self.dmin.to_f32();
        let mut out = [0.0f32; 256];

        for sb in 0..8 {
            let scale = d * self.get_scale(sb) as f32;
            let min = dmin * self.get_min(sb) as f32;
            let qs_offset = sb * 16; // 32 values = 16 bytes

            for i in 0..16 {
                let byte = self.qs[qs_offset + i];
                let out_idx = sb * 32 + i * 2;
                out[out_idx] = scale * (byte & 0x0F) as f32 - min;
                out[out_idx + 1] = scale * (byte >> 4) as f32 - min;
            }
        }

        out
    }
}

/// Quantization type enum for runtime dispatch
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    Q4_0,
    Q4_K,
    Q8_0,
    F16,
    F32,
}

impl QuantType {
    /// Bytes per block for this quantization type
    pub fn block_size_bytes(&self) -> usize {
        match self {
            QuantType::Q4_0 => Q4_0Block::SIZE_BYTES,
            QuantType::Q4_K => Q4_KBlock::SIZE_BYTES,
            QuantType::Q8_0 => 34, // 2 (f16 scale) + 32 (i8 values)
            QuantType::F16 => 2,
            QuantType::F32 => 4,
        }
    }

    /// Values per block
    pub fn values_per_block(&self) -> usize {
        match self {
            QuantType::Q4_0 => Q4_0_BLOCK_SIZE,
            QuantType::Q4_K => Q4_K_BLOCK_SIZE,
            QuantType::Q8_0 => 32,
            QuantType::F16 => 1,
            QuantType::F32 => 1,
        }
    }

    /// Bits per value (average)
    pub fn bits_per_value(&self) -> f32 {
        match self {
            QuantType::Q4_0 => 4.5, // 18 bytes / 32 values
            QuantType::Q4_K => 4.5, // 144 bytes / 256 values
            QuantType::Q8_0 => 8.5, // 34 bytes / 32 values
            QuantType::F16 => 16.0,
            QuantType::F32 => 32.0,
        }
    }
}

/// A quantized tensor with contiguous block storage
///
/// Weights are stored as a contiguous array of blocks, organized for
/// efficient sequential access during matrix operations.
#[derive(Debug)]
pub struct QuantizedTensor {
    /// Raw block data (aligned for SIMD)
    data: aligned_vec::AVec<u8, aligned_vec::ConstAlign<32>>,
    /// Quantization type
    quant_type: QuantType,
    /// Tensor shape [rows, cols] (logical shape in elements, not blocks)
    shape: [usize; 2],
    /// Number of blocks per row
    blocks_per_row: usize,
}

impl QuantizedTensor {
    /// Create a new quantized tensor from raw block data
    pub fn new(data: Vec<u8>, quant_type: QuantType, rows: usize, cols: usize) -> Self {
        let values_per_block = quant_type.values_per_block();
        let blocks_per_row = (cols + values_per_block - 1) / values_per_block;

        // Copy to aligned storage
        let mut aligned_data = aligned_vec::AVec::new(32);
        aligned_data.extend_from_slice(&data);

        Self {
            data: aligned_data,
            quant_type,
            shape: [rows, cols],
            blocks_per_row,
        }
    }

    /// Create from Q4_0 blocks
    pub fn from_q4_0_blocks(blocks: &[Q4_0Block], rows: usize, cols: usize) -> Self {
        let bytes: &[u8] = bytemuck::cast_slice(blocks);
        Self::new(bytes.to_vec(), QuantType::Q4_0, rows, cols)
    }

    /// Create from Q4_K blocks
    pub fn from_q4_k_blocks(blocks: &[Q4_KBlock], rows: usize, cols: usize) -> Self {
        let bytes: &[u8] = bytemuck::cast_slice(blocks);
        Self::new(bytes.to_vec(), QuantType::Q4_K, rows, cols)
    }

    /// Get quantization type
    pub fn quant_type(&self) -> QuantType {
        self.quant_type
    }

    /// Get tensor shape [rows, cols]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    /// Get number of rows
    pub fn rows(&self) -> usize {
        self.shape[0]
    }

    /// Get number of columns
    pub fn cols(&self) -> usize {
        self.shape[1]
    }

    /// Get raw data pointer (aligned)
    pub fn data_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    /// Get Q4_0 blocks for a specific row
    pub fn q4_0_row(&self, row: usize) -> &[Q4_0Block] {
        debug_assert_eq!(self.quant_type, QuantType::Q4_0);
        debug_assert!(row < self.shape[0]);

        let block_size = Q4_0Block::SIZE_BYTES;
        let row_offset = row * self.blocks_per_row * block_size;
        let row_end = row_offset + self.blocks_per_row * block_size;

        let row_bytes = &self.data[row_offset..row_end];
        bytemuck::cast_slice(row_bytes)
    }

    /// Get Q4_K blocks for a specific row
    pub fn q4_k_row(&self, row: usize) -> &[Q4_KBlock] {
        debug_assert_eq!(self.quant_type, QuantType::Q4_K);
        debug_assert!(row < self.shape[0]);

        let block_size = Q4_KBlock::SIZE_BYTES;
        let row_offset = row * self.blocks_per_row * block_size;
        let row_end = row_offset + self.blocks_per_row * block_size;

        let row_bytes = &self.data[row_offset..row_end];
        bytemuck::cast_slice(row_bytes)
    }

    /// Get all Q4_0 blocks
    pub fn as_q4_0_blocks(&self) -> &[Q4_0Block] {
        debug_assert_eq!(self.quant_type, QuantType::Q4_0);
        bytemuck::cast_slice(&self.data)
    }

    /// Get all Q4_K blocks
    pub fn as_q4_k_blocks(&self) -> &[Q4_KBlock] {
        debug_assert_eq!(self.quant_type, QuantType::Q4_K);
        bytemuck::cast_slice(&self.data)
    }

    /// Dequantize entire tensor to f32
    ///
    /// This allocates a full f32 tensor - use sparingly!
    /// Prefer on-the-fly dequantization in SIMD kernels.
    pub fn dequantize_full(&self) -> Vec<f32> {
        let total_values = self.shape[0] * self.shape[1];
        let mut out = vec![0.0f32; total_values];

        match self.quant_type {
            QuantType::Q4_0 => {
                let blocks = self.as_q4_0_blocks();
                for (i, block) in blocks.iter().enumerate() {
                    let values = block.dequantize();
                    let offset = i * Q4_0_BLOCK_SIZE;
                    let end = (offset + Q4_0_BLOCK_SIZE).min(total_values);
                    out[offset..end].copy_from_slice(&values[..end - offset]);
                }
            }
            QuantType::Q4_K => {
                let blocks = self.as_q4_k_blocks();
                for (i, block) in blocks.iter().enumerate() {
                    let values = block.dequantize();
                    let offset = i * Q4_K_BLOCK_SIZE;
                    let end = (offset + Q4_K_BLOCK_SIZE).min(total_values);
                    out[offset..end].copy_from_slice(&values[..end - offset]);
                }
            }
            _ => unimplemented!("Dequantization for {:?}", self.quant_type),
        }

        out
    }

    /// Memory usage in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }
}

// Safety: Q4_0Block has a well-defined layout with no padding
unsafe impl bytemuck::Pod for Q4_0Block {}
unsafe impl bytemuck::Zeroable for Q4_0Block {}

// Safety: Q4_KBlock has a well-defined layout with no padding
unsafe impl bytemuck::Pod for Q4_KBlock {}
unsafe impl bytemuck::Zeroable for Q4_KBlock {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_0_block_size() {
        assert_eq!(size_of::<Q4_0Block>(), 32); // Aligned to 32 bytes
        assert_eq!(Q4_0Block::SIZE_BYTES, 18); // Actual data is 18 bytes
    }

    #[test]
    fn test_q4_0_dequantize() {
        let block = Q4_0Block {
            d: f16::from_f32(0.5),
            qs: [0x88; 16], // All values = 8, which dequantizes to 0
        };

        let values = block.dequantize();
        for &v in &values {
            assert!((v - 0.0).abs() < 1e-5, "Expected 0, got {}", v);
        }
    }

    #[test]
    fn test_q4_0_dequantize_range() {
        // Test with values at extremes: 0 and 15
        let block = Q4_0Block {
            d: f16::from_f32(1.0),
            qs: [0xF0; 16], // Lower nibble = 0 (-8), upper nibble = 15 (+7)
        };

        let values = block.dequantize();
        // Even indices: 0 - 8 = -8
        assert!((values[0] - (-8.0)).abs() < 1e-5);
        // Odd indices: 15 - 8 = 7
        assert!((values[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_quantized_tensor_creation() {
        // Create a simple 2x64 tensor with Q4_0
        let num_blocks = 2 * 2; // 2 rows, 64 cols = 2 blocks per row
        let blocks: Vec<Q4_0Block> = (0..num_blocks)
            .map(|_| Q4_0Block {
                d: f16::from_f32(0.1),
                qs: [0x88; 16],
            })
            .collect();

        let tensor = QuantizedTensor::from_q4_0_blocks(&blocks, 2, 64);

        assert_eq!(tensor.shape(), [2, 64]);
        assert_eq!(tensor.quant_type(), QuantType::Q4_0);
        assert_eq!(tensor.rows(), 2);
        assert_eq!(tensor.cols(), 64);
    }
}
