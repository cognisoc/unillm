//! Precomputed Static Tensors for Performance
//!
//! This module provides caching utilities for static tensors that don't change
//! during inference. By precomputing these once at model initialization,
//! we avoid redundant computation on every forward pass.
//!
//! Supported caches:
//! - RoPECache: Precomputed rotary position embeddings (cos/sin frequencies)
//! - CausalMaskCache: Precomputed causal attention masks
//!
//! These caches work with ALL model architectures:
//! - RoPE: LLaMA, Qwen, Gemma, Mistral, Phi, DeepSeek, etc.
//! - Causal mask: All autoregressive decoder models

use anyhow::Result;
use crate::tensor_core::{Tensor, Device, DataType};

/// Precomputed RoPE (Rotary Position Embedding) frequencies
///
/// RoPE is used by most modern LLMs for position encoding.
/// Computing cos/sin values for each position is expensive, so we
/// precompute them once for max_seq_len positions.
///
/// # Example
/// ```ignore
/// let rope = RoPECache::new(128, 4096, 10000.0, &Device::CPU)?;
/// let (cos, sin) = rope.get(seq_len)?;
/// ```
pub struct RoPECache {
    /// Cosine values: [max_seq_len, head_dim/2]
    cos: Tensor,
    /// Sine values: [max_seq_len, head_dim/2]
    sin: Tensor,
    /// Head dimension
    head_dim: usize,
    /// Maximum sequence length
    max_seq_len: usize,
    /// RoPE base frequency (typically 10000.0)
    base: f32,
}

impl RoPECache {
    /// Create a new RoPE cache with precomputed frequencies
    ///
    /// # Arguments
    /// * `head_dim` - Dimension of each attention head
    /// * `max_seq_len` - Maximum sequence length to precompute
    /// * `base` - RoPE base frequency (typically 10000.0 for LLaMA)
    /// * `device` - Device to store tensors on
    pub fn new(head_dim: usize, max_seq_len: usize, base: f32, device: &Device) -> Result<Self> {
        let half_dim = head_dim / 2;

        // Compute inverse frequencies: 1 / (base^(2i/d))
        let mut inv_freq = vec![0.0f32; half_dim];
        for i in 0..half_dim {
            let exponent = (2.0 * i as f32) / head_dim as f32;
            inv_freq[i] = 1.0 / base.powf(exponent);
        }

        // Compute position indices
        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();

        // Compute angles: position * inv_freq for all positions
        // Result shape: [max_seq_len, half_dim]
        let mut cos_data = vec![0.0f32; max_seq_len * half_dim];
        let mut sin_data = vec![0.0f32; max_seq_len * half_dim];

        for pos in 0..max_seq_len {
            for freq_idx in 0..half_dim {
                let angle = positions[pos] * inv_freq[freq_idx];
                cos_data[pos * half_dim + freq_idx] = angle.cos();
                sin_data[pos * half_dim + freq_idx] = angle.sin();
            }
        }

        let cos = Tensor::from_f32_slice(&cos_data, &[max_seq_len, half_dim], device)?;
        let sin = Tensor::from_f32_slice(&sin_data, &[max_seq_len, half_dim], device)?;

        Ok(Self {
            cos,
            sin,
            head_dim,
            max_seq_len,
            base,
        })
    }

    /// Get precomputed cos/sin for a specific sequence length
    ///
    /// Returns slices of the precomputed tensors for positions [0, seq_len)
    pub fn get(&self, seq_len: usize) -> Result<(Tensor, Tensor)> {
        if seq_len > self.max_seq_len {
            return Err(anyhow::anyhow!(
                "Requested seq_len {} exceeds max_seq_len {}",
                seq_len,
                self.max_seq_len
            ));
        }

        // Return slices for the requested sequence length
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;

        Ok((cos, sin))
    }

    /// Get precomputed cos/sin for a range of positions
    ///
    /// Useful for KV cache scenarios where we only need positions [start, end)
    pub fn get_range(&self, start: usize, end: usize) -> Result<(Tensor, Tensor)> {
        if end > self.max_seq_len {
            return Err(anyhow::anyhow!(
                "Requested end {} exceeds max_seq_len {}",
                end,
                self.max_seq_len
            ));
        }
        if start >= end {
            return Err(anyhow::anyhow!(
                "Invalid range: start {} >= end {}",
                start,
                end
            ));
        }

        let len = end - start;
        let cos = self.cos.narrow(0, start, len)?;
        let sin = self.sin.narrow(0, start, len)?;

        Ok((cos, sin))
    }

    /// Get the head dimension
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get the maximum sequence length
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Get the RoPE base frequency
    pub fn base(&self) -> f32 {
        self.base
    }
}

/// Precomputed causal attention mask
///
/// Causal masks ensure that position i can only attend to positions <= i.
/// We precompute a mask for max_seq_len and slice as needed.
pub struct CausalMaskCache {
    /// Lower triangular mask: [max_seq_len, max_seq_len]
    /// Value 0.0 means "attend", NEG_INFINITY means "don't attend"
    mask: Tensor,
    /// Maximum sequence length
    max_seq_len: usize,
}

impl CausalMaskCache {
    /// Create a new causal mask cache
    ///
    /// # Arguments
    /// * `max_seq_len` - Maximum sequence length to precompute
    /// * `device` - Device to store the mask on
    pub fn new(max_seq_len: usize, device: &Device) -> Result<Self> {
        // Create causal mask: 0 for positions to attend, NEG_INFINITY for masked
        // This is the additive mask format used in attention computations
        let mut mask_data = vec![0.0f32; max_seq_len * max_seq_len];

        for i in 0..max_seq_len {
            for j in 0..max_seq_len {
                if j > i {
                    // Position j comes after position i, mask it
                    mask_data[i * max_seq_len + j] = f32::NEG_INFINITY;
                }
            }
        }

        let mask = Tensor::from_f32_slice(&mask_data, &[max_seq_len, max_seq_len], device)?;

        Ok(Self { mask, max_seq_len })
    }

    /// Get causal mask for a specific sequence length
    ///
    /// Returns a [seq_len, seq_len] mask
    pub fn get(&self, seq_len: usize) -> Result<Tensor> {
        if seq_len > self.max_seq_len {
            return Err(anyhow::anyhow!(
                "Requested seq_len {} exceeds max_seq_len {}",
                seq_len,
                self.max_seq_len
            ));
        }

        // Narrow both dimensions to get [seq_len, seq_len] slice
        let mask = self.mask.narrow(0, 0, seq_len)?.narrow(1, 0, seq_len)?;

        Ok(mask)
    }

    /// Get causal mask reshaped for attention broadcasting
    ///
    /// Returns mask with shape [1, 1, seq_len, seq_len] for broadcasting
    /// over [batch, heads, seq, seq] attention scores
    pub fn get_broadcast(&self, seq_len: usize) -> Result<Tensor> {
        let mask = self.get(seq_len)?;
        // Reshape to [1, 1, seq_len, seq_len] for broadcasting
        mask.reshape(&[1, 1, seq_len, seq_len])
    }

    /// Get the maximum sequence length
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

/// Precomputed sliding window attention mask
///
/// Each position can only attend to the previous `window_size` positions.
/// Used by Mistral, Mixtral, and other sliding window attention models.
pub struct SlidingWindowMaskCache {
    /// Sliding window mask: [max_seq_len, max_seq_len]
    mask: Tensor,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Window size
    window_size: usize,
}

impl SlidingWindowMaskCache {
    /// Create a new sliding window mask cache
    ///
    /// # Arguments
    /// * `max_seq_len` - Maximum sequence length to precompute
    /// * `window_size` - Number of positions to attend to
    /// * `device` - Device to store the mask on
    pub fn new(max_seq_len: usize, window_size: usize, device: &Device) -> Result<Self> {
        let mut mask_data = vec![f32::NEG_INFINITY; max_seq_len * max_seq_len];

        for i in 0..max_seq_len {
            // Can attend to positions from max(0, i - window_size + 1) to i
            let start = if i >= window_size { i - window_size + 1 } else { 0 };
            for j in start..=i {
                mask_data[i * max_seq_len + j] = 0.0;
            }
        }

        let mask = Tensor::from_f32_slice(&mask_data, &[max_seq_len, max_seq_len], device)?;

        Ok(Self {
            mask,
            max_seq_len,
            window_size,
        })
    }

    /// Get sliding window mask for a specific sequence length
    pub fn get(&self, seq_len: usize) -> Result<Tensor> {
        if seq_len > self.max_seq_len {
            return Err(anyhow::anyhow!(
                "Requested seq_len {} exceeds max_seq_len {}",
                seq_len,
                self.max_seq_len
            ));
        }

        let mask = self.mask.narrow(0, 0, seq_len)?.narrow(1, 0, seq_len)?;
        Ok(mask)
    }

    /// Get sliding window mask reshaped for attention broadcasting
    pub fn get_broadcast(&self, seq_len: usize) -> Result<Tensor> {
        let mask = self.get(seq_len)?;
        mask.reshape(&[1, 1, seq_len, seq_len])
    }

    /// Get the window size
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Get the maximum sequence length
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_cache_creation() {
        let rope = RoPECache::new(64, 128, 10000.0, &Device::CPU).unwrap();
        assert_eq!(rope.head_dim(), 64);
        assert_eq!(rope.max_seq_len(), 128);
    }

    #[test]
    fn test_rope_cache_get() {
        let rope = RoPECache::new(64, 128, 10000.0, &Device::CPU).unwrap();
        let (cos, sin) = rope.get(32).unwrap();
        assert_eq!(cos.shape(), &[32, 32]); // half_dim = 64/2 = 32
        assert_eq!(sin.shape(), &[32, 32]);
    }

    #[test]
    fn test_causal_mask_cache() {
        let cache = CausalMaskCache::new(64, &Device::CPU).unwrap();
        let mask = cache.get(8).unwrap();
        assert_eq!(mask.shape(), &[8, 8]);
    }

    #[test]
    fn test_causal_mask_values() {
        let cache = CausalMaskCache::new(4, &Device::CPU).unwrap();
        let mask = cache.get(4).unwrap();
        // Flatten to 1D to get values
        let mask_flat = mask.reshape(&[16]).unwrap();
        let values = mask_flat.to_vec_f32().unwrap();

        // Position 0 can only attend to position 0
        assert_eq!(values[0], 0.0);
        assert!(values[1].is_infinite() && values[1] < 0.0);

        // Position 3 can attend to all positions 0-3
        assert_eq!(values[12], 0.0); // pos 3, attend to 0
        assert_eq!(values[13], 0.0); // pos 3, attend to 1
        assert_eq!(values[14], 0.0); // pos 3, attend to 2
        assert_eq!(values[15], 0.0); // pos 3, attend to 3
    }

    #[test]
    fn test_sliding_window_mask() {
        let cache = SlidingWindowMaskCache::new(8, 3, &Device::CPU).unwrap();
        let mask = cache.get(8).unwrap();
        // Flatten to 1D to get values
        let mask_flat = mask.reshape(&[64]).unwrap();
        let values = mask_flat.to_vec_f32().unwrap();

        // Position 5 can attend to positions 3, 4, 5 (window_size=3)
        assert!(values[5 * 8 + 2].is_infinite()); // pos 5, can't attend to 2
        assert_eq!(values[5 * 8 + 3], 0.0); // pos 5, can attend to 3
        assert_eq!(values[5 * 8 + 4], 0.0); // pos 5, can attend to 4
        assert_eq!(values[5 * 8 + 5], 0.0); // pos 5, can attend to 5
    }
}
