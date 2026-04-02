//! KV Cache for efficient autoregressive generation
//!
//! This module provides a pre-allocated CPU-focused KV cache that stores computed
//! Key and Value tensors to avoid recomputation during text generation.
//!
//! # How it works
//!
//! During autoregressive generation, each transformer layer computes Key and Value
//! tensors that are reused when generating subsequent tokens. Without caching,
//! the model must recompute K,V for all previous tokens on each generation step,
//! leading to O(n^2) complexity.
//!
//! With KV caching:
//! - Prefill: Process entire prompt, cache K,V for each layer
//! - Decode: For each new token, only compute K,V for that token and append
//!   to cached values using slice_set (no allocation!)
//!
//! This reduces generation to O(n) complexity, providing 50-100x speedup.

use anyhow::Result;
use crate::tensor_core::Tensor;

/// Default maximum sequence length for pre-allocation
pub const DEFAULT_MAX_SEQ_LEN: usize = 2048;

/// Per-layer KV cache storing Key and Value tensors with pre-allocated buffers
#[derive(Debug, Clone)]
pub struct LayerKVCache {
    /// Pre-allocated Key buffer: [batch, num_kv_heads, max_seq_len, head_dim]
    k_buffer: Option<candle_core::Tensor>,
    /// Pre-allocated Value buffer: [batch, num_kv_heads, max_seq_len, head_dim]
    v_buffer: Option<candle_core::Tensor>,
    /// Current filled length (how many positions are used)
    current_len: usize,
    /// Maximum sequence length (buffer capacity)
    max_seq_len: usize,
}

impl LayerKVCache {
    /// Create an empty layer cache with default max sequence length
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_MAX_SEQ_LEN)
    }

    /// Create a layer cache with specified maximum sequence length
    pub fn with_capacity(max_seq_len: usize) -> Self {
        Self {
            k_buffer: None,
            v_buffer: None,
            current_len: 0,
            max_seq_len,
        }
    }

    /// Check if this layer has cached values
    pub fn is_empty(&self) -> bool {
        self.current_len == 0
    }

    /// Clear cached values (keeps buffers allocated for reuse)
    pub fn clear(&mut self) {
        self.current_len = 0;
        // Note: We keep buffers allocated for reuse
    }

    /// Fully deallocate buffers
    pub fn deallocate(&mut self) {
        self.k_buffer = None;
        self.v_buffer = None;
        self.current_len = 0;
    }

    /// Get the current cached sequence length
    pub fn seq_len(&self) -> usize {
        self.current_len
    }

    /// Get maximum sequence length (buffer capacity)
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Append new K,V tensors to the cache using slice_set (zero-allocation)
    ///
    /// # Arguments
    /// * `new_k` - New key tensor: [batch, num_kv_heads, new_seq_len, head_dim]
    /// * `new_v` - New value tensor: [batch, num_kv_heads, new_seq_len, head_dim]
    ///
    /// # Returns
    /// Ok(()) on success, Err if buffer overflow
    pub fn append(&mut self, new_k: &candle_core::Tensor, new_v: &candle_core::Tensor) -> Result<()> {
        let new_len = new_k.dims()[2];

        // Lazy allocation on first append
        if self.k_buffer.is_none() {
            let shape = new_k.dims();
            // [batch, num_kv_heads, max_seq_len, head_dim]
            let buffer_shape = (shape[0], shape[1], self.max_seq_len, shape[3]);
            self.k_buffer = Some(candle_core::Tensor::zeros(
                buffer_shape,
                new_k.dtype(),
                new_k.device(),
            )?);
            self.v_buffer = Some(candle_core::Tensor::zeros(
                buffer_shape,
                new_v.dtype(),
                new_v.device(),
            )?);
        }

        // Check for buffer overflow
        if self.current_len + new_len > self.max_seq_len {
            anyhow::bail!(
                "KV cache overflow: current_len={}, new_len={}, max_seq_len={}",
                self.current_len,
                new_len,
                self.max_seq_len
            );
        }

        // Use slice_set for zero-allocation append
        // slice_set(&src, dim, start) writes src into buffer at position start on dimension dim
        self.k_buffer
            .as_mut()
            .unwrap()
            .slice_set(new_k, 2, self.current_len)?;
        self.v_buffer
            .as_mut()
            .unwrap()
            .slice_set(new_v, 2, self.current_len)?;

        self.current_len += new_len;
        Ok(())
    }

    /// Get view of the filled portion of K,V buffers
    ///
    /// Returns (K, V) tensors narrowed to current sequence length
    pub fn get_kv(&self) -> Option<(candle_core::Tensor, candle_core::Tensor)> {
        match (&self.k_buffer, &self.v_buffer) {
            (Some(k), Some(v)) if self.current_len > 0 => {
                // narrow(dim, start, len) returns a view, no copy
                let k_view = k.narrow(2, 0, self.current_len).ok()?;
                let v_view = v.narrow(2, 0, self.current_len).ok()?;
                Some((k_view, v_view))
            }
            _ => None,
        }
    }

    // === Legacy compatibility methods ===

    /// Get cached K tensor (legacy interface)
    #[deprecated(note = "Use get_kv() instead for better performance")]
    pub fn k(&self) -> Option<Tensor> {
        self.get_kv().map(|(k, _)| Tensor::from_candle(k))
    }

    /// Get cached V tensor (legacy interface)
    #[deprecated(note = "Use get_kv() instead for better performance")]
    pub fn v(&self) -> Option<Tensor> {
        self.get_kv().map(|(_, v)| Tensor::from_candle(v))
    }

    /// Set K tensor (legacy interface - converts to append)
    #[deprecated(note = "Use append() instead for better performance")]
    pub fn set_k(&mut self, k: Tensor) {
        if let Ok(candle_k) = k.to_candle() {
            self.k_buffer = Some(candle_k);
            if let Some(ref k) = self.k_buffer {
                self.current_len = k.dims().get(2).copied().unwrap_or(0);
            }
        }
    }

    /// Set V tensor (legacy interface)
    #[deprecated(note = "Use append() instead for better performance")]
    pub fn set_v(&mut self, v: Tensor) {
        if let Ok(candle_v) = v.to_candle() {
            self.v_buffer = Some(candle_v);
        }
    }
}

impl Default for LayerKVCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Full model KV cache containing all layers
#[derive(Debug)]
pub struct KVCache {
    /// Per-layer caches
    layers: Vec<LayerKVCache>,
    /// Current total sequence length in cache
    seq_len: usize,
    /// Maximum sequence length for all layers
    max_seq_len: usize,
}

impl KVCache {
    /// Create a new KV cache with default max sequence length
    pub fn new(num_layers: usize) -> Self {
        Self::with_capacity(num_layers, DEFAULT_MAX_SEQ_LEN)
    }

    /// Create a new KV cache with specified max sequence length
    pub fn with_capacity(num_layers: usize, max_seq_len: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| LayerKVCache::with_capacity(max_seq_len))
                .collect(),
            seq_len: 0,
            max_seq_len,
        }
    }

    /// Get the number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get mutable reference to a specific layer's cache
    pub fn layer_mut(&mut self, layer_idx: usize) -> &mut LayerKVCache {
        &mut self.layers[layer_idx]
    }

    /// Get reference to a specific layer's cache
    pub fn layer(&self, layer_idx: usize) -> &LayerKVCache {
        &self.layers[layer_idx]
    }

    /// Get current cached sequence length
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Update the cached sequence length
    pub fn set_seq_len(&mut self, seq_len: usize) {
        self.seq_len = seq_len;
    }

    /// Get maximum sequence length
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Clear all cached values (keeps buffers allocated for reuse)
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
        self.seq_len = 0;
    }

    /// Fully deallocate all buffers
    pub fn deallocate(&mut self) {
        for layer in &mut self.layers {
            layer.deallocate();
        }
        self.seq_len = 0;
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_creation() {
        let cache = KVCache::new(32);
        assert_eq!(cache.num_layers(), 32);
        assert_eq!(cache.seq_len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.max_seq_len(), DEFAULT_MAX_SEQ_LEN);
    }

    #[test]
    fn test_kv_cache_with_capacity() {
        let cache = KVCache::with_capacity(16, 4096);
        assert_eq!(cache.num_layers(), 16);
        assert_eq!(cache.max_seq_len(), 4096);
    }

    #[test]
    fn test_layer_cache_empty() {
        let layer_cache = LayerKVCache::new();
        assert!(layer_cache.is_empty());
        assert_eq!(layer_cache.seq_len(), 0);
        assert!(layer_cache.get_kv().is_none());
    }

    #[test]
    fn test_layer_cache_append() {
        let mut layer_cache = LayerKVCache::with_capacity(512);

        // Create test tensors: [batch=1, heads=4, seq=10, head_dim=64]
        let k = candle_core::Tensor::zeros(
            (1, 4, 10, 64),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let v = candle_core::Tensor::zeros(
            (1, 4, 10, 64),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();

        // Append first batch
        layer_cache.append(&k, &v).unwrap();
        assert_eq!(layer_cache.seq_len(), 10);
        assert!(!layer_cache.is_empty());

        // Get K,V view
        let (k_view, v_view) = layer_cache.get_kv().unwrap();
        assert_eq!(k_view.dims(), &[1, 4, 10, 64]);
        assert_eq!(v_view.dims(), &[1, 4, 10, 64]);

        // Append more tokens
        let k2 = candle_core::Tensor::zeros(
            (1, 4, 5, 64),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let v2 = candle_core::Tensor::zeros(
            (1, 4, 5, 64),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();

        layer_cache.append(&k2, &v2).unwrap();
        assert_eq!(layer_cache.seq_len(), 15);

        let (k_view, v_view) = layer_cache.get_kv().unwrap();
        assert_eq!(k_view.dims(), &[1, 4, 15, 64]);
        assert_eq!(v_view.dims(), &[1, 4, 15, 64]);
    }

    #[test]
    fn test_layer_cache_clear() {
        let mut layer_cache = LayerKVCache::with_capacity(512);

        let k = candle_core::Tensor::zeros(
            (1, 4, 10, 64),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let v = candle_core::Tensor::zeros(
            (1, 4, 10, 64),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();

        layer_cache.append(&k, &v).unwrap();
        assert_eq!(layer_cache.seq_len(), 10);

        // Clear resets length but keeps buffers
        layer_cache.clear();
        assert_eq!(layer_cache.seq_len(), 0);
        assert!(layer_cache.is_empty());
        assert!(layer_cache.k_buffer.is_some()); // Buffer still allocated

        // Deallocate removes buffers
        layer_cache.deallocate();
        assert!(layer_cache.k_buffer.is_none());
    }

    #[test]
    fn test_kv_cache_clear() {
        let mut cache = KVCache::new(4);
        cache.set_seq_len(100);
        assert_eq!(cache.seq_len(), 100);

        cache.clear();
        assert_eq!(cache.seq_len(), 0);
        assert!(cache.is_empty());
    }
}
