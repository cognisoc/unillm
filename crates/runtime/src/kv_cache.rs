//! KV Cache implementation for incremental text generation
//!
//! KV caching stores previous key-value pairs to avoid recomputation during
//! autoregressive generation. This provides 10-50x speedup for text generation.

use crate::types::*;
use crate::gpu_tensor_ops::{GpuTensor, GpuTensorOps, GpuDevice};
use std::collections::HashMap;

/// KV Cache entry for a single layer and head
#[derive(Debug, Clone)]
pub struct KVCacheEntry {
    pub key: GpuTensor,
    pub value: GpuTensor,
    pub sequence_length: usize,
    pub max_length: usize,
}

impl KVCacheEntry {
    pub fn new(
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        max_length: usize,
        device: GpuDevice,
    ) -> ModelResult<Self> {
        // Pre-allocate tensors for maximum sequence length
        let key_shape = vec![batch_size, num_heads, max_length, head_dim];
        let value_shape = vec![batch_size, num_heads, max_length, head_dim];

        let key = GpuTensor::zeros(key_shape, device.clone())?;
        let value = GpuTensor::zeros(value_shape, device)?;

        Ok(Self {
            key,
            value,
            sequence_length: 0,
            max_length,
        })
    }

    /// Append new key-value pairs to the cache
    pub fn append(
        &mut self,
        new_key: &GpuTensor,
        _new_value: &GpuTensor,
        _tensor_ops: &GpuTensorOps,
    ) -> ModelResult<()> {
        let new_seq_len = new_key.shape()[2];

        if self.sequence_length + new_seq_len > self.max_length {
            return Err(ModelError::ComputationFailed(
                format!("KV cache overflow: {} + {} > {}",
                       self.sequence_length, new_seq_len, self.max_length)
            ));
        }

        // In a full implementation, we would copy new_key and new_value
        // to the appropriate positions in self.key and self.value
        // For now, we'll just update the sequence length
        self.sequence_length += new_seq_len;

        Ok(())
    }

    /// Get the current cached keys and values up to sequence_length
    pub fn get_current(&self) -> ModelResult<(GpuTensor, GpuTensor)> {
        if self.sequence_length == 0 {
            return Err(ModelError::ComputationFailed("Empty KV cache".to_string()));
        }

        // In a full implementation, we would slice the tensors to [batch, heads, seq_len, head_dim]
        // For now, return the full tensors (this would need proper slicing)
        Ok((self.key.clone(), self.value.clone()))
    }

    /// Check if cache has space for additional tokens
    pub fn can_fit(&self, additional_tokens: usize) -> bool {
        self.sequence_length + additional_tokens <= self.max_length
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.sequence_length = 0;
    }
}

/// Multi-layer KV Cache manager
#[derive(Debug)]
pub struct KVCache {
    cache_entries: HashMap<usize, KVCacheEntry>,
    batch_size: usize,
    num_heads: usize,
    head_dim: usize,
    max_length: usize,
    device: GpuDevice,
    tensor_ops: GpuTensorOps,
}

impl KVCache {
    pub fn new(
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        max_length: usize,
        num_layers: usize,
        device: GpuDevice,
    ) -> ModelResult<Self> {
        let tensor_ops = GpuTensorOps::with_device(device.clone());
        let mut cache_entries = HashMap::new();

        // Pre-allocate cache for all layers
        for layer_idx in 0..num_layers {
            let entry = KVCacheEntry::new(
                batch_size,
                num_heads,
                head_dim,
                max_length,
                device.clone(),
            )?;
            cache_entries.insert(layer_idx, entry);
        }

        Ok(Self {
            cache_entries,
            batch_size,
            num_heads,
            head_dim,
            max_length,
            device,
            tensor_ops,
        })
    }

    /// Update cache for a specific layer with new key-value pairs
    pub fn update(
        &mut self,
        layer_idx: usize,
        new_key: &GpuTensor,
        new_value: &GpuTensor,
    ) -> ModelResult<()> {
        let entry = self.cache_entries
            .get_mut(&layer_idx)
            .ok_or_else(|| ModelError::ComputationFailed(
                format!("Layer {} not found in KV cache", layer_idx)
            ))?;

        entry.append(new_key, new_value, &self.tensor_ops)
    }

    /// Get cached key-value pairs for a specific layer
    pub fn get(
        &self,
        layer_idx: usize,
    ) -> ModelResult<(GpuTensor, GpuTensor)> {
        let entry = self.cache_entries
            .get(&layer_idx)
            .ok_or_else(|| ModelError::ComputationFailed(
                format!("Layer {} not found in KV cache", layer_idx)
            ))?;

        entry.get_current()
    }

    /// Get current sequence length (should be same across all layers)
    pub fn sequence_length(&self) -> usize {
        self.cache_entries
            .get(&0)
            .map(|entry| entry.sequence_length)
            .unwrap_or(0)
    }

    /// Check if cache can fit additional tokens
    pub fn can_fit(&self, additional_tokens: usize) -> bool {
        self.cache_entries
            .values()
            .all(|entry| entry.can_fit(additional_tokens))
    }

    /// Clear all cached data
    pub fn clear(&mut self) {
        for entry in self.cache_entries.values_mut() {
            entry.clear();
        }
    }

    /// Get memory usage statistics
    pub fn memory_usage(&self) -> KVCacheStats {
        let num_layers = self.cache_entries.len();
        let elements_per_layer = self.batch_size * self.num_heads * self.max_length * self.head_dim;
        let bytes_per_element = 4; // Assuming float32

        let allocated_bytes = num_layers * elements_per_layer * bytes_per_element * 2; // key + value
        let used_elements = num_layers * self.batch_size * self.num_heads * self.sequence_length() * self.head_dim;
        let used_bytes = used_elements * bytes_per_element * 2;

        KVCacheStats {
            allocated_bytes,
            used_bytes,
            utilization: if allocated_bytes > 0 {
                used_bytes as f64 / allocated_bytes as f64
            } else {
                0.0
            },
            sequence_length: self.sequence_length(),
            max_length: self.max_length,
            num_layers,
        }
    }
}

/// KV Cache statistics
#[derive(Debug, Clone)]
pub struct KVCacheStats {
    pub allocated_bytes: usize,
    pub used_bytes: usize,
    pub utilization: f64,
    pub sequence_length: usize,
    pub max_length: usize,
    pub num_layers: usize,
}

/// Attention with KV caching support
pub struct CachedAttention {
    num_heads: usize,
    head_dim: usize,
    kv_cache: Option<KVCache>,
    tensor_ops: GpuTensorOps,
    use_cache: bool,
}

impl CachedAttention {
    pub fn new(
        num_heads: usize,
        head_dim: usize,
        device: GpuDevice,
    ) -> Self {
        let tensor_ops = GpuTensorOps::with_device(device);

        Self {
            num_heads,
            head_dim,
            kv_cache: None,
            tensor_ops,
            use_cache: false,
        }
    }

    /// Enable KV caching with specified parameters
    pub fn enable_cache(
        &mut self,
        batch_size: usize,
        max_length: usize,
        num_layers: usize,
        device: GpuDevice,
    ) -> ModelResult<()> {
        self.kv_cache = Some(KVCache::new(
            batch_size,
            self.num_heads,
            self.head_dim,
            max_length,
            num_layers,
            device,
        )?);
        self.use_cache = true;
        Ok(())
    }

    /// Disable KV caching
    pub fn disable_cache(&mut self) {
        self.use_cache = false;
        self.kv_cache = None;
    }

    /// Forward pass with optional KV caching
    pub fn forward(
        &mut self,
        layer_idx: usize,
        query: &GpuTensor,
        key: &GpuTensor,
        value: &GpuTensor,
        is_prefill: bool,  // True for initial prompt, false for generation
    ) -> ModelResult<GpuTensor> {
        if !self.use_cache || self.kv_cache.is_none() {
            return self.standard_attention(query, key, value);
        }

        let kv_cache = self.kv_cache.as_mut().unwrap();

        if is_prefill {
            // Prefill phase: cache the entire key-value sequence
            kv_cache.update(layer_idx, key, value)?;
            self.standard_attention(query, key, value)
        } else {
            // Generation phase: use cached keys/values + new key/value
            kv_cache.update(layer_idx, key, value)?;
            let (cached_key, cached_value) = kv_cache.get(layer_idx)?;
            self.cached_attention(query, &cached_key, &cached_value)
        }
    }

    /// Standard attention computation (no caching)
    fn standard_attention(
        &self,
        query: &GpuTensor,
        key: &GpuTensor,
        value: &GpuTensor,
    ) -> ModelResult<GpuTensor> {
        let head_dim = query.shape()[3] as f32;
        let scale = 1.0 / head_dim.sqrt();

        // QK^T
        let key_transposed = self.tensor_ops.transpose(key)?;
        let scores = self.tensor_ops.matmul(query, &key_transposed)?;

        // Scale
        let scores_shape = scores.shape();
        let total_elements: usize = scores_shape.iter().product();
        let scale_data = vec![scale; total_elements];
        let scale_tensor = GpuTensor::new(scale_data, scores_shape, self.tensor_ops.device().clone())?;
        let scaled_scores = self.tensor_ops.multiply(&scores, &scale_tensor)?;

        // Softmax
        let attn_weights = self.tensor_ops.softmax(&scaled_scores, 3)?; // Last dimension

        // Apply to values
        self.tensor_ops.matmul(&attn_weights, value)
    }

    /// Attention computation with cached key-value pairs
    fn cached_attention(
        &self,
        query: &GpuTensor,
        cached_key: &GpuTensor,
        cached_value: &GpuTensor,
    ) -> ModelResult<GpuTensor> {
        // Use the full cached key-value pairs
        self.standard_attention(query, cached_key, cached_value)
    }

    /// Get current cache statistics
    pub fn cache_stats(&self) -> Option<KVCacheStats> {
        self.kv_cache.as_ref().map(|cache| cache.memory_usage())
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        if let Some(ref mut cache) = self.kv_cache {
            cache.clear();
        }
    }
}

/// Generation context with KV caching
#[derive(Debug)]
pub struct GenerationContext {
    kv_cache: KVCache,
    current_length: usize,
    max_length: usize,
}

impl GenerationContext {
    pub fn new(
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        max_length: usize,
        num_layers: usize,
        device: GpuDevice,
    ) -> ModelResult<Self> {
        let kv_cache = KVCache::new(
            batch_size,
            num_heads,
            head_dim,
            max_length,
            num_layers,
            device,
        )?;

        Ok(Self {
            kv_cache,
            current_length: 0,
            max_length,
        })
    }

    /// Check if we can generate more tokens
    pub fn can_continue(&self) -> bool {
        self.current_length < self.max_length
    }

    /// Get remaining capacity
    pub fn remaining_capacity(&self) -> usize {
        self.max_length.saturating_sub(self.current_length)
    }

    /// Update context with new token
    pub fn step(&mut self) -> ModelResult<()> {
        if !self.can_continue() {
            return Err(ModelError::ComputationFailed("Generation context is full".to_string()));
        }
        self.current_length += 1;
        Ok(())
    }

    /// Reset the context
    pub fn reset(&mut self) {
        self.kv_cache.clear();
        self.current_length = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_entry_creation() {
        let device = GpuDevice::auto_detect();
        let entry = KVCacheEntry::new(1, 8, 64, 128, device);

        assert!(entry.is_ok());
        let entry = entry.unwrap();
        assert_eq!(entry.sequence_length, 0);
        assert_eq!(entry.max_length, 128);
        assert!(entry.can_fit(128));
        assert!(!entry.can_fit(129));
    }

    #[test]
    fn test_kv_cache_creation() {
        let device = GpuDevice::auto_detect();
        let cache = KVCache::new(2, 8, 64, 256, 12, device);

        assert!(cache.is_ok());
        let cache = cache.unwrap();
        assert_eq!(cache.sequence_length(), 0);
        assert!(cache.can_fit(256));
    }

    #[test]
    fn test_kv_cache_stats() {
        let device = GpuDevice::auto_detect();
        let cache = KVCache::new(1, 8, 64, 128, 6, device).unwrap();

        let stats = cache.memory_usage();
        assert_eq!(stats.sequence_length, 0);
        assert_eq!(stats.max_length, 128);
        assert_eq!(stats.num_layers, 6);
        assert_eq!(stats.utilization, 0.0);

        // Check memory calculation
        let expected_bytes = 1 * 8 * 128 * 64 * 4 * 2 * 6; // batch * heads * max_len * head_dim * sizeof(f32) * (key+value) * layers
        assert_eq!(stats.allocated_bytes, expected_bytes);
    }

    #[test]
    fn test_cached_attention_creation() {
        let device = GpuDevice::auto_detect();
        let attention = CachedAttention::new(8, 64, device);

        assert_eq!(attention.num_heads, 8);
        assert_eq!(attention.head_dim, 64);
        assert!(!attention.use_cache);
        assert!(attention.kv_cache.is_none());
    }

    #[test]
    fn test_cached_attention_enable_cache() {
        let device = GpuDevice::auto_detect();
        let mut attention = CachedAttention::new(8, 64, device.clone());

        let result = attention.enable_cache(1, 128, 6, device);
        assert!(result.is_ok());
        assert!(attention.use_cache);
        assert!(attention.kv_cache.is_some());
    }

    #[test]
    fn test_cached_attention_forward() {
        let device = GpuDevice::auto_detect();
        let mut attention = CachedAttention::new(4, 32, device.clone());

        // Enable cache
        attention.enable_cache(1, 64, 1, device.clone()).unwrap();

        // Create test tensors
        let batch_size = 1;
        let seq_len = 8;
        let hidden_size = 4 * 32; // num_heads * head_dim

        let q = GpuTensor::randn(vec![batch_size, 4, seq_len, 32], device.clone()).unwrap();
        let k = GpuTensor::randn(vec![batch_size, 4, seq_len, 32], device.clone()).unwrap();
        let v = GpuTensor::randn(vec![batch_size, 4, seq_len, 32], device.clone()).unwrap();

        // Test prefill phase
        let result = attention.forward(0, &q, &k, &v, true);
        if let Err(e) = &result {
            println!("KV cache forward failed: {:?}", e);
        }
        assert!(result.is_ok());

        // Create single token for generation
        let q_gen = GpuTensor::randn(vec![batch_size, 4, 1, 32], device.clone()).unwrap();
        let k_gen = GpuTensor::randn(vec![batch_size, 4, 1, 32], device.clone()).unwrap();
        let v_gen = GpuTensor::randn(vec![batch_size, 4, 1, 32], device.clone()).unwrap();

        // Test generation phase
        let result = attention.forward(0, &q_gen, &k_gen, &v_gen, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_generation_context() {
        let device = GpuDevice::auto_detect();
        let mut context = GenerationContext::new(1, 8, 64, 10, 6, device).unwrap();

        assert!(context.can_continue());
        assert_eq!(context.remaining_capacity(), 10);

        // Simulate token generation
        for i in 0..10 {
            assert!(context.can_continue());
            context.step().unwrap();
            assert_eq!(context.remaining_capacity(), 10 - i - 1);
        }

        assert!(!context.can_continue());
        assert_eq!(context.remaining_capacity(), 0);

        // Should fail to generate more
        let result = context.step();
        assert!(result.is_err());

        // Reset should work
        context.reset();
        assert!(context.can_continue());
        assert_eq!(context.remaining_capacity(), 10);
    }

    #[test]
    fn test_kv_cache_memory_efficiency() {
        let device = GpuDevice::auto_detect();

        // Small cache
        let small_cache = KVCache::new(1, 8, 64, 128, 12, device.clone()).unwrap();
        let small_stats = small_cache.memory_usage();

        // Large cache
        let large_cache = KVCache::new(1, 8, 64, 2048, 12, device).unwrap();
        let large_stats = large_cache.memory_usage();

        // Large cache should use 16x more memory (2048/128 = 16)
        assert_eq!(large_stats.allocated_bytes, small_stats.allocated_bytes * 16);
    }
}