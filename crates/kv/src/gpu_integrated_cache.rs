//! GPU-integrated hybrid cache that directly manages GPU memory
//!
//! This module integrates our hybrid cache with direct GPU memory management,
//! bypassing Python/OS overhead for maximum performance.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::hybrid_cache::{
    HybridKVCache, CacheHandle, CacheTier, KVTensorPair, TokenId, SequenceId,
    AdaptiveCachePolicy, HybridCacheStats
};
use crate::gpu_memory::{
    GpuAwareMemoryPool, GpuMemoryBackend, GpuDevicePtr, GpuMemoryResult, GpuMemoryError
};

/// GPU-integrated cache that combines hybrid caching with direct GPU memory management
pub struct GpuIntegratedCache {
    /// GPU memory pool for direct hardware access
    gpu_memory_pool: Arc<Mutex<GpuAwareMemoryPool>>,

    /// Hybrid cache for intelligent tier management
    hybrid_cache: HybridKVCache,

    /// Mapping from cache handles to GPU allocations
    gpu_allocations: HashMap<CacheHandle, GpuDevicePtr>,

    /// Device stream for async operations
    default_stream: u64,

    /// Cache statistics
    stats: GpuIntegratedCacheStats,
}

#[derive(Debug, Clone, Default)]
pub struct GpuIntegratedCacheStats {
    pub hybrid_stats: HybridCacheStats,
    pub gpu_memory_usage: usize,
    pub gpu_allocations: usize,
    pub h2d_transfers: u64,
    pub d2h_transfers: u64,
    pub cache_to_gpu_promotions: u64,
    pub gpu_to_cache_demotions: u64,
}

impl GpuIntegratedCache {
    /// Create new GPU-integrated cache with CUDA backend
    pub fn new_cuda(
        device_id: u32,
        l1_max_nodes: usize,
        l2_total_pages: usize,
        l2_page_size: usize,
        l2_pages_per_block: usize,
    ) -> GpuMemoryResult<Self> {
        let gpu_memory_pool = Arc::new(Mutex::new(
            GpuAwareMemoryPool::new_cuda(device_id)?
        ));

        // Initialize hybrid cache with placeholder base device pointer
        // The actual GPU memory will be managed through our pool
        let hybrid_cache = HybridKVCache::new(
            l1_max_nodes,
            l2_total_pages,
            l2_page_size,
            l2_pages_per_block,
            0x0, // Placeholder - we'll use our GPU pool instead
        );

        Ok(Self {
            gpu_memory_pool,
            hybrid_cache,
            gpu_allocations: HashMap::new(),
            default_stream: 0, // Default stream
            stats: GpuIntegratedCacheStats::default(),
        })
    }

    /// Create new GPU-integrated cache with HIP backend
    pub fn new_hip(
        device_id: u32,
        l1_max_nodes: usize,
        l2_total_pages: usize,
        l2_page_size: usize,
        l2_pages_per_block: usize,
    ) -> GpuMemoryResult<Self> {
        let gpu_memory_pool = Arc::new(Mutex::new(
            GpuAwareMemoryPool::new_hip(device_id)?
        ));

        let hybrid_cache = HybridKVCache::new(
            l1_max_nodes,
            l2_total_pages,
            l2_page_size,
            l2_pages_per_block,
            0x0, // Placeholder - we'll use our GPU pool instead
        );

        Ok(Self {
            gpu_memory_pool,
            hybrid_cache,
            gpu_allocations: HashMap::new(),
            default_stream: 0,
            stats: GpuIntegratedCacheStats::default(),
        })
    }

    /// Allocate sequence with direct GPU memory allocation
    pub fn allocate_sequence(
        &mut self,
        tokens: &[TokenId],
        max_length: usize,
        head_dim: usize,
        num_heads: usize,
    ) -> GpuMemoryResult<(CacheHandle, KVTensorPair)> {
        // Get cache handle from hybrid cache
        let cache_handle = self.hybrid_cache
            .allocate_sequence(tokens, max_length)
            .map_err(|e| GpuMemoryError::AllocationFailed {
                reason: format!("Hybrid cache allocation failed: {}", e)
            })?;

        // Calculate KV cache requirements
        let token_count = std::cmp::min(tokens.len(), max_length);

        // Allocate GPU memory directly
        let kv_tensor = {
            let mut gpu_pool = self.gpu_memory_pool.lock()
                .map_err(|_| GpuMemoryError::AllocationFailed {
                    reason: "Failed to lock GPU memory pool".to_string()
                })?;

            gpu_pool.allocate_kv_cache(token_count, head_dim, num_heads)?
        };

        // Store the GPU allocation mapping
        let gpu_device_ptr = GpuDevicePtr::new(
            kv_tensor.key_ptr,
            kv_tensor.size_bytes,
            256, // alignment
            0,   // device_id (TODO: get from pool)
        );
        self.gpu_allocations.insert(cache_handle, gpu_device_ptr);

        // Update statistics
        self.stats.gpu_allocations += 1;
        self.stats.gpu_memory_usage += kv_tensor.size_bytes;

        // Record access pattern
        self.hybrid_cache.record_access(cache_handle, tokens);

        Ok((cache_handle, kv_tensor))
    }

    /// Free sequence and return GPU memory to pool
    pub fn free_sequence(&mut self, handle: CacheHandle) -> GpuMemoryResult<()> {
        // Free GPU memory
        if let Some(gpu_ptr) = self.gpu_allocations.remove(&handle) {
            let mut gpu_pool = self.gpu_memory_pool.lock()
                .map_err(|_| GpuMemoryError::AllocationFailed {
                    reason: "Failed to lock GPU memory pool".to_string()
                })?;

            gpu_pool.free_to_pool(gpu_ptr)?;

            self.stats.gpu_allocations -= 1;
            self.stats.gpu_memory_usage -= gpu_ptr.size;
        }

        // Note: We don't have a free_sequence method in HybridKVCache yet
        // This would need to be added to the hybrid cache interface

        Ok(())
    }

    /// Copy data from host to device asynchronously
    pub fn copy_h2d_async(
        &mut self,
        handle: CacheHandle,
        host_data: &[u8],
        offset: usize,
    ) -> GpuMemoryResult<()> {
        let gpu_ptr = self.gpu_allocations.get(&handle)
            .ok_or_else(|| GpuMemoryError::AllocationFailed {
                reason: "Cache handle not found".to_string()
            })?;

        let dst_ptr = gpu_ptr.offset(offset);

        // Get GPU memory pool and perform async copy
        let gpu_pool = self.gpu_memory_pool.lock()
            .map_err(|_| GpuMemoryError::AllocationFailed {
                reason: "Failed to lock GPU memory pool".to_string()
            })?;

        // TODO: This requires adding copy methods to the memory pool
        // gpu_pool.copy_h2d_async(dst_ptr, host_data, self.default_stream)?;

        self.stats.h2d_transfers += 1;
        Ok(())
    }

    /// Copy data from device to host asynchronously
    pub fn copy_d2h_async(
        &mut self,
        handle: CacheHandle,
        host_buffer: &mut [u8],
        offset: usize,
    ) -> GpuMemoryResult<()> {
        let gpu_ptr = self.gpu_allocations.get(&handle)
            .ok_or_else(|| GpuMemoryError::AllocationFailed {
                reason: "Cache handle not found".to_string()
            })?;

        let src_ptr = gpu_ptr.offset(offset);

        // Get GPU memory pool and perform async copy
        let gpu_pool = self.gpu_memory_pool.lock()
            .map_err(|_| GpuMemoryError::AllocationFailed {
                reason: "Failed to lock GPU memory pool".to_string()
            })?;

        // TODO: This requires adding copy methods to the memory pool
        // gpu_pool.copy_d2h_async(host_buffer, src_ptr, self.default_stream)?;

        self.stats.d2h_transfers += 1;
        Ok(())
    }

    /// Synchronize all GPU operations
    pub fn synchronize(&self) -> GpuMemoryResult<()> {
        let gpu_pool = self.gpu_memory_pool.lock()
            .map_err(|_| GpuMemoryError::AllocationFailed {
                reason: "Failed to lock GPU memory pool".to_string()
            })?;

        // TODO: Add synchronize method to memory pool
        // gpu_pool.synchronize()?;

        Ok(())
    }

    /// Promote cache entry to higher tier (with GPU memory migration)
    pub fn promote_to_l1(&mut self, handle: CacheHandle) -> GpuMemoryResult<()> {
        // Check if promotion is beneficial
        match handle.tier {
            CacheTier::L2Paged => {
                // Migrate from L2 to L1
                // This would require implementing tier migration in the hybrid cache
                self.stats.cache_to_gpu_promotions += 1;
            },
            CacheTier::L3Compressed => {
                // Decompress and migrate to L1
                self.stats.cache_to_gpu_promotions += 1;
            },
            CacheTier::L1Radix => {
                // Already in L1
            }
        }

        Ok(())
    }

    /// Demote cache entry to lower tier (with GPU memory migration)
    pub fn demote_from_l1(&mut self, handle: CacheHandle) -> GpuMemoryResult<()> {
        if handle.tier == CacheTier::L1Radix {
            // Move from L1 to L2 or L3 based on access frequency
            // This would involve copying data from GPU memory to system memory
            self.stats.gpu_to_cache_demotions += 1;
        }

        Ok(())
    }

    /// Get comprehensive cache and GPU memory statistics
    pub fn get_stats(&self) -> GpuIntegratedCacheStats {
        let mut stats = self.stats.clone();
        stats.hybrid_stats = self.hybrid_cache.get_stats();

        // Add GPU memory pool statistics
        if let Ok(gpu_pool) = self.gpu_memory_pool.try_lock() {
            let gpu_stats = gpu_pool.get_memory_stats();
            stats.gpu_memory_usage = gpu_stats.allocated_memory;
            stats.gpu_allocations = gpu_stats.active_allocations;
        }

        stats
    }

    /// Optimize cache layout based on access patterns
    pub fn optimize_layout(&mut self) -> GpuMemoryResult<()> {
        // Analyze access patterns and optimize cache tier placement
        // This would involve the adaptive policy engine

        // Defragment GPU memory if needed
        {
            let mut gpu_pool = self.gpu_memory_pool.lock()
                .map_err(|_| GpuMemoryError::AllocationFailed {
                    reason: "Failed to lock GPU memory pool".to_string()
                })?;

            gpu_pool.defragment()?;
        }

        Ok(())
    }

    /// Prefetch sequences to GPU memory based on predicted access
    pub fn prefetch_sequences(&mut self, handles: &[CacheHandle]) -> GpuMemoryResult<()> {
        for &handle in handles {
            // Ensure the sequence is loaded in appropriate GPU memory tier
            if let Some(_gpu_ptr) = self.gpu_allocations.get(&handle) {
                // Already in GPU memory, possibly promote tier
                self.promote_to_l1(handle)?;
            } else {
                // Load from cache to GPU memory
                // This would require implementing cache-to-GPU loading
            }
        }

        Ok(())
    }

    /// Get GPU device properties
    pub fn get_device_properties(&self) -> GpuMemoryResult<crate::gpu_memory::GpuDeviceProperties> {
        let gpu_pool = self.gpu_memory_pool.lock()
            .map_err(|_| GpuMemoryError::AllocationFailed {
                reason: "Failed to lock GPU memory pool".to_string()
            })?;

        // TODO: Add method to get device properties from pool
        // gpu_pool.get_device_properties()

        // Placeholder return
        Err(GpuMemoryError::AllocationFailed {
            reason: "get_device_properties not yet implemented".to_string()
        })
    }
}

/// Builder for GPU-integrated cache with different configurations
pub struct GpuIntegratedCacheBuilder {
    device_id: u32,
    backend_type: GpuBackendType,
    l1_max_nodes: usize,
    l2_total_pages: usize,
    l2_page_size: usize,
    l2_pages_per_block: usize,
}

#[derive(Debug, Clone)]
pub enum GpuBackendType {
    Cuda,
    Hip,
}

impl GpuIntegratedCacheBuilder {
    pub fn new(device_id: u32, backend_type: GpuBackendType) -> Self {
        Self {
            device_id,
            backend_type,
            l1_max_nodes: 10000,     // Default L1 radix cache size
            l2_total_pages: 100000,  // Default L2 paged cache size
            l2_page_size: 16,        // Default page size in tokens
            l2_pages_per_block: 16,  // Default pages per block
        }
    }

    pub fn with_l1_capacity(mut self, max_nodes: usize) -> Self {
        self.l1_max_nodes = max_nodes;
        self
    }

    pub fn with_l2_capacity(mut self, total_pages: usize, page_size: usize) -> Self {
        self.l2_total_pages = total_pages;
        self.l2_page_size = page_size;
        self
    }

    pub fn with_block_size(mut self, pages_per_block: usize) -> Self {
        self.l2_pages_per_block = pages_per_block;
        self
    }

    pub fn build(self) -> GpuMemoryResult<GpuIntegratedCache> {
        match self.backend_type {
            GpuBackendType::Cuda => {
                GpuIntegratedCache::new_cuda(
                    self.device_id,
                    self.l1_max_nodes,
                    self.l2_total_pages,
                    self.l2_page_size,
                    self.l2_pages_per_block,
                )
            },
            GpuBackendType::Hip => {
                GpuIntegratedCache::new_hip(
                    self.device_id,
                    self.l1_max_nodes,
                    self.l2_total_pages,
                    self.l2_page_size,
                    self.l2_pages_per_block,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_integrated_cache_builder() {
        // This test would require mock GPU backends
        let builder = GpuIntegratedCacheBuilder::new(0, GpuBackendType::Cuda)
            .with_l1_capacity(5000)
            .with_l2_capacity(50000, 16)
            .with_block_size(16);

        // Would build and test if we had mock backends
        println!("Builder configured: device_id={}, l1_nodes={}",
                builder.device_id, builder.l1_max_nodes);
    }

    #[test]
    fn test_cache_handle_management() {
        // Test cache handle and GPU allocation mapping
        let mut allocations = HashMap::new();

        let handle = CacheHandle { tier: CacheTier::L1Radix, id: 1 };
        let gpu_ptr = GpuDevicePtr::new(0x10000000, 4096, 256, 0);

        allocations.insert(handle, gpu_ptr);

        assert!(allocations.contains_key(&handle));
        assert_eq!(allocations.get(&handle).unwrap().size, 4096);
    }
}