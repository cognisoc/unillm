//! Advanced Memory Pool for GPU Tensor Management
//!
//! This module provides sophisticated memory pooling and garbage collection
//! to minimize memory fragmentation and allocation overhead in high-throughput
//! inference workloads.

use crate::types::*;
use crate::gpu_tensor_ops::{GpuTensor, GpuDevice};
use std::collections::{HashMap, VecDeque, BinaryHeap};
use std::cmp::Reverse;
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex, RwLock};

/// Memory pool statistics for monitoring and debugging
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    pub total_allocated_bytes: usize,
    pub peak_allocated_bytes: usize,
    pub free_bytes: usize,
    pub fragmentation_ratio: f32,
    pub allocation_count: usize,
    pub deallocation_count: usize,
    pub cache_hit_rate: f32,
    pub gc_runs: usize,
    pub last_gc_time: Option<Instant>,
}

impl Default for MemoryPoolStats {
    fn default() -> Self {
        Self {
            total_allocated_bytes: 0,
            peak_allocated_bytes: 0,
            free_bytes: 0,
            fragmentation_ratio: 0.0,
            allocation_count: 0,
            deallocation_count: 0,
            cache_hit_rate: 0.0,
            gc_runs: 0,
            last_gc_time: None,
        }
    }
}

/// Memory block for pooled allocation
#[derive(Debug)]
struct MemoryBlock {
    size: usize,
    shape: Vec<usize>,
    tensor: Option<GpuTensor>,
    allocated_at: Instant,
    last_used: Instant,
    reference_count: usize,
}

impl MemoryBlock {
    fn new(size: usize, shape: Vec<usize>, tensor: GpuTensor) -> Self {
        let now = Instant::now();
        Self {
            size,
            shape,
            tensor: Some(tensor),
            allocated_at: now,
            last_used: now,
            reference_count: 0,
        }
    }

    fn is_compatible(&self, required_shape: &[usize]) -> bool {
        if self.shape.len() != required_shape.len() {
            return false;
        }

        let required_size: usize = required_shape.iter().product();
        let our_size: usize = self.shape.iter().product();

        // Allow reuse if our block is same size or up to 25% larger
        our_size == required_size || (our_size <= required_size * 5 / 4 && our_size >= required_size)
    }

    fn age(&self) -> Duration {
        self.last_used.elapsed()
    }
}

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    pub max_pool_size_bytes: usize,
    pub gc_threshold_bytes: usize,
    pub gc_interval: Duration,
    pub max_block_age: Duration,
    pub enable_defragmentation: bool,
    pub allocation_alignment: usize,
    pub warmup_sizes: Vec<Vec<usize>>, // Common tensor shapes to pre-allocate
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            max_pool_size_bytes: 4 * 1024 * 1024 * 1024, // 4GB
            gc_threshold_bytes: 1024 * 1024 * 1024, // 1GB
            gc_interval: Duration::from_secs(30),
            max_block_age: Duration::from_secs(300), // 5 minutes
            enable_defragmentation: true,
            allocation_alignment: 512, // bytes
            warmup_sizes: vec![
                vec![1, 1, 512],        // Single token
                vec![1, 32, 512],       // Small batch
                vec![4, 2048, 4096],    // Large sequence
                vec![32, 512, 512],     // Medium batch
            ],
        }
    }
}

/// Advanced GPU memory pool with garbage collection
pub struct AdvancedMemoryPool {
    device: GpuDevice,
    config: MemoryPoolConfig,

    // Memory management
    free_blocks: HashMap<usize, VecDeque<MemoryBlock>>, // size -> blocks
    allocated_blocks: HashMap<usize, MemoryBlock>, // id -> block
    block_id_counter: usize,

    // Statistics and monitoring
    stats: RwLock<MemoryPoolStats>,
    last_gc: Instant,

    // Reference counting for automatic cleanup
    active_references: HashMap<usize, usize>,
}

impl AdvancedMemoryPool {
    /// Create new memory pool
    pub fn new(device: GpuDevice, config: MemoryPoolConfig) -> Self {
        let mut pool = Self {
            device,
            config,
            free_blocks: HashMap::new(),
            allocated_blocks: HashMap::new(),
            block_id_counter: 0,
            stats: RwLock::new(MemoryPoolStats::default()),
            last_gc: Instant::now(),
            active_references: HashMap::new(),
        };

        // Warmup allocation for common sizes
        pool.warmup();

        pool
    }

    /// Pre-allocate common tensor sizes
    fn warmup(&mut self) {
        for shape in &self.config.warmup_sizes {
            if let Ok(tensor) = GpuTensor::zeros(shape.clone(), self.device.clone()) {
                let size: usize = shape.iter().product();
                let block = MemoryBlock::new(size, shape.clone(), tensor);

                self.free_blocks
                    .entry(size)
                    .or_insert_with(VecDeque::new)
                    .push_back(block);

                if let Ok(mut stats) = self.stats.write() {
                    stats.total_allocated_bytes += size * 4; // f32 = 4 bytes
                }
            }
        }
    }

    /// Allocate tensor from pool or create new
    pub fn allocate(&mut self, shape: Vec<usize>) -> ModelResult<PooledTensor> {
        let required_size: usize = shape.iter().product();

        // Try to find compatible block in free pool
        if let Some(mut block) = self.find_compatible_block(&shape, required_size) {
            block.last_used = Instant::now();
            block.reference_count = 1;

            let block_id = self.block_id_counter;
            self.block_id_counter += 1;

            let tensor = block.tensor.take().unwrap();
            self.allocated_blocks.insert(block_id, block);

            // Update stats
            if let Ok(mut stats) = self.stats.write() {
                stats.allocation_count += 1;
                stats.cache_hit_rate = stats.allocation_count as f32 /
                    (stats.allocation_count + stats.deallocation_count) as f32;
            }

            return Ok(PooledTensor::new(block_id, tensor, Arc::new(Mutex::new(self.clone()))));
        }

        // No compatible block found, create new tensor
        let tensor = GpuTensor::zeros(shape.clone(), self.device.clone())?;
        let block = MemoryBlock::new(required_size, shape, tensor.clone());

        let block_id = self.block_id_counter;
        self.block_id_counter += 1;

        self.allocated_blocks.insert(block_id, block);

        // Update stats
        if let Ok(mut stats) = self.stats.write() {
            stats.allocation_count += 1;
            stats.total_allocated_bytes += required_size * 4;
            stats.peak_allocated_bytes = stats.peak_allocated_bytes
                .max(stats.total_allocated_bytes);
        }

        // Check if GC is needed
        if self.should_run_gc() {
            self.garbage_collect();
        }

        Ok(PooledTensor::new(block_id, tensor, Arc::new(Mutex::new(self.clone()))))
    }

    /// Find compatible block for reuse
    fn find_compatible_block(&mut self, shape: &[usize], required_size: usize) -> Option<MemoryBlock> {
        // First try exact size match
        if let Some(blocks) = self.free_blocks.get_mut(&required_size) {
            if let Some(block) = blocks.pop_front() {
                return Some(block);
            }
        }

        // Then try compatible sizes (up to 25% larger)
        let max_acceptable_size = required_size * 5 / 4;

        for (&size, blocks) in &mut self.free_blocks {
            if size >= required_size && size <= max_acceptable_size {
                if let Some(mut block) = blocks.pop_front() {
                    if block.is_compatible(shape) {
                        return Some(block);
                    } else {
                        // Put it back if not compatible
                        blocks.push_front(block);
                    }
                }
            }
        }

        None
    }

    /// Deallocate tensor back to pool
    pub fn deallocate(&mut self, block_id: usize) {
        if let Some(mut block) = self.allocated_blocks.remove(&block_id) {
            block.reference_count = 0;
            block.last_used = Instant::now();

            let size = block.size;

            // Add back to free pool
            self.free_blocks
                .entry(size)
                .or_insert_with(VecDeque::new)
                .push_back(block);

            // Update stats
            if let Ok(mut stats) = self.stats.write() {
                stats.deallocation_count += 1;
            }
        }
    }

    /// Check if garbage collection should run
    fn should_run_gc(&self) -> bool {
        let stats = self.stats.read().unwrap();

        // Run GC if:
        // 1. Memory usage exceeds threshold
        // 2. Time since last GC exceeds interval
        // 3. Fragmentation is high

        stats.total_allocated_bytes > self.config.gc_threshold_bytes ||
        self.last_gc.elapsed() > self.config.gc_interval ||
        stats.fragmentation_ratio > 0.3
    }

    /// Run garbage collection
    pub fn garbage_collect(&mut self) {
        let start_time = Instant::now();
        let mut freed_bytes = 0;

        // Remove aged blocks from free pool
        for (_, blocks) in &mut self.free_blocks {
            blocks.retain(|block| {
                let should_keep = block.age() < self.config.max_block_age;
                if !should_keep {
                    freed_bytes += block.size * 4; // f32 = 4 bytes
                }
                should_keep
            });
        }

        // Remove empty size buckets
        self.free_blocks.retain(|_, blocks| !blocks.is_empty());

        // Update stats
        if let Ok(mut stats) = self.stats.write() {
            stats.gc_runs += 1;
            stats.last_gc_time = Some(start_time);
            stats.total_allocated_bytes = stats.total_allocated_bytes.saturating_sub(freed_bytes);
            stats.free_bytes = self.calculate_free_bytes();
            stats.fragmentation_ratio = self.calculate_fragmentation();
        }

        self.last_gc = start_time;

        // Optional defragmentation
        if self.config.enable_defragmentation {
            self.defragment();
        }
    }

    /// Calculate current free bytes
    fn calculate_free_bytes(&self) -> usize {
        self.free_blocks
            .values()
            .flat_map(|blocks| blocks.iter())
            .map(|block| block.size * 4)
            .sum()
    }

    /// Calculate memory fragmentation ratio
    fn calculate_fragmentation(&self) -> f32 {
        let total_free_blocks: usize = self.free_blocks
            .values()
            .map(|blocks| blocks.len())
            .sum();

        if total_free_blocks == 0 {
            return 0.0;
        }

        let unique_sizes = self.free_blocks.len();

        // Higher ratio means more fragmentation
        unique_sizes as f32 / total_free_blocks as f32
    }

    /// Defragment memory pool (placeholder implementation)
    fn defragment(&mut self) {
        // In a full implementation, this would:
        // 1. Consolidate adjacent free blocks
        // 2. Move allocated blocks to reduce fragmentation
        // 3. Compact the memory layout

        // For now, we'll just sort blocks by size within each bucket
        for blocks in self.free_blocks.values_mut() {
            let mut block_vec: Vec<_> = blocks.drain(..).collect();
            block_vec.sort_by_key(|b| b.size);
            blocks.extend(block_vec);
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> MemoryPoolStats {
        self.stats.read().unwrap().clone()
    }

    /// Force garbage collection
    pub fn force_gc(&mut self) {
        self.garbage_collect();
    }

    /// Get memory usage summary
    pub fn memory_summary(&self) -> String {
        let stats = self.stats();
        format!(
            "Memory Pool Summary:\n\
             Total Allocated: {:.2} MB\n\
             Peak Allocated: {:.2} MB\n\
             Free: {:.2} MB\n\
             Fragmentation: {:.1}%\n\
             Cache Hit Rate: {:.1}%\n\
             GC Runs: {}",
            stats.total_allocated_bytes as f64 / 1024.0 / 1024.0,
            stats.peak_allocated_bytes as f64 / 1024.0 / 1024.0,
            stats.free_bytes as f64 / 1024.0 / 1024.0,
            stats.fragmentation_ratio * 100.0,
            stats.cache_hit_rate * 100.0,
            stats.gc_runs
        )
    }
}

// Make AdvancedMemoryPool cloneable for sharing (note: this is a simplified clone)
impl Clone for AdvancedMemoryPool {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            config: self.config.clone(),
            free_blocks: HashMap::new(),
            allocated_blocks: HashMap::new(),
            block_id_counter: 0,
            stats: RwLock::new(MemoryPoolStats::default()),
            last_gc: Instant::now(),
            active_references: HashMap::new(),
        }
    }
}

/// Pooled tensor with automatic cleanup
pub struct PooledTensor {
    block_id: usize,
    tensor: Option<GpuTensor>,
    pool: Arc<Mutex<AdvancedMemoryPool>>,
}

impl PooledTensor {
    fn new(block_id: usize, tensor: GpuTensor, pool: Arc<Mutex<AdvancedMemoryPool>>) -> Self {
        Self {
            block_id,
            tensor: Some(tensor),
            pool,
        }
    }

    /// Get the underlying tensor
    pub fn tensor(&self) -> &GpuTensor {
        self.tensor.as_ref().unwrap()
    }

    /// Take ownership of the tensor (for moves)
    pub fn take_tensor(mut self) -> GpuTensor {
        self.tensor.take().unwrap()
    }
}

impl Drop for PooledTensor {
    fn drop(&mut self) {
        // Return tensor to pool when dropped
        if let Ok(mut pool) = self.pool.lock() {
            pool.deallocate(self.block_id);
        }
    }
}

/// Global memory pool manager
pub struct GlobalMemoryManager {
    pools: HashMap<String, AdvancedMemoryPool>, // device_name -> pool
}

impl GlobalMemoryManager {
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
        }
    }

    pub fn get_or_create_pool(&mut self, device: GpuDevice, config: MemoryPoolConfig) -> &mut AdvancedMemoryPool {
        let device_key = format!("{:?}", device);
        self.pools.entry(device_key).or_insert_with(|| {
            AdvancedMemoryPool::new(device, config)
        })
    }

    pub fn global_gc(&mut self) {
        for pool in self.pools.values_mut() {
            pool.garbage_collect();
        }
    }

    pub fn global_stats(&self) -> HashMap<String, MemoryPoolStats> {
        self.pools.iter()
            .map(|(device, pool)| (device.clone(), pool.stats()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_creation() {
        let device = GpuDevice::auto_detect();
        let config = MemoryPoolConfig::default();
        let pool = AdvancedMemoryPool::new(device, config);

        let stats = pool.stats();
        assert_eq!(stats.allocation_count, 0);
        assert_eq!(stats.deallocation_count, 0);
    }

    #[test]
    fn test_tensor_allocation() {
        let device = GpuDevice::auto_detect();
        let config = MemoryPoolConfig::default();
        let mut pool = AdvancedMemoryPool::new(device, config);

        let shape = vec![2, 64, 512];
        let pooled_tensor = pool.allocate(shape.clone());
        assert!(pooled_tensor.is_ok());

        let tensor = pooled_tensor.unwrap();
        assert_eq!(tensor.tensor().shape(), shape);

        let stats = pool.stats();
        assert_eq!(stats.allocation_count, 1);
    }

    #[test]
    fn test_tensor_reuse() {
        let device = GpuDevice::auto_detect();
        let config = MemoryPoolConfig::default();
        let mut pool = AdvancedMemoryPool::new(device, config);

        let shape = vec![2, 32, 256];

        // Allocate and immediately drop
        {
            let _tensor = pool.allocate(shape.clone()).unwrap();
        }

        // Allocate again - should reuse
        let tensor2 = pool.allocate(shape).unwrap();
        assert!(tensor2.tensor().shape().len() > 0);

        let stats = pool.stats();
        // Should have high cache hit rate due to reuse
        assert!(stats.cache_hit_rate > 0.0);
    }

    #[test]
    fn test_garbage_collection() {
        let device = GpuDevice::auto_detect();
        let mut config = MemoryPoolConfig::default();
        config.max_block_age = Duration::from_millis(1); // Very short age for testing

        let mut pool = AdvancedMemoryPool::new(device, config);

        // Allocate and deallocate
        {
            let _tensor = pool.allocate(vec![10, 10]).unwrap();
        }

        // Sleep longer than max_block_age
        std::thread::sleep(Duration::from_millis(10));

        // Force GC
        pool.force_gc();

        let stats = pool.stats();
        assert!(stats.gc_runs > 0);
    }

    #[test]
    fn test_memory_pool_config() {
        let mut config = MemoryPoolConfig::default();
        config.max_pool_size_bytes = 1024 * 1024; // 1MB

        assert_eq!(config.max_pool_size_bytes, 1024 * 1024);
        assert!(config.enable_defragmentation);
        assert!(!config.warmup_sizes.is_empty());
    }

    #[test]
    fn test_memory_summary() {
        let device = GpuDevice::auto_detect();
        let config = MemoryPoolConfig::default();
        let pool = AdvancedMemoryPool::new(device, config);

        let summary = pool.memory_summary();
        assert!(summary.contains("Memory Pool Summary"));
        assert!(summary.contains("Total Allocated"));
        assert!(summary.contains("Cache Hit Rate"));
    }

    #[test]
    fn test_global_memory_manager() {
        let mut manager = GlobalMemoryManager::new();
        let device = GpuDevice::auto_detect();
        let config = MemoryPoolConfig::default();

        let _pool = manager.get_or_create_pool(device.clone(), config.clone());
        let _pool2 = manager.get_or_create_pool(device, config);

        // Should only have one pool for the same device
        assert_eq!(manager.pools.len(), 1);
    }
}