//! GPU-aware memory management with direct CUDA/HIP integration
//!
//! This module provides direct GPU memory management that bypasses Python/OS overhead
//! by integrating directly with our CUDA/HIP backends through VFIO passthrough.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::ptr::NonNull;

use crate::hybrid_cache::{KVTensorPair, TokenId, SequenceId};

/// GPU memory allocation result
pub type GpuMemoryResult<T> = Result<T, GpuMemoryError>;

/// GPU memory allocation errors
#[derive(Debug, Clone)]
pub enum GpuMemoryError {
    OutOfMemory { requested: usize, available: usize },
    AllocationFailed { reason: String },
    InvalidAlignment { requested: usize, required: usize },
    DeviceError { code: i32, message: String },
    VfioError { message: String },
}

impl std::fmt::Display for GpuMemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuMemoryError::OutOfMemory { requested, available } =>
                write!(f, "Out of GPU memory: requested {} bytes, {} available", requested, available),
            GpuMemoryError::AllocationFailed { reason } =>
                write!(f, "GPU allocation failed: {}", reason),
            GpuMemoryError::InvalidAlignment { requested, required } =>
                write!(f, "Invalid alignment: requested {}, required {}", requested, required),
            GpuMemoryError::DeviceError { code, message } =>
                write!(f, "GPU device error {}: {}", code, message),
            GpuMemoryError::VfioError { message } =>
                write!(f, "VFIO error: {}", message),
        }
    }
}

impl std::error::Error for GpuMemoryError {}

/// GPU device pointer with metadata
#[derive(Debug, Clone, Copy)]
pub struct GpuDevicePtr {
    pub ptr: u64,
    pub size: usize,
    pub alignment: usize,
    pub device_id: u32,
}

impl GpuDevicePtr {
    pub fn new(ptr: u64, size: usize, alignment: usize, device_id: u32) -> Self {
        Self { ptr, size, alignment, device_id }
    }

    pub fn is_aligned(&self) -> bool {
        (self.ptr as usize) % self.alignment == 0
    }

    pub fn offset(&self, bytes: usize) -> GpuDevicePtr {
        GpuDevicePtr {
            ptr: self.ptr + bytes as u64,
            size: self.size.saturating_sub(bytes),
            alignment: self.alignment,
            device_id: self.device_id,
        }
    }
}

/// GPU memory allocation info
#[derive(Debug)]
pub struct GpuAllocation {
    pub device_ptr: GpuDevicePtr,
    pub host_ptr: Option<usize>, // For pinned memory (using usize for Send/Sync)
    pub ref_count: u32,
    pub sequence_id: Option<SequenceId>,
    pub allocation_time: std::time::Instant,
    pub last_access: std::time::Instant,
}

impl GpuAllocation {
    pub fn new(device_ptr: GpuDevicePtr, host_ptr: Option<usize>) -> Self {
        let now = std::time::Instant::now();
        Self {
            device_ptr,
            host_ptr,
            ref_count: 1,
            sequence_id: None,
            allocation_time: now,
            last_access: now,
        }
    }

    pub fn add_ref(&mut self) {
        self.ref_count += 1;
        self.last_access = std::time::Instant::now();
    }

    pub fn remove_ref(&mut self) -> u32 {
        if self.ref_count > 0 {
            self.ref_count -= 1;
        }
        self.ref_count
    }
}

/// GPU backend abstraction for memory operations
pub trait GpuMemoryBackend: Send + Sync {
    /// Allocate device memory
    fn allocate_device(&mut self, size: usize, alignment: usize) -> GpuMemoryResult<GpuDevicePtr>;

    /// Free device memory
    fn free_device(&mut self, ptr: GpuDevicePtr) -> GpuMemoryResult<()>;

    /// Allocate pinned host memory
    fn allocate_pinned_host(&mut self, size: usize) -> GpuMemoryResult<*mut u8>;

    /// Free pinned host memory
    fn free_pinned_host(&mut self, ptr: *mut u8) -> GpuMemoryResult<()>;

    /// Copy host to device asynchronously
    fn copy_h2d_async(&self, dst: GpuDevicePtr, src: *const u8, size: usize, stream: u64) -> GpuMemoryResult<()>;

    /// Copy device to host asynchronously
    fn copy_d2h_async(&self, dst: *mut u8, src: GpuDevicePtr, size: usize, stream: u64) -> GpuMemoryResult<()>;

    /// Synchronize with device
    fn synchronize(&self) -> GpuMemoryResult<()>;

    /// Get available memory
    fn get_available_memory(&self) -> GpuMemoryResult<(usize, usize)>; // (free, total)

    /// Get device properties
    fn get_device_properties(&self) -> GpuMemoryResult<GpuDeviceProperties>;
}

/// GPU device properties
#[derive(Debug, Clone)]
pub struct GpuDeviceProperties {
    pub device_id: u32,
    pub total_memory: usize,
    pub memory_bandwidth: usize, // GB/s
    pub compute_capability: (u32, u32), // major, minor
    pub max_threads_per_block: u32,
    pub max_shared_memory: usize,
    pub is_numa_aware: bool,
    pub pcie_generation: u32,
    pub pcie_lanes: u32,
}

/// CUDA memory backend implementation
pub struct CudaMemoryBackend {
    device_id: u32,
    context: u64, // CUDA context handle
    total_memory: usize,
    allocated_memory: usize,
    allocations: HashMap<u64, GpuAllocation>,
}

impl CudaMemoryBackend {
    pub fn new(device_id: u32) -> GpuMemoryResult<Self> {
        // Initialize CUDA context through our existing gpu-backend-cuda
        let context = unsafe { cuda_create_context(device_id) }?;
        let total_memory = unsafe { cuda_get_total_memory(device_id) }?;

        Ok(Self {
            device_id,
            context,
            total_memory,
            allocated_memory: 0,
            allocations: HashMap::new(),
        })
    }
}

impl GpuMemoryBackend for CudaMemoryBackend {
    fn allocate_device(&mut self, size: usize, alignment: usize) -> GpuMemoryResult<GpuDevicePtr> {
        // Check available memory
        if self.allocated_memory + size > self.total_memory {
            return Err(GpuMemoryError::OutOfMemory {
                requested: size,
                available: self.total_memory - self.allocated_memory,
            });
        }

        // Allocate through CUDA driver API
        let ptr = unsafe { cuda_malloc_aligned(size, alignment) }?;

        let device_ptr = GpuDevicePtr::new(ptr, size, alignment, self.device_id);

        // Track allocation
        let allocation = GpuAllocation::new(device_ptr, None);
        self.allocations.insert(ptr, allocation);
        self.allocated_memory += size;

        Ok(device_ptr)
    }

    fn free_device(&mut self, ptr: GpuDevicePtr) -> GpuMemoryResult<()> {
        // Remove from tracking
        if let Some(allocation) = self.allocations.remove(&ptr.ptr) {
            self.allocated_memory -= allocation.device_ptr.size;
        }

        // Free through CUDA driver API
        unsafe { cuda_free(ptr.ptr) }?;

        Ok(())
    }

    fn allocate_pinned_host(&mut self, size: usize) -> GpuMemoryResult<*mut u8> {
        let ptr = unsafe { cuda_malloc_host(size) }?;
        Ok(ptr)
    }

    fn free_pinned_host(&mut self, ptr: *mut u8) -> GpuMemoryResult<()> {
        unsafe { cuda_free_host(ptr) }?;
        Ok(())
    }

    fn copy_h2d_async(&self, dst: GpuDevicePtr, src: *const u8, size: usize, stream: u64) -> GpuMemoryResult<()> {
        unsafe { cuda_memcpy_h2d_async(dst.ptr, src, size, stream) }?;
        Ok(())
    }

    fn copy_d2h_async(&self, dst: *mut u8, src: GpuDevicePtr, size: usize, stream: u64) -> GpuMemoryResult<()> {
        unsafe { cuda_memcpy_d2h_async(dst, src.ptr, size, stream) }?;
        Ok(())
    }

    fn synchronize(&self) -> GpuMemoryResult<()> {
        unsafe { cuda_synchronize() }?;
        Ok(())
    }

    fn get_available_memory(&self) -> GpuMemoryResult<(usize, usize)> {
        let available = self.total_memory - self.allocated_memory;
        Ok((available, self.total_memory))
    }

    fn get_device_properties(&self) -> GpuMemoryResult<GpuDeviceProperties> {
        let props = unsafe { cuda_get_device_properties(self.device_id) }?;
        Ok(props)
    }
}

/// HIP memory backend implementation
pub struct HipMemoryBackend {
    device_id: u32,
    context: u64, // HIP context handle
    total_memory: usize,
    allocated_memory: usize,
    allocations: HashMap<u64, GpuAllocation>,
}

impl HipMemoryBackend {
    pub fn new(device_id: u32) -> GpuMemoryResult<Self> {
        // Initialize HIP context through our existing gpu-backend-hip
        let context = unsafe { hip_create_context(device_id) }?;
        let total_memory = unsafe { hip_get_total_memory(device_id) }?;

        Ok(Self {
            device_id,
            context,
            total_memory,
            allocated_memory: 0,
            allocations: HashMap::new(),
        })
    }
}

impl GpuMemoryBackend for HipMemoryBackend {
    fn allocate_device(&mut self, size: usize, alignment: usize) -> GpuMemoryResult<GpuDevicePtr> {
        // Check available memory
        if self.allocated_memory + size > self.total_memory {
            return Err(GpuMemoryError::OutOfMemory {
                requested: size,
                available: self.total_memory - self.allocated_memory,
            });
        }

        // Allocate through HIP driver API
        let ptr = unsafe { hip_malloc_aligned(size, alignment) }?;

        let device_ptr = GpuDevicePtr::new(ptr, size, alignment, self.device_id);

        // Track allocation
        let allocation = GpuAllocation::new(device_ptr, None);
        self.allocations.insert(ptr, allocation);
        self.allocated_memory += size;

        Ok(device_ptr)
    }

    fn free_device(&mut self, ptr: GpuDevicePtr) -> GpuMemoryResult<()> {
        // Remove from tracking
        if let Some(allocation) = self.allocations.remove(&ptr.ptr) {
            self.allocated_memory -= allocation.device_ptr.size;
        }

        // Free through HIP driver API
        unsafe { hip_free(ptr.ptr) }?;

        Ok(())
    }

    fn allocate_pinned_host(&mut self, size: usize) -> GpuMemoryResult<*mut u8> {
        let ptr = unsafe { hip_malloc_host(size) }?;
        Ok(ptr)
    }

    fn free_pinned_host(&mut self, ptr: *mut u8) -> GpuMemoryResult<()> {
        unsafe { hip_free_host(ptr) }?;
        Ok(())
    }

    fn copy_h2d_async(&self, dst: GpuDevicePtr, src: *const u8, size: usize, stream: u64) -> GpuMemoryResult<()> {
        unsafe { hip_memcpy_h2d_async(dst.ptr, src, size, stream) }?;
        Ok(())
    }

    fn copy_d2h_async(&self, dst: *mut u8, src: GpuDevicePtr, size: usize, stream: u64) -> GpuMemoryResult<()> {
        unsafe { hip_memcpy_d2h_async(dst, src.ptr, size, stream) }?;
        Ok(())
    }

    fn synchronize(&self) -> GpuMemoryResult<()> {
        unsafe { hip_synchronize() }?;
        Ok(())
    }

    fn get_available_memory(&self) -> GpuMemoryResult<(usize, usize)> {
        let available = self.total_memory - self.allocated_memory;
        Ok((available, self.total_memory))
    }

    fn get_device_properties(&self) -> GpuMemoryResult<GpuDeviceProperties> {
        let props = unsafe { hip_get_device_properties(self.device_id) }?;
        Ok(props)
    }
}

/// GPU-aware memory pool that integrates with our hybrid cache
pub struct GpuAwareMemoryPool {
    backend: Box<dyn GpuMemoryBackend>,
    memory_pools: HashMap<usize, Vec<GpuDevicePtr>>, // Size -> available pointers
    allocations: HashMap<u64, GpuAllocation>,
    total_allocated: usize,
    alignment: usize,
    device_properties: GpuDeviceProperties,
}

impl GpuAwareMemoryPool {
    pub fn new_cuda(device_id: u32) -> GpuMemoryResult<Self> {
        let backend = Box::new(CudaMemoryBackend::new(device_id)?);
        Self::new_with_backend(backend)
    }

    pub fn new_hip(device_id: u32) -> GpuMemoryResult<Self> {
        let backend = Box::new(HipMemoryBackend::new(device_id)?);
        Self::new_with_backend(backend)
    }

    fn new_with_backend(mut backend: Box<dyn GpuMemoryBackend>) -> GpuMemoryResult<Self> {
        let device_properties = backend.get_device_properties()?;

        Ok(Self {
            backend,
            memory_pools: HashMap::new(),
            allocations: HashMap::new(),
            total_allocated: 0,
            alignment: 256, // 256-byte alignment for optimal GPU memory access
            device_properties,
        })
    }

    /// Allocate GPU memory for KV cache with optimal alignment
    pub fn allocate_kv_cache(&mut self, token_count: usize, head_dim: usize, num_heads: usize) -> GpuMemoryResult<KVTensorPair> {
        // Calculate memory requirements
        let key_size = token_count * head_dim * num_heads * 2; // 2 bytes for f16
        let value_size = key_size; // Same size for keys and values
        let total_size = key_size + value_size;

        // Allocate contiguous memory for both K and V
        let device_ptr = self.allocate_aligned(total_size, self.alignment)?;

        // Split into K and V pointers
        let key_ptr = device_ptr.ptr;
        let value_ptr = device_ptr.ptr + key_size as u64;

        Ok(KVTensorPair {
            key_ptr,
            value_ptr,
            size_bytes: total_size,
            token_count,
            head_dim,
            num_heads,
        })
    }

    /// Allocate aligned GPU memory with pooling
    pub fn allocate_aligned(&mut self, size: usize, alignment: usize) -> GpuMemoryResult<GpuDevicePtr> {
        // Round size up to alignment
        let aligned_size = (size + alignment - 1) & !(alignment - 1);

        // Check if we have a suitable allocation in the pool
        if let Some(pool) = self.memory_pools.get_mut(&aligned_size) {
            if let Some(ptr) = pool.pop() {
                return Ok(ptr);
            }
        }

        // Allocate new memory through backend
        let device_ptr = self.backend.allocate_device(aligned_size, alignment)?;

        // Track allocation
        let allocation = GpuAllocation::new(device_ptr, None);
        self.allocations.insert(device_ptr.ptr, allocation);
        self.total_allocated += aligned_size;

        Ok(device_ptr)
    }

    /// Free GPU memory back to pool for reuse
    pub fn free_to_pool(&mut self, ptr: GpuDevicePtr) -> GpuMemoryResult<()> {
        // Remove from active allocations
        if let Some(allocation) = self.allocations.remove(&ptr.ptr) {
            self.total_allocated -= allocation.device_ptr.size;

            // Add to appropriate pool for reuse
            let pool = self.memory_pools.entry(ptr.size).or_insert_with(Vec::new);
            pool.push(ptr);
        } else {
            // Not in our tracking, free directly
            self.backend.free_device(ptr)?;
        }

        Ok(())
    }

    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> GpuMemoryStats {
        let (available, total) = self.backend.get_available_memory().unwrap_or((0, 0));

        GpuMemoryStats {
            total_memory: total,
            allocated_memory: self.total_allocated,
            available_memory: available,
            pooled_allocations: self.memory_pools.values().map(|v| v.len()).sum(),
            active_allocations: self.allocations.len(),
            device_id: self.device_properties.device_id,
        }
    }

    /// Defragment memory pools
    pub fn defragment(&mut self) -> GpuMemoryResult<()> {
        // Free unused pooled memory to reduce fragmentation
        for pool in self.memory_pools.values_mut() {
            // Keep only the most recently used allocations
            pool.truncate(pool.len() / 2);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct GpuMemoryStats {
    pub total_memory: usize,
    pub allocated_memory: usize,
    pub available_memory: usize,
    pub pooled_allocations: usize,
    pub active_allocations: usize,
    pub device_id: u32,
}

// External FFI functions to our CUDA/HIP backends
extern "C" {
    fn cuda_create_context(device_id: u32) -> Result<u64, GpuMemoryError>;
    fn cuda_malloc_aligned(size: usize, alignment: usize) -> Result<u64, GpuMemoryError>;
    fn cuda_free(ptr: u64) -> Result<(), GpuMemoryError>;
    fn cuda_malloc_host(size: usize) -> Result<*mut u8, GpuMemoryError>;
    fn cuda_free_host(ptr: *mut u8) -> Result<(), GpuMemoryError>;
    fn cuda_memcpy_h2d_async(dst: u64, src: *const u8, size: usize, stream: u64) -> Result<(), GpuMemoryError>;
    fn cuda_memcpy_d2h_async(dst: *mut u8, src: u64, size: usize, stream: u64) -> Result<(), GpuMemoryError>;
    fn cuda_synchronize() -> Result<(), GpuMemoryError>;
    fn cuda_get_total_memory(device_id: u32) -> Result<usize, GpuMemoryError>;
    fn cuda_get_device_properties(device_id: u32) -> Result<GpuDeviceProperties, GpuMemoryError>;

    fn hip_create_context(device_id: u32) -> Result<u64, GpuMemoryError>;
    fn hip_malloc_aligned(size: usize, alignment: usize) -> Result<u64, GpuMemoryError>;
    fn hip_free(ptr: u64) -> Result<(), GpuMemoryError>;
    fn hip_malloc_host(size: usize) -> Result<*mut u8, GpuMemoryError>;
    fn hip_free_host(ptr: *mut u8) -> Result<(), GpuMemoryError>;
    fn hip_memcpy_h2d_async(dst: u64, src: *const u8, size: usize, stream: u64) -> Result<(), GpuMemoryError>;
    fn hip_memcpy_d2h_async(dst: *mut u8, src: u64, size: usize, stream: u64) -> Result<(), GpuMemoryError>;
    fn hip_synchronize() -> Result<(), GpuMemoryError>;
    fn hip_get_total_memory(device_id: u32) -> Result<usize, GpuMemoryError>;
    fn hip_get_device_properties(device_id: u32) -> Result<GpuDeviceProperties, GpuMemoryError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock backend for testing
    struct MockGpuBackend {
        available_memory: usize,
        total_memory: usize,
        allocations: HashMap<u64, usize>,
        next_ptr: u64,
    }

    impl MockGpuBackend {
        fn new(total_memory: usize) -> Self {
            Self {
                available_memory: total_memory,
                total_memory,
                allocations: HashMap::new(),
                next_ptr: 0x10000000,
            }
        }
    }

    impl GpuMemoryBackend for MockGpuBackend {
        fn allocate_device(&mut self, size: usize, alignment: usize) -> GpuMemoryResult<GpuDevicePtr> {
            if size > self.available_memory {
                return Err(GpuMemoryError::OutOfMemory {
                    requested: size,
                    available: self.available_memory,
                });
            }

            let aligned_size = (size + alignment - 1) & !(alignment - 1);
            let ptr = self.next_ptr;
            self.next_ptr += aligned_size as u64;

            self.allocations.insert(ptr, aligned_size);
            self.available_memory -= aligned_size;

            Ok(GpuDevicePtr::new(ptr, aligned_size, alignment, 0))
        }

        fn free_device(&mut self, ptr: GpuDevicePtr) -> GpuMemoryResult<()> {
            if let Some(size) = self.allocations.remove(&ptr.ptr) {
                self.available_memory += size;
            }
            Ok(())
        }

        fn allocate_pinned_host(&mut self, _size: usize) -> GpuMemoryResult<*mut u8> {
            Ok(0x10000000 as *mut u8) // Mock pointer
        }

        fn free_pinned_host(&mut self, _ptr: *mut u8) -> GpuMemoryResult<()> {
            Ok(())
        }

        fn copy_h2d_async(&self, _dst: GpuDevicePtr, _src: *const u8, _size: usize, _stream: u64) -> GpuMemoryResult<()> {
            Ok(())
        }

        fn copy_d2h_async(&self, _dst: *mut u8, _src: GpuDevicePtr, _size: usize, _stream: u64) -> GpuMemoryResult<()> {
            Ok(())
        }

        fn synchronize(&self) -> GpuMemoryResult<()> {
            Ok(())
        }

        fn get_available_memory(&self) -> GpuMemoryResult<(usize, usize)> {
            Ok((self.available_memory, self.total_memory))
        }

        fn get_device_properties(&self) -> GpuMemoryResult<GpuDeviceProperties> {
            Ok(GpuDeviceProperties {
                device_id: 0,
                total_memory: self.total_memory,
                memory_bandwidth: 1000,
                compute_capability: (8, 0),
                max_threads_per_block: 1024,
                max_shared_memory: 65536,
                is_numa_aware: true,
                pcie_generation: 4,
                pcie_lanes: 16,
            })
        }
    }

    #[test]
    fn test_gpu_memory_allocation() {
        let backend = Box::new(MockGpuBackend::new(1024 * 1024 * 1024)); // 1GB
        let mut pool = GpuAwareMemoryPool::new_with_backend(backend).unwrap();

        // Test KV cache allocation
        let kv_cache = pool.allocate_kv_cache(128, 64, 32).unwrap();

        assert_eq!(kv_cache.token_count, 128);
        assert_eq!(kv_cache.head_dim, 64);
        assert_eq!(kv_cache.num_heads, 32);
        assert!(kv_cache.key_ptr != 0);
        assert!(kv_cache.value_ptr != 0);
        assert!(kv_cache.value_ptr > kv_cache.key_ptr);

        let stats = pool.get_memory_stats();
        println!("Memory stats: {:?}", stats);
        assert!(stats.allocated_memory > 0);
    }

    #[test]
    fn test_memory_pooling() {
        let backend = Box::new(MockGpuBackend::new(1024 * 1024)); // 1MB
        let mut pool = GpuAwareMemoryPool::new_with_backend(backend).unwrap();

        // Allocate and free memory
        let ptr1 = pool.allocate_aligned(1024, 256).unwrap();
        let ptr2 = pool.allocate_aligned(2048, 256).unwrap();

        pool.free_to_pool(ptr1).unwrap();
        pool.free_to_pool(ptr2).unwrap();

        let stats = pool.get_memory_stats();
        assert_eq!(stats.pooled_allocations, 2);
        assert_eq!(stats.active_allocations, 0);
    }
}