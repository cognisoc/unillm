//! CUDA memory management

use libc::{c_int, c_void, c_uint};

// CUDA memory API bindings
extern "C" {
    fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> c_int;
    fn cudaFree(devPtr: *mut c_void) -> c_int;
    fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> c_int;
    fn cudaFreeHost(ptr: *mut c_void) -> c_int;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_uint) -> c_int;
    fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: c_uint, stream: *mut c_void) -> c_int;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;
}

// CUDA memory copy kinds are defined in transfer.rs

/// CUDA device pointer
pub struct CudaDevicePtr {
    ptr: *mut libc::c_void,
    size: usize,
}

impl CudaDevicePtr {
    /// Create a new CUDA device pointer
    pub fn new(ptr: u64) -> Self {
        Self { 
            ptr: ptr as *mut libc::c_void,
            size: 0,
        }
    }
    
    /// Create from raw pointer
    pub fn from_raw_ptr(ptr: *mut libc::c_void, size: usize) -> Self {
        Self { ptr, size }
    }
    
    /// Get the pointer value
    pub fn as_ptr(&self) -> u64 {
        self.ptr as u64
    }
    
    /// Get the raw pointer
    pub fn as_raw_ptr(&self) -> *mut libc::c_void {
        self.ptr
    }
    
    /// Get the size
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for CudaDevicePtr {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                cudaFree(self.ptr);
            }
        }
    }
}

/// CUDA pinned memory
pub struct CudaPinnedMemory {
    ptr: *mut libc::c_void,
    size: usize,
}

impl CudaPinnedMemory {
    /// Create pinned memory
    pub fn new(size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let mut ptr: *mut libc::c_void = std::ptr::null_mut();
        let result = unsafe { cudaMallocHost(&mut ptr, size) };
        
        if result != 0 {
            return Err(format!("Failed to allocate pinned memory: {}", result).into());
        }
        
        Ok(Self { ptr, size })
    }
    
    /// Get the pointer
    pub fn as_ptr(&self) -> *const libc::c_void {
        self.ptr
    }
    
    /// Get the mutable pointer
    pub fn as_mut_ptr(&self) -> *mut libc::c_void {
        self.ptr
    }
    
    /// Get the size
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for CudaPinnedMemory {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                cudaFreeHost(self.ptr);
            }
        }
    }
}

/// CUDA memory manager
pub struct CudaMemoryManager {
    // Memory manager implementation
}

impl CudaMemoryManager {
    /// Create a new CUDA memory manager
    pub fn new() -> Self {
        Self {
            // Initialize memory manager
        }
    }
    
    /// Allocate memory on the device
    pub fn alloc(&self, bytes: usize) -> Result<CudaDevicePtr, Box<dyn std::error::Error>> {
        let mut ptr: *mut libc::c_void = std::ptr::null_mut();
        let result = unsafe { cudaMalloc(&mut ptr, bytes) };
        
        if result != 0 {
            return Err(format!("Failed to allocate CUDA memory: {}", result).into());
        }
        
        Ok(CudaDevicePtr::from_raw_ptr(ptr, bytes))
    }
    
    /// Allocate pinned memory
    pub fn alloc_pinned(&self, bytes: usize) -> Result<CudaPinnedMemory, Box<dyn std::error::Error>> {
        CudaPinnedMemory::new(bytes)
    }
    
    /// Get memory info
    pub fn get_memory_info(&self) -> Result<(usize, usize), Box<dyn std::error::Error>> {
        let mut free: usize = 0;
        let mut total: usize = 0;
        
        let result = unsafe { cudaMemGetInfo(&mut free, &mut total) };
        if result != 0 {
            return Err(format!("Failed to get CUDA memory info: {}", result).into());
        }
        
        Ok((free, total))
    }
}