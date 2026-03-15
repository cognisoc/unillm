//! HIP memory management

use libc::{c_int, c_void, c_uint};

// HIP memory API bindings
extern "C" {
    fn hipMalloc(devPtr: *mut *mut c_void, size: usize) -> c_int;
    fn hipFree(devPtr: *mut c_void) -> c_int;
    fn hipMallocHost(ptr: *mut *mut c_void, size: usize) -> c_int;
    fn hipFreeHost(ptr: *mut c_void) -> c_int;
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_uint) -> c_int;
    fn hipMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: c_uint, stream: *mut c_void) -> c_int;
    fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;
}

// HIP memory copy kinds are defined in transfer.rs

/// HIP device pointer
pub struct HipDevicePtr {
    ptr: *mut libc::c_void,
    size: usize,
}

impl HipDevicePtr {
    /// Create a new HIP device pointer
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

impl Drop for HipDevicePtr {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                hipFree(self.ptr);
            }
        }
    }
}

/// HIP pinned memory
pub struct HipPinnedMemory {
    ptr: *mut libc::c_void,
    size: usize,
}

impl HipPinnedMemory {
    /// Create pinned memory
    pub fn new(size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let mut ptr: *mut libc::c_void = std::ptr::null_mut();
        let result = unsafe { hipMallocHost(&mut ptr, size) };
        
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

impl Drop for HipPinnedMemory {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                hipFreeHost(self.ptr);
            }
        }
    }
}

/// HIP memory manager
pub struct HipMemoryManager {
    // Memory manager implementation
}

impl HipMemoryManager {
    /// Create a new HIP memory manager
    pub fn new() -> Self {
        Self {
            // Initialize memory manager
        }
    }
    
    /// Allocate memory on the device
    pub fn alloc(&self, bytes: usize) -> Result<HipDevicePtr, Box<dyn std::error::Error>> {
        let mut ptr: *mut libc::c_void = std::ptr::null_mut();
        let result = unsafe { hipMalloc(&mut ptr, bytes) };
        
        if result != 0 {
            return Err(format!("Failed to allocate HIP memory: {}", result).into());
        }
        
        Ok(HipDevicePtr::from_raw_ptr(ptr, bytes))
    }
    
    /// Allocate pinned memory
    pub fn alloc_pinned(&self, bytes: usize) -> Result<HipPinnedMemory, Box<dyn std::error::Error>> {
        HipPinnedMemory::new(bytes)
    }
    
    /// Get memory info
    pub fn get_memory_info(&self) -> Result<(usize, usize), Box<dyn std::error::Error>> {
        let mut free: usize = 0;
        let mut total: usize = 0;
        
        let result = unsafe { hipMemGetInfo(&mut free, &mut total) };
        if result != 0 {
            return Err(format!("Failed to get HIP memory info: {}", result).into());
        }
        
        Ok((free, total))
    }
}