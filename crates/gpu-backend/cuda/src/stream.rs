//! CUDA stream management

use libc::{c_int, c_void};

// CUDA stream API bindings
extern "C" {
    fn cudaStreamCreate(stream: *mut *mut c_void) -> c_int;
    fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: c_uint) -> c_int;
    fn cudaStreamCreateWithPriority(stream: *mut *mut c_void, flags: c_uint, priority: c_int) -> c_int;
    fn cudaStreamDestroy(stream: *mut c_void) -> c_int;
    fn cudaStreamSynchronize(stream: *mut c_void) -> c_int;
    fn cudaStreamQuery(stream: *mut c_void) -> c_int;
    fn cudaDeviceGetStreamPriorityRange(leastPriority: *mut c_int, greatestPriority: *mut c_int) -> c_int;
}

// CUDA stream flags
const CUDA_STREAM_DEFAULT: c_uint = 0x00;
const CUDA_STREAM_NON_BLOCKING: c_uint = 0x01;

/// CUDA stream
pub struct CudaStream {
    stream: *mut c_void,
    priority: i32,
}

impl CudaStream {
    /// Create a new CUDA stream with default priority
    pub fn new(priority: i32) -> Self {
        let mut stream: *mut c_void = std::ptr::null_mut();
        let result = unsafe { cudaStreamCreateWithPriority(&mut stream, CUDA_STREAM_DEFAULT, priority) };
        
        if result != 0 {
            // Fallback to regular stream creation
            let result = unsafe { cudaStreamCreate(&mut stream) };
            if result != 0 {
                panic!("Failed to create CUDA stream: {}", result);
            }
        }
        
        Self { stream, priority }
    }
    
    /// Create a new CUDA stream with flags
    pub fn new_with_flags(flags: u32, priority: i32) -> Self {
        let mut stream: *mut c_void = std::ptr::null_mut();
        let result = unsafe { cudaStreamCreateWithPriority(&mut stream, flags, priority) };
        
        if result != 0 {
            // Fallback to regular stream creation
            let result = unsafe { cudaStreamCreate(&mut stream) };
            if result != 0 {
                panic!("Failed to create CUDA stream: {}", result);
            }
        }
        
        Self { stream, priority }
    }
    
    /// Create a non-blocking stream
    pub fn new_non_blocking(priority: i32) -> Self {
        Self::new_with_flags(CUDA_STREAM_NON_BLOCKING, priority)
    }
    
    /// Synchronize the stream
    pub fn synchronize(&self) -> Result<(), Box<dyn std::error::Error>> {
        let result = unsafe { cudaStreamSynchronize(self.stream) };
        if result != 0 {
            return Err(format!("Failed to synchronize CUDA stream: {}", result).into());
        }
        Ok(())
    }
    
    /// Query if the stream is ready
    pub fn query(&self) -> Result<bool, Box<dyn std::error::Error>> {
        let result = unsafe { cudaStreamQuery(self.stream) };
        match result {
            0 => Ok(true),  // Stream is ready
            1 => Ok(false), // Stream is not ready
            _ => Err(format!("Failed to query CUDA stream: {}", result).into()),
        }
    }
    
    /// Get the raw stream pointer
    pub fn as_raw_ptr(&self) -> *mut c_void {
        self.stream
    }
    
    /// Get the stream priority
    pub fn priority(&self) -> i32 {
        self.priority
    }
    
    /// Get the priority range for the current device
    pub fn get_priority_range() -> Result<(i32, i32), Box<dyn std::error::Error>> {
        let mut least_priority = 0;
        let mut greatest_priority = 0;
        
        let result = unsafe { cudaDeviceGetStreamPriorityRange(&mut least_priority, &mut greatest_priority) };
        if result != 0 {
            return Err(format!("Failed to get CUDA stream priority range: {}", result).into());
        }
        
        Ok((least_priority, greatest_priority))
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe {
                cudaStreamDestroy(self.stream);
            }
        }
    }
}