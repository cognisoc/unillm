//! HIP stream management

use libc::{c_int, c_void};

// HIP stream API bindings
extern "C" {
    fn hipStreamCreate(stream: *mut *mut c_void) -> c_int;
    fn hipStreamCreateWithFlags(stream: *mut *mut c_void, flags: c_uint) -> c_int;
    fn hipStreamCreateWithPriority(stream: *mut *mut c_void, flags: c_uint, priority: c_int) -> c_int;
    fn hipStreamDestroy(stream: *mut c_void) -> c_int;
    fn hipStreamSynchronize(stream: *mut c_void) -> c_int;
    fn hipStreamQuery(stream: *mut c_void) -> c_int;
    fn hipDeviceGetStreamPriorityRange(leastPriority: *mut c_int, greatestPriority: *mut c_int) -> c_int;
}

// HIP stream flags
const HIP_STREAM_DEFAULT: c_uint = 0x00;
const HIP_STREAM_NON_BLOCKING: c_uint = 0x01;

/// HIP stream
pub struct HipStream {
    stream: *mut c_void,
    priority: i32,
}

impl HipStream {
    /// Create a new HIP stream with default priority
    pub fn new(priority: i32) -> Self {
        let mut stream: *mut c_void = std::ptr::null_mut();
        let result = unsafe { hipStreamCreateWithPriority(&mut stream, HIP_STREAM_DEFAULT, priority) };
        
        if result != 0 {
            // Fallback to regular stream creation
            let result = unsafe { hipStreamCreate(&mut stream) };
            if result != 0 {
                panic!("Failed to create HIP stream: {}", result);
            }
        }
        
        Self { stream, priority }
    }
    
    /// Create a new HIP stream with flags
    pub fn new_with_flags(flags: u32, priority: i32) -> Self {
        let mut stream: *mut c_void = std::ptr::null_mut();
        let result = unsafe { hipStreamCreateWithPriority(&mut stream, flags, priority) };
        
        if result != 0 {
            // Fallback to regular stream creation
            let result = unsafe { hipStreamCreate(&mut stream) };
            if result != 0 {
                panic!("Failed to create HIP stream: {}", result);
            }
        }
        
        Self { stream, priority }
    }
    
    /// Create a non-blocking stream
    pub fn new_non_blocking(priority: i32) -> Self {
        Self::new_with_flags(HIP_STREAM_NON_BLOCKING, priority)
    }
    
    /// Synchronize the stream
    pub fn synchronize(&self) -> Result<(), Box<dyn std::error::Error>> {
        let result = unsafe { hipStreamSynchronize(self.stream) };
        if result != 0 {
            return Err(format!("Failed to synchronize HIP stream: {}", result).into());
        }
        Ok(())
    }
    
    /// Query if the stream is ready
    pub fn query(&self) -> Result<bool, Box<dyn std::error::Error>> {
        let result = unsafe { hipStreamQuery(self.stream) };
        match result {
            0 => Ok(true),  // Stream is ready
            1 => Ok(false), // Stream is not ready
            _ => Err(format!("Failed to query HIP stream: {}", result).into()),
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
        
        let result = unsafe { hipDeviceGetStreamPriorityRange(&mut least_priority, &mut greatest_priority) };
        if result != 0 {
            return Err(format!("Failed to get HIP stream priority range: {}", result).into());
        }
        
        Ok((least_priority, greatest_priority))
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe {
                hipStreamDestroy(self.stream);
            }
        }
    }
}