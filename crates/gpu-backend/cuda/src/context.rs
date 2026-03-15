//! CUDA context management

use std::ptr;
use libc::{c_int, c_void};

// CUDA runtime API bindings
extern "C" {
    fn cudaSetDevice(device: c_int) -> c_int;
    fn cudaGetDevice(device: *mut c_int) -> c_int;
    fn cudaGetDeviceCount(count: *mut c_int) -> c_int;
    fn cudaDeviceSynchronize() -> c_int;
    fn cudaGetLastError() -> c_int;
    fn cudaGetErrorString(error: c_int) -> *const libc::c_char;
}

/// CUDA context
pub struct CudaContext {
    device_id: i32,
    initialized: bool,
}

impl CudaContext {
    /// Create a new CUDA context
    pub fn new(device_id: i32) -> Self {
        Self {
            device_id,
            initialized: false,
        }
    }
    
    /// Initialize the CUDA context
    pub fn init(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Set the CUDA device
        let result = unsafe { cudaSetDevice(self.device_id) };
        if result != 0 {
            return Err(format!("Failed to set CUDA device {}: {}", 
                              self.device_id, Self::get_error_string(result)).into());
        }
        
        // Synchronize to ensure device is ready
        let result = unsafe { cudaDeviceSynchronize() };
        if result != 0 {
            return Err(format!("Failed to synchronize CUDA device {}: {}", 
                              self.device_id, Self::get_error_string(result)).into());
        }
        
        self.initialized = true;
        println!("Initialized CUDA context for device {}", self.device_id);
        Ok(())
    }
    
    /// Get the device ID
    pub fn device_id(&self) -> i32 {
        self.device_id
    }
    
    /// Check if context is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
    
    /// Get the number of available CUDA devices
    pub fn get_device_count() -> Result<i32, Box<dyn std::error::Error>> {
        let mut count = 0;
        let result = unsafe { cudaGetDeviceCount(&mut count) };
        if result != 0 {
            return Err(format!("Failed to get CUDA device count: {}", 
                              Self::get_error_string(result)).into());
        }
        Ok(count)
    }
    
    /// Get current device
    pub fn get_current_device() -> Result<i32, Box<dyn std::error::Error>> {
        let mut device = 0;
        let result = unsafe { cudaGetDevice(&mut device) };
        if result != 0 {
            return Err(format!("Failed to get current CUDA device: {}", 
                              Self::get_error_string(result)).into());
        }
        Ok(device)
    }
    
    /// Synchronize the device
    pub fn synchronize(&self) -> Result<(), Box<dyn std::error::Error>> {
        if !self.initialized {
            return Err("CUDA context not initialized".into());
        }
        
        let result = unsafe { cudaDeviceSynchronize() };
        if result != 0 {
            return Err(format!("Failed to synchronize CUDA device: {}", 
                              Self::get_error_string(result)).into());
        }
        Ok(())
    }
    
    /// Get error string from CUDA error code
    fn get_error_string(error: i32) -> String {
        unsafe {
            let error_str = cudaGetErrorString(error);
            if error_str.is_null() {
                format!("Unknown CUDA error: {}", error)
            } else {
                std::ffi::CStr::from_ptr(error_str)
                    .to_string_lossy()
                    .to_string()
            }
        }
    }
}