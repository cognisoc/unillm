//! HIP context management

use libc::{c_int, c_void};

// HIP runtime API bindings
extern "C" {
    fn hipSetDevice(device: c_int) -> c_int;
    fn hipGetDevice(device: *mut c_int) -> c_int;
    fn hipGetDeviceCount(count: *mut c_int) -> c_int;
    fn hipDeviceSynchronize() -> c_int;
    fn hipGetLastError() -> c_int;
    fn hipGetErrorString(error: c_int) -> *const libc::c_char;
}

/// HIP context
pub struct HipContext {
    device_id: i32,
    initialized: bool,
}

impl HipContext {
    /// Create a new HIP context
    pub fn new(device_id: i32) -> Self {
        Self {
            device_id,
            initialized: false,
        }
    }
    
    /// Initialize the HIP context
    pub fn init(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Set the HIP device
        let result = unsafe { hipSetDevice(self.device_id) };
        if result != 0 {
            return Err(format!("Failed to set HIP device {}: {}", 
                              self.device_id, Self::get_error_string(result)).into());
        }
        
        // Synchronize to ensure device is ready
        let result = unsafe { hipDeviceSynchronize() };
        if result != 0 {
            return Err(format!("Failed to synchronize HIP device {}: {}", 
                              self.device_id, Self::get_error_string(result)).into());
        }
        
        self.initialized = true;
        println!("Initialized HIP context for device {}", self.device_id);
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
    
    /// Get the number of available HIP devices
    pub fn get_device_count() -> Result<i32, Box<dyn std::error::Error>> {
        let mut count = 0;
        let result = unsafe { hipGetDeviceCount(&mut count) };
        if result != 0 {
            return Err(format!("Failed to get HIP device count: {}", 
                              Self::get_error_string(result)).into());
        }
        Ok(count)
    }
    
    /// Get current device
    pub fn get_current_device() -> Result<i32, Box<dyn std::error::Error>> {
        let mut device = 0;
        let result = unsafe { hipGetDevice(&mut device) };
        if result != 0 {
            return Err(format!("Failed to get current HIP device: {}", 
                              Self::get_error_string(result)).into());
        }
        Ok(device)
    }
    
    /// Synchronize the device
    pub fn synchronize(&self) -> Result<(), Box<dyn std::error::Error>> {
        if !self.initialized {
            return Err("HIP context not initialized".into());
        }
        
        let result = unsafe { hipDeviceSynchronize() };
        if result != 0 {
            return Err(format!("Failed to synchronize HIP device: {}", 
                              Self::get_error_string(result)).into());
        }
        Ok(())
    }
    
    /// Get error string from HIP error code
    fn get_error_string(error: i32) -> String {
        unsafe {
            let error_str = hipGetErrorString(error);
            if error_str.is_null() {
                format!("Unknown HIP error: {}", error)
            } else {
                std::ffi::CStr::from_ptr(error_str)
                    .to_string_lossy()
                    .to_string()
            }
        }
    }
}