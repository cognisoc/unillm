//! HIP memory transfer operations

use libc::{c_int, c_void, c_uint};
use super::memory::HipDevicePtr;
use super::stream::HipStream;

// HIP memory copy kinds
const HIP_MEMCPY_HOST_TO_HOST: c_uint = 0;
const HIP_MEMCPY_HOST_TO_DEVICE: c_uint = 1;
const HIP_MEMCPY_DEVICE_TO_HOST: c_uint = 2;
const HIP_MEMCPY_DEVICE_TO_DEVICE: c_uint = 3;

// HIP runtime API bindings
extern "C" {
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_uint) -> c_int;
    fn hipMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: c_uint, stream: *mut c_void) -> c_int;
    fn hipMemcpyPeer(dst: *mut c_void, dstDevice: c_int, src: *const c_void, srcDevice: c_int, count: usize) -> c_int;
    fn hipMemcpyPeerAsync(dst: *mut c_void, dstDevice: c_int, src: *const c_void, srcDevice: c_int, count: usize, stream: *mut c_void) -> c_int;
}

/// Copy data from host to device synchronously
pub fn h2d_sync(dst: &HipDevicePtr, src: *const u8, n: usize) -> Result<(), Box<dyn std::error::Error>> {
    if src.is_null() {
        return Err("Source pointer is null".into());
    }
    
    if n == 0 {
        return Ok(());
    }
    
    let result = unsafe {
        hipMemcpy(
            dst.as_raw_ptr(),
            src as *const c_void,
            n,
            HIP_MEMCPY_HOST_TO_DEVICE,
        )
    };
    
    if result != 0 {
        return Err(format!("Failed to copy host to device: {}", result).into());
    }
    
    Ok(())
}

/// Copy data from host to device asynchronously
pub fn h2d_async(dst: &HipDevicePtr, src: *const u8, n: usize, stream: &HipStream) -> Result<(), Box<dyn std::error::Error>> {
    if src.is_null() {
        return Err("Source pointer is null".into());
    }
    
    if n == 0 {
        return Ok(());
    }
    
    let result = unsafe {
        hipMemcpyAsync(
            dst.as_raw_ptr(),
            src as *const c_void,
            n,
            HIP_MEMCPY_HOST_TO_DEVICE,
            stream.as_raw_ptr(),
        )
    };
    
    if result != 0 {
        return Err(format!("Failed to copy host to device async: {}", result).into());
    }
    
    Ok(())
}

/// Copy data from device to host synchronously
pub fn d2h_sync(dst: *mut u8, src: &HipDevicePtr, n: usize) -> Result<(), Box<dyn std::error::Error>> {
    if dst.is_null() {
        return Err("Destination pointer is null".into());
    }
    
    if n == 0 {
        return Ok(());
    }
    
    let result = unsafe {
        hipMemcpy(
            dst as *mut c_void,
            src.as_raw_ptr() as *const c_void,
            n,
            HIP_MEMCPY_DEVICE_TO_HOST,
        )
    };
    
    if result != 0 {
        return Err(format!("Failed to copy device to host: {}", result).into());
    }
    
    Ok(())
}

/// Copy data from device to host asynchronously
pub fn d2h_async(dst: *mut u8, src: &HipDevicePtr, n: usize, stream: &HipStream) -> Result<(), Box<dyn std::error::Error>> {
    if dst.is_null() {
        return Err("Destination pointer is null".into());
    }
    
    if n == 0 {
        return Ok(());
    }
    
    let result = unsafe {
        hipMemcpyAsync(
            dst as *mut c_void,
            src.as_raw_ptr() as *const c_void,
            n,
            HIP_MEMCPY_DEVICE_TO_HOST,
            stream.as_raw_ptr(),
        )
    };
    
    if result != 0 {
        return Err(format!("Failed to copy device to host async: {}", result).into());
    }
    
    Ok(())
}

/// Copy data between devices synchronously
pub fn d2d_sync(dst: &HipDevicePtr, dst_device: i32, src: &HipDevicePtr, src_device: i32, n: usize) -> Result<(), Box<dyn std::error::Error>> {
    if n == 0 {
        return Ok(());
    }
    
    let result = unsafe {
        hipMemcpyPeer(
            dst.as_raw_ptr(),
            dst_device,
            src.as_raw_ptr() as *const c_void,
            src_device,
            n,
        )
    };
    
    if result != 0 {
        return Err(format!("Failed to copy device to device: {}", result).into());
    }
    
    Ok(())
}

/// Copy data between devices asynchronously
pub fn d2d_async(dst: &HipDevicePtr, dst_device: i32, src: &HipDevicePtr, src_device: i32, n: usize, stream: &HipStream) -> Result<(), Box<dyn std::error::Error>> {
    if n == 0 {
        return Ok(());
    }
    
    let result = unsafe {
        hipMemcpyPeerAsync(
            dst.as_raw_ptr(),
            dst_device,
            src.as_raw_ptr() as *const c_void,
            src_device,
            n,
            stream.as_raw_ptr(),
        )
    };
    
    if result != 0 {
        return Err(format!("Failed to copy device to device async: {}", result).into());
    }
    
    Ok(())
}