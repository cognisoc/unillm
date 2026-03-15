//! CUDA memory transfer operations

use libc::{c_int, c_void, c_uint};
use super::memory::CudaDevicePtr;
use super::stream::CudaStream;

// CUDA memory copy kinds
const CUDA_MEMCPY_HOST_TO_HOST: c_uint = 0;
const CUDA_MEMCPY_HOST_TO_DEVICE: c_uint = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_uint = 2;
const CUDA_MEMCPY_DEVICE_TO_DEVICE: c_uint = 3;

// CUDA runtime API bindings
extern "C" {
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_uint) -> c_int;
    fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: c_uint, stream: *mut c_void) -> c_int;
    fn cudaMemcpyPeer(dst: *mut c_void, dstDevice: c_int, src: *const c_void, srcDevice: c_int, count: usize) -> c_int;
    fn cudaMemcpyPeerAsync(dst: *mut c_void, dstDevice: c_int, src: *const c_void, srcDevice: c_int, count: usize, stream: *mut c_void) -> c_int;
}

/// Copy data from host to device synchronously
pub fn h2d_sync(dst: &CudaDevicePtr, src: *const u8, n: usize) -> Result<(), Box<dyn std::error::Error>> {
    if src.is_null() {
        return Err("Source pointer is null".into());
    }
    
    if n == 0 {
        return Ok(());
    }
    
    let result = unsafe {
        cudaMemcpy(
            dst.as_raw_ptr(),
            src as *const c_void,
            n,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    };
    
    if result != 0 {
        return Err(format!("Failed to copy host to device: {}", result).into());
    }
    
    Ok(())
}

/// Copy data from host to device asynchronously
pub fn h2d_async(dst: &CudaDevicePtr, src: *const u8, n: usize, stream: &CudaStream) -> Result<(), Box<dyn std::error::Error>> {
    if src.is_null() {
        return Err("Source pointer is null".into());
    }
    
    if n == 0 {
        return Ok(());
    }
    
    let result = unsafe {
        cudaMemcpyAsync(
            dst.as_raw_ptr(),
            src as *const c_void,
            n,
            CUDA_MEMCPY_HOST_TO_DEVICE,
            stream.as_raw_ptr(),
        )
    };
    
    if result != 0 {
        return Err(format!("Failed to copy host to device async: {}", result).into());
    }
    
    Ok(())
}

/// Copy data from device to host synchronously
pub fn d2h_sync(dst: *mut u8, src: &CudaDevicePtr, n: usize) -> Result<(), Box<dyn std::error::Error>> {
    if dst.is_null() {
        return Err("Destination pointer is null".into());
    }
    
    if n == 0 {
        return Ok(());
    }
    
    let result = unsafe {
        cudaMemcpy(
            dst as *mut c_void,
            src.as_raw_ptr() as *const c_void,
            n,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    };
    
    if result != 0 {
        return Err(format!("Failed to copy device to host: {}", result).into());
    }
    
    Ok(())
}

/// Copy data from device to host asynchronously
pub fn d2h_async(dst: *mut u8, src: &CudaDevicePtr, n: usize, stream: &CudaStream) -> Result<(), Box<dyn std::error::Error>> {
    if dst.is_null() {
        return Err("Destination pointer is null".into());
    }
    
    if n == 0 {
        return Ok(());
    }
    
    let result = unsafe {
        cudaMemcpyAsync(
            dst as *mut c_void,
            src.as_raw_ptr() as *const c_void,
            n,
            CUDA_MEMCPY_DEVICE_TO_HOST,
            stream.as_raw_ptr(),
        )
    };
    
    if result != 0 {
        return Err(format!("Failed to copy device to host async: {}", result).into());
    }
    
    Ok(())
}

/// Copy data between devices synchronously
pub fn d2d_sync(dst: &CudaDevicePtr, dst_device: i32, src: &CudaDevicePtr, src_device: i32, n: usize) -> Result<(), Box<dyn std::error::Error>> {
    if n == 0 {
        return Ok(());
    }
    
    let result = unsafe {
        cudaMemcpyPeer(
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
pub fn d2d_async(dst: &CudaDevicePtr, dst_device: i32, src: &CudaDevicePtr, src_device: i32, n: usize, stream: &CudaStream) -> Result<(), Box<dyn std::error::Error>> {
    if n == 0 {
        return Ok(());
    }
    
    let result = unsafe {
        cudaMemcpyPeerAsync(
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