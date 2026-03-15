//! Transfer validation tests

use gpu_backend_cuda::CudaBackend;
use gpu_backend_hip::HipBackend;

/// Test CUDA transfers
pub fn test_cuda_transfers() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing CUDA transfers...");
    
    // Check if CUDA is available
    match gpu_backend_cuda::CudaContext::get_device_count() {
        Ok(count) => {
            if count == 0 {
                println!("No CUDA devices found, skipping CUDA tests");
                return Ok(());
            }
            println!("Found {} CUDA device(s)", count);
        },
        Err(e) => {
            println!("CUDA not available: {}, skipping CUDA tests", e);
            return Ok(());
        }
    }
    
    let mut backend = CudaBackend::new_default();
    backend.device_init()?;
    
    // Test memory allocation
    let device_ptr = backend.alloc(1024, false)?;
    println!("CUDA memory allocation successful: {:?}", device_ptr.as_ptr());
    
    // Test H2D transfer
    let test_data = vec![1.0f32, 2.0, 3.0, 4.0];
    backend.h2d_async(&*device_ptr, test_data.as_ptr() as *const u8, test_data.len() * 4, backend.stream())?;
    
    // Test D2H transfer
    let mut result_data = vec![0.0f32; 4];
    backend.d2h_async(result_data.as_mut_ptr() as *mut u8, &*device_ptr, result_data.len() * 4, backend.stream())?;
    
    // Verify the data
    for (i, &expected) in test_data.iter().enumerate() {
        if (result_data[i] - expected).abs() > 1e-6 {
            return Err(format!("Data mismatch at index {}: expected {}, got {}", i, expected, result_data[i]).into());
        }
    }
    
    println!("CUDA transfer test completed successfully");
    Ok(())
}

/// Test HIP transfers
pub fn test_hip_transfers() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing HIP transfers...");
    
    // Check if HIP is available
    match gpu_backend_hip::HipContext::get_device_count() {
        Ok(count) => {
            if count == 0 {
                println!("No HIP devices found, skipping HIP tests");
                return Ok(());
            }
            println!("Found {} HIP device(s)", count);
        },
        Err(e) => {
            println!("HIP not available: {}, skipping HIP tests", e);
            return Ok(());
        }
    }
    
    let mut backend = HipBackend::new_default();
    backend.device_init()?;
    
    // Test memory allocation
    let device_ptr = backend.alloc(1024, false)?;
    println!("HIP memory allocation successful: {:?}", device_ptr.as_ptr());
    
    // Test H2D transfer
    let test_data = vec![1.0f32, 2.0, 3.0, 4.0];
    backend.h2d_async(&*device_ptr, test_data.as_ptr() as *const u8, test_data.len() * 4, backend.stream())?;
    
    // Test D2H transfer
    let mut result_data = vec![0.0f32; 4];
    backend.d2h_async(result_data.as_mut_ptr() as *mut u8, &*device_ptr, result_data.len() * 4, backend.stream())?;
    
    // Verify the data
    for (i, &expected) in test_data.iter().enumerate() {
        if (result_data[i] - expected).abs() > 1e-6 {
            return Err(format!("Data mismatch at index {}: expected {}, got {}", i, expected, result_data[i]).into());
        }
    }
    
    println!("HIP transfer test completed successfully");
    Ok(())
}

/// Test memory info
pub fn test_memory_info() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing memory info...");
    
    // Test CUDA memory info
    if let Ok(_) = gpu_backend_cuda::CudaContext::get_device_count() {
        let mut backend = CudaBackend::new_default();
        if let Ok(_) = backend.device_init() {
            match backend.memory_manager().get_memory_info() {
                Ok((free, total)) => {
                    println!("CUDA Memory: {} MB free / {} MB total", free / 1024 / 1024, total / 1024 / 1024);
                },
                Err(e) => println!("Failed to get CUDA memory info: {}", e),
            }
        }
    }
    
    // Test HIP memory info
    if let Ok(_) = gpu_backend_hip::HipContext::get_device_count() {
        let mut backend = HipBackend::new_default();
        if let Ok(_) = backend.device_init() {
            match backend.memory_manager().get_memory_info() {
                Ok((free, total)) => {
                    println!("HIP Memory: {} MB free / {} MB total", free / 1024 / 1024, total / 1024 / 1024);
                },
                Err(e) => println!("Failed to get HIP memory info: {}", e),
            }
        }
    }
    
    Ok(())
}