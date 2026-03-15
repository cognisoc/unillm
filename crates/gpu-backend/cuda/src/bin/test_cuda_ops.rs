//! Test program for CUDA Flash Attention and GEMM operations

use gpu_backend_cuda::{CudaBackend, CudaDevicePtr, CudaStream, FlashAttentionVariant, CudaGemmConfig, CudaGemm};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing CUDA Flash Attention and GEMM operations...");
    
    // Create a CUDA backend
    let backend = CudaBackend::new_default();
    
    // Test Flash Attention
    println!("\n--- Testing Flash Attention ---");
    let flash_attention = backend.flash_attention();
    
    let q = CudaDevicePtr::new(0x1000);
    let k = CudaDevicePtr::new(0x2000);
    let v = CudaDevicePtr::new(0x3000);
    let mut output = CudaDevicePtr::new(0x4000);
    let stream = CudaStream::new(0);
    
    // Test FA2
    flash_attention.flash_attention(
        FlashAttentionVariant::FA2,
        &q,
        &k,
        &v,
        &mut output,
        &stream,
    )?;
    println!("Flash Attention 2 executed successfully");
    
    // Test FA3
    flash_attention.flash_attention(
        FlashAttentionVariant::FA3,
        &q,
        &k,
        &v,
        &mut output,
        &stream,
    )?;
    println!("Flash Attention 3 executed successfully");
    
    // Test GEMM
    println!("\n--- Testing GEMM ---");
    let gemm_config = CudaGemmConfig::new(false, false, 1.0, 0.0);
    let gemm = CudaGemm::new(gemm_config);
    
    let a = CudaDevicePtr::new(0x5000);
    let b = CudaDevicePtr::new(0x6000);
    let mut c = CudaDevicePtr::new(0x7000);
    let stream = CudaStream::new(0);
    
    gemm.gemm(64, 64, 64, &a, 64, &b, 64, &mut c, 64, &stream)?;
    println!("GEMM operation executed successfully");
    
    // Test batched GEMM
    gemm.batched_gemm(4, 64, 64, 64, &a, 64, &b, 64, &mut c, 64, &stream)?;
    println!("Batched GEMM operation executed successfully");
    
    println!("\nAll CUDA operations tested successfully!");
    
    Ok(())
}