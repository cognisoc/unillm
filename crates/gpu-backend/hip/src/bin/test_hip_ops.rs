//! Test program for HIP Flash Attention and GEMM operations

use gpu_backend_hip::{HipBackend, HipDevicePtr, HipStream, FlashAttentionVariant, HipGemmConfig, HipGemm};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing HIP Flash Attention and GEMM operations...");
    
    // Create a HIP backend
    let backend = HipBackend::new_default();
    
    // Test Flash Attention
    println!("\n--- Testing Flash Attention ---");
    let flash_attention = backend.flash_attention();
    
    let q = HipDevicePtr::new(0x1000);
    let k = HipDevicePtr::new(0x2000);
    let v = HipDevicePtr::new(0x3000);
    let mut output = HipDevicePtr::new(0x4000);
    let stream = HipStream::new(0);
    
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
    
    // Test Triton
    flash_attention.flash_attention(
        FlashAttentionVariant::Triton,
        &q,
        &k,
        &v,
        &mut output,
        &stream,
    )?;
    println!("Triton-based attention executed successfully");
    
    // Test GEMM
    println!("\n--- Testing GEMM ---");
    let gemm_config = HipGemmConfig::new(false, false, 1.0, 0.0);
    let gemm = HipGemm::new(gemm_config);
    
    let a = HipDevicePtr::new(0x5000);
    let b = HipDevicePtr::new(0x6000);
    let mut c = HipDevicePtr::new(0x7000);
    let stream = HipStream::new(0);
    
    gemm.gemm(64, 64, 64, &a, 64, &b, 64, &mut c, 64, &stream)?;
    println!("GEMM operation executed successfully");
    
    // Test batched GEMM
    gemm.batched_gemm(4, 64, 64, 64, &a, 64, &b, 64, &mut c, 64, &stream)?;
    println!("Batched GEMM operation executed successfully");
    
    println!("\nAll HIP operations tested successfully!");
    
    Ok(())
}