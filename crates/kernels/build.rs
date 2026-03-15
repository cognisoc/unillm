use std::env;

fn main() {
    // Check which GPU backend is enabled
    let cuda_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();
    let hip_enabled = env::var("CARGO_FEATURE_HIP").is_ok();
    
    if cuda_enabled {
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cublas");
        println!("cargo:rustc-link-lib=dylib=cublasLt");
        // TODO: Add nvcc compilation for CUDA kernels
    }
    
    if hip_enabled {
        println!("cargo:rustc-link-lib=dylib=amdhip64");
        println!("cargo:rustc-link-lib=dylib=rocblas");
        println!("cargo:rustc-link-lib=dylib=hipblaslt");
        // TODO: Add hipcc compilation for HIP kernels
    }
    
    // TODO: Implement kernel compilation based on features
}