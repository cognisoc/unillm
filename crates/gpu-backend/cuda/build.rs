fn main() {
    // Check for CUDA installation
    if let Ok(cuda_path) = std::env::var("CUDA_HOME") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cublasLt");
    } else if let Ok(cuda_path) = std::env::var("CUDA_ROOT") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cublasLt");
    } else {
        // Try common CUDA installation paths
        let common_paths = [
            "/usr/local/cuda",
            "/opt/cuda",
            "/usr/local/cuda-12.0",
            "/usr/local/cuda-11.8",
        ];
        
        for path in &common_paths {
            if std::path::Path::new(path).exists() {
                println!("cargo:rustc-link-search=native={}/lib64", path);
                println!("cargo:rustc-link-lib=cuda");
                println!("cargo:rustc-link-lib=cudart");
                println!("cargo:rustc-link-lib=cublas");
                println!("cargo:rustc-link-lib=cublasLt");
                break;
            }
        }
    }
    
    // Compile CUDA kernels if CUDA is available
    if cfg!(feature = "cuda") {
        let mut build = cc::Build::new();
        build
            .file("src/kernels.cu")
            .flag("-c")
            .flag("-o")
            .out_dir("target/debug/build/gpu-backend-cuda-*/out/");
        
        if let Ok(cuda_path) = std::env::var("CUDA_HOME") {
            build.include(format!("{}/include", cuda_path));
        } else if let Ok(cuda_path) = std::env::var("CUDA_ROOT") {
            build.include(format!("{}/include", cuda_path));
        } else {
            // Try common CUDA installation paths
            let common_paths = [
                "/usr/local/cuda",
                "/opt/cuda",
                "/usr/local/cuda-12.0",
                "/usr/local/cuda-11.8",
            ];
            
            for path in &common_paths {
                if std::path::Path::new(path).exists() {
                    build.include(format!("{}/include", path));
                    break;
                }
            }
        }
        
        build.compile("cuda_kernels");
    }
}