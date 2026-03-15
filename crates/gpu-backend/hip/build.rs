fn main() {
    // Check for ROCm/HIP installation
    if let Ok(rocm_path) = std::env::var("ROCM_PATH") {
        println!("cargo:rustc-link-search=native={}/lib", rocm_path);
        println!("cargo:rustc-link-lib=amdhip64");
        println!("cargo:rustc-link-lib=hipblas");
        println!("cargo:rustc-link-lib=hipblaslt");
    } else if let Ok(rocm_path) = std::env::var("ROCM_ROOT") {
        println!("cargo:rustc-link-search=native={}/lib", rocm_path);
        println!("cargo:rustc-link-lib=amdhip64");
        println!("cargo:rustc-link-lib=hipblas");
        println!("cargo:rustc-link-lib=hipblaslt");
    } else {
        // Try common ROCm installation paths
        let common_paths = [
            "/opt/rocm",
            "/usr/local/rocm",
            "/opt/rocm-5.7",
            "/opt/rocm-5.6",
        ];
        
        for path in &common_paths {
            if std::path::Path::new(path).exists() {
                println!("cargo:rustc-link-search=native={}/lib", path);
                println!("cargo:rustc-link-lib=amdhip64");
                println!("cargo:rustc-link-lib=hipblas");
                println!("cargo:rustc-link-lib=hipblaslt");
                break;
            }
        }
    }
    
    // Compile HIP kernels if HIP is available
    if cfg!(feature = "hip") {
        let mut build = cc::Build::new();
        build
            .file("src/kernels.hip")
            .flag("-c")
            .flag("-o")
            .out_dir("target/debug/build/gpu-backend-hip-*/out/");
        
        if let Ok(rocm_path) = std::env::var("ROCM_PATH") {
            build.include(format!("{}/include", rocm_path));
        } else if let Ok(rocm_path) = std::env::var("ROCM_ROOT") {
            build.include(format!("{}/include", rocm_path));
        } else {
            // Try common ROCm installation paths
            let common_paths = [
                "/opt/rocm",
                "/usr/local/rocm",
                "/opt/rocm-5.7",
                "/opt/rocm-5.6",
            ];
            
            for path in &common_paths {
                if std::path::Path::new(path).exists() {
                    build.include(format!("{}/include", path));
                    break;
                }
            }
        }
        
        build.compile("hip_kernels");
    }
}