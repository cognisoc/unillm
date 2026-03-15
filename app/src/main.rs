mod transfer_test;

fn main() {
    println!("UniLLM - Unikernel-based LLM inference engine");
    println!("Starting initialization...");
    
    // Run transfer validation tests
    if let Err(e) = transfer_test::test_cuda_transfers() {
        eprintln!("CUDA transfer test failed: {}", e);
    }
    
    if let Err(e) = transfer_test::test_hip_transfers() {
        eprintln!("HIP transfer test failed: {}", e);
    }
    
    // Test memory info
    if let Err(e) = transfer_test::test_memory_info() {
        eprintln!("Memory info test failed: {}", e);
    }
    
    println!("Initialization completed");
}