//! Test program for the sampler

use ndarray::ArrayD;
use runtime::{Runtime, GreedySampler};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing sampler...");
    
    // Create a runtime instance
    let runtime = Runtime::new();
    let sampler = runtime.sampler();
    
    // Test single sampling
    let logits = ArrayD::from_shape_vec(vec![5], vec![0.1, 0.2, 0.8, 0.3, 0.4])?;
    let token_id = sampler.sample(logits.view());
    println!("Sampled token ID: {}", token_id); // Should be 2 (highest logit)
    
    // Test batch sampling
    let batch_logits = ArrayD::from_shape_vec(vec![2, 4], vec![
        0.1, 0.7, 0.2, 0.3,  // First batch - max at index 1
        0.4, 0.1, 0.2, 0.9,  // Second batch - max at index 3
    ])?;
    let token_ids = sampler.sample_batch(batch_logits.view());
    println!("Batch sampled token IDs: {:?}", token_ids); // Should be [1, 3]
    
    Ok(())
}