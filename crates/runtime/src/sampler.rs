//! Greedy sampler implementation

use ndarray::{ArrayViewD, Axis};
use std::f32;

/// Greedy sampler that selects the token with the highest probability
pub struct GreedySampler;

impl GreedySampler {
    /// Create a new greedy sampler
    pub fn new() -> Self {
        Self
    }
    
    /// Sample the next token using greedy decoding
    /// 
    /// # Arguments
    /// * `logits` - The logits from the model (unnormalized log probabilities)
    /// 
    /// # Returns
    /// The token ID with the highest probability
    pub fn sample(&self, logits: ArrayViewD<f32>) -> usize {
        // For greedy sampling, we simply select the token with the highest logit
        // In a real implementation, we would:
        // 1. Apply temperature scaling if needed
        // 2. Apply top-k or top-p filtering if needed
        // 3. Select the token with the highest probability
        
        // Find the index of the maximum value
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        
        // Iterate through all elements to find the maximum
        for (i, &val) in logits.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        
        max_idx
    }
    
    /// Sample multiple tokens using greedy decoding
    /// 
    /// # Arguments
    /// * `logits` - The logits from the model (batch_size x vocab_size)
    /// 
    /// # Returns
    /// A vector of token IDs, one for each batch item
    pub fn sample_batch(&self, logits: ArrayViewD<f32>) -> Vec<usize> {
        // For batch sampling, we sample each batch item independently
        let batch_size = logits.shape()[0];
        
        let mut samples = Vec::with_capacity(batch_size);
        
        for batch_idx in 0..batch_size {
            // Extract logits for this batch item
            let batch_logits = logits.index_axis(Axis(0), batch_idx);
            
            // Sample using the regular sample method
            let token_id = self.sample(batch_logits.view());
            samples.push(token_id);
        }
        
        samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;
    
    #[test]
    fn test_greedy_sampler_single() {
        let sampler = GreedySampler::new();
        
        // Create test logits where the highest value is at index 3
        let logits = ArrayD::from_shape_vec(vec![5], vec![0.1, 0.2, 0.3, 0.9, 0.4]).unwrap();
        let logits_view = logits.view();
        
        let token_id = sampler.sample(logits_view);
        assert_eq!(token_id, 3);
    }
    
    #[test]
    fn test_greedy_sampler_batch() {
        let sampler = GreedySampler::new();
        
        // Create test logits for a batch of 2
        let logits = ArrayD::from_shape_vec(vec![2, 3], vec![
            0.1, 0.8, 0.3,  // First batch item - max at index 1
            0.6, 0.2, 0.9,  // Second batch item - max at index 2
        ]).unwrap();
        let logits_view = logits.view();
        
        let token_ids = sampler.sample_batch(logits_view);
        assert_eq!(token_ids, vec![1, 2]);
    }
}