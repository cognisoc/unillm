//! Decode loop implementation for steady-state inference

use ndarray::{ArrayD, ArrayViewD, Axis};
use std::collections::VecDeque;

/// State for the decode loop
#[derive(Debug, Clone)]
pub struct DecodeState {
    /// Current sequence of token IDs
    pub tokens: Vec<u32>,
    /// Current position in the sequence
    pub position: usize,
    /// Whether generation is complete
    pub finished: bool,
    /// Maximum sequence length
    pub max_length: usize,
    /// End-of-sequence token ID
    pub eos_token_id: u32,
}

impl DecodeState {
    /// Create a new decode state
    pub fn new(initial_tokens: Vec<u32>, max_length: usize, eos_token_id: u32) -> Self {
        Self {
            position: initial_tokens.len(),
            tokens: initial_tokens,
            finished: false,
            max_length,
            eos_token_id,
        }
    }
    
    /// Add a new token to the sequence
    pub fn add_token(&mut self, token_id: u32) {
        self.tokens.push(token_id);
        self.position += 1;
        
        // Check if we should stop
        if token_id == self.eos_token_id || self.position >= self.max_length {
            self.finished = true;
        }
    }
    
    /// Get the current sequence length
    pub fn length(&self) -> usize {
        self.tokens.len()
    }
    
    /// Get the last token
    pub fn last_token(&self) -> Option<u32> {
        self.tokens.last().copied()
    }
}

/// Decode loop for steady-state inference
pub struct DecodeLoop {
    /// KV cache for attention (placeholder for now)
    kv_cache: VecDeque<ArrayD<f32>>,
    /// Current decode state
    state: DecodeState,
}

impl DecodeLoop {
    /// Create a new decode loop
    pub fn new(initial_tokens: Vec<u32>, max_length: usize, eos_token_id: u32) -> Self {
        Self {
            kv_cache: VecDeque::new(),
            state: DecodeState::new(initial_tokens, max_length, eos_token_id),
        }
    }
    
    /// Run one step of the decode loop
    /// 
    /// # Arguments
    /// * `logits` - The logits from the model (vocab_size)
    /// * `sampler` - The sampler to use for token selection
    /// 
    /// # Returns
    /// The next token ID, or None if generation is complete
    pub fn step(&mut self, logits: ArrayViewD<f32>, sampler: &crate::sampler::GreedySampler) -> Option<u32> {
        if self.state.finished {
            return None;
        }
        
        // Sample the next token
        let token_id = sampler.sample(logits) as u32;
        
        // Add the token to the sequence
        self.state.add_token(token_id);
        
        Some(token_id)
    }
    
    /// Run the decode loop until completion
    /// 
    /// # Arguments
    /// * `model_fn` - Function that takes current tokens and returns logits
    /// * `sampler` - The sampler to use for token selection
    /// 
    /// # Returns
    /// The complete generated sequence
    pub fn run<F>(&mut self, mut model_fn: F, sampler: &crate::sampler::GreedySampler) -> Result<Vec<u32>, Box<dyn std::error::Error>>
    where
        F: FnMut(&[u32]) -> Result<ArrayD<f32>, Box<dyn std::error::Error>>,
    {
        let mut generated_tokens = Vec::new();
        
        while !self.state.finished {
            // Get logits for the current sequence
            let logits = model_fn(&self.state.tokens)?;
            
            // Run one step of the decode loop
            if let Some(token_id) = self.step(logits.view(), sampler) {
                generated_tokens.push(token_id);
            } else {
                break;
            }
        }
        
        Ok(generated_tokens)
    }
    
    /// Get the current decode state
    pub fn state(&self) -> &DecodeState {
        &self.state
    }
    
    /// Get the current sequence
    pub fn tokens(&self) -> &[u32] {
        &self.state.tokens
    }
    
    /// Check if generation is complete
    pub fn is_finished(&self) -> bool {
        self.state.finished
    }
    
    /// Reset the decode loop with new initial tokens
    pub fn reset(&mut self, initial_tokens: Vec<u32>, max_length: usize, eos_token_id: u32) {
        self.state = DecodeState::new(initial_tokens, max_length, eos_token_id);
        self.kv_cache.clear();
    }
}

/// Batch decode loop for processing multiple sequences
pub struct BatchDecodeLoop {
    /// Individual decode loops for each sequence
    loops: Vec<DecodeLoop>,
    /// Number of active sequences
    active_count: usize,
}

impl BatchDecodeLoop {
    /// Create a new batch decode loop
    pub fn new(initial_tokens_batch: Vec<Vec<u32>>, max_length: usize, eos_token_id: u32) -> Self {
        let loops: Vec<DecodeLoop> = initial_tokens_batch
            .into_iter()
            .map(|tokens| DecodeLoop::new(tokens, max_length, eos_token_id))
            .collect();
        
        let active_count = loops.len();
        
        Self { loops, active_count }
    }
    
    /// Run one step of the batch decode loop
    /// 
    /// # Arguments
    /// * `logits_batch` - The logits from the model (batch_size x vocab_size)
    /// * `sampler` - The sampler to use for token selection
    /// 
    /// # Returns
    /// Vector of next token IDs, with None for finished sequences
    pub fn step(&mut self, logits_batch: ArrayViewD<f32>, sampler: &crate::sampler::GreedySampler) -> Vec<Option<u32>> {
        let mut results = Vec::new();
        
        for (i, loop_state) in self.loops.iter_mut().enumerate() {
            if loop_state.is_finished() {
                results.push(None);
                continue;
            }
            
            // Extract logits for this sequence
            let batch_logits = logits_batch.index_axis(Axis(0), i);
            let token_id = loop_state.step(batch_logits.view(), sampler);
            results.push(token_id);
        }
        
        // Update active count
        self.active_count = self.loops.iter().filter(|l| !l.is_finished()).count();
        
        results
    }
    
    /// Get the number of active sequences
    pub fn active_count(&self) -> usize {
        self.active_count
    }
    
    /// Check if all sequences are finished
    pub fn all_finished(&self) -> bool {
        self.active_count == 0
    }
    
    /// Get all generated sequences
    pub fn get_sequences(&self) -> Vec<Vec<u32>> {
        self.loops.iter().map(|l| l.tokens().to_vec()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;
    
    #[test]
    fn test_decode_state() {
        let mut state = DecodeState::new(vec![1, 2, 3], 10, 0);
        
        assert_eq!(state.length(), 3);
        assert_eq!(state.position, 3);
        assert!(!state.finished);
        
        state.add_token(4);
        assert_eq!(state.length(), 4);
        assert_eq!(state.position, 4);
        
        state.add_token(0); // EOS token
        assert!(state.finished);
    }
    
    #[test]
    fn test_decode_loop() {
        let mut decode_loop = DecodeLoop::new(vec![1, 2], 5, 0);
        let sampler = crate::sampler::GreedySampler::new();
        
        // Mock model function that returns logits with max at index 3
        let model_fn = |_tokens: &[u32]| -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
            Ok(ArrayD::from_shape_vec(vec![5], vec![0.1, 0.2, 0.3, 0.9, 0.4]).unwrap())
        };
        
        let result = decode_loop.run(model_fn, &sampler).unwrap();
        assert_eq!(result, vec![3]);
        assert!(decode_loop.is_finished());
    }
}