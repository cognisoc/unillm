//! Runtime crate for model graph, loader, and sampler

mod models;
mod sampler;
mod decode_loop;

pub use models::{Model, ModelConfig};
pub use sampler::GreedySampler;
pub use decode_loop::{DecodeLoop, DecodeState};

/// Runtime implementation
pub struct Runtime {
    model: Option<Model>,
    sampler: GreedySampler,
}

impl Runtime {
    /// Create a new runtime instance
    pub fn new() -> Self {
        Self {
            model: None,
            sampler: GreedySampler::new(),
        }
    }
    
    /// Load a model
    pub fn load_model<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let model = Model::load(path)?;
        self.model = Some(model);
        Ok(())
    }
    
    /// Load a model from a directory (for models split across multiple files)
    pub fn load_model_from_directory<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let model = Model::load_from_directory(path)?;
        self.model = Some(model);
        Ok(())
    }
    
    /// Generate text using the loaded model
    pub fn generate(&self, prompt: &str, max_length: usize, eos_token_id: u32) -> Result<String, Box<dyn std::error::Error>> {
        let model = self.model.as_ref().ok_or("No model loaded")?;
        
        // For now, this is a placeholder implementation
        // In a real implementation, we would:
        // 1. Tokenize the prompt
        // 2. Run the model forward pass
        // 3. Use the decode loop to generate tokens
        // 4. Decode the tokens back to text
        
        println!("Generating text for prompt: '{}'", prompt);
        println!("Max length: {}, EOS token: {}", max_length, eos_token_id);
        
        // Placeholder response
        Ok(format!("Generated response for: {}", prompt))
    }
    
    /// Get the loaded model
    pub fn model(&self) -> Option<&Model> {
        self.model.as_ref()
    }
    
    /// Get the sampler
    pub fn sampler(&self) -> &GreedySampler {
        &self.sampler
    }
    
    /// Create a decode loop for the given initial tokens
    pub fn create_decode_loop(&self, initial_tokens: Vec<u32>, max_length: usize, eos_token_id: u32) -> DecodeLoop {
        DecodeLoop::new(initial_tokens, max_length, eos_token_id)
    }
}