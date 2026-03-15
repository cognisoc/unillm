//! Model loading and management

use ndarray::ArrayD;
use safetensors::{SafeTensors, tensor::TensorView};
use std::collections::HashMap;
use std::path::Path;
use std::fs;

/// A simple representation of a loaded model
pub struct Model {
    pub weights: HashMap<String, ArrayD<f32>>,
    pub config: ModelConfig,
}

/// Model configuration
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl ModelConfig {
    pub fn new(vocab_size: usize, hidden_size: usize, num_layers: usize, num_heads: usize) -> Self {
        Self {
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            head_dim: hidden_size / num_heads,
        }
    }
}

impl Model {
    /// Create a new model
    pub fn new(config: ModelConfig) -> Self {
        Self {
            weights: HashMap::new(),
            config,
        }
    }
    
    /// Load a model from a safetensors file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        println!("Loading model from: {:?}", path.as_ref());
        
        // Check if the file exists
        if !path.as_ref().exists() {
            return Err(format!("Model file not found: {:?}", path.as_ref()).into());
        }
        
        // Read the safetensors file
        let data = fs::read(path.as_ref())?;
        let tensors = SafeTensors::deserialize(&data)?;
        
        // Create a basic config (in a real implementation, this would come from a config.json)
        let config = ModelConfig::new(32000, 4096, 32, 32); // LLaMA-2 7B-like config
        let mut model = Model::new(config);
        
        // Load all tensors from the safetensors file
        for (name, tensor) in tensors.tensors() {
            let shape = tensor.shape();
            let data = tensor.data();
            
            // Convert the tensor data to our ArrayD format
            let array = match shape.len() {
                1 => ArrayD::from_shape_vec(vec![shape[0]], data.to_vec())?,
                2 => ArrayD::from_shape_vec(vec![shape[0], shape[1]], data.to_vec())?,
                3 => ArrayD::from_shape_vec(vec![shape[0], shape[1], shape[2]], data.to_vec())?,
                4 => ArrayD::from_shape_vec(vec![shape[0], shape[1], shape[2], shape[3]], data.to_vec())?,
                _ => {
                    println!("Warning: Skipping tensor {} with unsupported shape {:?}", name, shape);
                    continue;
                }
            };
            
            model.weights.insert(name.to_string(), array);
        }
        
        println!("Loaded model with {} tensors", model.weights.len());
        
        // Validate that we have the essential tensors
        Self::validate_model(&model)?;
        
        Ok(model)
    }
    
    /// Validate that the model has the essential tensors
    fn validate_model(model: &Model) -> Result<(), Box<dyn std::error::Error>> {
        let required_tensors = [
            "model.embed_tokens.weight",
            "model.norm.weight",
        ];
        
        for tensor_name in &required_tensors {
            if !model.weights.contains_key(*tensor_name) {
                return Err(format!("Missing required tensor: {}", tensor_name).into());
            }
        }
        
        // Check for at least one transformer layer
        let mut has_layers = false;
        for tensor_name in model.weights.keys() {
            if tensor_name.contains("model.layers.0.") {
                has_layers = true;
                break;
            }
        }
        
        if !has_layers {
            return Err("No transformer layers found in model".into());
        }
        
        Ok(())
    }
    
    /// Load a model from multiple safetensors files (for large models split across files)
    pub fn load_from_directory<P: AsRef<Path>>(dir_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        println!("Loading model from directory: {:?}", dir_path.as_ref());
        
        let dir = fs::read_dir(dir_path.as_ref())?;
        let mut all_tensors = HashMap::new();
        
        // Find all .safetensors files in the directory
        let mut safetensors_files = Vec::new();
        for entry in dir {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                safetensors_files.push(path);
            }
        }
        
        if safetensors_files.is_empty() {
            return Err("No .safetensors files found in directory".into());
        }
        
        // Sort files to ensure consistent loading order
        safetensors_files.sort();
        
        // Load tensors from each file
        for file_path in safetensors_files {
            println!("Loading tensors from: {:?}", file_path);
            let data = fs::read(&file_path)?;
            let tensors = SafeTensors::deserialize(&data)?;
            
            for (name, tensor) in tensors.tensors() {
                let shape = tensor.shape();
                let data = tensor.data();
                
                // Convert the tensor data to our ArrayD format
                let array = match shape.len() {
                    1 => ArrayD::from_shape_vec(vec![shape[0]], data.to_vec())?,
                    2 => ArrayD::from_shape_vec(vec![shape[0], shape[1]], data.to_vec())?,
                    3 => ArrayD::from_shape_vec(vec![shape[0], shape[1], shape[2]], data.to_vec())?,
                    4 => ArrayD::from_shape_vec(vec![shape[0], shape[1], shape[2], shape[3]], data.to_vec())?,
                    _ => {
                        println!("Warning: Skipping tensor {} with unsupported shape {:?}", name, shape);
                        continue;
                    }
                };
                
                all_tensors.insert(name.to_string(), array);
            }
        }
        
        // Create a basic config (in a real implementation, this would come from a config.json)
        let config = ModelConfig::new(32000, 4096, 32, 32); // LLaMA-2 7B-like config
        let mut model = Model::new(config);
        model.weights = all_tensors;
        
        println!("Loaded model with {} tensors from {} files", model.weights.len(), safetensors_files.len());
        
        // Validate that we have the essential tensors
        Self::validate_model(&model)?;
        
        Ok(model)
    }
    
    /// Get a weight tensor by name
    pub fn get_weight(&self, name: &str) -> Option<&ArrayD<f32>> {
        self.weights.get(name)
    }
    
    /// Get the model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
}