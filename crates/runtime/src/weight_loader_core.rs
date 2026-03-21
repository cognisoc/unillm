//! Core Weight Loading Abstraction
//!
//! This module provides unified weight loading that converts any supported
//! weight format into our core tensor abstraction.

use crate::tensor_core::{Tensor, Device, DataType, CpuStorage};
use crate::model_core::{ModelWeights, WeightMetadata, WeightFormat};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use anyhow::Result;
use serde_json::Value;
use safetensors::SafeTensors;

/// Core weight loader trait - abstracts loading from different formats
pub trait WeightLoader: Send + Sync {
    /// Load weights from path
    fn load_weights(&self, path: &Path) -> Result<ModelWeights>;

    /// Check if this loader supports the given path
    fn supports(&self, path: &Path) -> bool;

    /// Get format name
    fn format_name(&self) -> &str;
}

/// Unified weight loader that dispatches to appropriate format loaders
pub struct UnifiedWeightLoader {
    loaders: Vec<Box<dyn WeightLoader>>,
}

/// SafeTensors weight loader
pub struct SafeTensorsWeightLoader;

/// PyTorch weight loader
pub struct PyTorchWeightLoader;

/// GGUF weight loader
pub struct GGUFWeightLoader;

impl UnifiedWeightLoader {
    /// Create new unified loader with all supported formats
    pub fn new() -> Self {
        let mut loaders: Vec<Box<dyn WeightLoader>> = Vec::new();
        loaders.push(Box::new(SafeTensorsWeightLoader));
        loaders.push(Box::new(PyTorchWeightLoader));
        loaders.push(Box::new(GGUFWeightLoader));

        Self { loaders }
    }

    /// Load weights from path (auto-detect format)
    pub fn load_weights(&self, path: &Path) -> Result<ModelWeights> {
        // Try each loader until one works
        for loader in &self.loaders {
            if loader.supports(path) {
                println!("Loading weights using {} loader", loader.format_name());
                return loader.load_weights(path);
            }
        }

        Err(anyhow::anyhow!(
            "No suitable weight loader found for path: {}",
            path.display()
        ))
    }

    /// Detect weight format
    pub fn detect_format(&self, path: &Path) -> Option<&str> {
        for loader in &self.loaders {
            if loader.supports(path) {
                return Some(loader.format_name());
            }
        }
        None
    }

    /// Get all supported formats
    pub fn supported_formats(&self) -> Vec<&str> {
        self.loaders.iter().map(|l| l.format_name()).collect()
    }
}

impl SafeTensorsWeightLoader {
    /// Convert SafeTensors dtype to our DataType
    fn convert_dtype(&self, dtype: safetensors::Dtype) -> Result<DataType> {
        match dtype {
            safetensors::Dtype::F32 => Ok(DataType::Float32),
            safetensors::Dtype::F16 => Ok(DataType::Float16),
            safetensors::Dtype::BF16 => Ok(DataType::BFloat16),
            safetensors::Dtype::I32 => Ok(DataType::Int32),
            safetensors::Dtype::I64 => Ok(DataType::Int64),
            safetensors::Dtype::I8 => Ok(DataType::Int8),
            safetensors::Dtype::BOOL => Ok(DataType::Bool),
            _ => Err(anyhow::anyhow!("Unsupported dtype: {:?}", dtype)),
        }
    }

    /// Load single SafeTensors file
    fn load_safetensors_file(&self, file_path: &Path) -> Result<HashMap<String, Tensor>> {
        println!("Loading SafeTensors file: {}", file_path.display());

        let data = std::fs::read(file_path)?;
        let safetensors = SafeTensors::deserialize(&data)?;

        let mut tensors = HashMap::new();

        // Iterate through all tensors in the file
        for (name, tensor_view) in safetensors.tensors() {
            println!("  Loading tensor: {} {:?} {:?}", name, tensor_view.shape(), tensor_view.dtype());

            // Convert dtype
            let dtype = self.convert_dtype(tensor_view.dtype())?;

            // Get tensor data
            let tensor_data = tensor_view.data();

            // Create storage with actual data
            let storage = Arc::new(CpuStorage::new(tensor_data.to_vec(), Device::CPU));

            // Create our Tensor
            let tensor = Tensor::new(
                tensor_view.shape().to_vec(),
                dtype,
                Device::CPU,
                storage
            );

            tensors.insert(name.to_string(), tensor);
        }

        println!("  ✓ Loaded {} tensors", tensors.len());
        Ok(tensors)
    }
}

impl WeightLoader for SafeTensorsWeightLoader {
    fn load_weights(&self, path: &Path) -> Result<ModelWeights> {
        let mut all_tensors = HashMap::new();

        if path.is_file() && path.extension().map_or(false, |ext| ext == "safetensors") {
            // Single file
            let tensors = self.load_safetensors_file(path)?;
            all_tensors.extend(tensors);
        } else if path.is_dir() {
            // Directory with potentially multiple files
            let entries = std::fs::read_dir(path)?;
            for entry in entries {
                let entry = entry?;
                let file_path = entry.path();
                if file_path.extension().map_or(false, |ext| ext == "safetensors") {
                    let tensors = self.load_safetensors_file(&file_path)?;
                    all_tensors.extend(tensors);
                }
            }
        }

        if all_tensors.is_empty() {
            return Err(anyhow::anyhow!("No SafeTensors files found in {}", path.display()));
        }

        // Calculate total parameters
        let total_params: usize = all_tensors.values()
            .map(|tensor| tensor.numel())
            .sum();

        // Determine primary dtype
        let primary_dtype = all_tensors.values()
            .next()
            .map(|t| match t.dtype() {
                DataType::Float32 => "float32",
                DataType::Float16 => "float16",
                DataType::BFloat16 => "bfloat16",
                DataType::Int32 => "int32",
                DataType::Int64 => "int64",
                DataType::Int8 => "int8",
                DataType::Bool => "bool",
            })
            .unwrap_or("unknown");

        let metadata = WeightMetadata {
            architecture: "unknown".to_string(), // Will be filled from config.json
            total_params,
            format: WeightFormat::SafeTensors,
            dtype: primary_dtype.to_string(),
        };

        Ok(ModelWeights::new(all_tensors, metadata))
    }

    fn supports(&self, path: &Path) -> bool {
        if path.is_file() {
            return path.extension().map_or(false, |ext| ext == "safetensors");
        }

        if path.is_dir() {
            // Check if directory contains SafeTensors files
            if let Ok(entries) = std::fs::read_dir(path) {
                for entry in entries.flatten() {
                    if entry.path().extension().map_or(false, |ext| ext == "safetensors") {
                        return true;
                    }
                }
            }
        }

        false
    }

    fn format_name(&self) -> &str {
        "SafeTensors"
    }
}

impl WeightLoader for PyTorchWeightLoader {
    fn load_weights(&self, path: &Path) -> Result<ModelWeights> {
        println!("Loading PyTorch weights from: {}", path.display());

        // Placeholder implementation
        // Real implementation would use PyTorch bindings or pickle parsing

        let mut tensors = HashMap::new();

        // Create placeholder tensors
        let storage = Arc::new(CpuStorage::zeros(2048 * 4));
        let tensor = Tensor::new(
            vec![512, 4],
            DataType::Float32,
            Device::CPU,
            storage
        );

        tensors.insert("pytorch_weight".to_string(), tensor);

        let metadata = WeightMetadata {
            architecture: "unknown".to_string(),
            total_params: tensors.len(),
            format: WeightFormat::PyTorch,
            dtype: "float32".to_string(),
        };

        Ok(ModelWeights::new(tensors, metadata))
    }

    fn supports(&self, path: &Path) -> bool {
        if path.is_file() {
            return path.extension().map_or(false, |ext| ext == "bin" || ext == "pt");
        }

        if path.is_dir() {
            return path.join("pytorch_model.bin").exists() ||
                   path.join("model.pt").exists();
        }

        false
    }

    fn format_name(&self) -> &str {
        "PyTorch"
    }
}

impl WeightLoader for GGUFWeightLoader {
    fn load_weights(&self, path: &Path) -> Result<ModelWeights> {
        println!("Loading GGUF weights from: {}", path.display());

        // Placeholder implementation
        // Real implementation would parse GGUF format

        let mut tensors = HashMap::new();

        // Create placeholder tensors
        let storage = Arc::new(CpuStorage::zeros(1024 * 4));
        let tensor = Tensor::new(
            vec![256, 4],
            DataType::Float32,
            Device::CPU,
            storage
        );

        tensors.insert("gguf_weight".to_string(), tensor);

        let metadata = WeightMetadata {
            architecture: "unknown".to_string(),
            total_params: tensors.len(),
            format: WeightFormat::GGUF,
            dtype: "float32".to_string(),
        };

        Ok(ModelWeights::new(tensors, metadata))
    }

    fn supports(&self, path: &Path) -> bool {
        if path.is_file() {
            return path.extension().map_or(false, |ext| ext == "gguf");
        }

        if path.is_dir() {
            if let Ok(entries) = std::fs::read_dir(path) {
                for entry in entries.flatten() {
                    if entry.path().extension().map_or(false, |ext| ext == "gguf") {
                        return true;
                    }
                }
            }
        }

        false
    }

    fn format_name(&self) -> &str {
        "GGUF"
    }
}

/// Model configuration loader - loads config.json files
pub struct ConfigLoader;

impl ConfigLoader {
    /// Load model configuration from config.json
    pub fn load_config(&self, model_path: &Path) -> Result<Value> {
        let config_path = if model_path.is_file() {
            model_path.parent()
                .ok_or_else(|| anyhow::anyhow!("No parent directory"))?
                .join("config.json")
        } else {
            model_path.join("config.json")
        };

        if !config_path.exists() {
            return Err(anyhow::anyhow!("config.json not found at {}", config_path.display()));
        }

        let content = std::fs::read_to_string(&config_path)?;
        let config: Value = serde_json::from_str(&content)?;

        Ok(config)
    }

    /// Extract architecture from config
    pub fn get_architecture(&self, config: &Value) -> Result<String> {
        // Try common architecture field names
        if let Some(arch) = config.get("architectures").and_then(|a| a.as_array()) {
            if let Some(first_arch) = arch.first().and_then(|a| a.as_str()) {
                return Ok(first_arch.to_string());
            }
        }

        if let Some(arch) = config.get("model_type").and_then(|a| a.as_str()) {
            return Ok(arch.to_string());
        }

        if let Some(arch) = config.get("architecture").and_then(|a| a.as_str()) {
            return Ok(arch.to_string());
        }

        Err(anyhow::anyhow!("Could not determine model architecture from config"))
    }
}

/// Complete model loading pipeline
pub struct ModelLoader {
    weight_loader: UnifiedWeightLoader,
    config_loader: ConfigLoader,
}

impl ModelLoader {
    pub fn new() -> Self {
        Self {
            weight_loader: UnifiedWeightLoader::new(),
            config_loader: ConfigLoader,
        }
    }

    /// Load complete model (config + weights)
    pub fn load_model(&self, path: &Path) -> Result<(Value, ModelWeights)> {
        println!("Loading model from: {}", path.display());

        // Load configuration
        let config = self.config_loader.load_config(path)?;
        println!("✓ Configuration loaded");

        // Load weights
        let weights = self.weight_loader.load_weights(path)?;
        println!("✓ Weights loaded ({} tensors)", weights.tensors.len());

        Ok((config, weights))
    }

    /// Get supported formats
    pub fn supported_formats(&self) -> Vec<&str> {
        self.weight_loader.supported_formats()
    }
}

/// Global model loader
static MODEL_LOADER: std::sync::OnceLock<ModelLoader> = std::sync::OnceLock::new();

/// Get global model loader
pub fn loader() -> &'static ModelLoader {
    MODEL_LOADER.get_or_init(|| ModelLoader::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_weight_loader_creation() {
        let loader = UnifiedWeightLoader::new();
        let formats = loader.supported_formats();

        assert!(formats.contains(&"SafeTensors"));
        assert!(formats.contains(&"PyTorch"));
        assert!(formats.contains(&"GGUF"));
    }

    #[test]
    fn test_config_loader() {
        let loader = ConfigLoader;

        // Test with dummy config
        let config_json = r#"
        {
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 32000,
            "hidden_size": 4096
        }
        "#;

        let config: Value = serde_json::from_str(config_json).unwrap();
        let arch = loader.get_architecture(&config).unwrap();

        assert_eq!(arch, "LlamaForCausalLM");
    }
}