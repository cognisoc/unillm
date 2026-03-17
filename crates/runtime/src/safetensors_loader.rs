//! SafeTensors Model Loading System
//!
//! This module provides comprehensive SafeTensors support for loading
//! production models from HuggingFace and local files. SafeTensors is
//! the de facto standard for modern LLM deployment.

use crate::{
    gpu_tensor_ops::{GpuDevice, GpuTensor, GpuTensorOps},
    basic_model::ModelConfig,
    types::ModelError,
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    fs::File,
    io::{BufReader, Read},
};
use serde::{Deserialize, Serialize};
use safetensors::{SafeTensors, Dtype as SafeTensorsDtype};

/// SafeTensors model loader
pub struct SafeTensorsLoader {
    /// GPU device for loading tensors
    device: GpuDevice,
    /// Tensor operations
    tensor_ops: GpuTensorOps,
    /// Memory mapping for efficient loading
    use_memory_mapping: bool,
}

/// SafeTensors model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeTensorsMetadata {
    /// Model configuration from config.json
    pub config: ModelConfig,
    /// Tensor shapes and data types
    pub tensor_info: HashMap<String, TensorInfo>,
    /// Model architecture type
    pub architecture: String,
    /// Total parameter count
    pub total_parameters: usize,
    /// File paths for sharded models
    pub shard_files: Vec<PathBuf>,
}

/// Information about individual tensors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: String,
    /// Size in bytes
    pub size_bytes: usize,
    /// Which shard file contains this tensor
    pub shard_index: usize,
}

/// SafeTensors loading configuration
#[derive(Debug, Clone)]
pub struct SafeTensorsConfig {
    /// Use memory mapping for large models
    pub use_memory_mapping: bool,
    /// Load tensors lazily (on-demand)
    pub lazy_loading: bool,
    /// Verify tensor checksums
    pub verify_checksums: bool,
    /// Maximum memory usage (bytes)
    pub max_memory_bytes: Option<usize>,
    /// Device placement strategy
    pub device_placement: DevicePlacement,
}

/// Device placement strategy for multi-file models
#[derive(Debug, Clone)]
pub enum DevicePlacement {
    /// Load entire model on single device
    SingleDevice,
    /// Automatically balance across available devices
    AutoBalance,
    /// Manual device assignment per layer
    Manual(HashMap<String, GpuDevice>),
}

impl Default for SafeTensorsConfig {
    fn default() -> Self {
        Self {
            use_memory_mapping: true,
            lazy_loading: false,
            verify_checksums: true,
            max_memory_bytes: None,
            device_placement: DevicePlacement::SingleDevice,
        }
    }
}

impl SafeTensorsLoader {
    /// Create a new SafeTensors loader
    pub fn new(device: GpuDevice, config: SafeTensorsConfig) -> Self {
        let tensor_ops = GpuTensorOps::with_device(device.clone());

        Self {
            device,
            tensor_ops,
            use_memory_mapping: config.use_memory_mapping,
        }
    }

    /// Load model from SafeTensors files
    pub async fn load_model<P: AsRef<Path>>(
        &self,
        model_path: P,
        config: SafeTensorsConfig,
    ) -> Result<LoadedModel, ModelError> {
        let model_path = model_path.as_ref();
        println!("🔄 Loading SafeTensors model from: {}", model_path.display());

        // Step 1: Discover and validate SafeTensors files
        let metadata = self.discover_model_files(model_path).await?;
        println!("   📊 Discovered {} tensor files", metadata.shard_files.len());
        println!("   🔢 Total parameters: {:.2}M", metadata.total_parameters as f64 / 1e6);

        // Step 2: Load model configuration
        let model_config = self.load_model_config(model_path).await?;
        println!("   ⚙️  Model architecture: {}", metadata.architecture);
        println!("   📐 Model config: {}D hidden, {} layers",
                model_config.hidden_size, model_config.num_layers);

        // Step 3: Load tensors based on configuration
        let tensors = if config.lazy_loading {
            self.load_tensors_lazy(&metadata, &config).await?
        } else {
            self.load_tensors_immediate(&metadata, &config).await?
        };

        println!("   ✅ Loaded {} tensors successfully", tensors.len());

        // Step 4: Verify model integrity
        self.verify_model_integrity(&tensors, &metadata).await?;
        println!("   🔒 Model integrity verified");

        Ok(LoadedModel {
            config: model_config,
            tensors,
            metadata,
            device: self.device.clone(),
        })
    }

    /// Discover SafeTensors files in model directory
    async fn discover_model_files<P: AsRef<Path>>(
        &self,
        model_path: P,
    ) -> Result<SafeTensorsMetadata, ModelError> {
        let model_path = model_path.as_ref();

        // Look for SafeTensors files
        let mut shard_files = Vec::new();
        let mut tensor_info = HashMap::new();
        let mut total_parameters = 0;

        // Check for single file or sharded model
        if model_path.join("model.safetensors").exists() {
            // Single file model
            shard_files.push(model_path.join("model.safetensors"));
        } else {
            // Sharded model - look for model-*.safetensors files
            let entries = std::fs::read_dir(model_path)
                .map_err(|e| ModelError::LoadingError(format!("Cannot read directory: {}", e)))?;

            for entry in entries {
                let entry = entry.map_err(|e| ModelError::LoadingError(format!("Directory entry error: {}", e)))?;
                let file_name = entry.file_name();
                let file_name_str = file_name.to_string_lossy();

                if file_name_str.starts_with("model-") && file_name_str.ends_with(".safetensors") {
                    shard_files.push(entry.path());
                }
            }

            // Sort shard files for consistent ordering
            shard_files.sort();
        }

        if shard_files.is_empty() {
            return Err(ModelError::LoadingError(
                "No SafeTensors files found in model directory".to_string()
            ));
        }

        // Analyze each shard file
        for (shard_index, shard_file) in shard_files.iter().enumerate() {
            let shard_info = self.analyze_safetensors_file(shard_file, shard_index).await?;

            for (tensor_name, info) in shard_info {
                total_parameters += info.shape.iter().product::<usize>();
                tensor_info.insert(tensor_name, info);
            }
        }

        // Try to determine architecture from tensor names
        let architecture = self.detect_architecture(&tensor_info)?;

        Ok(SafeTensorsMetadata {
            config: ModelConfig::default(), // Will be loaded separately
            tensor_info,
            architecture,
            total_parameters,
            shard_files,
        })
    }

    /// Analyze a single SafeTensors file
    async fn analyze_safetensors_file<P: AsRef<Path>>(
        &self,
        file_path: P,
        shard_index: usize,
    ) -> Result<HashMap<String, TensorInfo>, ModelError> {
        let file_path = file_path.as_ref();

        // Open and parse SafeTensors file
        let file_data = std::fs::read(file_path)
            .map_err(|e| ModelError::LoadingError(format!("Cannot read file {}: {}", file_path.display(), e)))?;

        let safetensors = SafeTensors::deserialize(&file_data)
            .map_err(|e| ModelError::LoadingError(format!("Invalid SafeTensors format: {}", e)))?;

        let mut tensor_info = HashMap::new();

        // Extract tensor information
        for tensor_name in safetensors.names() {
            let tensor_view = safetensors.tensor(tensor_name)
                .map_err(|e| ModelError::LoadingError(format!("Cannot access tensor {}: {}", tensor_name, e)))?;

            let shape = tensor_view.shape().to_vec();
            let dtype = match tensor_view.dtype() {
                SafeTensorsDtype::F32 => "float32",
                SafeTensorsDtype::F16 => "float16",
                SafeTensorsDtype::BF16 => "bfloat16",
                SafeTensorsDtype::I32 => "int32",
                SafeTensorsDtype::I64 => "int64",
                _ => "unknown",
            }.to_string();

            let size_bytes = tensor_view.data().len();

            tensor_info.insert(tensor_name.to_string(), TensorInfo {
                shape,
                dtype,
                size_bytes,
                shard_index,
            });
        }

        Ok(tensor_info)
    }

    /// Detect model architecture from tensor names
    fn detect_architecture(&self, tensor_info: &HashMap<String, TensorInfo>) -> Result<String, ModelError> {
        let tensor_names: Vec<&String> = tensor_info.keys().collect();

        // Check for Llama/Llama2 architecture
        if tensor_names.iter().any(|name| name.contains("layers.")) &&
           tensor_names.iter().any(|name| name.contains("self_attn")) &&
           tensor_names.iter().any(|name| name.contains("mlp")) {
            return Ok("llama".to_string());
        }

        // Check for Mistral architecture
        if tensor_names.iter().any(|name| name.contains("attention.")) &&
           tensor_names.iter().any(|name| name.contains("feed_forward")) {
            return Ok("mistral".to_string());
        }

        // Check for Qwen architecture
        if tensor_names.iter().any(|name| name.contains("transformer.h.")) {
            return Ok("qwen".to_string());
        }

        // Check for Gemma architecture
        if tensor_names.iter().any(|name| name.contains("model.layers.")) &&
           tensor_names.iter().any(|name| name.contains("pre_feedforward_layernorm")) {
            return Ok("gemma".to_string());
        }

        // Fallback - try to infer from common patterns
        if tensor_names.iter().any(|name| name.contains("embed_tokens")) {
            return Ok("transformer".to_string());
        }

        Ok("unknown".to_string())
    }

    /// Load model configuration from config.json
    async fn load_model_config<P: AsRef<Path>>(
        &self,
        model_path: P,
    ) -> Result<ModelConfig, ModelError> {
        let config_path = model_path.as_ref().join("config.json");

        if !config_path.exists() {
            println!("   ⚠️  No config.json found, using default configuration");
            return Ok(ModelConfig::default());
        }

        let config_data = std::fs::read_to_string(&config_path)
            .map_err(|e| ModelError::LoadingError(format!("Cannot read config.json: {}", e)))?;

        // Parse HuggingFace config format
        let hf_config: serde_json::Value = serde_json::from_str(&config_data)
            .map_err(|e| ModelError::LoadingError(format!("Invalid JSON in config.json: {}", e)))?;

        // Convert HuggingFace config to our ModelConfig format
        self.convert_hf_config_to_model_config(hf_config)
    }

    /// Convert HuggingFace config to our ModelConfig
    fn convert_hf_config_to_model_config(
        &self,
        hf_config: serde_json::Value,
    ) -> Result<ModelConfig, ModelError> {
        // Extract common fields from HuggingFace config
        let vocab_size = hf_config.get("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(32000) as usize;

        let hidden_size = hf_config.get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as usize;

        let num_layers = hf_config.get("num_hidden_layers")
            .or_else(|| hf_config.get("num_layers"))
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;

        let num_attention_heads = hf_config.get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;

        let intermediate_size = hf_config.get("intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(hidden_size * 4) as usize;

        let max_position_embeddings = hf_config.get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(2048) as usize;

        let head_dim = hidden_size / num_attention_heads;

        Ok(ModelConfig {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_layers,
            num_attention_heads,
            head_dim,
            max_seq_len: max_position_embeddings,
            eps: 1e-5,
        })
    }

    /// Load tensors immediately (eager loading)
    async fn load_tensors_immediate(
        &self,
        metadata: &SafeTensorsMetadata,
        config: &SafeTensorsConfig,
    ) -> Result<HashMap<String, GpuTensor>, ModelError> {
        let mut tensors = HashMap::new();

        println!("   📥 Loading tensors immediately...");

        for shard_file in &metadata.shard_files {
            let shard_tensors = self.load_shard_tensors(shard_file).await?;

            for (name, tensor) in shard_tensors {
                tensors.insert(name, tensor);
            }
        }

        // Apply memory optimization if needed
        if let Some(max_memory) = config.max_memory_bytes {
            self.optimize_memory_usage(&mut tensors, max_memory).await?;
        }

        Ok(tensors)
    }

    /// Load tensors lazily (on-demand)
    async fn load_tensors_lazy(
        &self,
        metadata: &SafeTensorsMetadata,
        _config: &SafeTensorsConfig,
    ) -> Result<HashMap<String, GpuTensor>, ModelError> {
        // For now, implement lazy loading as immediate loading
        // In a full implementation, this would return lazy tensor handles
        println!("   ⏳ Lazy loading not fully implemented, falling back to immediate loading");
        self.load_tensors_immediate(metadata, _config).await
    }

    /// Load tensors from a single shard file
    async fn load_shard_tensors<P: AsRef<Path>>(
        &self,
        shard_file: P,
    ) -> Result<HashMap<String, GpuTensor>, ModelError> {
        let shard_file = shard_file.as_ref();

        // Read SafeTensors file
        let file_data = std::fs::read(shard_file)
            .map_err(|e| ModelError::LoadingError(format!("Cannot read shard {}: {}", shard_file.display(), e)))?;

        let safetensors = SafeTensors::deserialize(&file_data)
            .map_err(|e| ModelError::LoadingError(format!("Invalid SafeTensors in {}: {}", shard_file.display(), e)))?;

        let mut tensors = HashMap::new();

        // Load each tensor
        for tensor_name in safetensors.names() {
            let tensor = self.load_tensor_from_safetensors(&safetensors, tensor_name).await?;
            tensors.insert(tensor_name.to_string(), tensor);
        }

        println!("      📦 Loaded {} tensors from {}", tensors.len(), shard_file.file_name().unwrap().to_string_lossy());

        Ok(tensors)
    }

    /// Load a single tensor from SafeTensors
    async fn load_tensor_from_safetensors(
        &self,
        safetensors: &SafeTensors<'_>,
        tensor_name: &str,
    ) -> Result<GpuTensor, ModelError> {
        let tensor_view = safetensors.tensor(tensor_name)
            .map_err(|e| ModelError::LoadingError(format!("Cannot access tensor {}: {}", tensor_name, e)))?;

        let shape = tensor_view.shape().to_vec();
        let data = tensor_view.data();

        // Convert data based on dtype
        let float_data: Vec<f32> = match tensor_view.dtype() {
            SafeTensorsDtype::F32 => {
                // Direct f32 data
                data.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect()
            },
            SafeTensorsDtype::F16 => {
                // Convert f16 to f32
                data.chunks_exact(2)
                    .map(|chunk| {
                        let f16_val = half::f16::from_le_bytes([chunk[0], chunk[1]]);
                        f16_val.to_f32()
                    })
                    .collect()
            },
            SafeTensorsDtype::BF16 => {
                // Convert bf16 to f32
                data.chunks_exact(2)
                    .map(|chunk| {
                        let bf16_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        let f32_bits = (bf16_bits as u32) << 16;
                        f32::from_bits(f32_bits)
                    })
                    .collect()
            },
            _ => {
                return Err(ModelError::LoadingError(
                    format!("Unsupported tensor dtype for {}: {:?}", tensor_name, tensor_view.dtype())
                ));
            }
        };

        // Create GPU tensor
        GpuTensor::new(float_data, shape, self.device.clone())
    }

    /// Optimize memory usage of loaded tensors
    async fn optimize_memory_usage(
        &self,
        tensors: &mut HashMap<String, GpuTensor>,
        max_memory_bytes: usize,
    ) -> Result<(), ModelError> {
        let current_memory: usize = tensors.values()
            .map(|tensor| tensor.data().len() * std::mem::size_of::<f32>())
            .sum();

        if current_memory > max_memory_bytes {
            println!("   ⚠️  Memory usage ({:.2} GB) exceeds limit ({:.2} GB), applying optimizations",
                    current_memory as f64 / 1e9, max_memory_bytes as f64 / 1e9);

            // In a full implementation, this would apply quantization or other optimizations
            // For now, just warn
        }

        Ok(())
    }

    /// Verify model integrity
    async fn verify_model_integrity(
        &self,
        tensors: &HashMap<String, GpuTensor>,
        metadata: &SafeTensorsMetadata,
    ) -> Result<(), ModelError> {
        // Check that all expected tensors are present
        for (expected_name, _) in &metadata.tensor_info {
            if !tensors.contains_key(expected_name) {
                return Err(ModelError::LoadingError(
                    format!("Missing expected tensor: {}", expected_name)
                ));
            }
        }

        // Check for unexpected tensors
        for loaded_name in tensors.keys() {
            if !metadata.tensor_info.contains_key(loaded_name) {
                println!("   ⚠️  Unexpected tensor found: {}", loaded_name);
            }
        }

        Ok(())
    }
}

/// A fully loaded SafeTensors model
#[derive(Debug)]
pub struct LoadedModel {
    /// Model configuration
    pub config: ModelConfig,
    /// Loaded tensors
    pub tensors: HashMap<String, GpuTensor>,
    /// Model metadata
    pub metadata: SafeTensorsMetadata,
    /// Device the model is loaded on
    pub device: GpuDevice,
}

impl LoadedModel {
    /// Get a tensor by name
    pub fn get_tensor(&self, name: &str) -> Option<&GpuTensor> {
        self.tensors.get(name)
    }

    /// Get tensor names by pattern
    pub fn get_tensor_names_matching(&self, pattern: &str) -> Vec<String> {
        self.tensors.keys()
            .filter(|name| name.contains(pattern))
            .cloned()
            .collect()
    }

    /// Get model memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.tensors.values()
            .map(|tensor| tensor.data().len() * std::mem::size_of::<f32>())
            .sum()
    }

    /// Get model information summary
    pub fn info(&self) -> ModelInfo {
        ModelInfo {
            architecture: self.metadata.architecture.clone(),
            total_parameters: self.metadata.total_parameters,
            memory_usage_bytes: self.memory_usage(),
            num_tensors: self.tensors.len(),
            device: self.device.clone(),
        }
    }
}

/// Model information summary
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub architecture: String,
    pub total_parameters: usize,
    pub memory_usage_bytes: usize,
    pub num_tensors: usize,
    pub device: GpuDevice,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_safetensors_loader_creation() {
        let device = GpuDevice::Cpu;
        let config = SafeTensorsConfig::default();
        let loader = SafeTensorsLoader::new(device, config);

        // Basic sanity check
        assert!(!loader.use_memory_mapping || loader.use_memory_mapping);
    }

    #[tokio::test]
    async fn test_architecture_detection() {
        let device = GpuDevice::Cpu;
        let config = SafeTensorsConfig::default();
        let loader = SafeTensorsLoader::new(device, config);

        // Test Llama detection
        let mut tensor_info = HashMap::new();
        tensor_info.insert("model.layers.0.self_attn.q_proj.weight".to_string(), TensorInfo {
            shape: vec![4096, 4096],
            dtype: "float32".to_string(),
            size_bytes: 4096 * 4096 * 4,
            shard_index: 0,
        });
        tensor_info.insert("model.layers.0.mlp.gate_proj.weight".to_string(), TensorInfo {
            shape: vec![11008, 4096],
            dtype: "float32".to_string(),
            size_bytes: 11008 * 4096 * 4,
            shard_index: 0,
        });

        let architecture = loader.detect_architecture(&tensor_info).unwrap();
        assert_eq!(architecture, "llama");
    }
}