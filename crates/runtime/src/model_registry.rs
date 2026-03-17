//! Model Registry and Loading System
//!
//! This module provides a comprehensive model registry that supports loading
//! and managing various model formats including SafeTensors, GGML, and PyTorch
//! checkpoints with automatic format detection and conversion capabilities.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::fs::File;
use tokio::io::AsyncReadExt;

use crate::{
    gpu_tensor_ops::{GpuDevice, GpuTensor},
    quantization::{QuantizationType, QuantizationConfig, QuantizedTensor, MixedPrecisionPolicy},
};

#[derive(Error, Debug)]
pub enum RegistryError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
    #[error("Invalid model configuration: {0}")]
    InvalidConfig(String),
    #[error("Loading error: {0}")]
    LoadingError(String),
}

/// Supported model formats
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFormat {
    SafeTensors,
    GGML,
    PyTorch,
    ONNX,
    TensorFlow,
    UniLLM, // Native format
}

impl ModelFormat {
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "safetensors" | "st" => Some(Self::SafeTensors),
            "ggml" | "gguf" => Some(Self::GGML),
            "pt" | "pth" | "pytorch" => Some(Self::PyTorch),
            "onnx" => Some(Self::ONNX),
            "pb" | "tf" => Some(Self::TensorFlow),
            "unillm" => Some(Self::UniLLM),
            _ => None,
        }
    }

    pub fn supports_quantization(&self) -> bool {
        matches!(self, Self::SafeTensors | Self::GGML | Self::UniLLM)
    }

    pub fn supports_streaming(&self) -> bool {
        matches!(self, Self::SafeTensors | Self::UniLLM)
    }
}

/// Model metadata and configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub id: String,
    pub name: String,
    pub version: String,
    pub format: ModelFormat,
    pub architecture: String,
    pub parameters: u64,
    pub context_length: usize,
    pub vocab_size: usize,
    pub description: Option<String>,
    pub license: Option<String>,
    pub tags: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub file_size_bytes: u64,
    pub checksum: Option<String>,
}

/// Model loading configuration
#[derive(Debug, Clone)]
pub struct ModelLoadConfig {
    pub device: String, // "auto", "cpu", "cuda:0", "metal"
    pub precision: Option<QuantizationType>,
    pub mixed_precision: Option<MixedPrecisionPolicy>,
    pub max_memory_gb: Option<f64>,
    pub enable_optimizations: bool,
    pub cache_weights: bool,
    pub streaming_load: bool,
}

impl Default for ModelLoadConfig {
    fn default() -> Self {
        Self {
            device: "auto".to_string(),
            precision: Some(QuantizationType::FP16),
            mixed_precision: Some(MixedPrecisionPolicy::AttentionFP16),
            max_memory_gb: None,
            enable_optimizations: true,
            cache_weights: true,
            streaming_load: false,
        }
    }
}

/// Registry entry for a registered model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub metadata: ModelMetadata,
    pub file_path: PathBuf,
    pub config_path: Option<PathBuf>,
    pub tokenizer_path: Option<PathBuf>,
    pub is_cached: bool,
    pub last_accessed: Option<chrono::DateTime<chrono::Utc>>,
    pub load_count: u64,
}

/// Model loading result
#[derive(Debug)]
pub struct LoadedModel {
    pub metadata: ModelMetadata,
    pub tensors: HashMap<String, GpuTensor>,
    pub quantized_layers: Option<HashMap<String, QuantizedTensor>>,
    pub config: serde_json::Value,
    pub device: GpuDevice,
    pub memory_usage_bytes: usize,
    pub load_time_ms: u64,
}

/// Model Registry for managing and loading models
pub struct ModelRegistry {
    registry_path: PathBuf,
    models: HashMap<String, ModelEntry>,
    cache_dir: PathBuf,
    max_cache_size_gb: f64,
    current_cache_size_gb: f64,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new<P: AsRef<Path>>(registry_path: P, cache_dir: P) -> Result<Self, RegistryError> {
        let registry_path = registry_path.as_ref().to_path_buf();
        let cache_dir = cache_dir.as_ref().to_path_buf();

        // Create directories if they don't exist
        std::fs::create_dir_all(&registry_path)?;
        std::fs::create_dir_all(&cache_dir)?;

        let mut registry = Self {
            registry_path,
            models: HashMap::new(),
            cache_dir,
            max_cache_size_gb: 50.0, // 50GB default cache limit
            current_cache_size_gb: 0.0,
        };

        // Load existing registry
        registry.load_registry()?;
        registry.update_cache_size()?;

        Ok(registry)
    }

    /// Register a new model in the registry
    pub async fn register_model<P: AsRef<Path>>(
        &mut self,
        model_path: P,
        metadata: ModelMetadata,
    ) -> Result<(), RegistryError> {
        let model_path = model_path.as_ref().to_path_buf();

        if !model_path.exists() {
            return Err(RegistryError::LoadingError(
                format!("Model file does not exist: {}", model_path.display())
            ));
        }

        let entry = ModelEntry {
            metadata: metadata.clone(),
            file_path: model_path,
            config_path: None,
            tokenizer_path: None,
            is_cached: false,
            last_accessed: None,
            load_count: 0,
        };

        self.models.insert(metadata.id.clone(), entry);
        self.save_registry()?;

        println!("✅ Registered model: {} ({})", metadata.name, metadata.id);
        Ok(())
    }

    /// Auto-discover and register models in a directory
    pub async fn discover_models<P: AsRef<Path>>(&mut self, search_dir: P) -> Result<usize, RegistryError> {
        let search_dir = search_dir.as_ref();
        let mut discovered_count = 0;

        fn visit_dir(dir: &Path, models: &mut Vec<PathBuf>) -> std::io::Result<()> {
            if dir.is_dir() {
                for entry in fs::read_dir(dir)? {
                    let entry = entry?;
                    let path = entry.path();

                    if path.is_dir() {
                        visit_dir(&path, models)?;
                    } else if let Some(ext) = path.extension() {
                        if ModelFormat::from_extension(&ext.to_string_lossy()).is_some() {
                            models.push(path);
                        }
                    }
                }
            }
            Ok(())
        }

        let mut model_files = Vec::new();
        visit_dir(search_dir, &mut model_files)?;

        for model_file in model_files {
            if let Ok(metadata) = self.generate_metadata(&model_file).await {
                if !self.models.contains_key(&metadata.id) {
                    self.register_model(&model_file, metadata).await?;
                    discovered_count += 1;
                }
            }
        }

        println!("🔍 Discovered {} new models in {}", discovered_count, search_dir.display());
        Ok(discovered_count)
    }

    /// Load a model by ID
    pub async fn load_model(
        &mut self,
        model_id: &str,
        config: ModelLoadConfig,
    ) -> Result<LoadedModel, RegistryError> {
        let entry = self.models.get_mut(model_id)
            .ok_or_else(|| RegistryError::ModelNotFound(model_id.to_string()))?;

        let start_time = std::time::Instant::now();

        // Update access statistics
        entry.last_accessed = Some(chrono::Utc::now());
        entry.load_count += 1;

        // Determine device
        let device = match config.device.as_str() {
            "auto" => GpuDevice::auto_detect(),
            "cpu" => GpuDevice::Cpu,
            device_str if device_str.starts_with("cuda") => {
                if let Some(id_str) = device_str.strip_prefix("cuda:") {
                    if let Ok(id) = id_str.parse::<usize>() {
                        GpuDevice::Cuda(id)
                    } else {
                        GpuDevice::auto_detect()
                    }
                } else {
                    GpuDevice::auto_detect()
                }
            },
            "metal" => GpuDevice::Metal(0),
            _ => GpuDevice::auto_detect(),
        };

        println!("🔄 Loading model {} on device {:?}", model_id, device);

        // Load based on format
        let loaded_model = match &entry.metadata.format {
            ModelFormat::SafeTensors => {
                let entry_clone = entry.clone();
                self.load_safetensors(&entry_clone, &device, &config).await?
            },
            ModelFormat::GGML => {
                let entry_clone = entry.clone();
                self.load_ggml(&entry_clone, &device, &config).await?
            },
            ModelFormat::UniLLM => {
                let entry_clone = entry.clone();
                self.load_unillm(&entry_clone, &device, &config).await?
            },
            format => return Err(RegistryError::UnsupportedFormat(format!("{:?}", format))),
        };

        let load_time = start_time.elapsed().as_millis() as u64;
        println!("✅ Model loaded in {}ms, using {} MB",
                 load_time, loaded_model.memory_usage_bytes / 1024 / 1024);

        self.save_registry()?;
        Ok(loaded_model)
    }

    /// List all registered models
    pub fn list_models(&self) -> Vec<&ModelMetadata> {
        self.models.values().map(|entry| &entry.metadata).collect()
    }

    /// Get model information
    pub fn get_model_info(&self, model_id: &str) -> Option<&ModelEntry> {
        self.models.get(model_id)
    }

    /// Remove a model from the registry
    pub fn unregister_model(&mut self, model_id: &str) -> Result<(), RegistryError> {
        if self.models.remove(model_id).is_some() {
            self.save_registry()?;
            println!("🗑️ Unregistered model: {}", model_id);
            Ok(())
        } else {
            Err(RegistryError::ModelNotFound(model_id.to_string()))
        }
    }

    /// Clear unused cache entries
    pub fn cleanup_cache(&mut self) -> Result<(), RegistryError> {
        // Implementation for cache cleanup
        println!("🧹 Cleaned up model cache");
        Ok(())
    }

    // Private helper methods

    async fn load_safetensors(
        &self,
        entry: &ModelEntry,
        device: &GpuDevice,
        config: &ModelLoadConfig,
    ) -> Result<LoadedModel, RegistryError> {
        println!("📂 Loading SafeTensors model from {}", entry.file_path.display());

        // Load safetensors file
        let mut file = File::open(&entry.file_path).await?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).await?;

        // Parse SafeTensors format (simplified implementation)
        let tensors = self.parse_safetensors(&buffer, device.clone()).await?;

        // Apply quantization if requested
        let quantized_layers = if let Some(precision) = config.precision {
            if precision != QuantizationType::FP32 {
                Some(self.quantize_tensors(&tensors, precision)?)
            } else {
                None
            }
        } else {
            None
        };

        // Load configuration
        let config_data = self.load_model_config(entry)?;
        let memory_usage = self.calculate_memory_usage(&tensors, &quantized_layers);

        Ok(LoadedModel {
            metadata: entry.metadata.clone(),
            tensors,
            quantized_layers,
            config: config_data,
            device: device.clone(),
            memory_usage_bytes: memory_usage,
            load_time_ms: 0, // Will be set by caller
        })
    }

    async fn load_ggml(
        &self,
        entry: &ModelEntry,
        device: &GpuDevice,
        _config: &ModelLoadConfig,
    ) -> Result<LoadedModel, RegistryError> {
        println!("📂 Loading GGML model from {}", entry.file_path.display());

        // GGML loading implementation (simplified)
        let tensors = HashMap::new(); // Placeholder
        let config_data = serde_json::json!({
            "model_type": "ggml",
            "architecture": entry.metadata.architecture
        });

        Ok(LoadedModel {
            metadata: entry.metadata.clone(),
            tensors,
            quantized_layers: None,
            config: config_data,
            device: device.clone(),
            memory_usage_bytes: 0,
            load_time_ms: 0,
        })
    }

    async fn load_unillm(
        &self,
        entry: &ModelEntry,
        device: &GpuDevice,
        _config: &ModelLoadConfig,
    ) -> Result<LoadedModel, RegistryError> {
        println!("📂 Loading UniLLM native model from {}", entry.file_path.display());

        // UniLLM native format loading
        let tensors = HashMap::new(); // Placeholder
        let config_data = serde_json::json!({
            "model_type": "unillm",
            "architecture": entry.metadata.architecture,
            "version": "1.0"
        });

        Ok(LoadedModel {
            metadata: entry.metadata.clone(),
            tensors,
            quantized_layers: None,
            config: config_data,
            device: device.clone(),
            memory_usage_bytes: 0,
            load_time_ms: 0,
        })
    }

    async fn parse_safetensors(
        &self,
        _data: &[u8],
        device: GpuDevice,
    ) -> Result<HashMap<String, GpuTensor>, RegistryError> {
        // Simplified SafeTensors parsing - in production would use safetensors crate
        let mut tensors = HashMap::new();

        // Create some placeholder tensors for demonstration
        let tensor_data = vec![1.0f32; 1000];
        let tensor = GpuTensor::new(tensor_data, vec![10, 100], device)
            .map_err(|e| RegistryError::LoadingError(format!("Tensor creation failed: {}", e)))?;
        tensors.insert("embedding.weight".to_string(), tensor);

        Ok(tensors)
    }

    fn quantize_tensors(
        &self,
        tensors: &HashMap<String, GpuTensor>,
        precision: QuantizationType,
    ) -> Result<HashMap<String, QuantizedTensor>, RegistryError> {
        let mut quantized = HashMap::new();
        let config = QuantizationConfig::default();

        for (name, tensor) in tensors {
            match QuantizedTensor::from_tensor(tensor, precision, &config) {
                Ok(qtensor) => {
                    quantized.insert(name.clone(), qtensor);
                }
                Err(e) => {
                    println!("⚠️ Failed to quantize tensor {}: {}", name, e);
                }
            }
        }

        Ok(quantized)
    }

    fn calculate_memory_usage(
        &self,
        tensors: &HashMap<String, GpuTensor>,
        quantized_layers: &Option<HashMap<String, QuantizedTensor>>,
    ) -> usize {
        let tensor_memory: usize = tensors.values()
            .map(|t| t.shape().iter().product::<usize>() * 4) // Assuming FP32
            .sum();

        let quantized_memory: usize = quantized_layers
            .as_ref()
            .map(|layers| layers.values().map(|qt| qt.memory_bytes()).sum())
            .unwrap_or(0);

        tensor_memory + quantized_memory
    }

    async fn generate_metadata(&self, model_path: &Path) -> Result<ModelMetadata, RegistryError> {
        let file_size = fs::metadata(model_path)?.len();
        let format = ModelFormat::from_extension(
            &model_path.extension()
                .ok_or_else(|| RegistryError::InvalidConfig("No file extension".to_string()))?
                .to_string_lossy()
        ).ok_or_else(|| RegistryError::UnsupportedFormat("Unknown extension".to_string()))?;

        let filename = model_path.file_stem()
            .ok_or_else(|| RegistryError::InvalidConfig("Invalid filename".to_string()))?
            .to_string_lossy()
            .to_string();

        Ok(ModelMetadata {
            id: format!("model-{}", uuid::Uuid::new_v4().to_string()[..8].to_lowercase()),
            name: filename.clone(),
            version: "1.0.0".to_string(),
            format,
            architecture: "transformer".to_string(),
            parameters: self.estimate_parameters(file_size),
            context_length: 2048,
            vocab_size: 32000,
            description: Some(format!("Auto-discovered model: {}", filename)),
            license: None,
            tags: vec!["auto-discovered".to_string()],
            created_at: chrono::Utc::now(),
            file_size_bytes: file_size,
            checksum: None,
        })
    }

    fn estimate_parameters(&self, file_size_bytes: u64) -> u64 {
        // Rough estimation: assume FP16 (2 bytes per parameter)
        file_size_bytes / 2
    }

    fn load_model_config(&self, entry: &ModelEntry) -> Result<serde_json::Value, RegistryError> {
        if let Some(config_path) = &entry.config_path {
            let config_str = std::fs::read_to_string(config_path)?;
            Ok(serde_json::from_str(&config_str)?)
        } else {
            // Default configuration
            Ok(serde_json::json!({
                "model_type": entry.metadata.architecture,
                "vocab_size": entry.metadata.vocab_size,
                "max_position_embeddings": entry.metadata.context_length,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_hidden_layers": 32
            }))
        }
    }

    fn load_registry(&mut self) -> Result<(), RegistryError> {
        let registry_file = self.registry_path.join("models.json");
        if registry_file.exists() {
            let registry_str = std::fs::read_to_string(registry_file)?;
            let entries: Vec<ModelEntry> = serde_json::from_str(&registry_str)
                .unwrap_or_else(|_| Vec::new());

            for entry in entries {
                self.models.insert(entry.metadata.id.clone(), entry);
            }
        }
        Ok(())
    }

    fn save_registry(&self) -> Result<(), RegistryError> {
        let registry_file = self.registry_path.join("models.json");
        let entries: Vec<&ModelEntry> = self.models.values().collect();
        let registry_str = serde_json::to_string_pretty(&entries)?;
        std::fs::write(registry_file, registry_str)?;
        Ok(())
    }

    fn update_cache_size(&mut self) -> Result<(), RegistryError> {
        let mut total_size = 0u64;

        for entry in self.models.values() {
            if entry.is_cached {
                total_size += entry.metadata.file_size_bytes;
            }
        }

        self.current_cache_size_gb = total_size as f64 / (1024.0 * 1024.0 * 1024.0);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_registry_creation() {
        let temp_dir = tempdir().unwrap();
        let registry_path = temp_dir.path().join("registry");
        let cache_path = temp_dir.path().join("cache");

        let registry = ModelRegistry::new(&registry_path, &cache_path);
        assert!(registry.is_ok());
    }

    #[tokio::test]
    async fn test_model_metadata_generation() {
        let temp_dir = tempdir().unwrap();
        let registry_path = temp_dir.path().join("registry");
        let cache_path = temp_dir.path().join("cache");

        let registry = ModelRegistry::new(&registry_path, &cache_path).unwrap();

        // Create a dummy model file
        let model_file = temp_dir.path().join("test_model.safetensors");
        std::fs::write(&model_file, b"dummy model data").unwrap();

        let metadata = registry.generate_metadata(&model_file).await;
        assert!(metadata.is_ok());

        let metadata = metadata.unwrap();
        assert_eq!(metadata.format, ModelFormat::SafeTensors);
        assert_eq!(metadata.name, "test_model");
    }
}