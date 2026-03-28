//! Core Weight Loading Abstraction
//!
//! This module provides unified weight loading that converts any supported
//! weight format into our core tensor abstraction.

use crate::tensor_core::{Tensor, Device, DataType, CpuStorage, CandleStorage};
use crate::model_core::{ModelWeights, WeightMetadata, WeightFormat};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use anyhow::{Result, anyhow};
use serde_json::Value;
use safetensors::SafeTensors;
use candle_core::quantized::gguf_file;

/// Configuration extracted from GGUF metadata
#[derive(Debug, Clone, Default)]
pub struct GGUFModelConfig {
    pub architecture: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}

impl GGUFModelConfig {
    /// Extract config from GGUF metadata
    pub fn from_gguf_metadata(metadata: &HashMap<String, gguf_file::Value>) -> Self {
        let architecture = extract_gguf_string(metadata, "general.architecture")
            .unwrap_or_else(|| "llama".to_string());

        // Try architecture-specific keys first, then fall back to llama keys
        let arch_prefix = &architecture;

        let vocab_size = extract_gguf_u32(metadata, &format!("{}.vocab_size", arch_prefix))
            .or_else(|| extract_gguf_u32(metadata, "llama.vocab_size"))
            .unwrap_or(32000) as usize;

        let hidden_size = extract_gguf_u32(metadata, &format!("{}.embedding_length", arch_prefix))
            .or_else(|| extract_gguf_u32(metadata, "llama.embedding_length"))
            .unwrap_or(4096) as usize;

        let intermediate_size = extract_gguf_u32(metadata, &format!("{}.feed_forward_length", arch_prefix))
            .or_else(|| extract_gguf_u32(metadata, "llama.feed_forward_length"))
            .unwrap_or(11008) as usize;

        let num_hidden_layers = extract_gguf_u32(metadata, &format!("{}.block_count", arch_prefix))
            .or_else(|| extract_gguf_u32(metadata, "llama.block_count"))
            .unwrap_or(32) as usize;

        let num_attention_heads = extract_gguf_u32(metadata, &format!("{}.attention.head_count", arch_prefix))
            .or_else(|| extract_gguf_u32(metadata, "llama.attention.head_count"))
            .unwrap_or(32) as usize;

        let num_key_value_heads = extract_gguf_u32(metadata, &format!("{}.attention.head_count_kv", arch_prefix))
            .or_else(|| extract_gguf_u32(metadata, "llama.attention.head_count_kv"))
            .unwrap_or(num_attention_heads as u32) as usize;

        let head_dim = if num_attention_heads > 0 {
            hidden_size / num_attention_heads
        } else {
            128
        };

        let rms_norm_eps = extract_gguf_f32(metadata, &format!("{}.attention.layer_norm_rms_epsilon", arch_prefix))
            .or_else(|| extract_gguf_f32(metadata, "llama.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);

        let rope_theta = extract_gguf_f32(metadata, &format!("{}.rope.freq_base", arch_prefix))
            .or_else(|| extract_gguf_f32(metadata, "llama.rope.freq_base"))
            .unwrap_or(10000.0);

        let max_position_embeddings = extract_gguf_u32(metadata, &format!("{}.context_length", arch_prefix))
            .or_else(|| extract_gguf_u32(metadata, "llama.context_length"))
            .unwrap_or(2048) as usize;

        Self {
            architecture,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            rope_theta,
            max_position_embeddings,
        }
    }
}

/// Special token IDs extracted from GGUF
#[derive(Debug, Clone, Default)]
pub struct GGUFSpecialTokens {
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub unk_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
}

/// Tokenizer data extracted from GGUF metadata
#[derive(Debug, Clone)]
pub struct GGUFTokenizer {
    pub tokens: Vec<String>,
    pub token_types: Option<Vec<i32>>,
    pub scores: Option<Vec<f32>>,
    pub model_type: String,
    pub special_tokens: GGUFSpecialTokens,
}

impl GGUFTokenizer {
    /// Extract tokenizer data from GGUF metadata
    pub fn from_gguf_metadata(metadata: &HashMap<String, gguf_file::Value>) -> Option<Self> {
        // Extract token list
        let tokens = extract_gguf_string_array(metadata, "tokenizer.ggml.tokens")?;

        if tokens.is_empty() {
            return None;
        }

        // Extract token types (optional)
        let token_types = extract_gguf_i32_array(metadata, "tokenizer.ggml.token_type");

        // Extract BPE scores (optional)
        let scores = extract_gguf_f32_array(metadata, "tokenizer.ggml.scores");

        // Extract model type
        let model_type = extract_gguf_string(metadata, "tokenizer.ggml.model")
            .unwrap_or_else(|| "llama".to_string());

        // Extract special token IDs
        let special_tokens = GGUFSpecialTokens {
            bos_token_id: extract_gguf_u32(metadata, "tokenizer.ggml.bos_token_id"),
            eos_token_id: extract_gguf_u32(metadata, "tokenizer.ggml.eos_token_id"),
            unk_token_id: extract_gguf_u32(metadata, "tokenizer.ggml.unknown_token_id"),
            pad_token_id: extract_gguf_u32(metadata, "tokenizer.ggml.padding_token_id"),
        };

        Some(Self {
            tokens,
            token_types,
            scores,
            model_type,
            special_tokens,
        })
    }

    /// Get vocab size
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
}

/// Extract string array from GGUF metadata
fn extract_gguf_string_array(metadata: &HashMap<String, gguf_file::Value>, key: &str) -> Option<Vec<String>> {
    match metadata.get(key) {
        Some(gguf_file::Value::Array(arr)) => {
            let strings: Vec<String> = arr.iter()
                .filter_map(|v| {
                    if let gguf_file::Value::String(s) = v {
                        Some(s.clone())
                    } else {
                        None
                    }
                })
                .collect();
            if strings.is_empty() { None } else { Some(strings) }
        }
        _ => None,
    }
}

/// Extract i32 array from GGUF metadata
fn extract_gguf_i32_array(metadata: &HashMap<String, gguf_file::Value>, key: &str) -> Option<Vec<i32>> {
    match metadata.get(key) {
        Some(gguf_file::Value::Array(arr)) => {
            let nums: Vec<i32> = arr.iter()
                .filter_map(|v| {
                    match v {
                        gguf_file::Value::I32(n) => Some(*n),
                        gguf_file::Value::U32(n) => Some(*n as i32),
                        _ => None,
                    }
                })
                .collect();
            if nums.is_empty() { None } else { Some(nums) }
        }
        _ => None,
    }
}

/// Extract f32 array from GGUF metadata
fn extract_gguf_f32_array(metadata: &HashMap<String, gguf_file::Value>, key: &str) -> Option<Vec<f32>> {
    match metadata.get(key) {
        Some(gguf_file::Value::Array(arr)) => {
            let nums: Vec<f32> = arr.iter()
                .filter_map(|v| {
                    match v {
                        gguf_file::Value::F32(n) => Some(*n),
                        gguf_file::Value::F64(n) => Some(*n as f32),
                        _ => None,
                    }
                })
                .collect();
            if nums.is_empty() { None } else { Some(nums) }
        }
        _ => None,
    }
}

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

        // Open the GGUF file
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow!("Failed to read GGUF file: {}", e))?;

        println!("GGUF file loaded - {} tensors, {} metadata keys",
            content.tensor_infos.len(),
            content.metadata.len());

        // Extract model config from GGUF metadata
        let gguf_config = GGUFModelConfig::from_gguf_metadata(&content.metadata);

        println!("Model architecture: {}", gguf_config.architecture);
        println!("Model config: vocab_size={}, hidden_size={}, num_layers={}, heads={}, kv_heads={}",
            gguf_config.vocab_size, gguf_config.hidden_size, gguf_config.num_hidden_layers,
            gguf_config.num_attention_heads, gguf_config.num_key_value_heads);

        // Extract tokenizer from GGUF metadata
        let gguf_tokenizer = GGUFTokenizer::from_gguf_metadata(&content.metadata);
        if let Some(ref tok) = gguf_tokenizer {
            println!("Tokenizer extracted: vocab_size={}, model_type={}",
                tok.vocab_size(), tok.model_type);
        } else {
            println!("No tokenizer data found in GGUF metadata");
        }

        // Load tensors
        let device = candle_core::Device::Cpu;
        let mut tensors = HashMap::new();
        let mut total_params = 0usize;

        for (name, tensor_info) in content.tensor_infos.iter() {
            // Read the quantized tensor
            let qtensor = tensor_info.read(&mut file, content.tensor_data_offset, &device)
                .map_err(|e| anyhow!("Failed to read tensor '{}': {}", name, e))?;

            // Dequantize to f32 for compatibility with our existing ops
            let candle_tensor = qtensor.dequantize(&device)
                .map_err(|e| anyhow!("Failed to dequantize tensor '{}': {}", name, e))?;

            // Track parameters
            total_params += candle_tensor.elem_count();

            // Convert GGUF tensor name to HuggingFace-style name
            let hf_name = gguf_to_hf_name(name);

            // Wrap in our Tensor type using CandleStorage
            let tensor = Tensor::from_candle(candle_tensor);

            tensors.insert(hf_name, tensor);
        }

        println!("Loaded {} tensors, {} total parameters",
            tensors.len(), total_params);

        let metadata = WeightMetadata {
            architecture: gguf_config.architecture.clone(),
            total_params,
            format: WeightFormat::GGUF,
            dtype: "quantized".to_string(),
        };

        Ok(ModelWeights::with_gguf_config_and_tokenizer(tensors, metadata, gguf_config, gguf_tokenizer))
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

// === GGUF Helper Functions ===

/// Extract a string value from GGUF metadata
fn extract_gguf_string(metadata: &HashMap<String, gguf_file::Value>, key: &str) -> Option<String> {
    metadata.get(key).and_then(|v| {
        if let gguf_file::Value::String(s) = v {
            Some(s.clone())
        } else {
            None
        }
    })
}

/// Extract a u32 value from GGUF metadata
fn extract_gguf_u32(metadata: &HashMap<String, gguf_file::Value>, key: &str) -> Option<u32> {
    metadata.get(key).and_then(|v| {
        match v {
            gguf_file::Value::U32(n) => Some(*n),
            gguf_file::Value::I32(n) => Some(*n as u32),
            gguf_file::Value::U64(n) => Some(*n as u32),
            gguf_file::Value::I64(n) => Some(*n as u32),
            _ => None,
        }
    })
}

/// Extract a f32 value from GGUF metadata
fn extract_gguf_f32(metadata: &HashMap<String, gguf_file::Value>, key: &str) -> Option<f32> {
    metadata.get(key).and_then(|v| {
        match v {
            gguf_file::Value::F32(n) => Some(*n),
            gguf_file::Value::F64(n) => Some(*n as f32),
            _ => None,
        }
    })
}

/// Convert GGUF tensor names to HuggingFace-style names
/// GGUF uses different naming conventions than HuggingFace models
fn gguf_to_hf_name(gguf_name: &str) -> String {
    // Common GGUF to HF name mappings
    // NOTE: Order matters! More specific patterns must come before less specific ones
    // e.g., ".attn_output.weight" must be replaced before "output.weight"
    let name = gguf_name
        // Token embeddings
        .replace("token_embd.weight", "model.embed_tokens.weight")
        // Block/layer prefix
        .replace("blk.", "model.layers.")
        // Attention layers - weights (MUST come before output.weight replacement!)
        .replace(".attn_output.weight", ".self_attn.o_proj.weight")
        .replace(".attn_q.weight", ".self_attn.q_proj.weight")
        .replace(".attn_k.weight", ".self_attn.k_proj.weight")
        .replace(".attn_v.weight", ".self_attn.v_proj.weight")
        // Attention layers - biases
        .replace(".attn_output.bias", ".self_attn.o_proj.bias")
        .replace(".attn_q.bias", ".self_attn.q_proj.bias")
        .replace(".attn_k.bias", ".self_attn.k_proj.bias")
        .replace(".attn_v.bias", ".self_attn.v_proj.bias")
        // Output layer (MUST come after attn_output to avoid partial match)
        .replace("output_norm.weight", "model.norm.weight")
        .replace("output.weight", "lm_head.weight")
        // MLP layers - weights
        .replace(".ffn_gate.weight", ".mlp.gate_proj.weight")
        .replace(".ffn_up.weight", ".mlp.up_proj.weight")
        .replace(".ffn_down.weight", ".mlp.down_proj.weight")
        // MLP layers - biases
        .replace(".ffn_gate.bias", ".mlp.gate_proj.bias")
        .replace(".ffn_up.bias", ".mlp.up_proj.bias")
        .replace(".ffn_down.bias", ".mlp.down_proj.bias")
        // Layer norms - weights
        .replace(".attn_norm.weight", ".input_layernorm.weight")
        .replace(".ffn_norm.weight", ".post_attention_layernorm.weight")
        // Layer norms - biases (some models have these)
        .replace(".attn_norm.bias", ".input_layernorm.bias")
        .replace(".ffn_norm.bias", ".post_attention_layernorm.bias");

    name
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