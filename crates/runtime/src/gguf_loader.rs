//! GGUF Model Loading System
//!
//! This module provides comprehensive GGUF (GPT-Generated Unified Format)
//! support for loading quantized models. GGUF is the modern successor to
//! GGML and is widely used for efficient local model deployment.

use crate::{
    gpu_tensor_ops::{GpuDevice, GpuTensor, GpuTensorOps},
    basic_model::ModelConfig,
    types::ModelError,
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    io::{BufReader, Read, Seek, SeekFrom},
    fs::File,
};
use serde::{Deserialize, Serialize};

/// GGUF file format constants
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian
const GGUF_VERSION: u32 = 3;

/// GGUF model loader
pub struct GGUFLoader {
    /// GPU device for loading tensors
    device: GpuDevice,
    /// Tensor operations
    tensor_ops: GpuTensorOps,
}

/// GGUF file header
#[derive(Debug, Clone)]
pub struct GGUFHeader {
    /// Magic number (should be "GGUF")
    pub magic: u32,
    /// Version number
    pub version: u32,
    /// Number of tensors
    pub tensor_count: u64,
    /// Number of metadata key-value pairs
    pub metadata_kv_count: u64,
}

/// GGUF metadata key-value pair
#[derive(Debug, Clone)]
pub struct GGUFMetadata {
    /// Key name
    pub key: String,
    /// Value type
    pub value_type: GGUFValueType,
    /// Value data
    pub value: GGUFValue,
}

/// GGUF value types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GGUFValueType {
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
}

/// GGUF value data
#[derive(Debug, Clone)]
pub enum GGUFValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}

/// GGUF tensor information
#[derive(Debug, Clone)]
pub struct GGUFTensorInfo {
    /// Tensor name
    pub name: String,
    /// Tensor dimensions
    pub dimensions: Vec<u64>,
    /// Tensor data type
    pub tensor_type: GGUFTensorType,
    /// Offset in file where tensor data starts
    pub offset: u64,
}

/// GGUF tensor data types (quantization formats)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GGUFTensorType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
}

/// Loaded GGUF model
#[derive(Debug)]
pub struct GGUFModel {
    /// Model configuration extracted from metadata
    pub config: ModelConfig,
    /// Model metadata
    pub metadata: HashMap<String, GGUFValue>,
    /// Tensor information
    pub tensor_info: HashMap<String, GGUFTensorInfo>,
    /// Loaded tensors (lazy loaded)
    pub tensors: HashMap<String, GpuTensor>,
    /// File path for lazy loading
    pub file_path: PathBuf,
    /// Device the model is loaded on
    pub device: GpuDevice,
}

impl GGUFLoader {
    /// Create a new GGUF loader
    pub fn new(device: GpuDevice) -> Self {
        let tensor_ops = GpuTensorOps::with_device(device.clone());

        Self {
            device,
            tensor_ops,
        }
    }

    /// Load GGUF model from file
    pub async fn load_model<P: AsRef<Path>>(&self, file_path: P) -> Result<GGUFModel, ModelError> {
        let file_path = file_path.as_ref();
        println!("🔄 Loading GGUF model from: {}", file_path.display());

        // Open file
        let mut file = File::open(file_path)
            .map_err(|e| ModelError::LoadingError(format!("Cannot open GGUF file: {}", e)))?;

        let mut reader = BufReader::new(file);

        // Read header
        let header = self.read_header(&mut reader)?;
        println!("   📊 GGUF version: {}, tensors: {}, metadata entries: {}",
                header.version, header.tensor_count, header.metadata_kv_count);

        // Read metadata
        let metadata = self.read_metadata(&mut reader, header.metadata_kv_count)?;
        println!("   ⚙️  Loaded {} metadata entries", metadata.len());

        // Read tensor information
        let tensor_info = self.read_tensor_info(&mut reader, header.tensor_count)?;
        println!("   🔢 Loaded info for {} tensors", tensor_info.len());

        // Extract model configuration from metadata
        let config = self.extract_model_config(&metadata)?;
        println!("   📐 Model config: {}D hidden, {} layers",
                config.hidden_size, config.num_layers);

        // Create model (tensors will be loaded lazily)
        let model = GGUFModel {
            config,
            metadata,
            tensor_info,
            tensors: HashMap::new(),
            file_path: file_path.to_path_buf(),
            device: self.device.clone(),
        };

        println!("   ✅ GGUF model loaded successfully");

        Ok(model)
    }

    /// Read GGUF file header
    fn read_header<R: Read>(&self, reader: &mut R) -> Result<GGUFHeader, ModelError> {
        let magic = self.read_u32(reader)?;
        if magic != GGUF_MAGIC {
            return Err(ModelError::LoadingError(
                format!("Invalid GGUF magic number: 0x{:x}", magic)
            ));
        }

        let version = self.read_u32(reader)?;
        if version != GGUF_VERSION {
            println!("   ⚠️  GGUF version {} may not be fully supported (expected {})", version, GGUF_VERSION);
        }

        let tensor_count = self.read_u64(reader)?;
        let metadata_kv_count = self.read_u64(reader)?;

        Ok(GGUFHeader {
            magic,
            version,
            tensor_count,
            metadata_kv_count,
        })
    }

    /// Read metadata key-value pairs
    fn read_metadata<R: Read>(&self, reader: &mut R, count: u64) -> Result<HashMap<String, GGUFValue>, ModelError> {
        let mut metadata = HashMap::new();

        for _ in 0..count {
            let key = self.read_string(reader)?;
            let value_type = self.read_u32(reader)?;
            let value = self.read_value(reader, value_type)?;

            metadata.insert(key, value);
        }

        Ok(metadata)
    }

    /// Read tensor information
    fn read_tensor_info<R: Read>(&self, reader: &mut R, count: u64) -> Result<HashMap<String, GGUFTensorInfo>, ModelError> {
        let mut tensor_info = HashMap::new();

        for _ in 0..count {
            let name = self.read_string(reader)?;
            let n_dimensions = self.read_u32(reader)?;

            let mut dimensions = Vec::new();
            for _ in 0..n_dimensions {
                dimensions.push(self.read_u64(reader)?);
            }

            let tensor_type_raw = self.read_u32(reader)?;
            let tensor_type = self.parse_tensor_type(tensor_type_raw)?;

            let offset = self.read_u64(reader)?;

            tensor_info.insert(name.clone(), GGUFTensorInfo {
                name,
                dimensions,
                tensor_type,
                offset,
            });
        }

        Ok(tensor_info)
    }

    /// Extract model configuration from metadata
    fn extract_model_config(&self, metadata: &HashMap<String, GGUFValue>) -> Result<ModelConfig, ModelError> {
        // Helper function to get integer value from metadata
        let get_int = |key: &str| -> usize {
            metadata.get(key)
                .and_then(|v| match v {
                    GGUFValue::UInt32(n) => Some(*n as usize),
                    GGUFValue::UInt64(n) => Some(*n as usize),
                    GGUFValue::Int32(n) => Some(*n as usize),
                    GGUFValue::Int64(n) => Some(*n as usize),
                    _ => None,
                })
                .unwrap_or(0)
        };

        // Extract common model parameters
        let vocab_size = get_int("llama.vocab_size")
            .max(get_int("tokenizer.ggml.tokens.length"))
            .max(32000); // Fallback

        let hidden_size = get_int("llama.embedding_length")
            .max(get_int("llama.n_embd"))
            .max(4096); // Fallback

        let num_layers = get_int("llama.block_count")
            .max(get_int("llama.n_layer"))
            .max(32); // Fallback

        let num_attention_heads = get_int("llama.attention.head_count")
            .max(get_int("llama.n_head"))
            .max(32); // Fallback

        let intermediate_size = get_int("llama.feed_forward_length")
            .max(hidden_size * 4); // Common ratio

        let max_seq_len = get_int("llama.context_length")
            .max(get_int("llama.n_ctx"))
            .max(2048); // Fallback

        let head_dim = hidden_size / num_attention_heads;

        Ok(ModelConfig {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_layers,
            num_attention_heads,
            head_dim,
            max_seq_len,
            eps: 1e-5,
        })
    }

    /// Parse tensor type from raw value
    fn parse_tensor_type(&self, raw_type: u32) -> Result<GGUFTensorType, ModelError> {
        match raw_type {
            0 => Ok(GGUFTensorType::F32),
            1 => Ok(GGUFTensorType::F16),
            2 => Ok(GGUFTensorType::Q4_0),
            3 => Ok(GGUFTensorType::Q4_1),
            6 => Ok(GGUFTensorType::Q5_0),
            7 => Ok(GGUFTensorType::Q5_1),
            8 => Ok(GGUFTensorType::Q8_0),
            9 => Ok(GGUFTensorType::Q8_1),
            10 => Ok(GGUFTensorType::Q2_K),
            11 => Ok(GGUFTensorType::Q3_K),
            12 => Ok(GGUFTensorType::Q4_K),
            13 => Ok(GGUFTensorType::Q5_K),
            14 => Ok(GGUFTensorType::Q6_K),
            15 => Ok(GGUFTensorType::Q8_K),
            _ => Err(ModelError::LoadingError(
                format!("Unsupported tensor type: {}", raw_type)
            )),
        }
    }

    /// Read GGUF value based on type
    fn read_value<R: Read>(&self, reader: &mut R, value_type: u32) -> Result<GGUFValue, ModelError> {
        match value_type {
            0 => Ok(GGUFValue::UInt8(self.read_u8(reader)?)),
            1 => Ok(GGUFValue::Int8(self.read_i8(reader)?)),
            2 => Ok(GGUFValue::UInt16(self.read_u16(reader)?)),
            3 => Ok(GGUFValue::Int16(self.read_i16(reader)?)),
            4 => Ok(GGUFValue::UInt32(self.read_u32(reader)?)),
            5 => Ok(GGUFValue::Int32(self.read_i32(reader)?)),
            6 => Ok(GGUFValue::Float32(self.read_f32(reader)?)),
            7 => Ok(GGUFValue::Bool(self.read_u8(reader)? != 0)),
            8 => Ok(GGUFValue::String(self.read_string(reader)?)),
            10 => Ok(GGUFValue::UInt64(self.read_u64(reader)?)),
            11 => Ok(GGUFValue::Int64(self.read_i64(reader)?)),
            12 => Ok(GGUFValue::Float64(self.read_f64(reader)?)),
            9 => {
                // Array type
                let array_type = self.read_u32(reader)?;
                let array_length = self.read_u64(reader)?;

                let mut array = Vec::new();
                for _ in 0..array_length {
                    array.push(self.read_value(reader, array_type)?);
                }

                Ok(GGUFValue::Array(array))
            },
            _ => Err(ModelError::LoadingError(
                format!("Unsupported value type: {}", value_type)
            )),
        }
    }

    /// Read string from GGUF file
    fn read_string<R: Read>(&self, reader: &mut R) -> Result<String, ModelError> {
        let length = self.read_u64(reader)?;
        let mut buffer = vec![0u8; length as usize];
        reader.read_exact(&mut buffer)
            .map_err(|e| ModelError::LoadingError(format!("Cannot read string: {}", e)))?;

        String::from_utf8(buffer)
            .map_err(|e| ModelError::LoadingError(format!("Invalid UTF-8 string: {}", e)))
    }

    /// Helper functions for reading primitive types
    fn read_u8<R: Read>(&self, reader: &mut R) -> Result<u8, ModelError> {
        let mut buffer = [0u8; 1];
        reader.read_exact(&mut buffer)
            .map_err(|e| ModelError::LoadingError(format!("Cannot read u8: {}", e)))?;
        Ok(buffer[0])
    }

    fn read_i8<R: Read>(&self, reader: &mut R) -> Result<i8, ModelError> {
        Ok(self.read_u8(reader)? as i8)
    }

    fn read_u16<R: Read>(&self, reader: &mut R) -> Result<u16, ModelError> {
        let mut buffer = [0u8; 2];
        reader.read_exact(&mut buffer)
            .map_err(|e| ModelError::LoadingError(format!("Cannot read u16: {}", e)))?;
        Ok(u16::from_le_bytes(buffer))
    }

    fn read_i16<R: Read>(&self, reader: &mut R) -> Result<i16, ModelError> {
        Ok(self.read_u16(reader)? as i16)
    }

    fn read_u32<R: Read>(&self, reader: &mut R) -> Result<u32, ModelError> {
        let mut buffer = [0u8; 4];
        reader.read_exact(&mut buffer)
            .map_err(|e| ModelError::LoadingError(format!("Cannot read u32: {}", e)))?;
        Ok(u32::from_le_bytes(buffer))
    }

    fn read_i32<R: Read>(&self, reader: &mut R) -> Result<i32, ModelError> {
        Ok(self.read_u32(reader)? as i32)
    }

    fn read_u64<R: Read>(&self, reader: &mut R) -> Result<u64, ModelError> {
        let mut buffer = [0u8; 8];
        reader.read_exact(&mut buffer)
            .map_err(|e| ModelError::LoadingError(format!("Cannot read u64: {}", e)))?;
        Ok(u64::from_le_bytes(buffer))
    }

    fn read_i64<R: Read>(&self, reader: &mut R) -> Result<i64, ModelError> {
        Ok(self.read_u64(reader)? as i64)
    }

    fn read_f32<R: Read>(&self, reader: &mut R) -> Result<f32, ModelError> {
        let mut buffer = [0u8; 4];
        reader.read_exact(&mut buffer)
            .map_err(|e| ModelError::LoadingError(format!("Cannot read f32: {}", e)))?;
        Ok(f32::from_le_bytes(buffer))
    }

    fn read_f64<R: Read>(&self, reader: &mut R) -> Result<f64, ModelError> {
        let mut buffer = [0u8; 8];
        reader.read_exact(&mut buffer)
            .map_err(|e| ModelError::LoadingError(format!("Cannot read f64: {}", e)))?;
        Ok(f64::from_le_bytes(buffer))
    }
}

impl GGUFModel {
    /// Load a tensor by name (lazy loading)
    pub async fn load_tensor(&mut self, tensor_name: &str) -> Result<&GpuTensor, ModelError> {
        // Return if already loaded
        if self.tensors.contains_key(tensor_name) {
            return Ok(self.tensors.get(tensor_name).unwrap());
        }

        // Get tensor info
        let tensor_info = self.tensor_info.get(tensor_name)
            .ok_or_else(|| ModelError::LoadingError(format!("Tensor not found: {}", tensor_name)))?;

        // Load tensor data from file
        let tensor = self.load_tensor_data(tensor_info).await?;

        // Store loaded tensor
        self.tensors.insert(tensor_name.to_string(), tensor);

        Ok(self.tensors.get(tensor_name).unwrap())
    }

    /// Load tensor data from file
    async fn load_tensor_data(&self, tensor_info: &GGUFTensorInfo) -> Result<GpuTensor, ModelError> {
        let mut file = File::open(&self.file_path)
            .map_err(|e| ModelError::LoadingError(format!("Cannot open GGUF file: {}", e)))?;

        // Seek to tensor data
        file.seek(SeekFrom::Start(tensor_info.offset))
            .map_err(|e| ModelError::LoadingError(format!("Cannot seek to tensor data: {}", e)))?;

        // Calculate tensor size
        let element_count: usize = tensor_info.dimensions.iter().map(|&d| d as usize).product();

        // Read and dequantize tensor data
        let tensor_data = self.read_and_dequantize_tensor(&mut file, &tensor_info.tensor_type, element_count)?;

        // Convert dimensions to usize
        let shape: Vec<usize> = tensor_info.dimensions.iter().map(|&d| d as usize).collect();

        // Create GPU tensor
        GpuTensor::new(tensor_data, shape, self.device.clone())
    }

    /// Read and dequantize tensor data
    fn read_and_dequantize_tensor(
        &self,
        file: &mut File,
        tensor_type: &GGUFTensorType,
        element_count: usize,
    ) -> Result<Vec<f32>, ModelError> {
        match tensor_type {
            GGUFTensorType::F32 => {
                // Read f32 data directly
                let mut buffer = vec![0u8; element_count * 4];
                file.read_exact(&mut buffer)
                    .map_err(|e| ModelError::LoadingError(format!("Cannot read F32 tensor data: {}", e)))?;

                let float_data: Vec<f32> = buffer.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();

                Ok(float_data)
            },

            GGUFTensorType::F16 => {
                // Read f16 data and convert to f32
                let mut buffer = vec![0u8; element_count * 2];
                file.read_exact(&mut buffer)
                    .map_err(|e| ModelError::LoadingError(format!("Cannot read F16 tensor data: {}", e)))?;

                let float_data: Vec<f32> = buffer.chunks_exact(2)
                    .map(|chunk| {
                        let f16_val = half::f16::from_le_bytes([chunk[0], chunk[1]]);
                        f16_val.to_f32()
                    })
                    .collect();

                Ok(float_data)
            },

            GGUFTensorType::Q4_0 => {
                // Dequantize Q4_0 format
                self.dequantize_q4_0(file, element_count)
            },

            GGUFTensorType::Q8_0 => {
                // Dequantize Q8_0 format
                self.dequantize_q8_0(file, element_count)
            },

            _ => {
                // For other quantization formats, return error for now
                Err(ModelError::LoadingError(
                    format!("Quantization format {:?} not yet implemented", tensor_type)
                ))
            }
        }
    }

    /// Dequantize Q4_0 format
    fn dequantize_q4_0(&self, file: &mut File, element_count: usize) -> Result<Vec<f32>, ModelError> {
        const BLOCK_SIZE: usize = 32;
        const BLOCK_BYTES: usize = 18; // 2 bytes for scale + 16 bytes for quantized data

        let num_blocks = (element_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let mut result = Vec::with_capacity(element_count);

        for _ in 0..num_blocks {
            // Read block
            let mut block_data = [0u8; BLOCK_BYTES];
            file.read_exact(&mut block_data)
                .map_err(|e| ModelError::LoadingError(format!("Cannot read Q4_0 block: {}", e)))?;

            // Extract scale (first 2 bytes as f16)
            let scale_bytes = [block_data[0], block_data[1]];
            let scale = half::f16::from_le_bytes(scale_bytes).to_f32();

            // Dequantize 32 elements from 16 bytes
            for i in 0..16 {
                let byte = block_data[2 + i];

                // Extract two 4-bit values
                let val1 = ((byte & 0x0F) as i8) - 8;
                let val2 = (((byte >> 4) & 0x0F) as i8) - 8;

                if result.len() < element_count {
                    result.push(val1 as f32 * scale);
                }
                if result.len() < element_count {
                    result.push(val2 as f32 * scale);
                }
            }
        }

        result.truncate(element_count);
        Ok(result)
    }

    /// Dequantize Q8_0 format
    fn dequantize_q8_0(&self, file: &mut File, element_count: usize) -> Result<Vec<f32>, ModelError> {
        const BLOCK_SIZE: usize = 32;
        const BLOCK_BYTES: usize = 34; // 2 bytes for scale + 32 bytes for quantized data

        let num_blocks = (element_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let mut result = Vec::with_capacity(element_count);

        for _ in 0..num_blocks {
            // Read block
            let mut block_data = [0u8; BLOCK_BYTES];
            file.read_exact(&mut block_data)
                .map_err(|e| ModelError::LoadingError(format!("Cannot read Q8_0 block: {}", e)))?;

            // Extract scale (first 2 bytes as f16)
            let scale_bytes = [block_data[0], block_data[1]];
            let scale = half::f16::from_le_bytes(scale_bytes).to_f32();

            // Dequantize 32 elements
            for i in 0..32 {
                if result.len() < element_count {
                    let quantized_val = block_data[2 + i] as i8;
                    result.push(quantized_val as f32 * scale);
                }
            }
        }

        result.truncate(element_count);
        Ok(result)
    }

    /// Get tensor names by pattern
    pub fn get_tensor_names_matching(&self, pattern: &str) -> Vec<String> {
        self.tensor_info.keys()
            .filter(|name| name.contains(pattern))
            .cloned()
            .collect()
    }

    /// Get model memory usage (for loaded tensors)
    pub fn memory_usage(&self) -> usize {
        self.tensors.values()
            .map(|tensor| tensor.data().len() * std::mem::size_of::<f32>())
            .sum()
    }

    /// Get metadata value by key
    pub fn get_metadata(&self, key: &str) -> Option<&GGUFValue> {
        self.metadata.get(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_loader_creation() {
        let device = GpuDevice::Cpu;
        let loader = GGUFLoader::new(device);

        // Basic sanity check
        assert_eq!(loader.device, GpuDevice::Cpu);
    }

    #[test]
    fn test_tensor_type_parsing() {
        let device = GpuDevice::Cpu;
        let loader = GGUFLoader::new(device);

        assert_eq!(loader.parse_tensor_type(0).unwrap(), GGUFTensorType::F32);
        assert_eq!(loader.parse_tensor_type(2).unwrap(), GGUFTensorType::Q4_0);
        assert_eq!(loader.parse_tensor_type(8).unwrap(), GGUFTensorType::Q8_0);
    }
}