//! Quantization Support for Memory-Efficient Inference
//!
//! This module provides comprehensive quantization support including:
//! - INT8 dynamic quantization for activations
//! - FP16 half-precision for weights and activations
//! - INT4 aggressive quantization (experimental)
//! - Mixed-precision inference strategies

use crate::types::*;
use crate::gpu_tensor_ops::{GpuTensor, GpuDevice};
use std::collections::HashMap;

/// Quantization data types
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum QuantizationType {
    /// Full precision (32-bit float)
    FP32,
    /// Half precision (16-bit float)
    FP16,
    /// 8-bit integer quantization
    INT8,
    /// 4-bit integer quantization (experimental)
    INT4,
    /// Mixed precision (different layers use different precisions)
    Mixed,
}

impl QuantizationType {
    /// Get memory reduction factor compared to FP32
    pub fn memory_reduction(&self) -> f32 {
        match self {
            Self::FP32 => 1.0,
            Self::FP16 => 2.0,
            Self::INT8 => 4.0,
            Self::INT4 => 8.0,
            Self::Mixed => 2.5, // Typical mixed precision savings
        }
    }

    /// Get bytes per parameter
    pub fn bytes_per_param(&self) -> usize {
        match self {
            Self::FP32 => 4,
            Self::FP16 => 2,
            Self::INT8 => 1,
            Self::INT4 => 1, // Packed, but still 1 byte minimum
            Self::Mixed => 2, // Average
        }
    }
}

/// Quantization configuration
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    pub weight_quantization: QuantizationType,
    pub activation_quantization: QuantizationType,
    pub enable_dynamic_quantization: bool,
    pub calibration_samples: usize,
    pub symmetric_quantization: bool,
    pub per_channel_quantization: bool,
    pub quantization_error_threshold: f32,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            weight_quantization: QuantizationType::FP16,
            activation_quantization: QuantizationType::FP32,
            enable_dynamic_quantization: true,
            calibration_samples: 512,
            symmetric_quantization: true,
            per_channel_quantization: false,
            quantization_error_threshold: 0.01, // 1% max error
        }
    }
}

/// Quantization statistics for a tensor
#[derive(Debug, Clone)]
pub struct QuantizationStats {
    pub min_val: f32,
    pub max_val: f32,
    pub scale: f32,
    pub zero_point: i32,
    pub channel_stats: Option<Vec<(f32, f32)>>, // per-channel (scale, zero_point)
}

impl QuantizationStats {
    /// Calculate quantization parameters from tensor data
    pub fn from_tensor(tensor: &GpuTensor, qtype: QuantizationType, symmetric: bool) -> ModelResult<Self> {
        let data = tensor.to_vec()?;

        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let (scale, zero_point) = match qtype {
            QuantizationType::INT8 => Self::calculate_int8_params(min_val, max_val, symmetric),
            QuantizationType::INT4 => Self::calculate_int4_params(min_val, max_val, symmetric),
            _ => (1.0, 0), // No quantization for FP types
        };

        Ok(Self {
            min_val,
            max_val,
            scale,
            zero_point,
            channel_stats: None,
        })
    }

    fn calculate_int8_params(min_val: f32, max_val: f32, symmetric: bool) -> (f32, i32) {
        if symmetric {
            let abs_max = min_val.abs().max(max_val.abs());
            let scale = abs_max / 127.0;
            (scale, 0)
        } else {
            let scale = (max_val - min_val) / 255.0;
            let zero_point = (-min_val / scale).round() as i32 - 128;
            (scale, zero_point)
        }
    }

    fn calculate_int4_params(min_val: f32, max_val: f32, symmetric: bool) -> (f32, i32) {
        if symmetric {
            let abs_max = min_val.abs().max(max_val.abs());
            let scale = abs_max / 7.0;
            (scale, 0)
        } else {
            let scale = (max_val - min_val) / 15.0;
            let zero_point = (-min_val / scale).round() as i32 - 8;
            (scale, zero_point)
        }
    }
}

/// Quantized tensor representation
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub qtype: QuantizationType,
    pub stats: QuantizationStats,
    pub device: GpuDevice,
}

impl QuantizedTensor {
    /// Create quantized tensor from float tensor
    pub fn from_tensor(
        tensor: &GpuTensor,
        qtype: QuantizationType,
        config: &QuantizationConfig,
    ) -> ModelResult<Self> {
        let shape = tensor.shape();
        let device = tensor.device.clone();
        let stats = QuantizationStats::from_tensor(tensor, qtype, config.symmetric_quantization)?;

        let data = match qtype {
            QuantizationType::FP32 => Self::quantize_fp32(tensor)?,
            QuantizationType::FP16 => Self::quantize_fp16(tensor)?,
            QuantizationType::INT8 => Self::quantize_int8(tensor, &stats)?,
            QuantizationType::INT4 => Self::quantize_int4(tensor, &stats)?,
            QuantizationType::Mixed => {
                return Err(ModelError::ComputationFailed(
                    "Mixed quantization not implemented for single tensor".to_string()
                ));
            }
        };

        Ok(Self {
            data,
            shape,
            qtype,
            stats,
            device,
        })
    }

    fn quantize_fp32(tensor: &GpuTensor) -> ModelResult<Vec<u8>> {
        let float_data = tensor.to_vec()?;
        let mut bytes = Vec::with_capacity(float_data.len() * 4);

        for &val in &float_data {
            bytes.extend_from_slice(&val.to_le_bytes());
        }

        Ok(bytes)
    }

    fn quantize_fp16(tensor: &GpuTensor) -> ModelResult<Vec<u8>> {
        let float_data = tensor.to_vec()?;
        let mut bytes = Vec::with_capacity(float_data.len() * 2);

        for &val in &float_data {
            let fp16_bits = half::f16::from_f32(val).to_bits();
            bytes.extend_from_slice(&fp16_bits.to_le_bytes());
        }

        Ok(bytes)
    }

    fn quantize_int8(tensor: &GpuTensor, stats: &QuantizationStats) -> ModelResult<Vec<u8>> {
        let float_data = tensor.to_vec()?;
        let mut quantized = Vec::with_capacity(float_data.len());

        for &val in &float_data {
            let quantized_val = ((val / stats.scale) + stats.zero_point as f32)
                .round()
                .clamp(-128.0, 127.0) as i8;

            quantized.push(quantized_val as u8);
        }

        Ok(quantized)
    }

    fn quantize_int4(tensor: &GpuTensor, stats: &QuantizationStats) -> ModelResult<Vec<u8>> {
        let float_data = tensor.to_vec()?;
        let mut quantized = Vec::with_capacity((float_data.len() + 1) / 2);

        // Pack two 4-bit values into one byte
        for chunk in float_data.chunks(2) {
            let val1 = ((chunk[0] / stats.scale) + stats.zero_point as f32)
                .round()
                .clamp(-8.0, 7.0) as i8;

            let val2 = if chunk.len() > 1 {
                ((chunk[1] / stats.scale) + stats.zero_point as f32)
                    .round()
                    .clamp(-8.0, 7.0) as i8
            } else {
                0
            };

            // Pack into single byte: lower 4 bits for val1, upper 4 bits for val2
            let packed = ((val2 & 0x0F) << 4) | (val1 & 0x0F);
            quantized.push(packed as u8);
        }

        Ok(quantized)
    }

    /// Dequantize back to float tensor
    pub fn dequantize(&self) -> ModelResult<GpuTensor> {
        let float_data = match self.qtype {
            QuantizationType::FP32 => self.dequantize_fp32()?,
            QuantizationType::FP16 => self.dequantize_fp16()?,
            QuantizationType::INT8 => self.dequantize_int8()?,
            QuantizationType::INT4 => self.dequantize_int4()?,
            QuantizationType::Mixed => {
                return Err(ModelError::ComputationFailed(
                    "Mixed quantization dequantization not implemented".to_string()
                ));
            }
        };

        GpuTensor::new(float_data, self.shape.clone(), self.device.clone())
    }

    fn dequantize_fp32(&self) -> ModelResult<Vec<f32>> {
        let mut floats = Vec::with_capacity(self.data.len() / 4);

        for chunk in self.data.chunks_exact(4) {
            let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
            let val = f32::from_le_bytes(bytes);
            floats.push(val);
        }

        Ok(floats)
    }

    fn dequantize_fp16(&self) -> ModelResult<Vec<f32>> {
        let mut floats = Vec::with_capacity(self.data.len() / 2);

        for chunk in self.data.chunks_exact(2) {
            let bytes = [chunk[0], chunk[1]];
            let fp16_bits = u16::from_le_bytes(bytes);
            let fp16_val = half::f16::from_bits(fp16_bits);
            floats.push(fp16_val.to_f32());
        }

        Ok(floats)
    }

    fn dequantize_int8(&self) -> ModelResult<Vec<f32>> {
        let mut floats = Vec::with_capacity(self.data.len());

        for &byte in &self.data {
            let quantized_val = byte as i8;
            let float_val = (quantized_val as f32 - self.stats.zero_point as f32) * self.stats.scale;
            floats.push(float_val);
        }

        Ok(floats)
    }

    fn dequantize_int4(&self) -> ModelResult<Vec<f32>> {
        let mut floats = Vec::with_capacity(self.data.len() * 2);

        for &byte in &self.data {
            // Unpack two 4-bit values
            let val1 = ((byte & 0x0F) as i8) << 4 >> 4; // Sign extend
            let val2 = ((byte & 0xF0) as i8) >> 4;

            let float1 = (val1 as f32 - self.stats.zero_point as f32) * self.stats.scale;
            let float2 = (val2 as f32 - self.stats.zero_point as f32) * self.stats.scale;

            floats.push(float1);
            floats.push(float2);
        }

        Ok(floats)
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.data.len()
    }

    /// Calculate compression ratio vs FP32
    pub fn compression_ratio(&self) -> f32 {
        let fp32_size = self.shape.iter().product::<usize>() * 4;
        fp32_size as f32 / self.data.len() as f32
    }
}

/// Dynamic quantization calibrator
pub struct DynamicQuantizer {
    config: QuantizationConfig,
    calibration_data: HashMap<String, Vec<f32>>,
    layer_stats: HashMap<String, QuantizationStats>,
}

impl DynamicQuantizer {
    pub fn new(config: QuantizationConfig) -> Self {
        Self {
            config,
            calibration_data: HashMap::new(),
            layer_stats: HashMap::new(),
        }
    }

    /// Add calibration sample for a layer
    pub fn add_calibration_sample(&mut self, layer_name: String, tensor: &GpuTensor) -> ModelResult<()> {
        let data = tensor.to_vec()?;

        self.calibration_data
            .entry(layer_name)
            .or_insert_with(Vec::new)
            .extend(data);

        Ok(())
    }

    /// Finalize calibration and compute quantization parameters
    pub fn finalize_calibration(&mut self) -> ModelResult<()> {
        for (layer_name, data) in &self.calibration_data {
            if data.len() < self.config.calibration_samples {
                continue;
            }

            let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            let (scale, zero_point) = match self.config.activation_quantization {
                QuantizationType::INT8 => {
                    QuantizationStats::calculate_int8_params(min_val, max_val, self.config.symmetric_quantization)
                }
                QuantizationType::INT4 => {
                    QuantizationStats::calculate_int4_params(min_val, max_val, self.config.symmetric_quantization)
                }
                _ => (1.0, 0),
            };

            let stats = QuantizationStats {
                min_val,
                max_val,
                scale,
                zero_point,
                channel_stats: None,
            };

            self.layer_stats.insert(layer_name.clone(), stats);
        }

        Ok(())
    }

    /// Get quantization stats for a layer
    pub fn get_layer_stats(&self, layer_name: &str) -> Option<&QuantizationStats> {
        self.layer_stats.get(layer_name)
    }
}

/// Mixed precision inference manager
pub struct MixedPrecisionManager {
    layer_precision: HashMap<String, QuantizationType>,
    precision_policy: MixedPrecisionPolicy,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum MixedPrecisionPolicy {
    /// Attention layers in FP16, others in INT8
    AttentionFP16,
    /// First and last layers in FP16, middle in INT8
    EdgesFP16,
    /// Custom per-layer configuration
    Custom(HashMap<String, QuantizationType>),
}

impl MixedPrecisionManager {
    pub fn new(policy: MixedPrecisionPolicy) -> Self {
        Self {
            layer_precision: HashMap::new(),
            precision_policy: policy,
        }
    }

    /// Initialize precision assignment based on policy
    pub fn initialize_precision(&mut self, layer_names: &[String]) {
        match &self.precision_policy {
            MixedPrecisionPolicy::AttentionFP16 => {
                for name in layer_names {
                    if name.contains("attention") || name.contains("attn") {
                        self.layer_precision.insert(name.clone(), QuantizationType::FP16);
                    } else {
                        self.layer_precision.insert(name.clone(), QuantizationType::INT8);
                    }
                }
            }
            MixedPrecisionPolicy::EdgesFP16 => {
                for (i, name) in layer_names.iter().enumerate() {
                    if i == 0 || i == layer_names.len() - 1 {
                        self.layer_precision.insert(name.clone(), QuantizationType::FP16);
                    } else {
                        self.layer_precision.insert(name.clone(), QuantizationType::INT8);
                    }
                }
            }
            MixedPrecisionPolicy::Custom(custom_map) => {
                self.layer_precision = custom_map.clone();
            }
        }
    }

    /// Get precision for a layer
    pub fn get_layer_precision(&self, layer_name: &str) -> QuantizationType {
        self.layer_precision.get(layer_name).copied().unwrap_or(QuantizationType::FP16)
    }
}

/// Quantization-aware model wrapper
pub struct QuantizedModel {
    quantized_layers: HashMap<String, QuantizedTensor>,
    config: QuantizationConfig,
    mixed_precision: Option<MixedPrecisionManager>,
    device: GpuDevice,
}

impl QuantizedModel {
    pub fn new(config: QuantizationConfig, device: GpuDevice) -> Self {
        Self {
            quantized_layers: HashMap::new(),
            config,
            mixed_precision: None,
            device,
        }
    }

    /// Enable mixed precision inference
    pub fn enable_mixed_precision(&mut self, policy: MixedPrecisionPolicy) {
        self.mixed_precision = Some(MixedPrecisionManager::new(policy));
    }

    /// Add quantized layer
    pub fn add_quantized_layer(&mut self, name: String, tensor: QuantizedTensor) {
        self.quantized_layers.insert(name, tensor);
    }

    /// Get layer in appropriate precision
    pub fn get_layer(&self, name: &str) -> Option<ModelResult<GpuTensor>> {
        self.quantized_layers.get(name).map(|qt| qt.dequantize())
    }

    /// Get total memory usage
    pub fn memory_usage(&self) -> usize {
        self.quantized_layers.values().map(|qt| qt.memory_bytes()).sum()
    }

    /// Get total compression ratio
    pub fn compression_ratio(&self) -> f32 {
        let total_compressed: usize = self.quantized_layers.values().map(|qt| qt.memory_bytes()).sum();
        let total_uncompressed: usize = self.quantized_layers.values()
            .map(|qt| qt.shape.iter().product::<usize>() * 4)
            .sum();

        total_uncompressed as f32 / total_compressed as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_types() {
        assert_eq!(QuantizationType::FP32.memory_reduction(), 1.0);
        assert_eq!(QuantizationType::FP16.memory_reduction(), 2.0);
        assert_eq!(QuantizationType::INT8.memory_reduction(), 4.0);
        assert_eq!(QuantizationType::INT4.memory_reduction(), 8.0);
    }

    #[test]
    fn test_quantization_config() {
        let config = QuantizationConfig::default();
        assert_eq!(config.weight_quantization, QuantizationType::FP16);
        assert!(config.enable_dynamic_quantization);
        assert_eq!(config.calibration_samples, 512);
    }

    #[test]
    fn test_quantized_tensor_fp16() {
        let device = GpuDevice::auto_detect();
        let config = QuantizationConfig::default();

        let original = GpuTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], device).unwrap();
        let quantized = QuantizedTensor::from_tensor(&original, QuantizationType::FP16, &config).unwrap();

        assert_eq!(quantized.qtype, QuantizationType::FP16);
        assert_eq!(quantized.data.len(), 8); // 4 floats * 2 bytes each

        let dequantized = quantized.dequantize().unwrap();
        let recovered_data = dequantized.to_vec().unwrap();

        // FP16 should be very close to original
        for (orig, recovered) in vec![1.0, 2.0, 3.0, 4.0].iter().zip(recovered_data.iter()) {
            assert!((orig - recovered).abs() < 0.01);
        }
    }

    #[test]
    fn test_quantized_tensor_int8() {
        let device = GpuDevice::auto_detect();
        let config = QuantizationConfig::default();

        let original = GpuTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], device).unwrap();
        let quantized = QuantizedTensor::from_tensor(&original, QuantizationType::INT8, &config).unwrap();

        assert_eq!(quantized.qtype, QuantizationType::INT8);
        assert_eq!(quantized.data.len(), 4); // 4 floats * 1 byte each

        let dequantized = quantized.dequantize().unwrap();
        assert_eq!(dequantized.shape(), vec![2, 2]);

        // Check compression ratio
        assert_eq!(quantized.compression_ratio(), 4.0);
    }

    #[test]
    fn test_dynamic_quantizer() {
        let config = QuantizationConfig::default();
        let mut quantizer = DynamicQuantizer::new(config);

        let device = GpuDevice::auto_detect();
        let tensor = GpuTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], device).unwrap();

        // Add calibration sample
        let result = quantizer.add_calibration_sample("layer1".to_string(), &tensor);
        assert!(result.is_ok());

        // Should have calibration data
        assert!(quantizer.calibration_data.contains_key("layer1"));
    }

    #[test]
    fn test_mixed_precision_manager() {
        let mut manager = MixedPrecisionManager::new(MixedPrecisionPolicy::AttentionFP16);
        let layers = vec![
            "embedding".to_string(),
            "attention_layer_1".to_string(),
            "mlp_layer_1".to_string(),
            "attention_layer_2".to_string(),
        ];

        manager.initialize_precision(&layers);

        assert_eq!(manager.get_layer_precision("embedding"), QuantizationType::INT8);
        assert_eq!(manager.get_layer_precision("attention_layer_1"), QuantizationType::FP16);
        assert_eq!(manager.get_layer_precision("mlp_layer_1"), QuantizationType::INT8);
        assert_eq!(manager.get_layer_precision("attention_layer_2"), QuantizationType::FP16);
    }

    #[test]
    fn test_quantized_model() {
        let device = GpuDevice::auto_detect();
        let config = QuantizationConfig::default();
        let mut model = QuantizedModel::new(config.clone(), device.clone());

        // Create and add a quantized layer
        let tensor = GpuTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], device).unwrap();
        let quantized = QuantizedTensor::from_tensor(&tensor, QuantizationType::FP16, &config).unwrap();

        model.add_quantized_layer("test_layer".to_string(), quantized);

        // Test retrieval
        let retrieved = model.get_layer("test_layer");
        assert!(retrieved.is_some());
        assert!(retrieved.unwrap().is_ok());

        // Test metrics
        assert!(model.memory_usage() > 0);
        assert!(model.compression_ratio() > 1.0);
    }
}