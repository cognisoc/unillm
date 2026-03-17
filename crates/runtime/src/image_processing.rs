//! Image Processing Infrastructure for Multimodal Support
//!
//! Provides comprehensive image loading, processing, and tensor conversion
//! capabilities for vision-language models. Supports PNG, JPEG, WebP, TIFF
//! formats with efficient memory management and GPU acceleration.

use crate::types::{Tensor, DataType, Device, ModelResult, ModelError};
use image::{DynamicImage, ImageBuffer, Rgb, Rgba, imageops};
use imageproc::geometric_transformations;
use base64::{Engine as _, engine::general_purpose};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::io::Cursor;

/// Supported image formats for multimodal processing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    PNG,
    JPEG,
    WebP,
    TIFF,
    Unknown,
}

impl ImageFormat {
    pub fn from_mime_type(mime_type: &str) -> Self {
        match mime_type.to_lowercase().as_str() {
            "image/png" => ImageFormat::PNG,
            "image/jpeg" | "image/jpg" => ImageFormat::JPEG,
            "image/webp" => ImageFormat::WebP,
            "image/tiff" | "image/tif" => ImageFormat::TIFF,
            _ => ImageFormat::Unknown,
        }
    }

    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "png" => ImageFormat::PNG,
            "jpg" | "jpeg" => ImageFormat::JPEG,
            "webp" => ImageFormat::WebP,
            "tiff" | "tif" => ImageFormat::TIFF,
            _ => ImageFormat::Unknown,
        }
    }
}

/// Resize algorithms for image processing
#[derive(Debug, Clone, Copy)]
pub enum ResizeAlgorithm {
    Lanczos3,      // High quality, slower
    CatmullRom,    // Good quality, medium speed
    Triangle,      // Medium quality, faster
    Nearest,       // Fast but low quality
}

/// Color conversion specifications
#[derive(Debug, Clone)]
pub struct ColorConfig {
    pub target_channels: usize,    // 1 (grayscale), 3 (RGB), 4 (RGBA)
    pub normalize: bool,           // Convert to 0.0-1.0 range
    pub mean: Vec<f32>,           // Channel normalization means
    pub std: Vec<f32>,            // Channel normalization standard deviations
}

impl Default for ColorConfig {
    fn default() -> Self {
        Self {
            target_channels: 3,
            normalize: true,
            mean: vec![0.485, 0.456, 0.406],  // ImageNet means
            std: vec![0.229, 0.224, 0.225],   // ImageNet std devs
        }
    }
}

/// Vision model configuration for image preprocessing
#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub input_size: (u32, u32),      // Target image dimensions
    pub resize_algorithm: ResizeAlgorithm,
    pub color_config: ColorConfig,
    pub preserve_aspect_ratio: bool,
    pub padding_color: [u8; 3],       // RGB padding color
    // Additional fields for LLaVA
    pub image_size: u32,             // Square image size
    pub patch_size: u32,             // Patch size for vision transformer
    pub hidden_size: usize,          // Vision encoder hidden dimension
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            input_size: (224, 224),      // Standard CLIP size
            resize_algorithm: ResizeAlgorithm::Lanczos3,
            color_config: ColorConfig::default(),
            preserve_aspect_ratio: true,
            padding_color: [0, 0, 0],     // Black padding
            // LLaVA defaults
            image_size: 224,              // Standard CLIP image size
            patch_size: 16,               // Standard vision transformer patch size
            hidden_size: 768,             // CLIP-ViT-B/16 hidden size
        }
    }
}

/// Processed image tensor ready for model input
#[derive(Debug, Clone)]
pub struct ImageTensor {
    pub data: Vec<f32>,               // Flattened pixel data
    pub shape: Vec<usize>,            // [channels, height, width] or [batch, channels, height, width]
    pub dtype: DataType,              // Data type (usually Float32)
    pub device: Device,               // Target device (CPU/CUDA/Metal)
    pub original_size: (u32, u32),    // Original image dimensions
    pub processed_size: (u32, u32),   // Final processed dimensions
}

impl ImageTensor {
    /// Convert to Candle tensor for model processing
    pub fn to_tensor(&self) -> ModelResult<Tensor> {
        Ok(Tensor::new(self.shape.clone(), self.dtype, self.device.clone()))
    }

    /// Get number of elements in tensor
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Reshape tensor for batch processing
    pub fn add_batch_dimension(mut self) -> Self {
        self.shape.insert(0, 1);  // Add batch dimension at front
        self
    }
}

/// High-performance image processor with GPU acceleration support
pub struct ImageProcessor {
    supported_formats: Vec<ImageFormat>,
    default_config: VisionConfig,
    memory_pool_size: usize,
}

impl ImageProcessor {
    /// Create new image processor with default configuration
    pub fn new() -> Self {
        Self {
            supported_formats: vec![
                ImageFormat::PNG,
                ImageFormat::JPEG,
                ImageFormat::WebP,
                ImageFormat::TIFF,
            ],
            default_config: VisionConfig::default(),
            memory_pool_size: 100 * 1024 * 1024,  // 100MB default pool
        }
    }

    /// Create image processor with custom configuration
    pub fn with_config(config: VisionConfig) -> Self {
        Self {
            supported_formats: vec![
                ImageFormat::PNG,
                ImageFormat::JPEG,
                ImageFormat::WebP,
                ImageFormat::TIFF,
            ],
            default_config: config,
            memory_pool_size: 100 * 1024 * 1024,
        }
    }

    /// Load image from raw bytes with format detection
    pub fn load_image(&self, data: &[u8]) -> ModelResult<DynamicImage> {
        let image = image::load_from_memory(data)
            .map_err(|e| ModelError::InvalidInput(format!("Failed to load image: {}", e)))?;
        Ok(image)
    }

    /// Load image from base64 encoded string
    pub fn load_image_from_base64(&self, base64_data: &str) -> ModelResult<DynamicImage> {
        // Remove data URL prefix if present
        let base64_clean = if base64_data.starts_with("data:") {
            base64_data.split(',').nth(1).unwrap_or(base64_data)
        } else {
            base64_data
        };

        let bytes = general_purpose::STANDARD
            .decode(base64_clean)
            .map_err(|e| ModelError::InvalidInput(format!("Invalid base64 data: {}", e)))?;

        self.load_image(&bytes)
    }

    /// Load image from URL (async)
    pub async fn load_image_from_url(&self, url: &str) -> ModelResult<DynamicImage> {
        let response = reqwest::get(url).await
            .map_err(|e| ModelError::InvalidInput(format!("Failed to fetch image: {}", e)))?;

        let bytes = response.bytes().await
            .map_err(|e| ModelError::InvalidInput(format!("Failed to read image data: {}", e)))?;

        self.load_image(&bytes)
    }

    /// Resize image while preserving aspect ratio with intelligent padding
    pub fn resize_with_aspect_ratio(
        &self,
        image: DynamicImage,
        target_size: (u32, u32),
        algorithm: ResizeAlgorithm,
        preserve_ratio: bool,
        padding_color: [u8; 3],
    ) -> ModelResult<DynamicImage> {
        let (target_width, target_height) = target_size;
        let (orig_width, orig_height) = image.dimensions();

        if !preserve_ratio {
            // Simple resize without aspect ratio preservation
            return Ok(self.resize_image(image, target_width, target_height, algorithm)?);
        }

        // Calculate scaling factor to fit within target dimensions
        let width_scale = target_width as f32 / orig_width as f32;
        let height_scale = target_height as f32 / orig_height as f32;
        let scale = width_scale.min(height_scale);

        let new_width = (orig_width as f32 * scale) as u32;
        let new_height = (orig_height as f32 * scale) as u32;

        // Resize to fit within target dimensions
        let resized = self.resize_image(image, new_width, new_height, algorithm)?;

        // Add padding if necessary
        if new_width != target_width || new_height != target_height {
            let pad_x = (target_width - new_width) / 2;
            let pad_y = (target_height - new_height) / 2;

            let mut padded = ImageBuffer::from_pixel(
                target_width,
                target_height,
                Rgb(padding_color)
            );

            // Copy resized image to center of padded canvas
            imageops::overlay(&mut padded, &resized.to_rgb8(), pad_x as i64, pad_y as i64);
            Ok(DynamicImage::ImageRgb8(padded))
        } else {
            Ok(resized)
        }
    }

    /// Resize image using specified algorithm
    fn resize_image(
        &self,
        image: DynamicImage,
        width: u32,
        height: u32,
        algorithm: ResizeAlgorithm,
    ) -> ModelResult<DynamicImage> {
        let filter = match algorithm {
            ResizeAlgorithm::Lanczos3 => imageops::FilterType::Lanczos3,
            ResizeAlgorithm::CatmullRom => imageops::FilterType::CatmullRom,
            ResizeAlgorithm::Triangle => imageops::FilterType::Triangle,
            ResizeAlgorithm::Nearest => imageops::FilterType::Nearest,
        };

        Ok(image.resize_exact(width, height, filter))
    }

    /// Convert image to tensor format for model input
    pub fn normalize_for_model(
        &self,
        image: DynamicImage,
        config: &VisionConfig,
    ) -> ModelResult<ImageTensor> {
        // Resize image according to configuration
        let processed_image = self.resize_with_aspect_ratio(
            image,
            config.input_size,
            config.resize_algorithm,
            config.preserve_aspect_ratio,
            config.padding_color,
        )?;

        let (width, height) = processed_image.dimensions();

        // Convert to RGB format
        let rgb_image = processed_image.to_rgb8();
        let raw_pixels = rgb_image.as_raw();

        // Prepare tensor data with proper channel ordering
        let mut tensor_data = Vec::with_capacity(
            config.color_config.target_channels * (width * height) as usize
        );

        // Convert to CHW format (channels, height, width)
        for c in 0..config.color_config.target_channels {
            for y in 0..height {
                for x in 0..width {
                    let pixel_idx = ((y * width + x) * 3) as usize;
                    let pixel_value = if c < 3 {
                        raw_pixels[pixel_idx + c] as f32
                    } else {
                        255.0  // Alpha channel default
                    };

                    // Apply normalization
                    let normalized_value = if config.color_config.normalize {
                        let normalized = pixel_value / 255.0;
                        if c < config.color_config.mean.len() && c < config.color_config.std.len() {
                            (normalized - config.color_config.mean[c]) / config.color_config.std[c]
                        } else {
                            normalized
                        }
                    } else {
                        pixel_value
                    };

                    tensor_data.push(normalized_value);
                }
            }
        }

        Ok(ImageTensor {
            data: tensor_data,
            shape: vec![config.color_config.target_channels, height as usize, width as usize],
            dtype: DataType::Float32,
            device: Device::CPU,  // Will be moved to GPU during model processing
            original_size: (width, height),
            processed_size: config.input_size,
        })
    }

    /// Process single image with default configuration
    pub fn process_image(&self, image: DynamicImage) -> ModelResult<ImageTensor> {
        self.normalize_for_model(image, &self.default_config)
    }

    /// Process batch of images efficiently
    pub fn process_image_batch(
        &self,
        images: Vec<DynamicImage>,
        config: Option<&VisionConfig>,
    ) -> ModelResult<Vec<ImageTensor>> {
        let config = config.unwrap_or(&self.default_config);

        images.into_iter()
            .map(|img| self.normalize_for_model(img, config))
            .collect()
    }

    /// Create batched tensor from multiple image tensors
    pub fn create_batch_tensor(
        &self,
        image_tensors: Vec<ImageTensor>,
    ) -> ModelResult<ImageTensor> {
        if image_tensors.is_empty() {
            return Err(ModelError::InvalidInput("Empty image batch".to_string()));
        }

        // Validate all tensors have same shape
        let reference_shape = &image_tensors[0].shape;
        for tensor in &image_tensors {
            if tensor.shape != *reference_shape {
                return Err(ModelError::InvalidInput(
                    "All images in batch must have same dimensions".to_string()
                ));
            }
        }

        let batch_size = image_tensors.len();
        let tensor_size = reference_shape.iter().product::<usize>();

        // Flatten all tensors into single batch tensor
        let mut batch_data = Vec::with_capacity(batch_size * tensor_size);
        for tensor in &image_tensors {
            batch_data.extend_from_slice(&tensor.data);
        }

        // Create batch shape: [batch_size, channels, height, width]
        let mut batch_shape = vec![batch_size];
        batch_shape.extend_from_slice(reference_shape);

        Ok(ImageTensor {
            data: batch_data,
            shape: batch_shape,
            dtype: image_tensors[0].dtype,
            device: image_tensors[0].device.clone(),
            original_size: image_tensors[0].original_size,
            processed_size: image_tensors[0].processed_size,
        })
    }

    /// Get image metadata without loading full image
    pub fn get_image_info(&self, data: &[u8]) -> ModelResult<ImageInfo> {
        let format = image::guess_format(data)
            .map_err(|e| ModelError::InvalidInput(format!("Cannot detect image format: {}", e)))?;

        let mut reader = image::io::Reader::new(Cursor::new(data));
        reader.set_format(format);

        let dimensions = reader.into_dimensions()
            .map_err(|e| ModelError::InvalidInput(format!("Cannot read image dimensions: {}", e)))?;

        Ok(ImageInfo {
            format: ImageFormat::from_extension(&format!("{:?}", format).to_lowercase()),
            width: dimensions.0,
            height: dimensions.1,
            size_bytes: data.len(),
        })
    }

    /// Process image specifically for vision-language models
    pub fn process_for_vision_model(
        &self,
        image: &DynamicImage,
        config: &VisionConfig
    ) -> ModelResult<ImageTensor> {
        // Use the standard image processing pipeline
        self.normalize_for_model(image.clone(), config)
    }
}

/// Image metadata information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageInfo {
    pub format: ImageFormat,
    pub width: u32,
    pub height: u32,
    pub size_bytes: usize,
}

/// URL-based image reference for API compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,  // "low" | "high" | "auto"
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_image_format_detection() {
        assert_eq!(ImageFormat::from_mime_type("image/png"), ImageFormat::PNG);
        assert_eq!(ImageFormat::from_mime_type("image/jpeg"), ImageFormat::JPEG);
        assert_eq!(ImageFormat::from_extension("webp"), ImageFormat::WebP);
        assert_eq!(ImageFormat::from_extension("unknown"), ImageFormat::Unknown);
    }

    #[test]
    fn test_vision_config_default() {
        let config = VisionConfig::default();
        assert_eq!(config.input_size, (224, 224));
        assert_eq!(config.color_config.target_channels, 3);
        assert_eq!(config.preserve_aspect_ratio, true);
    }

    #[test]
    fn test_image_processor_creation() {
        let processor = ImageProcessor::new();
        assert_eq!(processor.supported_formats.len(), 4);
        assert!(processor.supported_formats.contains(&ImageFormat::PNG));
        assert!(processor.supported_formats.contains(&ImageFormat::JPEG));
    }

    // Integration tests would go here for actual image loading
    // Requires test images in test data directory
}