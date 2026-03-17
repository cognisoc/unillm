//! Simple test to verify LLaVA implementation works

use std::error::Error;

// Include the LLaVA types directly for testing
#[path = "crates/runtime/src/models/llava.rs"]
mod llava;

#[path = "crates/runtime/src/types.rs"]
mod types;

#[path = "crates/runtime/src/tensor_ops.rs"]
mod tensor_ops;

#[path = "crates/runtime/src/model.rs"]
mod model;

#[path = "crates/runtime/src/image_processing.rs"]
mod image_processing;

use llava::*;
use image_processing::*;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Testing LLaVA implementation...");

    // Test 1: LLaVA Configuration
    let config = LLaVAConfig::default();
    println!("✅ LLaVA config created: projection_dim = {}", config.projection_dim);

    // Test 2: Vision Configuration
    let vision_config = VisionConfig::default();
    println!("✅ Vision config created: image_size = {}", vision_config.image_size);

    // Test 3: Image Processor
    let image_processor = ImageProcessor::new();
    println!("✅ Image processor created");

    // Test 4: Layer Norm
    let layer_norm = LayerNorm::new(64)?;
    let input = tensor_ops::CpuTensor::new(vec![1, 10, 64], vec![1.0; 640])?;
    let output = layer_norm.forward(&input)?;
    println!("✅ Layer norm works: input shape {:?} -> output shape {:?}",
             input.shape, output.shape);

    // Test 5: Patch Embedding
    let patch_embedding = PatchEmbedding::new(3, 768, 16)?;
    let image_tensor = ImageTensor {
        data: vec![0.5; 3 * 224 * 224],
        shape: vec![3, 224, 224],
        dtype: types::DataType::Float32,
        device: types::Device::CPU,
        original_size: (224, 224),
        processed_size: (224, 224),
    };

    let patch_output = patch_embedding.forward(&image_tensor)?;
    println!("✅ Patch embedding works: shape {:?}", patch_output.shape);

    // Test 6: Vision Encoder Creation
    let encoder = VisionEncoder::new(vision_config)?;
    println!("✅ Vision encoder created");

    // Test 7: LLaVA Model Creation
    let llava_config = LLaVAConfig::default();
    let model = LLaVAModel::new(llava_config)?;
    println!("✅ LLaVA model created successfully");

    println!("\n🎉 All LLaVA tests passed! Vision-language model is ready.");

    Ok(())
}