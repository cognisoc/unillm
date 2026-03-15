//! Test program for the model loader

use runtime::Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing model loader...");
    
    // Create a model (this would normally load from a file)
    let model = Model::load("test_model.safetensors")?;
    
    println!("Model loaded successfully!");
    println!("Model config: {:?}", model.config());
    println!("Number of weights: {}", model.weights.len());
    
    // Try to access a specific weight
    if let Some(embedding) = model.get_weight("model.embed_tokens.weight") {
        println!("Embedding weight shape: {:?}", embedding.shape());
    } else {
        println!("Embedding weight not found");
    }
    
    Ok(())
}