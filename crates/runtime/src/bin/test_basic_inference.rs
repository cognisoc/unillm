//! Basic inference pipeline test
//!
//! This binary tests the basic inference pipeline with a minimal Llama model.

use runtime::{
    inference::InferencePipelineBuilder,
    models_v2::llama::LlamaConfig,
    model_core::{GenerationConfig, ModelConfig},
};

fn main() -> anyhow::Result<()> {
    println!("🚀 Testing basic inference pipeline...");

    // Create a minimal Llama config for testing
    let config = LlamaConfig {
        vocab_size: 1000,
        hidden_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 4,
        intermediate_size: 128,
        max_position_embeddings: 128,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-6,
        ..Default::default()
    };

    println!("📝 Model config: vocab_size={}, hidden_size={}, layers={}",
             config.vocab_size(), config.hidden_size(), config.num_layers());

    // Create pipeline
    let pipeline = InferencePipelineBuilder::new()
        .with_model_config(config)
        .build()?;

    println!("✅ Pipeline created successfully!");

    // Test generation
    let gen_config = GenerationConfig {
        max_new_tokens: 10,
        temperature: 1.0,
        do_sample: false, // Use greedy for deterministic output
        eos_token_id: 2,
        ..Default::default()
    };

    println!("🤖 Testing generation...");
    let prompt = "Hello world";

    match pipeline.generate(prompt, &gen_config) {
        Ok(output) => {
            println!("✅ Generation successful!");
            println!("📤 Input:  '{}'", prompt);
            println!("📥 Output: '{}'", output);
        },
        Err(e) => {
            println!("❌ Generation failed: {}", e);
            return Err(e.into()); // Convert ModelError to anyhow::Error
        }
    }

    println!("🎉 Basic inference pipeline test completed!");

    Ok(())
}