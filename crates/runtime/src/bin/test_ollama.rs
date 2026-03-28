//! Test Ollama Registry Integration
//!
//! Downloads a model from Ollama registry and tests inference.
//!
//! Usage:
//!   cargo run --bin test_ollama -p runtime
//!   cargo run --bin test_ollama -p runtime -- --model tinyllama:latest
//!   cargo run --bin test_ollama -p runtime -- --list-cached

use anyhow::Result;
use runtime::ollama::OllamaRegistry;
use runtime::weight_loader_core::UnifiedWeightLoader;
use runtime::model_core::{Model, ModelInputs, ModelOutputs};
use runtime::models_v2::llama::{LlamaModelV2, LlamaConfig};
use runtime::tokenizer::Tokenizer;
use runtime::tensor_core::Tensor;
use std::env;

// Use TinyLlama by default - it's Llama-compatible and ~600MB
const DEFAULT_MODEL: &str = "tinyllama:latest";

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== UniLLM Ollama Registry Test ===\n");

    let args: Vec<String> = env::args().collect();

    // Handle --list-cached flag
    if args.contains(&"--list-cached".to_string()) {
        let registry = OllamaRegistry::new()?;
        println!("Cached models in {}:", registry.cache_dir().display());
        for model in registry.list_cached() {
            println!("  - {}", model);
        }
        return Ok(());
    }

    // Get model name from args or use default
    let model_name = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or(DEFAULT_MODEL);

    println!("Model: {}", model_name);

    // Step 1: Download model from Ollama registry
    println!("\n[1/5] Downloading model from Ollama registry...");
    let registry = OllamaRegistry::new()?;
    let model_path = registry.pull(model_name).await?;
    println!("Model cached at: {}", model_path.display());

    // Step 2: Load GGUF weights
    println!("\n[2/5] Loading GGUF weights...");
    let loader = UnifiedWeightLoader::new();
    let weights = loader.load_weights(&model_path)?;
    println!("Architecture: {}", weights.metadata.architecture);
    println!("Total parameters: {}", weights.metadata.total_params);
    println!("Tensors loaded: {}", weights.tensors.len());

    // Print GGUF config if available
    if let Some(ref gguf_config) = weights.gguf_config {
        println!("\nGGUF Model Config:");
        println!("  vocab_size: {}", gguf_config.vocab_size);
        println!("  hidden_size: {}", gguf_config.hidden_size);
        println!("  intermediate_size: {}", gguf_config.intermediate_size);
        println!("  num_hidden_layers: {}", gguf_config.num_hidden_layers);
        println!("  num_attention_heads: {}", gguf_config.num_attention_heads);
        println!("  num_key_value_heads: {}", gguf_config.num_key_value_heads);
        println!("  head_dim: {}", gguf_config.head_dim);
    }

    // Step 3: Create tokenizer from GGUF
    println!("\n[3/5] Creating tokenizer from GGUF...");
    let tokenizer = Tokenizer::from_model_weights(&weights)?;
    println!("Tokenizer vocab size: {}", tokenizer.vocab_size());
    println!("BOS token ID: {}", tokenizer.bos_token_id());
    println!("EOS token ID: {}", tokenizer.eos_token_id());

    // Print some sample tokens
    println!("\nSample tokens:");
    for id in [0, 1, 2, 3, 100, 1000, 10000] {
        if let Some(token) = tokenizer.id_to_token(id) {
            println!("  ID {}: {:?}", id, token);
        }
    }

    // Step 4: Create model from weights
    println!("\n[4/5] Creating model from weights...");

    // Extract config from GGUF metadata
    let config = if let Some(ref gguf_config) = weights.gguf_config {
        LlamaConfig::from_gguf_config(gguf_config)
    } else {
        // Fallback to TinyLlama defaults
        LlamaConfig {
            vocab_size: 32000,
            hidden_size: 2048,
            intermediate_size: 5632,
            num_hidden_layers: 22,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            rms_norm_eps: 1e-5,
            ..Default::default()
        }
    };

    println!("Using config: vocab_size={}, hidden={}, layers={}, heads={}/{}",
        config.vocab_size, config.hidden_size, config.num_hidden_layers,
        config.num_attention_heads, config.num_key_value_heads);

    let model = LlamaModelV2::from_weights(config, weights)?;
    println!("Model created successfully!");

    // Step 5: Test generation with real tokenizer
    println!("\n[5/5] Testing generation with GGUF tokenizer...");

    // Simple prompt for testing
    let prompt = "1+1=";
    println!("Prompt: {}", prompt);

    // Encode prompt
    let mut tokens: Vec<u32> = tokenizer.encode_with_special_tokens(prompt, true, false);
    println!("Encoded tokens: {:?}", tokens);

    // Generation loop - just a few tokens for quick testing
    let max_new_tokens = 10;
    let device = runtime::tensor_core::Device::CPU;

    for _ in 0..max_new_tokens {
        // Create input tensor
        let tokens_i64: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
        let input_tensor = Tensor::from_i64_slice(&tokens_i64, &[1, tokens.len()], &device)?;

        let inputs = ModelInputs::Text {
            input_ids: input_tensor,
            attention_mask: None,
            position_ids: None,
        };

        // Forward pass
        let outputs = model.forward(&inputs)?;

        // Get logits
        let logits = match outputs {
            ModelOutputs::Logits { logits, .. } => logits,
            _ => return Err(anyhow::anyhow!("Expected logits output")),
        };

        // Get last token logits and sample (greedy)
        let logits_candle = logits.to_candle()?;
        let shape = logits_candle.dims();

        let last_logits = if shape.len() == 3 {
            let seq_len = shape[1];
            logits_candle
                .narrow(1, seq_len - 1, 1)?
                .squeeze(1)?
                .squeeze(0)?
        } else {
            let seq_len = shape[0];
            logits_candle
                .narrow(0, seq_len - 1, 1)?
                .squeeze(0)?
        };

        // Greedy sampling
        let logits_vec: Vec<f32> = last_logits.to_vec1()?;
        let mut max_idx = 0;
        let mut max_val = logits_vec[0];
        for (idx, &val) in logits_vec.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = idx;
            }
        }
        let next_token = max_idx as u32;

        // Check for EOS
        if next_token == tokenizer.eos_token_id() {
            break;
        }

        tokens.push(next_token);
    }

    // Decode output using GGUF tokenizer
    let output = tokenizer.decode(&tokens);
    println!("\nGenerated tokens: {:?}", tokens);
    println!("Output: {}", output);

    println!("\n=== Test Complete ===");
    Ok(())
}
