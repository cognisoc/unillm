//! UniLLM CLI - Run LLM inference from the command line
//!
//! Usage:
//!   cargo run --bin unillm -p runtime -- generate --prompt "Hello world"
//!   cargo run --bin unillm -p runtime -- generate --model llama2:7b --prompt "Explain gravity"
//!   cargo run --bin unillm -p runtime -- models

use anyhow::Result;
use clap::{Parser, Subcommand};
use runtime::model_core::{Model, ModelInputs, ModelOutputs};
use runtime::models_v2::llama::{LlamaConfig, LlamaModelV2};
use runtime::ollama::OllamaRegistry;
use runtime::tensor_core::{Device, Tensor};
use runtime::tokenizer::Tokenizer;
use runtime::weight_loader_core::UnifiedWeightLoader;
use std::io::{self, Write};

const DEFAULT_MODEL: &str = "tinyllama:latest";

#[derive(Parser)]
#[command(name = "unillm", about = "UniLLM - LLM inference runtime")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate text from a prompt
    Generate {
        /// The prompt to generate from
        #[arg(short, long)]
        prompt: String,

        /// Model to use (Ollama format, e.g. tinyllama:latest, llama2:7b)
        #[arg(short, long, default_value = DEFAULT_MODEL)]
        model: String,

        /// Maximum number of tokens to generate
        #[arg(long, default_value_t = 100)]
        max_tokens: usize,

        /// Sampling temperature (0.0 = greedy, higher = more random)
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
    },

    /// List locally cached models
    Models,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            prompt,
            model,
            max_tokens,
            temperature,
        } => {
            generate(&model, &prompt, max_tokens, temperature).await?;
        }
        Commands::Models => {
            list_models()?;
        }
    }

    Ok(())
}

async fn generate(model_name: &str, prompt: &str, max_tokens: usize, temperature: f32) -> Result<()> {
    // Step 1: Download model
    eprint!("Downloading {}...", model_name);
    let registry = OllamaRegistry::new()?;
    let model_path = registry.pull(model_name).await?;
    eprintln!(" done");

    // Step 2: Load weights
    eprint!("Loading weights...");
    let loader = UnifiedWeightLoader::new();
    let weights = loader.load_weights(&model_path)?;
    eprintln!(" done ({} tensors)", weights.tensors.len());

    // Step 3: Create config from GGUF metadata
    let config = if let Some(ref gguf_config) = weights.gguf_config {
        LlamaConfig::from_gguf_config(gguf_config)
    } else {
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

    // Step 4: Create tokenizer
    let tokenizer = Tokenizer::from_model_weights(&weights)?;

    // Step 5: Create model
    eprint!("Loading model...");
    let model = LlamaModelV2::from_weights(config, weights)?;
    eprintln!(" done");

    // Step 6: Generate
    let mut tokens: Vec<u32> = tokenizer.encode_with_special_tokens(prompt, true, false);
    let device = Device::CPU;

    // Print prompt
    print!("{}", prompt);
    io::stdout().flush()?;

    for _ in 0..max_tokens {
        let tokens_i64: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
        let input_tensor = Tensor::from_i64_slice(&tokens_i64, &[1, tokens.len()], &device)?;

        let inputs = ModelInputs::Text {
            input_ids: input_tensor,
            attention_mask: None,
            position_ids: None,
        };

        let outputs = model.forward(&inputs)?;

        let logits = match outputs {
            ModelOutputs::Logits { logits, .. } => logits,
            _ => return Err(anyhow::anyhow!("Expected logits output")),
        };

        let logits_candle = logits.to_candle()?;
        let shape = logits_candle.dims();

        let last_logits = if shape.len() == 3 {
            let seq_len = shape[1];
            logits_candle.narrow(1, seq_len - 1, 1)?.squeeze(1)?.squeeze(0)?
        } else {
            let seq_len = shape[0];
            logits_candle.narrow(0, seq_len - 1, 1)?.squeeze(0)?
        };

        let logits_vec: Vec<f32> = last_logits.to_vec1()?;

        let next_token = if temperature > 0.0 {
            sample_temperature(&logits_vec, temperature)
        } else {
            sample_greedy(&logits_vec)
        };

        if next_token == tokenizer.eos_token_id() {
            break;
        }

        tokens.push(next_token);

        // Stream token to stdout
        let token_text = tokenizer.decode(&[next_token]);
        print!("{}", token_text);
        io::stdout().flush()?;
    }

    println!();
    Ok(())
}

fn sample_greedy(logits: &[f32]) -> u32 {
    let mut max_idx = 0;
    let mut max_val = logits[0];
    for (idx, &val) in logits.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = idx;
        }
    }
    max_idx as u32
}

fn sample_temperature(logits: &[f32], temperature: f32) -> u32 {
    use rand::Rng;

    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&l| ((l - max_logit) / temperature).exp()).sum();
    let probs: Vec<f32> = logits
        .iter()
        .map(|&l| ((l - max_logit) / temperature).exp() / exp_sum)
        .collect();

    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;
    for (idx, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= r {
            return idx as u32;
        }
    }
    (probs.len() - 1) as u32
}

fn list_models() -> Result<()> {
    let registry = OllamaRegistry::new()?;
    let models = registry.list_cached();

    if models.is_empty() {
        println!("No cached models.");
        println!("Download one with: unillm generate --model tinyllama:latest --prompt \"hello\"");
    } else {
        println!("Cached models:");
        for model in models {
            println!("  {}", model);
        }
    }

    Ok(())
}
