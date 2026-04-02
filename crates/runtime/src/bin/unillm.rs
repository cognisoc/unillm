//! UniLLM - Rust-native LLM inference engine
//!
//! Usage:
//!   unillm -m tinyllama:latest -p "Hello world" -n 128
//!   unillm -m model.gguf -p "The meaning of life is" -v
//!   unillm -m tinyllama:latest -i
//!   unillm pull tinyllama:latest
//!   unillm list
//!   unillm serve -m tinyllama:latest --port 8080

use anyhow::Result;
use clap::{Parser, Subcommand};
use runtime::model_core::{GenerationConfig, GenerationMetrics, Model, ModelRunner};
use runtime::models_v2::llama::{LlamaConfig, LlamaModelV2};
use runtime::ollama::OllamaRegistry;
use runtime::tokenizer::Tokenizer;
use runtime::weight_loader_core::UnifiedWeightLoader;
use runtime::chat_template::{ChatMessage, ChatTemplateFormat};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "unillm", about = "UniLLM - Rust-native LLM inference engine")]
struct Cli {
    /// GGUF file path or Ollama model name (e.g. tinyllama:latest)
    #[arg(short = 'm', long)]
    model: Option<String>,

    /// Generation prompt
    #[arg(short = 'p', long)]
    prompt: Option<String>,

    /// Maximum tokens to generate
    #[arg(short = 'n', long = "n-predict", default_value = "128")]
    n_predict: usize,

    /// Temperature for sampling (0 = greedy)
    #[arg(long, default_value = "0.8")]
    temp: f32,

    /// Top-p (nucleus) sampling threshold
    #[arg(long = "top-p", default_value = "0.95")]
    top_p: f32,

    /// Top-k sampling (0 = disabled)
    #[arg(long = "top-k")]
    top_k: Option<usize>,

    /// Repetition penalty (1.0 = disabled)
    #[arg(long = "repeat-penalty", default_value = "1.1")]
    repeat_penalty: f32,

    /// Context size
    #[arg(short = 'c', long = "ctx-size", default_value = "2048")]
    ctx_size: usize,

    /// Number of CPU threads
    #[arg(short = 't', long)]
    threads: Option<usize>,

    /// Random seed
    #[arg(short = 's', long)]
    seed: Option<u64>,

    /// Interactive chat mode
    #[arg(short = 'i', long)]
    interactive: bool,

    /// Print performance metrics
    #[arg(short = 'v', long)]
    verbose: bool,

    /// Device selection
    #[arg(long, default_value = "cpu")]
    gpu: String,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Pull a model from Ollama registry
    Pull {
        /// Model name (e.g. tinyllama:latest)
        model: String,
    },
    /// List cached models
    List,
    /// Start OpenAI-compatible HTTP server
    Serve {
        /// Model path or Ollama model name
        #[arg(short = 'm', long)]
        model: String,
        /// Port to listen on
        #[arg(long, default_value = "8080")]
        port: u16,
        /// Host to bind to
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        /// Max concurrent requests
        #[arg(long, default_value = "4")]
        concurrency: usize,
        /// Context size
        #[arg(short = 'c', long = "ctx-size", default_value = "2048")]
        ctx_size: usize,
        /// Device selection
        #[arg(long, default_value = "cpu")]
        gpu: String,
    },
}

// ============================================================================
// ModelRunner implementation for LlamaModelV2
// ============================================================================

/// Wraps a LlamaModelV2 + Tokenizer to implement ModelRunner
struct LlamaRunner {
    model: LlamaModelV2,
    tokenizer: Tokenizer,
    config: LlamaConfig,
}

impl ModelRunner for LlamaRunner {
    fn generate_streaming(
        &self,
        tokens: &[u32],
        config: &GenerationConfig,
        callback: &mut dyn FnMut(u32, &str) -> bool,
    ) -> Result<Vec<u32>> {
        self.model.generate_streaming_with_tokenizer(
            &self.tokenizer,
            tokens,
            config,
            callback,
        )
    }

    fn eos_token_id(&self) -> u32 {
        self.tokenizer.eos_token_id()
    }

    fn architecture(&self) -> &str {
        "llama"
    }

    fn context_size(&self) -> usize {
        self.config.max_position_embeddings
    }
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Handle subcommands
    match &cli.command {
        Some(Commands::Pull { model }) => {
            return pull_model(model).await;
        }
        Some(Commands::List) => {
            return list_models();
        }
        Some(Commands::Serve {
            model,
            port,
            host,
            concurrency,
            ctx_size,
            gpu,
        }) => {
            return serve(model, host, *port, *concurrency, *ctx_size, gpu).await;
        }
        None => {}
    }

    // Default: generation mode
    let model_name = cli.model.as_deref().unwrap_or_else(|| {
        eprintln!("Error: --model is required. Use -m <model> or --model <model>");
        eprintln!("  e.g.: unillm -m tinyllama:latest -p \"Hello\"");
        std::process::exit(1);
    });

    if cli.interactive {
        run_interactive(model_name, &cli).await
    } else {
        let prompt = cli.prompt.as_deref().unwrap_or_else(|| {
            eprintln!("Error: --prompt is required for non-interactive mode.");
            eprintln!("  e.g.: unillm -m tinyllama:latest -p \"Hello world\"");
            eprintln!("  Or use -i for interactive mode.");
            std::process::exit(1);
        });
        run_generation(model_name, prompt, &cli).await
    }
}

// ============================================================================
// Model Loading
// ============================================================================

/// Resolve model path - either a local GGUF file or Ollama model name
async fn resolve_model_path(model: &str) -> Result<PathBuf> {
    let path = Path::new(model);
    if path.exists() && path.extension().map_or(false, |e| e == "gguf") {
        return Ok(path.to_path_buf());
    }

    // Try Ollama registry
    eprintln!("Pulling model from Ollama registry: {}", model);
    let registry = OllamaRegistry::new()?;
    let model_path = registry.pull(model).await?;
    Ok(model_path)
}

/// Load model, tokenizer, and config from a GGUF file path
fn load_model(
    model_path: &Path,
    ctx_size: usize,
    gpu: &str,
    metrics: &mut GenerationMetrics,
) -> Result<(LlamaModelV2, Tokenizer, LlamaConfig)> {
    let load_start = Instant::now();

    // Load GGUF weights
    let loader = UnifiedWeightLoader::new();
    let weights = loader.load_weights(model_path)?;

    // Create tokenizer
    let tokenizer = Tokenizer::from_model_weights(&weights)?;

    // Create model config
    let mut config = if let Some(ref gguf_config) = weights.gguf_config {
        LlamaConfig::from_gguf_config(gguf_config)
    } else {
        LlamaConfig::default()
    };

    // Override context size if specified
    if ctx_size > 0 {
        config.max_position_embeddings = ctx_size;
    }

    // Update EOS token from tokenizer
    config.eos_token_id = tokenizer.eos_token_id() as i64;

    // Create model
    let mut model = LlamaModelV2::from_weights(config.clone(), weights)?;

    // Move to device if requested
    let device = parse_device(gpu);
    if !matches!(device, runtime::Device::CPU) {
        model.to_device(&device)?;
    }

    metrics.model_load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    Ok((model, tokenizer, config))
}

fn parse_device(gpu: &str) -> runtime::Device {
    match gpu.to_lowercase().as_str() {
        "metal" => runtime::Device::Metal(0),
        "cuda" | "cuda:0" => runtime::Device::CUDA(0),
        "auto" => {
            // Try Metal first (macOS), then CUDA, fallback to CPU
            #[cfg(target_os = "macos")]
            { return runtime::Device::Metal(0); }
            #[cfg(not(target_os = "macos"))]
            { runtime::Device::CPU }
        }
        _ => runtime::Device::CPU,
    }
}

// ============================================================================
// Generation Mode
// ============================================================================

async fn run_generation(model_name: &str, prompt: &str, cli: &Cli) -> Result<()> {
    let mut metrics = GenerationMetrics::new();

    let model_path = resolve_model_path(model_name).await?;
    let (model, tokenizer, config) = load_model(&model_path, cli.ctx_size, &cli.gpu, &mut metrics)?;

    if cli.verbose {
        eprintln!(
            "Model loaded: {} layers, {} heads, {} hidden, {:.0}ms",
            config.num_hidden_layers,
            config.num_attention_heads,
            config.hidden_size,
            metrics.model_load_ms
        );
    }

    // Tokenize prompt
    let prompt_tokens = tokenizer.encode_with_special_tokens(prompt, true, false);
    metrics.prompt_tokens = prompt_tokens.len();

    let gen_config = GenerationConfig {
        max_new_tokens: cli.n_predict,
        temperature: cli.temp,
        top_p: cli.top_p,
        top_k: cli.top_k,
        do_sample: cli.temp > 0.0,
        repetition_penalty: cli.repeat_penalty,
        eos_token_id: tokenizer.eos_token_id(),
        ..Default::default()
    };

    // Generate with streaming
    let gen_start = Instant::now();
    let mut prefill_done = false;
    let mut prefill_time = 0.0f64;
    let mut gen_tokens = 0usize;

    let _generated = model.generate_streaming_with_tokenizer(
        &tokenizer,
        &prompt_tokens,
        &gen_config,
        &mut |_token_id, text| {
            if !prefill_done {
                prefill_time = gen_start.elapsed().as_secs_f64() * 1000.0;
                prefill_done = true;
            }
            gen_tokens += 1;
            print!("{}", text);
            std::io::stdout().flush().ok();
            true
        },
    )?;

    println!(); // Final newline

    // Compute metrics
    metrics.prompt_eval_ms = prefill_time;
    metrics.generated_tokens = gen_tokens;
    metrics.generation_ms = gen_start.elapsed().as_secs_f64() * 1000.0 - prefill_time;
    metrics.finalize();

    if cli.verbose {
        metrics.print_summary();
    }

    Ok(())
}

// ============================================================================
// Interactive Chat Mode
// ============================================================================

async fn run_interactive(model_name: &str, cli: &Cli) -> Result<()> {
    let mut metrics = GenerationMetrics::new();
    let model_path = resolve_model_path(model_name).await?;
    let (model, tokenizer, config) = load_model(&model_path, cli.ctx_size, &cli.gpu, &mut metrics)?;

    eprintln!("Model loaded in {:.0}ms", metrics.model_load_ms);
    eprintln!(
        "Architecture: {} | Layers: {} | Context: {}",
        "llama", config.num_hidden_layers, config.max_position_embeddings
    );
    eprintln!("Type your message and press Enter. Type 'quit' or Ctrl+C to exit.\n");

    let template = ChatTemplateFormat::detect("llama", model_name);
    let mut history: Vec<ChatMessage> = Vec::new();

    loop {
        // Print prompt
        eprint!("> ");
        std::io::stderr().flush().ok();

        // Read user input
        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).is_err() {
            break;
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "quit" || input == "exit" {
            break;
        }
        if input == "/clear" {
            history.clear();
            eprintln!("Chat history cleared.");
            continue;
        }

        // Add user message to history
        history.push(ChatMessage {
            role: "user".to_string(),
            content: input.to_string(),
        });

        // Render prompt with template
        let prompt = template.render(&history);
        let prompt_tokens = tokenizer.encode_with_special_tokens(&prompt, true, false);

        let gen_config = GenerationConfig {
            max_new_tokens: cli.n_predict,
            temperature: cli.temp,
            top_p: cli.top_p,
            top_k: cli.top_k,
            do_sample: cli.temp > 0.0,
            repetition_penalty: cli.repeat_penalty,
            eos_token_id: tokenizer.eos_token_id(),
            ..Default::default()
        };

        // Generate response
        let gen_start = Instant::now();
        let mut response_text = String::new();

        let result = model.generate_streaming_with_tokenizer(
            &tokenizer,
            &prompt_tokens,
            &gen_config,
            &mut |_token_id, text| {
                response_text.push_str(text);
                print!("{}", text);
                std::io::stdout().flush().ok();
                true
            },
        );

        println!(); // Newline after response

        match result {
            Ok(_) => {
                // Add assistant response to history
                history.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: response_text,
                });

                if cli.verbose {
                    let elapsed = gen_start.elapsed().as_secs_f64() * 1000.0;
                    eprintln!("[{:.0}ms]", elapsed);
                }
            }
            Err(e) => {
                eprintln!("\nGeneration error: {}", e);
            }
        }
        println!();
    }

    Ok(())
}

// ============================================================================
// Pull & List Commands
// ============================================================================

async fn pull_model(model: &str) -> Result<()> {
    eprintln!("Pulling model: {}", model);
    let registry = OllamaRegistry::new()?;
    let path = registry.pull(model).await?;
    eprintln!("Model cached at: {}", path.display());
    Ok(())
}

fn list_models() -> Result<()> {
    let registry = OllamaRegistry::new()?;
    let models = registry.list_cached();
    if models.is_empty() {
        eprintln!("No cached models found.");
        eprintln!("Pull a model with: unillm pull tinyllama:latest");
    } else {
        eprintln!("Cached models (in {}):", registry.cache_dir().display());
        for model in &models {
            println!("  {}", model);
        }
    }
    Ok(())
}

// ============================================================================
// Serve Command
// ============================================================================

async fn serve(
    model_name: &str,
    host: &str,
    port: u16,
    concurrency: usize,
    ctx_size: usize,
    gpu: &str,
) -> Result<()> {
    let mut metrics = GenerationMetrics::new();
    let model_path = resolve_model_path(model_name).await?;
    let (model, tokenizer, config) = load_model(&model_path, ctx_size, gpu, &mut metrics)?;

    eprintln!("Model loaded in {:.0}ms", metrics.model_load_ms);
    eprintln!(
        "Architecture: {} | Layers: {} | Context: {} | Concurrency: {}",
        "llama", config.num_hidden_layers, config.max_position_embeddings, concurrency
    );

    // Build a second tokenizer for the server state from the model weights
    // (Tokenizer doesn't implement Clone, so we reload from the same path)
    let weights2 = UnifiedWeightLoader::new().load_weights(&model_path)?;
    let tokenizer2 = Tokenizer::from_model_weights(&weights2)?;

    let runner = LlamaRunner {
        model,
        tokenizer,
        config: config.clone(),
    };

    let template = ChatTemplateFormat::detect("llama", model_name);

    let state = std::sync::Arc::new(runtime::server::ModelState {
        model: Box::new(runner),
        tokenizer: tokenizer2,
        model_name: model_name.to_string(),
        chat_template: template,
        semaphore: tokio::sync::Semaphore::new(concurrency),
    });

    runtime::server::start_server(state, host, port).await
}
