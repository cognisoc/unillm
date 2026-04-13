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
use runtime::service::InferenceService;
use runtime::model_core::{GenerationConfig, GenerationMetrics, Model, ModelRunner, ModelWeights};
use runtime::server::{ServerConfig, create_shared_state_with_config};
use runtime::models_v2::llama::{LlamaConfig, LlamaModelV2};
use runtime::models_v2::qwen::QwenModelV2;
use runtime::models_v2::mistral::MistralModelV2;
use runtime::models_v2::gemma::GemmaModelV2;
use runtime::models_v2::phi::PhiModelV2;
use runtime::models_v2::deepseek::DeepSeekModelV2;
use runtime::models_v2::mixtral::MixtralModelV2;
use runtime::models_v2::yi::YiModelV2;
use runtime::ollama::OllamaRegistry;
use runtime::tokenizer::Tokenizer;
use runtime::weight_loader_core::UnifiedWeightLoader;
use runtime::chat_template::{ChatMessage, ChatTemplateFormat};
use inference::UniLLMInferenceEngine;
use inference::types::SamplingParams;
use inference::InferenceEngine;
use inference::request::RequestBuilder;
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
        /// Request timeout in seconds
        #[arg(long, default_value = "300")]
        timeout: u64,
        /// Enable inference engine with batching/scheduling
        #[arg(long)]
        engine: bool,
    },
}

// ============================================================================
// ModelRunner implementations for each architecture
// ============================================================================

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

struct QwenRunner {
    model: QwenModelV2,
    tokenizer: Tokenizer,
    config: runtime::models_v2::qwen::QwenConfig,
}

impl ModelRunner for QwenRunner {
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
        "qwen"
    }

    fn context_size(&self) -> usize {
        self.config.max_position_embeddings
    }
}

struct MistralRunner {
    model: MistralModelV2,
    tokenizer: Tokenizer,
    config: runtime::models_v2::mistral::MistralConfig,
}

impl ModelRunner for MistralRunner {
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
        "mistral"
    }

    fn context_size(&self) -> usize {
        self.config.sliding_window
    }
}

struct GemmaRunner {
    model: GemmaModelV2,
    tokenizer: Tokenizer,
    config: runtime::models_v2::gemma::GemmaConfig,
}

impl ModelRunner for GemmaRunner {
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
        "gemma"
    }

    fn context_size(&self) -> usize {
        self.config.max_position_embeddings
    }
}

struct PhiRunner {
    model: PhiModelV2,
    tokenizer: Tokenizer,
    config: runtime::models_v2::phi::PhiConfig,
}

impl ModelRunner for PhiRunner {
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
        "phi"
    }

    fn context_size(&self) -> usize {
        self.config.max_position_embeddings
    }
}

struct DeepSeekRunner {
    model: DeepSeekModelV2,
    tokenizer: Tokenizer,
    config: runtime::models_v2::deepseek::DeepSeekConfig,
}

impl ModelRunner for DeepSeekRunner {
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
        "deepseek"
    }

    fn context_size(&self) -> usize {
        self.config.max_position_embeddings
    }
}

struct MixtralRunner {
    model: MixtralModelV2,
    tokenizer: Tokenizer,
    config: runtime::models_v2::mixtral::MixtralConfig,
}

impl ModelRunner for MixtralRunner {
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
        "mixtral"
    }

    fn context_size(&self) -> usize {
        self.config.sliding_window
    }
}

struct YiRunner {
    model: YiModelV2,
    tokenizer: Tokenizer,
    config: runtime::models_v2::yi::YiConfig,
}

impl ModelRunner for YiRunner {
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
        "yi"
    }

    fn context_size(&self) -> usize {
        self.config.max_position_embeddings
    }
}

/// Detect model architecture from weight metadata
fn detect_architecture(weights: &ModelWeights) -> &str {
    if let Some(ref gguf_config) = weights.gguf_config {
        if !gguf_config.architecture.is_empty() {
            let arch = gguf_config.architecture.to_lowercase();
            if arch.contains("llama") || arch.contains("codellama") {
                return "llama";
            }
            if arch.contains("qwen") {
                return "qwen";
            }
            if arch.contains("mixtral") {
                return "mixtral";
            }
            if arch.contains("mistral") {
                return "mistral";
            }
            if arch.contains("gemma") {
                return "gemma";
            }
            if arch.contains("phi") {
                return "phi";
            }
            if arch.contains("deepseek") {
                return "deepseek";
            }
            if arch.contains("yi") {
                return "yi";
            }
            return &gguf_config.architecture;
        }
    }
    "llama"
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

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
            timeout,
            engine,
        }) => {
            return serve(model, host, *port, *concurrency, *ctx_size, gpu, *timeout, *engine).await;
        }
        None => {}
    }

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

async fn resolve_model_path(model: &str) -> Result<PathBuf> {
    let path = Path::new(model);
    if path.exists() && path.extension().map_or(false, |e| e == "gguf") {
        return Ok(path.to_path_buf());
    }

    eprintln!("Pulling model from Ollama registry: {}", model);
    let registry = OllamaRegistry::new()?;
    let model_path = registry.pull(model).await?;
    Ok(model_path)
}

fn load_model_runner(
    model_path: &Path,
    ctx_size: usize,
    gpu: &str,
    metrics: &mut GenerationMetrics,
) -> Result<(Box<dyn ModelRunner>, String, usize)> {
    let load_start = Instant::now();

    let loader = UnifiedWeightLoader::new();
    let weights = loader.load_weights(model_path)?;

    let tokenizer = Tokenizer::from_model_weights(&weights)?;

    let arch = detect_architecture(&weights).to_string();
    eprintln!("Detected architecture: {}", arch);

    let runner: Box<dyn ModelRunner> = match arch.as_str() {
        "llama" => {
            let mut config = if let Some(ref gguf_config) = weights.gguf_config {
                LlamaConfig::from_gguf_config(gguf_config)
            } else {
                LlamaConfig::default()
            };
            if ctx_size > 0 {
                config.max_position_embeddings = ctx_size;
            }
            config.eos_token_id = tokenizer.eos_token_id() as i64;
            let mut model = LlamaModelV2::from_weights(config.clone(), weights)?;
            let device = parse_device(gpu);
            if !matches!(device, runtime::Device::CPU) {
                model.to_device(&device)?;
            }
            Box::new(LlamaRunner { model, tokenizer, config })
        }
        "qwen" => {
            let mut config = if let Some(ref gguf_config) = weights.gguf_config {
                runtime::models_v2::qwen::QwenConfig::from_gguf_config(gguf_config)
            } else {
                runtime::models_v2::qwen::QwenConfig::default()
            };
            if ctx_size > 0 {
                config.max_position_embeddings = ctx_size;
            }
            config.eos_token_id = tokenizer.eos_token_id() as i64;
            let mut model = QwenModelV2::from_weights(config.clone(), weights)?;
            let device = parse_device(gpu);
            if !matches!(device, runtime::Device::CPU) {
                model.to_device(&device)?;
            }
            Box::new(QwenRunner { model, tokenizer, config })
        }
        "mistral" => {
            let mut config = if let Some(ref gguf_config) = weights.gguf_config {
                runtime::models_v2::mistral::MistralConfig::from_gguf_config(gguf_config)
            } else {
                runtime::models_v2::mistral::MistralConfig::default()
            };
            if ctx_size > 0 {
                config.sliding_window = ctx_size;
            }
            config.eos_token_id = tokenizer.eos_token_id() as i64;
            let mut model = MistralModelV2::from_weights(config.clone(), weights)?;
            let device = parse_device(gpu);
            if !matches!(device, runtime::Device::CPU) {
                model.to_device(&device)?;
            }
            Box::new(MistralRunner { model, tokenizer, config })
        }
        "gemma" => {
            let mut config = if let Some(ref gguf_config) = weights.gguf_config {
                runtime::models_v2::gemma::GemmaConfig::from_gguf_config(gguf_config)
            } else {
                runtime::models_v2::gemma::GemmaConfig::default()
            };
            if ctx_size > 0 {
                config.max_position_embeddings = ctx_size;
            }
            let mut model = GemmaModelV2::from_weights(config.clone(), weights)?;
            let device = parse_device(gpu);
            if !matches!(device, runtime::Device::CPU) {
                model.to_device(&device)?;
            }
            Box::new(GemmaRunner { model, tokenizer, config })
        }
        "phi" => {
            let mut config = if let Some(ref gguf_config) = weights.gguf_config {
                runtime::models_v2::phi::PhiConfig::from_gguf_config(gguf_config)
            } else {
                runtime::models_v2::phi::PhiConfig::default()
            };
            if ctx_size > 0 {
                config.max_position_embeddings = ctx_size;
            }
            let mut model = PhiModelV2::from_weights(config.clone(), weights)?;
            let device = parse_device(gpu);
            if !matches!(device, runtime::Device::CPU) {
                model.to_device(&device)?;
            }
            Box::new(PhiRunner { model, tokenizer, config })
        }
        "deepseek" => {
            let mut config = if let Some(ref gguf_config) = weights.gguf_config {
                runtime::models_v2::deepseek::DeepSeekConfig::from_gguf_config(gguf_config)
            } else {
                runtime::models_v2::deepseek::DeepSeekConfig::default()
            };
            if ctx_size > 0 {
                config.max_position_embeddings = ctx_size;
            }
            config.eos_token_id = tokenizer.eos_token_id() as i64;
            let mut model = DeepSeekModelV2::from_weights(config.clone(), weights)?;
            let device = parse_device(gpu);
            if !matches!(device, runtime::Device::CPU) {
                model.to_device(&device)?;
            }
            Box::new(DeepSeekRunner { model, tokenizer, config })
        }
        "mixtral" => {
            let mut config = if let Some(ref gguf_config) = weights.gguf_config {
                runtime::models_v2::mixtral::MixtralConfig::from_gguf_config(gguf_config)
            } else {
                runtime::models_v2::mixtral::MixtralConfig::default()
            };
            if ctx_size > 0 {
                config.sliding_window = ctx_size;
            }
            config.eos_token_id = tokenizer.eos_token_id() as i64;
            let mut model = MixtralModelV2::from_weights(config.clone(), weights)?;
            let device = parse_device(gpu);
            if !matches!(device, runtime::Device::CPU) {
                model.to_device(&device)?;
            }
            Box::new(MixtralRunner { model, tokenizer, config })
        }
        "yi" => {
            let mut config = if let Some(ref gguf_config) = weights.gguf_config {
                runtime::models_v2::yi::YiConfig::from_gguf_config(gguf_config)
            } else {
                runtime::models_v2::yi::YiConfig::default()
            };
            if ctx_size > 0 {
                config.max_position_embeddings = ctx_size;
            }
            config.eos_token_id = tokenizer.eos_token_id() as i64;
            let mut model = YiModelV2::from_weights(config.clone(), weights)?;
            let device = parse_device(gpu);
            if !matches!(device, runtime::Device::CPU) {
                model.to_device(&device)?;
            }
            Box::new(YiRunner { model, tokenizer, config })
        }
        _ => {
            anyhow::bail!("Unsupported architecture: {}. Supported: llama, qwen, mistral, gemma, phi, deepseek, mixtral, yi", arch)
        }
    };

    let ctx = runner.context_size();
    metrics.model_load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    Ok((runner, arch.to_string(), ctx))
}

fn parse_device(gpu: &str) -> runtime::Device {
    match gpu.to_lowercase().as_str() {
        "metal" => runtime::Device::Metal(0),
        "cuda" | "cuda:0" => runtime::Device::CUDA(0),
        "auto" => {
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
    let (runner, arch, _ctx_size) = load_model_runner(&model_path, cli.ctx_size, &cli.gpu, &mut metrics)?;

    if cli.verbose {
        eprintln!(
            "Model loaded: architecture={}, context={}, {:.0}ms",
            arch, runner.context_size(), metrics.model_load_ms
        );
    }

    let loader = UnifiedWeightLoader::new();
    let weights_enc = loader.load_weights(&model_path)?;
    let tokenizer = Tokenizer::from_model_weights(&weights_enc)?;
    let prompt_tokens = tokenizer.encode_with_special_tokens(prompt, true, false);
    metrics.prompt_tokens = prompt_tokens.len();

    let gen_config = GenerationConfig {
        max_new_tokens: cli.n_predict,
        temperature: cli.temp,
        top_p: cli.top_p,
        top_k: cli.top_k,
        do_sample: cli.temp > 0.0,
        repetition_penalty: cli.repeat_penalty,
        eos_token_id: runner.eos_token_id(),
        ..Default::default()
    };

    let gen_start = Instant::now();
    let mut prefill_done = false;
    let mut prefill_time = 0.0f64;
    let mut gen_tokens = 0usize;

    runner.generate_streaming(
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

    println!();

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
    let (runner, arch, _ctx_size) = load_model_runner(&model_path, cli.ctx_size, &cli.gpu, &mut metrics)?;

    eprintln!("Model loaded in {:.0}ms", metrics.model_load_ms);
    eprintln!(
        "Architecture: {} | Context: {}",
        arch, runner.context_size()
    );
    eprintln!("Type your message and press Enter. Type 'quit' or Ctrl+C to exit.\n");

    let template = ChatTemplateFormat::detect(&arch, model_name);
    let mut history: Vec<ChatMessage> = Vec::new();

    let loader = UnifiedWeightLoader::new();
    let weights2 = loader.load_weights(&model_path)?;
    let tokenizer = Tokenizer::from_model_weights(&weights2)?;

    loop {
        eprint!("> ");
        std::io::stderr().flush().ok();

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

        history.push(ChatMessage {
            role: "user".to_string(),
            content: input.to_string(),
        });

        let prompt = template.render(&history);
        let prompt_tokens = tokenizer.encode_with_special_tokens(&prompt, true, false);

        let gen_config = GenerationConfig {
            max_new_tokens: cli.n_predict,
            temperature: cli.temp,
            top_p: cli.top_p,
            top_k: cli.top_k,
            do_sample: cli.temp > 0.0,
            repetition_penalty: cli.repeat_penalty,
            eos_token_id: runner.eos_token_id(),
            ..Default::default()
        };

        let gen_start = Instant::now();
        let mut response_text = String::new();

        let result = runner.generate_streaming(
            &prompt_tokens,
            &gen_config,
            &mut |_token_id, text| {
                response_text.push_str(text);
                print!("{}", text);
                std::io::stdout().flush().ok();
                true
            },
        );

        println!();

        match result {
            Ok(_) => {
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
    timeout_secs: u64,
    use_engine: bool,
) -> Result<()> {
    let mut metrics = GenerationMetrics::new();
    let model_path = resolve_model_path(model_name).await?;
    let (runner, arch, ctx) = load_model_runner(&model_path, ctx_size, gpu, &mut metrics)?;

    eprintln!("Model loaded in {:.0}ms", metrics.model_load_ms);
    eprintln!(
        "Architecture: {} | Context: {} | Concurrency: {} | Timeout: {}s",
        arch, ctx, concurrency, timeout_secs
    );

    let loader = UnifiedWeightLoader::new();
    let weights2 = loader.load_weights(&model_path)?;
    let tokenizer = Tokenizer::from_model_weights(&weights2)?;

    let template = ChatTemplateFormat::detect(&arch, model_name);

    let service = InferenceService::new(
        runner,
        tokenizer,
        template,
        model_name.to_string(),
        ctx,
        concurrency,
    );

    let config = ServerConfig {
        host: host.to_string(),
        port,
        max_concurrency: concurrency,
        request_timeout_secs: timeout_secs,
        api_key: None,
    };

    let cancel = tokio_util::sync::CancellationToken::new();

    let state = if use_engine {
        let arc_runner: std::sync::Arc<dyn runtime::model_core::ModelRunner> = {
            let (runner2, _arch2, _ctx2) = load_model_runner(&model_path, ctx_size, gpu, &mut GenerationMetrics::new())?;
            std::sync::Arc::from(runner2)
        };
        let engine = create_engine_from_runner(arc_runner, concurrency)?;
        eprintln!("Inference engine enabled (batching/scheduling active)");
        runtime::server::create_shared_state_with_engine(
            service,
            engine,
            model_name.to_string(),
            arch,
            config,
            cancel,
        )
    } else {
        create_shared_state_with_config(
            service,
            model_name.to_string(),
            arch,
            config,
            cancel,
        )
    };

    runtime::server::start_server(state, host, port).await
}

// ============================================================================
// Engine Integration
// ============================================================================

struct EngineProcessorAdapter {
    engine: UniLLMInferenceEngine,
}

#[async_trait::async_trait]
impl runtime::server::EngineProcessor for EngineProcessorAdapter {
    async fn process(
        &self,
        prompt: String,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        repetition_penalty: f32,
    ) -> Result<runtime::server::EngineOutput, String> {
        let sampling_params = SamplingParams {
            temperature,
            top_p,
            top_k: None,
            max_tokens: Some(max_tokens),
            stop_sequences: vec![],
            seed: None,
            repetition_penalty,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        };

        let request = RequestBuilder::new(prompt)
            .with_sampling_params(sampling_params)
            .build();

        if !self.engine.is_running().await {
            self.engine.start().await.map_err(|e| format!("Engine start failed: {}", e))?;
        }

        let response = self.engine
            .process_request(request)
            .await
            .map_err(|e| format!("Engine error: {}", e))?;

        Ok(runtime::server::EngineOutput {
            text: response.text,
            finish_reason: format!("{:?}", response.finish_reason),
            prompt_tokens: response.stats.prompt_tokens,
            completion_tokens: response.stats.completion_tokens,
            total_tokens: response.stats.total_tokens,
        })
    }
}

fn create_engine_from_runner(
    model: std::sync::Arc<dyn runtime::model_core::ModelRunner>,
    concurrency: usize,
) -> Result<std::sync::Arc<dyn runtime::server::EngineProcessor>> {
    let rt = tokio::runtime::Handle::current();
    let tokenizer = runtime::tokenizer::Tokenizer::new();

    let engine = rt.block_on(UniLLMInferenceEngine::new_cpu(
        model,
        tokenizer,
        concurrency,
    )).map_err(|e| anyhow::anyhow!("Engine creation failed: {}", e))?;

    rt.block_on(engine.start()).map_err(|e| anyhow::anyhow!("Engine start failed: {}", e))?;

    Ok(std::sync::Arc::new(EngineProcessorAdapter { engine }))
}