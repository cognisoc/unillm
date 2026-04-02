//! Benchmark comparison: UniLLM vs llama.cpp
//!
//! Compares inference performance between UniLLM and llama.cpp using the same
//! GGUF model file downloaded from Ollama registry.
//!
//! Usage:
//!   # UniLLM only (no llama.cpp dependency)
//!   cargo run --release --bin benchmark_comparison -p runtime
//!
//!   # With llama.cpp comparison (requires benchmark feature and system deps)
//!   cargo run --release --bin benchmark_comparison -p runtime --features benchmark
//!
//!   # Custom options
//!   cargo run --release --bin benchmark_comparison -p runtime -- \
//!       --model tinyllama:latest \
//!       --warmup 5 \
//!       --iterations 20 \
//!       --max-tokens 100

use anyhow::Result;
use clap::Parser;
use runtime::benchmark::{
    print_comparison_table, print_single_results, BenchmarkComparison, BenchmarkConfig,
    BenchmarkRunner, InferenceBackend, UniLLMBackend,
};
use runtime::ollama::OllamaRegistry;

#[cfg(feature = "benchmark")]
use runtime::benchmark::LlamaCppBackend;

/// Benchmark comparison between UniLLM and llama.cpp
#[derive(Parser, Debug)]
#[command(name = "benchmark_comparison")]
#[command(about = "Compare UniLLM and llama.cpp inference performance")]
struct Args {
    /// Model name (Ollama format, e.g., tinyllama:latest)
    #[arg(long, default_value = "tinyllama:latest")]
    model: String,

    /// Number of warmup iterations
    #[arg(long, default_value = "3")]
    warmup: usize,

    /// Number of measurement iterations
    #[arg(long, default_value = "10")]
    iterations: usize,

    /// Maximum tokens to generate per prompt
    #[arg(long, default_value = "50")]
    max_tokens: usize,

    /// Only run UniLLM benchmark (skip llama.cpp)
    #[arg(long)]
    unillm_only: bool,

    /// Only run llama.cpp benchmark (skip UniLLM)
    #[arg(long)]
    #[cfg(feature = "benchmark")]
    llama_cpp_only: bool,
}

const BENCHMARK_PROMPTS: &[&str] = &[
    "The capital of France is",
    "In machine learning, a neural network is",
    "The quick brown fox",
];

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           UniLLM vs llama.cpp Benchmark                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Step 1: Download model via Ollama registry
    println!("[1/4] Downloading model: {}", args.model);
    let registry = OllamaRegistry::new()?;
    let model_path = registry.pull(&args.model).await?;
    println!("      Model path: {}", model_path.display());

    // Get file size
    let file_size = std::fs::metadata(&model_path)?.len();
    println!("      Model size: {:.1} MB\n", file_size as f64 / (1024.0 * 1024.0));

    // Create benchmark config
    let config = BenchmarkConfig {
        model_path: model_path.clone(),
        prompts: BENCHMARK_PROMPTS.iter().map(|s| s.to_string()).collect(),
        max_new_tokens: args.max_tokens,
        warmup_iterations: args.warmup,
        measurement_iterations: args.iterations,
        seed: 42,
    };

    let runner = BenchmarkRunner::new(config.clone());

    // Check what to run
    #[cfg(feature = "benchmark")]
    let run_llama_cpp = !args.unillm_only;
    #[cfg(not(feature = "benchmark"))]
    let run_llama_cpp = false;

    let run_unillm = {
        #[cfg(feature = "benchmark")]
        {
            !args.llama_cpp_only
        }
        #[cfg(not(feature = "benchmark"))]
        {
            true
        }
    };

    // Run llama.cpp benchmark if enabled
    #[cfg(feature = "benchmark")]
    let llama_results = if run_llama_cpp {
        println!("[2/4] Running llama.cpp benchmark...");
        let mut llama_cpp = LlamaCppBackend::new();
        let results = runner.run(&mut llama_cpp)?;
        llama_cpp.unload();

        // Give time for memory cleanup
        std::thread::sleep(std::time::Duration::from_millis(500));

        print_single_results(&results);
        Some(results)
    } else {
        println!("[2/4] Skipping llama.cpp benchmark (--unillm-only)");
        None
    };

    #[cfg(not(feature = "benchmark"))]
    let llama_results: Option<runtime::benchmark::BenchmarkResults> = {
        println!("[2/4] Skipping llama.cpp benchmark (benchmark feature not enabled)");
        println!("      To enable: cargo run --release --bin benchmark_comparison -p runtime --features benchmark");
        None
    };

    // Run UniLLM benchmark
    let unillm_results = if run_unillm {
        println!("\n[3/4] Running UniLLM benchmark...");
        let mut unillm = UniLLMBackend::new();
        let results = runner.run(&mut unillm)?;
        unillm.unload();

        print_single_results(&results);
        Some(results)
    } else {
        println!("\n[3/4] Skipping UniLLM benchmark (--llama-cpp-only)");
        None
    };

    // Print comparison if both were run
    println!("\n[4/4] Generating comparison report...");

    match (llama_results, unillm_results) {
        (Some(llama), Some(unillm)) => {
            let comparison = BenchmarkComparison {
                llama_cpp: llama,
                unillm,
            };
            print_comparison_table(&comparison);
        }
        (None, Some(unillm)) => {
            println!("\n=== UniLLM Results Only ===");
            println!("(llama.cpp comparison not available)\n");
            print_single_results(&unillm);
            print_fairness_notes();
        }
        (Some(llama), None) => {
            println!("\n=== llama.cpp Results Only ===");
            println!("(UniLLM comparison not available)\n");
            print_single_results(&llama);
        }
        (None, None) => {
            println!("\nNo benchmarks were run.");
        }
    }

    println!("\n=== Benchmark Complete ===");
    Ok(())
}

fn print_fairness_notes() {
    println!("\n--- Notes ---");
    println!("* UniLLM now uses KV caching for efficient autoregressive generation");
    println!("* UniLLM runs on CPU only (no GPU acceleration yet)");
    println!("* UniLLM dequantizes GGUF weights to F32 (higher memory, simpler compute)");
    println!("* llama.cpp uses quantized inference with optimized SIMD kernels");
}
