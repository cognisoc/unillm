//! Benchmark runner implementation

use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use prettytable::{row, Table};
use sysinfo::{Pid, System};

use super::{BenchmarkComparison, BenchmarkConfig, BenchmarkResults, GenerationResult, InferenceMetrics};

/// Trait for inference backends that can be benchmarked
pub trait InferenceBackend: Send + Sync {
    /// Backend identifier name
    fn name(&self) -> &str;

    /// Load model from path, return load time in milliseconds
    fn load_model(&mut self, path: &Path) -> Result<f64>;

    /// Generate text from prompt, return generation result
    fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<GenerationResult>;

    /// Get current memory usage in bytes
    fn memory_usage(&self) -> u64;

    /// Unload model and free resources
    fn unload(&mut self);
}

/// Get current process memory usage in bytes
pub fn get_process_memory() -> u64 {
    let mut sys = System::new_all();
    let pid = Pid::from_u32(std::process::id());
    sys.refresh_all();

    sys.process(pid).map(|p| p.memory()).unwrap_or(0)
}

/// Benchmark runner that executes benchmarks on backends
pub struct BenchmarkRunner {
    config: BenchmarkConfig,
}

impl BenchmarkRunner {
    /// Create new benchmark runner with configuration
    pub fn new(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Run benchmark on a backend
    pub fn run<B: InferenceBackend>(&self, backend: &mut B) -> Result<BenchmarkResults> {
        let mut runs = Vec::new();

        // Memory before loading
        let memory_before = get_process_memory();

        // Load model and measure time
        println!("  Loading model...");
        let load_time = backend.load_model(&self.config.model_path)?;
        println!("  Model loaded in {:.2}ms", load_time);

        // Memory after loading
        let memory_after = get_process_memory();
        let model_memory = memory_after.saturating_sub(memory_before);
        println!(
            "  Model memory: {:.2} MB",
            model_memory as f64 / (1024.0 * 1024.0)
        );

        // Warmup phase
        println!(
            "  Running {} warmup iterations...",
            self.config.warmup_iterations
        );
        for i in 0..self.config.warmup_iterations {
            for prompt in &self.config.prompts {
                let _ = backend.generate(prompt, self.config.max_new_tokens)?;
            }
            print!("    Warmup {}/{}\r", i + 1, self.config.warmup_iterations);
        }
        println!();

        // Measurement phase
        println!(
            "  Running {} measurement iterations...",
            self.config.measurement_iterations
        );
        let mut peak_memory = memory_after;

        for i in 0..self.config.measurement_iterations {
            for prompt in &self.config.prompts {
                let result = backend.generate(prompt, self.config.max_new_tokens)?;

                // Track peak memory
                let current_memory = get_process_memory();
                peak_memory = peak_memory.max(current_memory);

                let tokens_per_sec = if result.total_time_ms > 0.0 {
                    result.tokens_generated as f64 / (result.total_time_ms / 1000.0)
                } else {
                    0.0
                };

                runs.push(InferenceMetrics {
                    load_time_ms: load_time,
                    time_to_first_token_ms: result.time_to_first_token_ms,
                    total_inference_time_ms: result.total_time_ms,
                    tokens_generated: result.tokens_generated,
                    tokens_per_second: tokens_per_sec,
                    prompt_tokens: result.prompt_tokens,
                    memory_before_load_bytes: memory_before,
                    memory_after_load_bytes: memory_after,
                    peak_memory_bytes: peak_memory,
                });
            }
            print!(
                "    Iteration {}/{}\r",
                i + 1,
                self.config.measurement_iterations
            );
        }
        println!();

        Ok(BenchmarkResults::aggregate(
            backend.name(),
            runs,
            load_time,
            peak_memory,
        ))
    }
}

/// Print comparison results as a formatted table
pub fn print_comparison_table(comparison: &BenchmarkComparison) {
    let mut table = Table::new();

    // Header
    table.add_row(row![
        "Metric",
        "llama.cpp",
        "UniLLM",
        "Diff"
    ]);

    // Load time
    table.add_row(row![
        "Load Time (ms)",
        format!("{:.1}", comparison.llama_cpp.load_time_ms),
        format!("{:.1}", comparison.unillm.load_time_ms),
        BenchmarkComparison::format_diff(
            comparison.llama_cpp.load_time_ms,
            comparison.unillm.load_time_ms
        )
    ]);

    // TTFT
    table.add_row(row![
        "TTFT (ms)",
        format!("{:.1}", comparison.llama_cpp.avg_ttft_ms),
        format!("{:.1}", comparison.unillm.avg_ttft_ms),
        BenchmarkComparison::format_diff(
            comparison.llama_cpp.avg_ttft_ms,
            comparison.unillm.avg_ttft_ms
        )
    ]);

    // Tokens/sec (avg)
    table.add_row(row![
        "Tokens/sec (avg)",
        format!("{:.1}", comparison.llama_cpp.avg_tokens_per_sec),
        format!("{:.1}", comparison.unillm.avg_tokens_per_sec),
        BenchmarkComparison::format_diff(
            comparison.llama_cpp.avg_tokens_per_sec,
            comparison.unillm.avg_tokens_per_sec
        )
    ]);

    // Tokens/sec (p50)
    table.add_row(row![
        "Tokens/sec (p50)",
        format!("{:.1}", comparison.llama_cpp.p50_tokens_per_sec),
        format!("{:.1}", comparison.unillm.p50_tokens_per_sec),
        BenchmarkComparison::format_diff(
            comparison.llama_cpp.p50_tokens_per_sec,
            comparison.unillm.p50_tokens_per_sec
        )
    ]);

    // Memory (avg)
    table.add_row(row![
        "Memory (MB)",
        format!("{:.1}", comparison.llama_cpp.avg_memory_mb),
        format!("{:.1}", comparison.unillm.avg_memory_mb),
        BenchmarkComparison::format_diff(
            comparison.llama_cpp.avg_memory_mb,
            comparison.unillm.avg_memory_mb
        )
    ]);

    // Peak memory
    table.add_row(row![
        "Peak Memory (MB)",
        format!("{:.1}", comparison.llama_cpp.peak_memory_mb),
        format!("{:.1}", comparison.unillm.peak_memory_mb),
        BenchmarkComparison::format_diff(
            comparison.llama_cpp.peak_memory_mb,
            comparison.unillm.peak_memory_mb
        )
    ]);

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║         UniLLM vs llama.cpp Benchmark Results                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    table.printstd();

    // Notes
    println!("\n--- Notes ---");
    println!("* UniLLM uses KV caching for efficient autoregressive generation");
    println!("* UniLLM keeps weights in quantized format (Q4/Q8) for 4-5x memory savings");
    println!("* UniLLM runs on CPU only (no GPU acceleration yet)");
    println!("* llama.cpp uses quantized inference with optimized SIMD kernels");
}

/// Print results for a single backend
pub fn print_single_results(results: &BenchmarkResults) {
    println!("\n=== {} Results ===", results.backend_name);
    println!("Load Time:      {:.1} ms", results.load_time_ms);
    println!("Avg TTFT:       {:.1} ms", results.avg_ttft_ms);
    println!("Avg Tokens/sec: {:.1}", results.avg_tokens_per_sec);
    println!("P50 Tokens/sec: {:.1}", results.p50_tokens_per_sec);
    println!("P99 Tokens/sec: {:.1}", results.p99_tokens_per_sec);
    println!("Avg Memory:     {:.1} MB", results.avg_memory_mb);
    println!("Peak Memory:    {:.1} MB", results.peak_memory_mb);
}
