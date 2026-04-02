//! Benchmark metrics and results structures

use std::path::PathBuf;

/// Configuration for benchmark runs
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Path to the model file (GGUF)
    pub model_path: PathBuf,
    /// Prompts to use for benchmarking
    pub prompts: Vec<String>,
    /// Maximum new tokens to generate per prompt
    pub max_new_tokens: usize,
    /// Number of warmup iterations before measurement
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    pub measurement_iterations: usize,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            prompts: vec![
                "The capital of France is".to_string(),
                "In machine learning, a neural network is".to_string(),
                "The quick brown fox".to_string(),
            ],
            max_new_tokens: 50,
            warmup_iterations: 3,
            measurement_iterations: 10,
            seed: 42,
        }
    }
}

/// Metrics collected per inference run
#[derive(Debug, Clone)]
pub struct InferenceMetrics {
    /// Model load time in milliseconds
    pub load_time_ms: f64,
    /// Time to first token in milliseconds
    pub time_to_first_token_ms: f64,
    /// Total inference time in milliseconds
    pub total_inference_time_ms: f64,
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Tokens per second
    pub tokens_per_second: f64,
    /// Number of prompt tokens
    pub prompt_tokens: usize,
    /// Memory before model load (bytes)
    pub memory_before_load_bytes: u64,
    /// Memory after model load (bytes)
    pub memory_after_load_bytes: u64,
    /// Peak memory during inference (bytes)
    pub peak_memory_bytes: u64,
}

/// Result of a single generation
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// Generated text
    pub output_text: String,
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Number of prompt tokens
    pub prompt_tokens: usize,
    /// Time to first token in milliseconds
    pub time_to_first_token_ms: f64,
    /// Total time in milliseconds
    pub total_time_ms: f64,
}

/// Aggregated benchmark results for a backend
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Backend name
    pub backend_name: String,
    /// All individual run metrics
    pub runs: Vec<InferenceMetrics>,
    /// Model load time in milliseconds
    pub load_time_ms: f64,
    /// Average time to first token in milliseconds
    pub avg_ttft_ms: f64,
    /// Average tokens per second
    pub avg_tokens_per_sec: f64,
    /// Median (p50) tokens per second
    pub p50_tokens_per_sec: f64,
    /// 99th percentile tokens per second
    pub p99_tokens_per_sec: f64,
    /// Average memory usage in MB
    pub avg_memory_mb: f64,
    /// Peak memory usage in MB
    pub peak_memory_mb: f64,
}

impl BenchmarkResults {
    /// Aggregate metrics from individual runs
    pub fn aggregate(
        backend_name: &str,
        runs: Vec<InferenceMetrics>,
        load_time_ms: f64,
        peak_memory_bytes: u64,
    ) -> Self {
        let n = runs.len() as f64;

        // Calculate averages
        let avg_ttft_ms = runs.iter().map(|r| r.time_to_first_token_ms).sum::<f64>() / n;
        let avg_tokens_per_sec = runs.iter().map(|r| r.tokens_per_second).sum::<f64>() / n;
        let avg_memory_bytes =
            runs.iter().map(|r| r.memory_after_load_bytes).sum::<u64>() as f64 / n;

        // Calculate percentiles for tokens/sec
        let mut tps_values: Vec<f64> = runs.iter().map(|r| r.tokens_per_second).collect();
        tps_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p50_idx = (tps_values.len() as f64 * 0.5) as usize;
        let p99_idx = (tps_values.len() as f64 * 0.99) as usize;

        let p50_tokens_per_sec = tps_values.get(p50_idx).copied().unwrap_or(0.0);
        let p99_tokens_per_sec = tps_values
            .get(p99_idx.min(tps_values.len() - 1))
            .copied()
            .unwrap_or(0.0);

        Self {
            backend_name: backend_name.to_string(),
            runs,
            load_time_ms,
            avg_ttft_ms,
            avg_tokens_per_sec,
            p50_tokens_per_sec,
            p99_tokens_per_sec,
            avg_memory_mb: avg_memory_bytes / (1024.0 * 1024.0),
            peak_memory_mb: peak_memory_bytes as f64 / (1024.0 * 1024.0),
        }
    }
}

/// Comparison between two backends
#[derive(Debug)]
pub struct BenchmarkComparison {
    pub llama_cpp: BenchmarkResults,
    pub unillm: BenchmarkResults,
}

impl BenchmarkComparison {
    /// Calculate percentage difference (positive means UniLLM is slower/uses more)
    pub fn diff_percent(baseline: f64, comparison: f64) -> f64 {
        if baseline == 0.0 {
            return 0.0;
        }
        ((comparison - baseline) / baseline) * 100.0
    }

    /// Format difference as string with sign
    pub fn format_diff(baseline: f64, comparison: f64) -> String {
        let diff = Self::diff_percent(baseline, comparison);
        if diff >= 0.0 {
            format!("+{:.0}%", diff)
        } else {
            format!("{:.0}%", diff)
        }
    }
}
