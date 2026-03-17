//! UniLLM Benchmark Suite
//!
//! Comprehensive benchmarking tool for UniLLM performance testing.

use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::sleep;

// Import UniLLM components for internal benchmarking
use inference::UniLLMInferenceEngine;
use kv::HybridKVCache;
use scheduler::IntelligentScheduler;
use kernels::KernelFramework;

#[derive(Parser)]
#[command(name = "unillm-benchmark")]
#[command(about = "UniLLM Performance Benchmark Suite")]
struct Args {
    /// Benchmark type
    #[arg(value_enum, default_value = "throughput")]
    benchmark_type: BenchmarkType,

    /// Number of requests to run
    #[arg(long, default_value = "100")]
    requests: usize,

    /// Number of concurrent clients
    #[arg(long, default_value = "1")]
    concurrency: usize,

    /// Input sequence length
    #[arg(long, default_value = "512")]
    input_length: usize,

    /// Output sequence length
    #[arg(long, default_value = "128")]
    output_length: usize,

    /// Batch size for batched inference
    #[arg(long, default_value = "32")]
    batch_size: usize,

    /// Warmup requests
    #[arg(long, default_value = "10")]
    warmup: usize,

    /// Results output file (JSON format)
    #[arg(long)]
    output: Option<String>,

    /// Compare with baseline results
    #[arg(long)]
    baseline: Option<String>,

    /// GPU target for testing
    #[arg(long, default_value = "cuda")]
    gpu_target: String,
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum BenchmarkType {
    /// Throughput benchmark (requests/second)
    Throughput,
    /// Latency benchmark (response time distribution)
    Latency,
    /// Memory efficiency benchmark
    Memory,
    /// Cache performance benchmark
    Cache,
    /// GPU utilization benchmark
    Gpu,
    /// Comprehensive benchmark (all above)
    Comprehensive,
}

#[derive(Serialize, Deserialize)]
struct BenchmarkResults {
    benchmark_type: String,
    timestamp: String,
    system_info: SystemInfo,
    parameters: BenchmarkParameters,
    results: BenchmarkMetrics,
}

#[derive(Serialize, Deserialize)]
struct SystemInfo {
    gpu_target: String,
    total_memory_gb: f64,
    gpu_memory_gb: f64,
    cpu_cores: usize,
    rust_version: String,
    unillm_version: String,
}

#[derive(Serialize, Deserialize)]
struct BenchmarkParameters {
    requests: usize,
    concurrency: usize,
    input_length: usize,
    output_length: usize,
    batch_size: usize,
    warmup: usize,
}

#[derive(Serialize, Deserialize)]
struct BenchmarkMetrics {
    throughput_rps: f64,
    latency_stats: LatencyStats,
    memory_stats: MemoryBenchmarkStats,
    cache_stats: CacheStats,
    gpu_stats: GpuStats,
    error_rate: f64,
}

#[derive(Serialize, Deserialize)]
struct LatencyStats {
    mean_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

#[derive(Serialize, Deserialize)]
struct MemoryBenchmarkStats {
    peak_memory_mb: f64,
    average_memory_mb: f64,
    memory_efficiency: f64, // MB per request
}

#[derive(Serialize, Deserialize)]
struct CacheStats {
    hit_rate: f64,
    miss_rate: f64,
    eviction_rate: f64,
    memory_usage_mb: f64,
}

#[derive(Serialize, Deserialize)]
struct GpuStats {
    peak_utilization: f64,
    average_utilization: f64,
    memory_usage_mb: f64,
    kernel_launch_rate: f64,
}

struct BenchmarkRunner {
    inference_engine: UniLLMInferenceEngine,
    args: Args,
}

impl BenchmarkRunner {
    async fn new(args: Args) -> Result<Self, Box<dyn std::error::Error>> {
        println!("🔧 Initializing UniLLM for benchmarking...");

        // Create kernel framework
        let kernel_framework = std::sync::Arc::new(
            KernelFramework::new()
                .map_err(|e| format!("Failed to initialize kernel framework: {}", e))?
        );

        // Create hybrid KV cache with larger size for benchmarking
        let kv_cache = std::sync::Arc::new(
            HybridKVCache::new(4 * 1024 * 1024 * 1024) // 4GB cache
                .map_err(|e| format!("Failed to initialize KV cache: {}", e))?
        );

        // Create intelligent scheduler
        let scheduler = std::sync::Arc::new(
            IntelligentScheduler::new(kv_cache.clone(), kernel_framework.clone())
        );

        // Create inference engine
        let inference_engine = UniLLMInferenceEngine::new(
            kv_cache,
            scheduler,
            kernel_framework,
        ).await
            .map_err(|e| format!("Failed to initialize inference engine: {}", e))?;

        Ok(Self {
            inference_engine,
            args,
        })
    }

    async fn run(&self) -> Result<BenchmarkResults, Box<dyn std::error::Error>> {
        println!("🏁 Starting benchmark: {:?}", self.args.benchmark_type);

        // Warmup
        if self.args.warmup > 0 {
            println!("🔥 Running {} warmup requests...", self.args.warmup);
            self.run_warmup().await?;
        }

        // Run benchmark based on type
        let results = match self.args.benchmark_type {
            BenchmarkType::Throughput => self.run_throughput_benchmark().await?,
            BenchmarkType::Latency => self.run_latency_benchmark().await?,
            BenchmarkType::Memory => self.run_memory_benchmark().await?,
            BenchmarkType::Cache => self.run_cache_benchmark().await?,
            BenchmarkType::Gpu => self.run_gpu_benchmark().await?,
            BenchmarkType::Comprehensive => self.run_comprehensive_benchmark().await?,
        };

        Ok(results)
    }

    async fn run_warmup(&self) -> Result<(), Box<dyn std::error::Error>> {
        for _ in 0..self.args.warmup {
            let request = self.create_test_request();
            let _ = self.inference_engine.process_request(request).await;
        }
        Ok(())
    }

    async fn run_throughput_benchmark(&self) -> Result<BenchmarkResults, Box<dyn std::error::Error>> {
        println!("📈 Running throughput benchmark...");

        let start_time = Instant::now();
        let mut successful_requests = 0;
        let mut latencies = Vec::new();

        // Run requests
        for i in 0..self.args.requests {
            let request_start = Instant::now();
            let request = self.create_test_request();

            match self.inference_engine.process_request(request).await {
                Ok(_) => {
                    successful_requests += 1;
                    latencies.push(request_start.elapsed().as_millis() as f64);
                }
                Err(e) => {
                    eprintln!("Request {} failed: {}", i, e);
                }
            }

            // Progress indicator
            if (i + 1) % 10 == 0 {
                print!(".");
                if (i + 1) % 100 == 0 {
                    println!(" {}/{}", i + 1, self.args.requests);
                }
            }
        }

        let total_time = start_time.elapsed();
        let throughput = successful_requests as f64 / total_time.as_secs_f64();

        println!("\n✅ Throughput benchmark complete");
        println!("  Throughput: {:.2} requests/second", throughput);

        Ok(BenchmarkResults {
            benchmark_type: "throughput".to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            system_info: self.get_system_info(),
            parameters: self.get_benchmark_parameters(),
            results: BenchmarkMetrics {
                throughput_rps: throughput,
                latency_stats: self.calculate_latency_stats(&latencies),
                memory_stats: self.get_memory_stats().await,
                cache_stats: self.get_cache_stats().await,
                gpu_stats: self.get_gpu_stats().await,
                error_rate: 1.0 - (successful_requests as f64 / self.args.requests as f64),
            },
        })
    }

    async fn run_latency_benchmark(&self) -> Result<BenchmarkResults, Box<dyn std::error::Error>> {
        println!("⚡ Running latency benchmark...");

        let mut latencies = Vec::new();
        let mut successful_requests = 0;

        for i in 0..self.args.requests {
            let start = Instant::now();
            let request = self.create_test_request();

            match self.inference_engine.process_request(request).await {
                Ok(_) => {
                    successful_requests += 1;
                    latencies.push(start.elapsed().as_millis() as f64);
                }
                Err(e) => {
                    eprintln!("Request {} failed: {}", i, e);
                }
            }

            // Small delay between requests for latency testing
            sleep(Duration::from_millis(10)).await;
        }

        println!("✅ Latency benchmark complete");

        Ok(BenchmarkResults {
            benchmark_type: "latency".to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            system_info: self.get_system_info(),
            parameters: self.get_benchmark_parameters(),
            results: BenchmarkMetrics {
                throughput_rps: 0.0, // Not applicable for latency benchmark
                latency_stats: self.calculate_latency_stats(&latencies),
                memory_stats: self.get_memory_stats().await,
                cache_stats: self.get_cache_stats().await,
                gpu_stats: self.get_gpu_stats().await,
                error_rate: 1.0 - (successful_requests as f64 / self.args.requests as f64),
            },
        })
    }

    async fn run_memory_benchmark(&self) -> Result<BenchmarkResults, Box<dyn std::error::Error>> {
        println!("💾 Running memory benchmark...");
        // Implementation for memory-focused benchmarking
        self.run_throughput_benchmark().await // Placeholder
    }

    async fn run_cache_benchmark(&self) -> Result<BenchmarkResults, Box<dyn std::error::Error>> {
        println!("🎯 Running cache benchmark...");
        // Implementation for cache-focused benchmarking
        self.run_throughput_benchmark().await // Placeholder
    }

    async fn run_gpu_benchmark(&self) -> Result<BenchmarkResults, Box<dyn std::error::Error>> {
        println!("🎮 Running GPU benchmark...");
        // Implementation for GPU-focused benchmarking
        self.run_throughput_benchmark().await // Placeholder
    }

    async fn run_comprehensive_benchmark(&self) -> Result<BenchmarkResults, Box<dyn std::error::Error>> {
        println!("🔍 Running comprehensive benchmark...");
        // Implementation for comprehensive benchmarking
        self.run_throughput_benchmark().await // Placeholder
    }

    fn create_test_request(&self) -> inference::InferenceRequest {
        inference::InferenceRequest {
            prompt_tokens: self.args.input_length,
            max_output_length: self.args.output_length,
            temperature: 1.0,
            top_p: 0.9,
            stop_sequences: vec![],
            stream: false,
        }
    }

    fn calculate_latency_stats(&self, latencies: &[f64]) -> LatencyStats {
        if latencies.is_empty() {
            return LatencyStats {
                mean_ms: 0.0,
                p50_ms: 0.0,
                p95_ms: 0.0,
                p99_ms: 0.0,
                min_ms: 0.0,
                max_ms: 0.0,
            };
        }

        let mut sorted_latencies = latencies.to_vec();
        sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let p50 = sorted_latencies[sorted_latencies.len() / 2];
        let p95 = sorted_latencies[(sorted_latencies.len() as f64 * 0.95) as usize];
        let p99 = sorted_latencies[(sorted_latencies.len() as f64 * 0.99) as usize];
        let min = sorted_latencies[0];
        let max = sorted_latencies[sorted_latencies.len() - 1];

        LatencyStats {
            mean_ms: mean,
            p50_ms: p50,
            p95_ms: p95,
            p99_ms: p99,
            min_ms: min,
            max_ms: max,
        }
    }

    fn get_system_info(&self) -> SystemInfo {
        SystemInfo {
            gpu_target: self.args.gpu_target.clone(),
            total_memory_gb: 16.0, // Placeholder
            gpu_memory_gb: 24.0,   // Placeholder
            cpu_cores: num_cpus::get(),
            rust_version: env!("CARGO_PKG_VERSION").to_string(),
            unillm_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    fn get_benchmark_parameters(&self) -> BenchmarkParameters {
        BenchmarkParameters {
            requests: self.args.requests,
            concurrency: self.args.concurrency,
            input_length: self.args.input_length,
            output_length: self.args.output_length,
            batch_size: self.args.batch_size,
            warmup: self.args.warmup,
        }
    }

    async fn get_memory_stats(&self) -> MemoryBenchmarkStats {
        MemoryBenchmarkStats {
            peak_memory_mb: 2048.0,    // Placeholder
            average_memory_mb: 1536.0, // Placeholder
            memory_efficiency: 15.36,  // MB per request
        }
    }

    async fn get_cache_stats(&self) -> CacheStats {
        let engine_stats = self.inference_engine.get_statistics().await;
        CacheStats {
            hit_rate: engine_stats.cache_hit_rate,
            miss_rate: 1.0 - engine_stats.cache_hit_rate,
            eviction_rate: 0.05,  // Placeholder
            memory_usage_mb: engine_stats.cache_memory_mb,
        }
    }

    async fn get_gpu_stats(&self) -> GpuStats {
        let engine_stats = self.inference_engine.get_statistics().await;
        GpuStats {
            peak_utilization: engine_stats.gpu_utilization,
            average_utilization: engine_stats.gpu_utilization * 0.8, // Placeholder
            memory_usage_mb: 8192.0, // Placeholder
            kernel_launch_rate: 1000.0, // kernels/second, placeholder
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("🚀 UniLLM Benchmark Suite");
    println!("========================");

    // Initialize benchmark runner
    let runner = BenchmarkRunner::new(args).await?;

    // Run benchmark
    let results = runner.run().await?;

    // Print results
    println!("\n📊 Benchmark Results:");
    println!("====================");
    println!("Benchmark Type: {}", results.benchmark_type);
    println!("Throughput: {:.2} requests/second", results.results.throughput_rps);
    println!("Latency (P50): {:.1}ms", results.results.latency_stats.p50_ms);
    println!("Latency (P95): {:.1}ms", results.results.latency_stats.p95_ms);
    println!("Cache Hit Rate: {:.1}%", results.results.cache_stats.hit_rate * 100.0);
    println!("GPU Utilization: {:.1}%", results.results.gpu_stats.average_utilization * 100.0);
    println!("Error Rate: {:.2}%", results.results.error_rate * 100.0);

    // Save results to file if specified
    if let Some(output_file) = &runner.args.output {
        let json_results = serde_json::to_string_pretty(&results)?;
        std::fs::write(output_file, json_results)?;
        println!("\n💾 Results saved to: {}", output_file);
    }

    // Compare with baseline if specified
    if let Some(baseline_file) = &runner.args.baseline {
        println!("\n📈 Comparing with baseline...");
        compare_with_baseline(&results, baseline_file)?;
    }

    Ok(())
}

fn compare_with_baseline(
    current: &BenchmarkResults,
    baseline_file: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let baseline_content = std::fs::read_to_string(baseline_file)?;
    let baseline: BenchmarkResults = serde_json::from_str(&baseline_content)?;

    println!("Comparison with baseline:");

    let throughput_change = (current.results.throughput_rps / baseline.results.throughput_rps - 1.0) * 100.0;
    let latency_change = (current.results.latency_stats.p50_ms / baseline.results.latency_stats.p50_ms - 1.0) * 100.0;

    println!("  Throughput: {:.1}% change", throughput_change);
    println!("  Latency (P50): {:.1}% change", latency_change);

    if throughput_change > 5.0 {
        println!("  🎉 Significant throughput improvement!");
    } else if throughput_change < -5.0 {
        println!("  ⚠️  Throughput regression detected");
    }

    Ok(())
}