//! UniLLM Client
//!
//! Command-line client for interacting with UniLLM inference server.

use clap::Parser;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::io::{self, Write};
use tokio::time::{Duration, Instant};
use inference::request::{RequestBuilder, ChatMessage};
use inference::types::{SamplingParams};

#[derive(Parser)]
#[command(name = "unillm-client")]
#[command(about = "UniLLM Inference Client")]
struct Args {
    /// Server URL
    #[arg(long, default_value = "http://localhost:8080")]
    server: String,

    /// Interactive mode
    #[arg(long, short)]
    interactive: bool,

    /// Prompt text (for non-interactive mode)
    #[arg(long, short)]
    prompt: Option<String>,

    /// Maximum tokens to generate
    #[arg(long, default_value = "100")]
    max_tokens: usize,

    /// Temperature for sampling
    #[arg(long, default_value = "1.0")]
    temperature: f32,

    /// Top-p for nucleus sampling
    #[arg(long, default_value = "0.9")]
    top_p: f32,

    /// Show server statistics
    #[arg(long)]
    stats: bool,

    /// Health check
    #[arg(long)]
    health: bool,

    /// Benchmark mode (send multiple requests)
    #[arg(long)]
    benchmark: Option<usize>,
}

// Remove this struct since we'll use the one from the inference crate

#[derive(Deserialize)]
struct InferenceResponse {
    generated_text: String,
    tokens_generated: usize,
    inference_time_ms: u64,
    cache_hits: usize,
    gpu_utilization: f64,
}

#[derive(Deserialize)]
struct HealthResponse {
    status: String,
    version: String,
    gpu_target: String,
    runtime_mode: String,
    memory_usage_mb: f64,
    gpu_memory_usage_mb: f64,
}

#[derive(Deserialize)]
struct StatsResponse {
    total_requests: u64,
    total_tokens_generated: u64,
    average_latency_ms: f64,
    cache_hit_rate: f64,
    gpu_utilization: f64,
    memory_stats: MemoryStats,
}

#[derive(Deserialize)]
struct MemoryStats {
    total_memory_mb: f64,
    used_memory_mb: f64,
    cache_memory_mb: f64,
    gpu_memory_mb: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let client = Client::new();

    println!("🚀 UniLLM Client");
    println!("Server: {}", args.server);

    // Health check
    if args.health {
        health_check(&client, &args.server).await?;
        return Ok(());
    }

    // Statistics
    if args.stats {
        show_stats(&client, &args.server).await?;
        return Ok(());
    }

    // Benchmark mode
    if let Some(num_requests) = args.benchmark {
        run_benchmark(&client, &args.server, num_requests, &args).await?;
        return Ok(());
    }

    // Interactive mode
    if args.interactive {
        run_interactive_mode(&client, &args.server, &args).await?;
    } else if let Some(ref prompt) = args.prompt {
        // Single inference
        let start = Instant::now();
        let response = send_inference_request(&client, &args.server, &prompt, &args).await?;
        let total_time = start.elapsed();

        println!("\n📝 Generated Text:");
        println!("{}", response.generated_text);
        println!("\n📊 Performance:");
        println!("  Tokens generated: {}", response.tokens_generated);
        println!("  Inference time: {}ms", response.inference_time_ms);
        println!("  Total time: {}ms", total_time.as_millis());
        println!("  Cache hits: {}", response.cache_hits);
        println!("  GPU utilization: {:.1}%", response.gpu_utilization * 100.0);
    } else {
        eprintln!("❌ Please provide a prompt with --prompt or use --interactive mode");
        std::process::exit(1);
    }

    Ok(())
}

async fn health_check(client: &Client, server: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Checking server health...");

    let response = client
        .get(&format!("{}/health", server))
        .send()
        .await?;

    if response.status().is_success() {
        let health: HealthResponse = response.json().await?;
        println!("✅ Server is healthy");
        println!("  Status: {}", health.status);
        println!("  Version: {}", health.version);
        println!("  GPU Target: {}", health.gpu_target);
        println!("  Runtime Mode: {}", health.runtime_mode);
        println!("  Memory Usage: {:.1} MB", health.memory_usage_mb);
        println!("  GPU Memory: {:.1} MB", health.gpu_memory_usage_mb);
    } else {
        println!("❌ Server health check failed: {}", response.status());
    }

    Ok(())
}

async fn show_stats(client: &Client, server: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Server Statistics:");

    let response = client
        .get(&format!("{}/stats", server))
        .send()
        .await?;

    if response.status().is_success() {
        let stats: StatsResponse = response.json().await?;
        println!("  Total Requests: {}", stats.total_requests);
        println!("  Total Tokens Generated: {}", stats.total_tokens_generated);
        println!("  Average Latency: {:.1}ms", stats.average_latency_ms);
        println!("  Cache Hit Rate: {:.1}%", stats.cache_hit_rate * 100.0);
        println!("  GPU Utilization: {:.1}%", stats.gpu_utilization * 100.0);
        println!("\n💾 Memory Stats:");
        println!("  Total Memory: {:.1} MB", stats.memory_stats.total_memory_mb);
        println!("  Used Memory: {:.1} MB", stats.memory_stats.used_memory_mb);
        println!("  Cache Memory: {:.1} MB", stats.memory_stats.cache_memory_mb);
        println!("  GPU Memory: {:.1} MB", stats.memory_stats.gpu_memory_mb);
    } else {
        println!("❌ Failed to get server stats: {}", response.status());
    }

    Ok(())
}

async fn run_interactive_mode(
    client: &Client,
    server: &str,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎮 Interactive Mode - Type 'quit' to exit");

    loop {
        print!("\n💬 Prompt: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let prompt = input.trim();

        if prompt.is_empty() {
            continue;
        }

        if prompt == "quit" || prompt == "exit" {
            break;
        }

        if prompt == "stats" {
            show_stats(client, server).await?;
            continue;
        }

        let start = Instant::now();
        match send_inference_request(client, server, prompt, args).await {
            Ok(response) => {
                let total_time = start.elapsed();
                println!("\n🤖 Response: {}", response.generated_text);
                println!("⚡ Generated {} tokens in {}ms (total: {}ms)",
                    response.tokens_generated,
                    response.inference_time_ms,
                    total_time.as_millis()
                );
            }
            Err(e) => {
                eprintln!("❌ Error: {}", e);
            }
        }
    }

    println!("👋 Goodbye!");
    Ok(())
}

async fn run_benchmark(
    client: &Client,
    server: &str,
    num_requests: usize,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🏁 Running benchmark with {} requests...", num_requests);

    let prompt = args.prompt.as_deref().unwrap_or("Hello, how are you?");
    let mut latencies = Vec::new();
    let mut total_tokens = 0;

    let benchmark_start = Instant::now();

    for i in 0..num_requests {
        let start = Instant::now();
        match send_inference_request(client, server, prompt, args).await {
            Ok(response) => {
                let latency = start.elapsed();
                latencies.push(latency.as_millis() as f64);
                total_tokens += response.tokens_generated;
                print!(".");
                if (i + 1) % 10 == 0 {
                    println!(" {}/{}", i + 1, num_requests);
                }
            }
            Err(e) => {
                eprintln!("\n❌ Request {} failed: {}", i + 1, e);
            }
        }
    }

    let total_time = benchmark_start.elapsed();
    println!("\n\n📊 Benchmark Results:");
    println!("  Total Requests: {}", num_requests);
    println!("  Successful Requests: {}", latencies.len());
    println!("  Total Time: {:.2}s", total_time.as_secs_f64());
    println!("  Requests/sec: {:.2}", latencies.len() as f64 / total_time.as_secs_f64());
    println!("  Total Tokens: {}", total_tokens);
    println!("  Tokens/sec: {:.2}", total_tokens as f64 / total_time.as_secs_f64());

    if !latencies.is_empty() {
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let p50 = latencies[latencies.len() / 2];
        let p95 = latencies[(latencies.len() as f64 * 0.95) as usize];
        let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];

        println!("\n⚡ Latency Statistics:");
        println!("  Average: {:.1}ms", avg_latency);
        println!("  P50: {:.1}ms", p50);
        println!("  P95: {:.1}ms", p95);
        println!("  P99: {:.1}ms", p99);
    }

    Ok(())
}

async fn send_inference_request(
    client: &Client,
    server: &str,
    prompt: &str,
    args: &Args,
) -> Result<InferenceResponse, Box<dyn std::error::Error>> {
    let sampling_params = SamplingParams {
        max_tokens: Some(args.max_tokens),
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: None,
        seed: None,
        stop_sequences: vec![],
        repetition_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
    };

    let request = RequestBuilder::new(prompt.to_string())
        .with_sampling_params(sampling_params)
        .build();

    let response = client
        .post(&format!("{}/v1/generate", server))
        .json(&request)
        .send()
        .await?;

    if response.status().is_success() {
        let inference_response: InferenceResponse = response.json().await?;
        Ok(inference_response)
    } else {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
        Err(format!("Request failed with status {}: {}", status, error_text).into())
    }
}