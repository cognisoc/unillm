//! Benchmark and performance validation framework

use crate::{Telemetry, ValidationResults, ValidationResult};
use std::time::Duration;

/// Benchmark configuration
pub struct BenchmarkConfig {
    pub name: String,
    pub iterations: usize,
    pub warmup_iterations: usize,
    pub target_latency_p50_ms: f64,
    pub target_throughput_tps: f64,
    pub target_memory_mb: u64,
}

/// Benchmark results
pub struct BenchmarkResults {
    pub config: BenchmarkConfig,
    pub validation: ValidationResult,
    pub execution_time: Duration,
    pub iterations_completed: usize,
}

impl BenchmarkConfig {
    /// Create a new benchmark configuration
    pub fn new(name: String) -> Self {
        Self {
            name,
            iterations: 1000,
            warmup_iterations: 100,
            target_latency_p50_ms: 20.0,
            target_throughput_tps: 1000.0,
            target_memory_mb: 1024,
        }
    }
    
    /// Set the number of iterations
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }
    
    /// Set the number of warmup iterations
    pub fn with_warmup(mut self, warmup: usize) -> Self {
        self.warmup_iterations = warmup;
        self
    }
    
    /// Set target latency for 50th percentile
    pub fn with_target_latency_p50(mut self, latency_ms: f64) -> Self {
        self.target_latency_p50_ms = latency_ms;
        self
    }
    
    /// Set target throughput
    pub fn with_target_throughput(mut self, tps: f64) -> Self {
        self.target_throughput_tps = tps;
        self
    }
    
    /// Set target memory usage
    pub fn with_target_memory(mut self, mb: u64) -> Self {
        self.target_memory_mb = mb;
        self
    }
}

/// Performance benchmark runner
pub struct BenchmarkRunner {
    telemetry: Telemetry,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner
    pub fn new() -> Self {
        Self {
            telemetry: Telemetry::new(),
        }
    }
    
    /// Run a benchmark
    pub fn run_benchmark<F>(&mut self, config: BenchmarkConfig, benchmark_fn: F) -> BenchmarkResults
    where
        F: Fn() -> Result<(), Box<dyn std::error::Error>>,
    {
        println!("Running benchmark: {}", config.name);
        println!("Warmup iterations: {}", config.warmup_iterations);
        println!("Test iterations: {}", config.iterations);
        
        let start_time = std::time::Instant::now();
        
        // Warmup phase
        for i in 0..config.warmup_iterations {
            let trace_handle = self.telemetry.start_trace(format!("warmup_{}", i));
            if let Err(e) = benchmark_fn() {
                eprintln!("Warmup iteration {} failed: {}", i, e);
            }
            self.telemetry.end_trace(trace_handle);
        }
        
        // Test phase
        let mut successful_iterations = 0;
        for i in 0..config.iterations {
            let trace_handle = self.telemetry.start_trace(format!("iteration_{}", i));
            match benchmark_fn() {
                Ok(()) => {
                    successful_iterations += 1;
                }
                Err(e) => {
                    eprintln!("Test iteration {} failed: {}", i, e);
                }
            }
            self.telemetry.end_trace(trace_handle);
        }
        
        let execution_time = start_time.elapsed();
        
        // Calculate metrics
        let total_tokens = successful_iterations * 100; // Assume 100 tokens per iteration
        let throughput = total_tokens as f64 / execution_time.as_secs_f64();
        
        self.telemetry.record_metric(
            "throughput".to_string(),
            throughput,
            "tokens/sec".to_string(),
        );
        
        self.telemetry.record_metric(
            "execution_time".to_string(),
            execution_time.as_secs_f64(),
            "seconds".to_string(),
        );
        
        // Create baseline for comparison
        let baseline = ValidationResults {
            latency_p50: config.target_latency_p50_ms,
            latency_p90: config.target_latency_p50_ms * 1.5,
            latency_p99: config.target_latency_p50_ms * 2.0,
            throughput: config.target_throughput_tps,
            memory_usage: config.target_memory_mb * 1024 * 1024,
        };
        
        let validation = self.telemetry.compare_against_baseline(&baseline);
        
        BenchmarkResults {
            config,
            validation,
            execution_time,
            iterations_completed: successful_iterations,
        }
    }
    
    /// Get the underlying telemetry instance
    pub fn telemetry(&self) -> &Telemetry {
        &self.telemetry
    }
    
    /// Get mutable access to the underlying telemetry instance
    pub fn telemetry_mut(&mut self) -> &mut Telemetry {
        &mut self.telemetry
    }
}

impl BenchmarkResults {
    /// Print benchmark results
    pub fn print_results(&self) {
        println!("\n=== Benchmark Results: {} ===", self.config.name);
        println!("Execution time: {:.2?}", self.execution_time);
        println!("Iterations completed: {}", self.iterations_completed);
        println!("Success rate: {:.2}%", 
            (self.iterations_completed as f64 / self.config.iterations as f64) * 100.0);
        
        let validation = &self.validation;
        println!("\n--- Performance Validation ---");
        println!("Meets requirements: {}", validation.meets_requirements);
        println!("Latency p50 difference: {:+.2}%", validation.latency_p50_diff_percent);
        println!("Throughput difference: {:+.2}%", validation.throughput_diff_percent);
        
        println!("\n--- Current Results ---");
        println!("Latency p50: {:.2} ms", validation.current_results.latency_p50);
        println!("Throughput: {:.2} tokens/sec", validation.current_results.throughput);
        println!("Memory usage: {:.2} MB", 
            validation.current_results.memory_usage as f64 / (1024.0 * 1024.0));
        
        println!("\n--- Baseline Targets ---");
        println!("Target latency p50: {:.2} ms", validation.baseline_results.latency_p50);
        println!("Target throughput: {:.2} tokens/sec", validation.baseline_results.throughput);
        println!("Target memory: {:.2} MB", 
            validation.baseline_results.memory_usage as f64 / (1024.0 * 1024.0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    
    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig::new("test_benchmark".to_string())
            .with_iterations(500)
            .with_warmup(50)
            .with_target_latency_p50(15.0)
            .with_target_throughput(1200.0)
            .with_target_memory(512);
        
        assert_eq!(config.name, "test_benchmark");
        assert_eq!(config.iterations, 500);
        assert_eq!(config.warmup_iterations, 50);
        assert_eq!(config.target_latency_p50_ms, 15.0);
        assert_eq!(config.target_throughput_tps, 1200.0);
        assert_eq!(config.target_memory_mb, 512);
    }
    
    #[test]
    fn test_benchmark_runner() {
        let mut runner = BenchmarkRunner::new();
        let config = BenchmarkConfig::new("simple_test".to_string())
            .with_iterations(10)
            .with_warmup(2);
        
        let results = runner.run_benchmark(config, || {
            // Simulate some work
            std::thread::sleep(Duration::from_millis(1));
            Ok(())
        });
        
        assert_eq!(results.config.iterations, 10);
        assert_eq!(results.config.warmup_iterations, 2);
        assert!(results.execution_time > Duration::from_millis(0));
        assert_eq!(results.iterations_completed, 10);
    }
}