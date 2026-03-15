//! Telemetry crate for tracing, counters, and performance validation

mod benchmark;

use std::collections::HashMap;
use std::time::{Instant, Duration};

pub use benchmark::{BenchmarkConfig, BenchmarkRunner, BenchmarkResults};

/// Telemetry implementation for performance monitoring
pub struct Telemetry {
    metrics: HashMap<String, Metric>,
    traces: Vec<Trace>,
}

/// A metric for tracking performance
pub struct Metric {
    name: String,
    value: f64,
    unit: String,
    timestamp: Instant,
}

/// A trace for tracking execution flow
pub struct Trace {
    name: String,
    start_time: Instant,
    end_time: Option<Instant>,
    duration: Option<Duration>,
}

/// Performance validation results
#[derive(Clone)]
pub struct ValidationResults {
    pub latency_p50: f64,  // 50th percentile latency in milliseconds
    pub latency_p90: f64,  // 90th percentile latency in milliseconds
    pub latency_p99: f64,  // 99th percentile latency in milliseconds
    pub throughput: f64,   // Tokens per second
    pub memory_usage: u64, // Memory usage in bytes
}

impl Telemetry {
    /// Create a new telemetry instance
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            traces: Vec::new(),
        }
    }
    
    /// Record a metric
    pub fn record_metric(&mut self, name: String, value: f64, unit: String) {
        let metric = Metric {
            name: name.clone(),
            value,
            unit,
            timestamp: Instant::now(),
        };
        self.metrics.insert(name, metric);
    }
    
    /// Start a trace
    pub fn start_trace(&mut self, name: String) -> TraceHandle {
        let trace = Trace {
            name: name.clone(),
            start_time: Instant::now(),
            end_time: None,
            duration: None,
        };
        self.traces.push(trace);
        TraceHandle {
            index: self.traces.len() - 1,
        }
    }
    
    /// End a trace
    pub fn end_trace(&mut self, handle: TraceHandle) {
        if let Some(trace) = self.traces.get_mut(handle.index) {
            trace.end_time = Some(Instant::now());
            trace.duration = Some(trace.end_time.unwrap() - trace.start_time);
        }
    }
    
    /// Get a metric by name
    pub fn get_metric(&self, name: &str) -> Option<&Metric> {
        self.metrics.get(name)
    }
    
    /// Get all traces
    pub fn get_traces(&self) -> &[Trace] {
        &self.traces
    }
    
    /// Calculate performance validation results
    pub fn calculate_validation_results(&self) -> ValidationResults {
        // In a real implementation, we would:
        // 1. Analyze trace durations for latency percentiles
        // 2. Calculate throughput from completed operations
        // 3. Measure memory usage
        // 4. Compare against baseline benchmarks
        
        // For now, we'll return placeholder results
        ValidationResults {
            latency_p50: 15.2,
            latency_p90: 22.8,
            latency_p99: 35.1,
            throughput: 1250.0,
            memory_usage: 2048 * 1024 * 1024, // 2GB
        }
    }
    
    /// Compare results against baseline
    pub fn compare_against_baseline(&self, baseline: &ValidationResults) -> ValidationResult {
        let current = self.calculate_validation_results();
        
        let latency_p50_diff = ((current.latency_p50 - baseline.latency_p50) / baseline.latency_p50) * 100.0;
        let throughput_diff = ((current.throughput - baseline.throughput) / baseline.throughput) * 100.0;
        
        ValidationResult {
            meets_requirements: latency_p50_diff <= 5.0 && throughput_diff >= -5.0,
            latency_p50_diff_percent: latency_p50_diff,
            throughput_diff_percent: throughput_diff,
            current_results: current,
            baseline_results: baseline.clone(),
        }
    }
}

/// Handle for managing trace lifecycle
pub struct TraceHandle {
    index: usize,
}

/// Result of performance validation
pub struct ValidationResult {
    pub meets_requirements: bool,
    pub latency_p50_diff_percent: f64,
    pub throughput_diff_percent: f64,
    pub current_results: ValidationResults,
    pub baseline_results: ValidationResults,
}

impl Metric {
    /// Get the metric name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get the metric value
    pub fn value(&self) -> f64 {
        self.value
    }
    
    /// Get the metric unit
    pub fn unit(&self) -> &str {
        &self.unit
    }
}

impl Trace {
    /// Get the trace name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get the trace duration in milliseconds
    pub fn duration_ms(&self) -> Option<f64> {
        self.duration.map(|d| d.as_secs_f64() * 1000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    
    #[test]
    fn test_telemetry_creation() {
        let telemetry = Telemetry::new();
        assert_eq!(telemetry.metrics.len(), 0);
        assert_eq!(telemetry.traces.len(), 0);
    }
    
    #[test]
    fn test_record_metric() {
        let mut telemetry = Telemetry::new();
        telemetry.record_metric("latency".to_string(), 15.5, "ms".to_string());
        
        let metric = telemetry.get_metric("latency");
        assert!(metric.is_some());
        assert_eq!(metric.unwrap().value(), 15.5);
        assert_eq!(metric.unwrap().unit(), "ms");
    }
    
    #[test]
    fn test_trace_lifecycle() {
        let mut telemetry = Telemetry::new();
        let handle = telemetry.start_trace("test_operation".to_string());
        std::thread::sleep(Duration::from_millis(10)); // Small delay
        telemetry.end_trace(handle);
        
        assert_eq!(telemetry.traces.len(), 1);
        let trace = &telemetry.traces[0];
        assert_eq!(trace.name(), "test_operation");
        assert!(trace.duration_ms().is_some());
    }
    
    #[test]
    fn test_validation_results() {
        let telemetry = Telemetry::new();
        let results = telemetry.calculate_validation_results();
        
        // Check that we get reasonable placeholder values
        assert!(results.latency_p50 > 0.0);
        assert!(results.throughput > 0.0);
        assert!(results.memory_usage > 0);
    }
}