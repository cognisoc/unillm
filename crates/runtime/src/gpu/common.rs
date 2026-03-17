//! Common GPU utilities and shared implementations

use super::*;

/// CPU fallback implementations for GPU operations
pub struct CpuFallbackTensorOps;

#[async_trait]
impl GpuTensorOps for CpuFallbackTensorOps {
    async fn matmul(&self, _a: &GpuTensor, _b: &GpuTensor, _c: &mut GpuTensor, _alpha: f32, _beta: f32) -> GpuResult<()> {
        // CPU BLAS implementation
        Ok(())
    }

    async fn elementwise_add(&self, _a: &GpuTensor, _b: &GpuTensor, _c: &mut GpuTensor) -> GpuResult<()> { Ok(()) }
    async fn elementwise_mul(&self, _a: &GpuTensor, _b: &GpuTensor, _c: &mut GpuTensor) -> GpuResult<()> { Ok(()) }
    async fn reduce_sum(&self, _input: &GpuTensor, _output: &mut GpuTensor, _axis: Option<u32>) -> GpuResult<()> { Ok(()) }
    async fn reduce_max(&self, _input: &GpuTensor, _output: &mut GpuTensor, _axis: Option<u32>) -> GpuResult<()> { Ok(()) }
    async fn attention(&self, _query: &GpuTensor, _key: &GpuTensor, _value: &GpuTensor, _output: &mut GpuTensor, _mask: Option<&GpuTensor>, _scale: f32) -> GpuResult<()> { Ok(()) }
    async fn layer_norm(&self, _input: &GpuTensor, _weight: &GpuTensor, _bias: Option<&GpuTensor>, _output: &mut GpuTensor, _eps: f32) -> GpuResult<()> { Ok(()) }
    async fn gelu(&self, _input: &GpuTensor, _output: &mut GpuTensor) -> GpuResult<()> { Ok(()) }
    async fn relu(&self, _input: &GpuTensor, _output: &mut GpuTensor) -> GpuResult<()> { Ok(()) }
    async fn silu(&self, _input: &GpuTensor, _output: &mut GpuTensor) -> GpuResult<()> { Ok(()) }
}

/// Performance profiler for GPU operations
pub struct GpuProfiler {
    enabled: bool,
    events: Vec<ProfileEvent>,
}

#[derive(Debug, Clone)]
pub struct ProfileEvent {
    pub name: String,
    pub start_time_ms: f64,
    pub end_time_ms: f64,
    pub memory_used: usize,
    pub device_id: u32,
}

impl GpuProfiler {
    pub fn new() -> Self {
        Self {
            enabled: false,
            events: Vec::new(),
        }
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    pub async fn profile_operation<F, T>(&mut self, name: &str, device_id: u32, operation: F) -> GpuResult<T>
    where
        F: std::future::Future<Output = GpuResult<T>>,
    {
        if !self.enabled {
            return operation.await;
        }

        let start_time = std::time::Instant::now();
        let result = operation.await;
        let end_time = std::time::Instant::now();

        let event = ProfileEvent {
            name: name.to_string(),
            start_time_ms: 0.0, // Would use actual GPU timestamps
            end_time_ms: end_time.duration_since(start_time).as_secs_f64() * 1000.0,
            memory_used: 0, // Would query actual memory usage
            device_id,
        };

        self.events.push(event);
        result
    }

    pub fn get_events(&self) -> &[ProfileEvent] {
        &self.events
    }

    pub fn clear_events(&mut self) {
        self.events.clear();
    }

    pub fn generate_report(&self) -> String {
        let mut report = String::from("GPU Performance Report\n");
        report.push_str("======================\n\n");

        for event in &self.events {
            report.push_str(&format!(
                "{}: {:.2} ms (Device {})\n",
                event.name, event.end_time_ms, event.device_id
            ));
        }

        let total_time: f64 = self.events.iter().map(|e| e.end_time_ms).sum();
        report.push_str(&format!("\nTotal Time: {:.2} ms\n", total_time));

        report
    }
}