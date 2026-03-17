use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Error types for GPU driver operations
#[derive(Debug, thiserror::Error)]
pub enum GpuDriverError {
    #[error("Driver initialization failed: {0}")]
    DriverLoadError(String),

    #[error("Function loading failed: {0}")]
    FunctionLoadError(String),

    #[error("GPU driver initialization failed: {0}")]
    InitializationError(String),

    #[error("Device error: {0}")]
    DeviceError(String),

    #[error("Context error: {0}")]
    ContextError(String),

    #[error("Memory allocation error: {0}")]
    AllocationError(String),

    #[error("Memory operation error: {0}")]
    MemoryError(String),

    #[error("Module operation error: {0}")]
    ModuleError(String),

    #[error("Kernel execution error: {0}")]
    ExecutionError(String),

    #[error("Stream operation error: {0}")]
    StreamError(String),

    #[error("Synchronization error: {0}")]
    SynchronizationError(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Hardware detection failed: {0}")]
    HardwareDetection(String),

    #[error("Template not found: {0}")]
    TemplateNotFound(String),

    #[error("Compilation failed: {0}")]
    CompilationFailed(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for GPU driver operations
pub type GpuDriverResult<T> = std::result::Result<T, GpuDriverError>;

/// GPU vendor enum
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Unknown,
}

/// Kernel parameters for template instantiation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelParameters {
    pub head_dim: usize,
    pub num_heads: usize,
    pub block_size: usize,
    pub cache_line_size: usize,
    pub warp_size: usize,
    pub memory_coalescing_factor: usize,
    pub additional_params: HashMap<String, ParameterValue>,
}

impl Default for KernelParameters {
    fn default() -> Self {
        Self {
            head_dim: 128,
            num_heads: 32,
            block_size: 256,
            cache_line_size: 128,
            warp_size: 32,
            memory_coalescing_factor: 4,
            additional_params: HashMap::new(),
        }
    }
}

/// Parameter value types for kernel templates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterValue {
    Int(i64),
    UInt(u64),
    Float(f64),
    Bool(bool),
    String(String),
}

/// Hardware information structure
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub gpu_architecture: crate::hardware_detection::GpuArchitecture,
    pub compute_units: usize,
    pub memory_size_gb: usize,
    pub memory_bandwidth_gb_s: f64,
    pub tensor_cores_available: bool,
    pub vendor: GpuVendor,
}

impl HardwareInfo {
    pub fn mock_nvidia() -> Self {
        Self {
            gpu_architecture: crate::hardware_detection::GpuArchitecture::Ampere,
            compute_units: 108,
            memory_size_gb: 40,
            memory_bandwidth_gb_s: 1555.0,
            tensor_cores_available: true,
            vendor: GpuVendor::Nvidia,
        }
    }

    pub fn mock_amd() -> Self {
        Self {
            gpu_architecture: crate::hardware_detection::GpuArchitecture::RDNA3,
            compute_units: 96,
            memory_size_gb: 32,
            memory_bandwidth_gb_s: 1000.0,
            tensor_cores_available: false,
            vendor: GpuVendor::Amd,
        }
    }
}

/// Placeholder for missing types that were causing compilation errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedKernel {
    pub name: String,
    pub source_code: String,
    pub parameters: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct CompiledKernel {
    pub name: String,
    pub binary_data: Vec<u8>,
    pub function_name: String,
}

#[derive(Debug, Clone)]
pub struct BatchExecutionConfig {
    pub batch_size: usize,
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
}

#[derive(Debug, Clone)]
pub struct KernelExecutionResult {
    pub performance: KernelPerformance,
    pub output_data: Vec<u8>,
    pub execution_time_ms: f64,
}

#[derive(Debug, Clone)]
pub struct KernelPerformance {
    pub throughput_tflops: f64,
    pub memory_bandwidth_gb_s: f64,
    pub power_consumption_watts: f64,
    pub energy_efficiency: f64,
    pub kernel_execution_time_us: f64,
    pub memory_transfer_time_us: f64,
}