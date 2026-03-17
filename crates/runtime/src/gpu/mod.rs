//! Comprehensive GPU support for UniLLM
//!
//! This module implements support for all major GPU architectures:
//! - NVIDIA CUDA (all compute capabilities)
//! - AMD ROCm (all RDNA and CDNA architectures)
//! - Intel XPU (oneAPI/SYCL)
//! - Apple Metal (all Apple Silicon)

pub mod cuda;
pub mod rocm;
pub mod intel;
pub mod metal;
pub mod common;

use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::models::traits::*;

/// GPU backend types supported by UniLLM
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GpuBackend {
    /// NVIDIA CUDA
    Cuda,
    /// AMD ROCm
    Rocm,
    /// Intel oneAPI/XPU
    Intel,
    /// Apple Metal
    Metal,
    /// CPU fallback
    Cpu,
}

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    pub device_id: u32,
    pub name: String,
    pub backend: GpuBackend,
    pub compute_capability: Option<String>,
    pub memory_total: usize,
    pub memory_free: usize,
    pub core_count: u32,
    pub max_threads_per_block: u32,
    pub max_shared_memory: usize,
    pub tensor_cores: bool,
    pub fp16_support: bool,
    pub bf16_support: bool,
    pub fp8_support: bool,
    pub int8_support: bool,
    pub int4_support: bool,
    pub nvlink_support: bool,
    pub pcie_generation: u32,
    pub ecc_enabled: bool,
    pub driver_version: String,
    pub cuda_version: Option<String>,
    pub rocm_version: Option<String>,
    pub oneapi_version: Option<String>,
    pub metal_version: Option<String>,
}

/// GPU context for operations
#[derive(Debug, Clone)]
pub struct GpuContext {
    pub device_info: GpuDeviceInfo,
    pub device_handle: u64, // Opaque handle to actual device context
    pub stream_handles: Vec<u64>, // Compute streams
    pub memory_pool: Option<Arc<dyn GpuMemoryPool>>,
    pub kernel_cache: HashMap<String, u64>, // Compiled kernels
}

/// GPU memory pool trait
#[async_trait]
pub trait GpuMemoryPool: Send + Sync {
    /// Allocate GPU memory
    async fn allocate(&self, size: usize, alignment: usize) -> GpuResult<GpuMemoryHandle>;

    /// Free GPU memory
    async fn free(&self, handle: GpuMemoryHandle) -> GpuResult<()>;

    /// Get memory usage statistics
    fn get_memory_stats(&self) -> GpuMemoryStats;

    /// Copy data between CPU and GPU
    async fn copy_host_to_device(&self, src: &[u8], dst: GpuMemoryHandle) -> GpuResult<()>;
    async fn copy_device_to_host(&self, src: GpuMemoryHandle, dst: &mut [u8]) -> GpuResult<()>;
    async fn copy_device_to_device(&self, src: GpuMemoryHandle, dst: GpuMemoryHandle, size: usize) -> GpuResult<()>;

    /// Memory pool defragmentation
    async fn defragment(&self) -> GpuResult<()>;
}

/// GPU memory handle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuMemoryHandle(pub u64);

/// GPU memory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMemoryStats {
    pub total_memory: usize,
    pub allocated_memory: usize,
    pub free_memory: usize,
    pub fragmentation_ratio: f32,
    pub allocation_count: u64,
    pub deallocation_count: u64,
    pub peak_usage: usize,
    pub cache_hit_rate: f32,
}

/// GPU kernel execution trait
#[async_trait]
pub trait GpuKernel: Send + Sync {
    /// Compile kernel from source
    async fn compile(&self, source: &str, function_name: &str, options: &[String]) -> GpuResult<u64>;

    /// Launch kernel with parameters
    async fn launch(
        &self,
        kernel_handle: u64,
        grid_size: (u32, u32, u32),
        block_size: (u32, u32, u32),
        shared_memory: usize,
        stream: u64,
        parameters: &[KernelParameter],
    ) -> GpuResult<()>;

    /// Synchronize kernel execution
    async fn synchronize(&self, stream: u64) -> GpuResult<()>;

    /// Get kernel execution statistics
    fn get_kernel_stats(&self, kernel_handle: u64) -> GpuResult<KernelStats>;
}

/// Kernel parameter types
#[derive(Debug, Clone)]
pub enum KernelParameter {
    Buffer(GpuMemoryHandle),
    Scalar(ScalarValue),
    LocalMemory(usize),
}

/// Scalar values for kernel parameters
#[derive(Debug, Clone)]
pub enum ScalarValue {
    Int32(i32),
    UInt32(u32),
    Int64(i64),
    UInt64(u64),
    Float32(f32),
    Float64(f64),
}

/// Kernel execution statistics
#[derive(Debug, Clone)]
pub struct KernelStats {
    pub execution_time_ms: f64,
    pub occupancy: f32,
    pub memory_throughput_gb_s: f32,
    pub compute_utilization: f32,
    pub cache_hit_rate: f32,
    pub register_usage: u32,
    pub shared_memory_usage: u32,
}

/// GPU tensor operations trait
#[async_trait]
pub trait GpuTensorOps: Send + Sync {
    /// Matrix multiplication
    async fn matmul(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &mut GpuTensor,
        alpha: f32,
        beta: f32,
    ) -> GpuResult<()>;

    /// Element-wise operations
    async fn elementwise_add(&self, a: &GpuTensor, b: &GpuTensor, c: &mut GpuTensor) -> GpuResult<()>;
    async fn elementwise_mul(&self, a: &GpuTensor, b: &GpuTensor, c: &mut GpuTensor) -> GpuResult<()>;

    /// Reduction operations
    async fn reduce_sum(&self, input: &GpuTensor, output: &mut GpuTensor, axis: Option<u32>) -> GpuResult<()>;
    async fn reduce_max(&self, input: &GpuTensor, output: &mut GpuTensor, axis: Option<u32>) -> GpuResult<()>;

    /// Attention operations
    async fn attention(
        &self,
        query: &GpuTensor,
        key: &GpuTensor,
        value: &GpuTensor,
        output: &mut GpuTensor,
        mask: Option<&GpuTensor>,
        scale: f32,
    ) -> GpuResult<()>;

    /// Layer normalization
    async fn layer_norm(
        &self,
        input: &GpuTensor,
        weight: &GpuTensor,
        bias: Option<&GpuTensor>,
        output: &mut GpuTensor,
        eps: f32,
    ) -> GpuResult<()>;

    /// Activation functions
    async fn gelu(&self, input: &GpuTensor, output: &mut GpuTensor) -> GpuResult<()>;
    async fn relu(&self, input: &GpuTensor, output: &mut GpuTensor) -> GpuResult<()>;
    async fn silu(&self, input: &GpuTensor, output: &mut GpuTensor) -> GpuResult<()>;
}

/// GPU tensor representation
#[derive(Debug, Clone)]
pub struct GpuTensor {
    pub memory_handle: GpuMemoryHandle,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub dtype: DataType,
    pub device_id: u32,
    pub backend: GpuBackend,
}

impl GpuTensor {
    /// Create a new GPU tensor
    pub fn new(
        memory_handle: GpuMemoryHandle,
        shape: Vec<usize>,
        dtype: DataType,
        device_id: u32,
        backend: GpuBackend,
    ) -> Self {
        let strides = Self::compute_strides(&shape);
        Self {
            memory_handle,
            shape,
            strides,
            dtype,
            device_id,
            backend,
        }
    }

    /// Compute default strides for row-major layout
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Get total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get tensor size in bytes
    pub fn size_bytes(&self) -> usize {
        let element_size = match self.dtype {
            DataType::Float32 => 4,
            DataType::Float16 | DataType::BFloat16 => 2,
            DataType::Int8 => 1,
            DataType::Int4 => 1, // Packed
        };
        self.numel() * element_size
    }

    /// Check if tensor is contiguous
    pub fn is_contiguous(&self) -> bool {
        let expected_strides = Self::compute_strides(&self.shape);
        self.strides == expected_strides
    }

    /// Reshape tensor (view operation)
    pub fn reshape(&self, new_shape: Vec<usize>) -> GpuResult<Self> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(GpuError::InvalidOperation(
                format!("Cannot reshape tensor from {} to {} elements", self.numel(), new_numel)
            ));
        }

        Ok(Self {
            memory_handle: self.memory_handle,
            shape: new_shape.clone(),
            strides: Self::compute_strides(&new_shape),
            dtype: self.dtype,
            device_id: self.device_id,
            backend: self.backend,
        })
    }

    /// Transpose tensor
    pub fn transpose(&self, dim0: usize, dim1: usize) -> GpuResult<Self> {
        if dim0 >= self.shape.len() || dim1 >= self.shape.len() {
            return Err(GpuError::InvalidOperation(
                "Transpose dimensions out of bounds".to_string()
            ));
        }

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();

        new_shape.swap(dim0, dim1);
        new_strides.swap(dim0, dim1);

        Ok(Self {
            memory_handle: self.memory_handle,
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
            device_id: self.device_id,
            backend: self.backend,
        })
    }
}

/// GPU backend manager
pub struct GpuManager {
    backends: HashMap<GpuBackend, Box<dyn GpuBackendInterface>>,
    devices: Vec<GpuDeviceInfo>,
    default_device: Option<(GpuBackend, u32)>,
}

/// GPU backend interface
#[async_trait]
pub trait GpuBackendInterface: Send + Sync {
    /// Initialize the backend
    async fn initialize(&mut self) -> GpuResult<()>;

    /// Get available devices
    fn get_devices(&self) -> GpuResult<Vec<GpuDeviceInfo>>;

    /// Create GPU context
    async fn create_context(&self, device_id: u32) -> GpuResult<GpuContext>;

    /// Get memory pool
    fn get_memory_pool(&self, device_id: u32) -> GpuResult<Arc<dyn GpuMemoryPool>>;

    /// Get kernel interface
    fn get_kernel_interface(&self, device_id: u32) -> GpuResult<Arc<dyn GpuKernel>>;

    /// Get tensor operations
    fn get_tensor_ops(&self, device_id: u32) -> GpuResult<Arc<dyn GpuTensorOps>>;

    /// Check if backend is available
    fn is_available(&self) -> bool;

    /// Get backend capabilities
    fn get_capabilities(&self) -> GpuBackendCapabilities;
}

/// Backend capabilities
#[derive(Debug, Clone)]
pub struct GpuBackendCapabilities {
    pub supports_fp16: bool,
    pub supports_bf16: bool,
    pub supports_fp8: bool,
    pub supports_int8: bool,
    pub supports_int4: bool,
    pub supports_tensor_cores: bool,
    pub supports_flash_attention: bool,
    pub supports_unified_memory: bool,
    pub supports_peer_to_peer: bool,
    pub max_compute_capability: Option<String>,
    pub max_memory_per_device: usize,
    pub max_threads_per_block: u32,
}

impl GpuManager {
    /// Create a new GPU manager
    pub fn new() -> Self {
        Self {
            backends: HashMap::new(),
            devices: Vec::new(),
            default_device: None,
        }
    }

    /// Initialize all available backends
    pub async fn initialize(&mut self) -> GpuResult<()> {
        // Try to initialize CUDA
        if let Ok(mut cuda_backend) = cuda::CudaBackend::new() {
            if cuda_backend.is_available() {
                cuda_backend.initialize().await?;
                let devices = cuda_backend.get_devices()?;
                self.devices.extend(devices);
                self.backends.insert(GpuBackend::Cuda, Box::new(cuda_backend));

                if self.default_device.is_none() {
                    self.default_device = Some((GpuBackend::Cuda, 0));
                }
            }
        }

        // Try to initialize ROCm
        if let Ok(mut rocm_backend) = rocm::RocmBackend::new() {
            if rocm_backend.is_available() {
                rocm_backend.initialize().await?;
                let devices = rocm_backend.get_devices()?;
                self.devices.extend(devices);
                self.backends.insert(GpuBackend::Rocm, Box::new(rocm_backend));

                if self.default_device.is_none() {
                    self.default_device = Some((GpuBackend::Rocm, 0));
                }
            }
        }

        // Try to initialize Intel
        if let Ok(mut intel_backend) = intel::IntelBackend::new() {
            if intel_backend.is_available() {
                intel_backend.initialize().await?;
                let devices = intel_backend.get_devices()?;
                self.devices.extend(devices);
                self.backends.insert(GpuBackend::Intel, Box::new(intel_backend));

                if self.default_device.is_none() {
                    self.default_device = Some((GpuBackend::Intel, 0));
                }
            }
        }

        // Try to initialize Metal
        #[cfg(target_os = "macos")]
        if let Ok(mut metal_backend) = metal::MetalBackend::new() {
            if metal_backend.is_available() {
                metal_backend.initialize().await?;
                let devices = metal_backend.get_devices()?;
                self.devices.extend(devices);
                self.backends.insert(GpuBackend::Metal, Box::new(metal_backend));

                if self.default_device.is_none() {
                    self.default_device = Some((GpuBackend::Metal, 0));
                }
            }
        }

        println!("Initialized GPU backends. Found {} devices", self.devices.len());
        for device in &self.devices {
            println!("  - {} ({}): {:.1} GB", device.name, device.backend as u32, device.memory_total as f64 / 1e9);
        }

        Ok(())
    }

    /// Get all available devices
    pub fn get_devices(&self) -> &[GpuDeviceInfo] {
        &self.devices
    }

    /// Get device by backend and ID
    pub fn get_device(&self, backend: GpuBackend, device_id: u32) -> Option<&GpuDeviceInfo> {
        self.devices.iter().find(|d| d.backend == backend && d.device_id == device_id)
    }

    /// Create GPU context for device
    pub async fn create_context(&self, backend: GpuBackend, device_id: u32) -> GpuResult<GpuContext> {
        if let Some(backend_impl) = self.backends.get(&backend) {
            backend_impl.create_context(device_id).await
        } else {
            Err(GpuError::BackendNotAvailable(format!("{:?}", backend)))
        }
    }

    /// Get default device
    pub fn get_default_device(&self) -> Option<(GpuBackend, u32)> {
        self.default_device
    }

    /// Set default device
    pub fn set_default_device(&mut self, backend: GpuBackend, device_id: u32) -> GpuResult<()> {
        if self.get_device(backend, device_id).is_some() {
            self.default_device = Some((backend, device_id));
            Ok(())
        } else {
            Err(GpuError::DeviceNotFound(format!("{:?}:{}", backend, device_id)))
        }
    }

    /// Get tensor operations for device
    pub fn get_tensor_ops(&self, backend: GpuBackend, device_id: u32) -> GpuResult<Arc<dyn GpuTensorOps>> {
        if let Some(backend_impl) = self.backends.get(&backend) {
            backend_impl.get_tensor_ops(device_id)
        } else {
            Err(GpuError::BackendNotAvailable(format!("{:?}", backend)))
        }
    }

    /// Detect optimal device for model
    pub fn detect_optimal_device(&self, model_size_bytes: usize, requirements: &GpuRequirements) -> Option<(GpuBackend, u32)> {
        let mut best_device = None;
        let mut best_score = 0.0f32;

        for device in &self.devices {
            if device.memory_total < model_size_bytes * 2 {
                continue; // Need at least 2x model size for activations
            }

            let mut score = 0.0f32;

            // Memory score (more is better)
            score += (device.memory_total as f32 / 1e9) * 10.0;

            // Compute capability score
            if device.tensor_cores && requirements.tensor_cores {
                score += 50.0;
            }

            // Precision support
            if device.fp16_support && requirements.fp16_support {
                score += 20.0;
            }
            if device.bf16_support && requirements.bf16_support {
                score += 20.0;
            }
            if device.fp8_support && requirements.fp8_support {
                score += 30.0;
            }

            // Backend preference (CUDA > ROCm > Intel > Metal for ML workloads)
            match device.backend {
                GpuBackend::Cuda => score += 100.0,
                GpuBackend::Rocm => score += 80.0,
                GpuBackend::Intel => score += 60.0,
                GpuBackend::Metal => score += 40.0,
                GpuBackend::Cpu => score += 0.0,
            }

            if score > best_score {
                best_score = score;
                best_device = Some((device.backend, device.device_id));
            }
        }

        best_device
    }
}

/// GPU requirements for device selection
#[derive(Debug, Clone, Default)]
pub struct GpuRequirements {
    pub min_memory_gb: f32,
    pub tensor_cores: bool,
    pub fp16_support: bool,
    pub bf16_support: bool,
    pub fp8_support: bool,
    pub flash_attention: bool,
    pub multi_gpu: bool,
}

/// GPU operation results
pub type GpuResult<T> = Result<T, GpuError>;

/// GPU errors
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("Backend not available: {0}")]
    BackendNotAvailable(String),

    #[error("Device not found: {0}")]
    DeviceNotFound(String),

    #[error("Out of memory: {0}")]
    OutOfMemory(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Kernel compilation failed: {0}")]
    KernelCompilationFailed(String),

    #[error("Kernel execution failed: {0}")]
    KernelExecutionFailed(String),

    #[error("Memory copy failed: {0}")]
    MemoryCopyFailed(String),

    #[error("Synchronization failed: {0}")]
    SynchronizationFailed(String),

    #[error("Driver error: {0}")]
    DriverError(String),

    #[error("Runtime error: {0}")]
    RuntimeError(String),
}

/// Global GPU manager instance
static mut GPU_MANAGER: Option<GpuManager> = None;
static GPU_MANAGER_INIT: std::sync::Once = std::sync::Once::new();

/// Get global GPU manager
pub fn get_gpu_manager() -> &'static mut GpuManager {
    unsafe {
        GPU_MANAGER_INIT.call_once(|| {
            GPU_MANAGER = Some(GpuManager::new());
        });
        GPU_MANAGER.as_mut().unwrap()
    }
}

/// Initialize GPU subsystem
pub async fn initialize_gpu() -> GpuResult<()> {
    get_gpu_manager().initialize().await
}

/// Utility functions for GPU operations
pub mod utils {
    use super::*;

    /// Get optimal block size for kernel launch
    pub fn get_optimal_block_size(device: &GpuDeviceInfo, kernel_complexity: u32) -> (u32, u32, u32) {
        let max_threads = device.max_threads_per_block;

        // For simple kernels, use larger blocks
        let block_size = if kernel_complexity <= 1 {
            (max_threads.min(1024), 1, 1)
        } else if kernel_complexity <= 3 {
            (max_threads.min(512), 1, 1)
        } else {
            (max_threads.min(256), 1, 1)
        };

        block_size
    }

    /// Calculate grid size for given problem size
    pub fn calculate_grid_size(problem_size: (u32, u32, u32), block_size: (u32, u32, u32)) -> (u32, u32, u32) {
        (
            (problem_size.0 + block_size.0 - 1) / block_size.0,
            (problem_size.1 + block_size.1 - 1) / block_size.1,
            (problem_size.2 + block_size.2 - 1) / block_size.2,
        )
    }

    /// Estimate memory bandwidth utilization
    pub fn estimate_bandwidth_utilization(
        data_size: usize,
        execution_time_ms: f64,
        peak_bandwidth_gb_s: f32,
    ) -> f32 {
        let data_gb = data_size as f64 / 1e9;
        let execution_time_s = execution_time_ms / 1000.0;
        let achieved_bandwidth = data_gb / execution_time_s;

        (achieved_bandwidth / peak_bandwidth_gb_s as f64) as f32 * 100.0
    }

    /// Check if tensors are compatible for operation
    pub fn tensors_compatible(a: &GpuTensor, b: &GpuTensor) -> bool {
        a.backend == b.backend && a.device_id == b.device_id && a.dtype == b.dtype
    }

    /// Get memory alignment for data type
    pub fn get_memory_alignment(dtype: DataType) -> usize {
        match dtype {
            DataType::Float32 => 4,
            DataType::Float16 | DataType::BFloat16 => 2,
            DataType::Int8 => 1,
            DataType::Int4 => 1,
        }
    }
}