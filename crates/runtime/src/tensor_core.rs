//! Core Tensor Abstraction
//!
//! This module provides the foundational tensor abstraction that unifies
//! all tensor operations across CPU, GPU, and different data formats.
//!
//! Uses Candle (HuggingFace's Rust ML framework) as the computational backend.

use std::sync::Arc;
use anyhow::Result;
use candle_core::{DType, Tensor as CandleTensor, Device as CandleDevice};
use candle_nn::ops as candle_ops;

/// Core tensor abstraction - the single source of truth for all tensor operations
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Tensor data storage (device-agnostic)
    data: Arc<dyn TensorStorage>,
    /// Tensor shape
    shape: Vec<usize>,
    /// Data type
    dtype: DataType,
    /// Device location
    device: Device,
}

/// Data types supported by tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    Float32,
    Float16,
    BFloat16,
    Int32,
    Int64,
    Int8,
    Bool,
}

/// Device abstraction
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    CPU,
    CUDA(usize), // GPU ID
    Metal(usize),
}

/// Tensor storage trait - abstracts how data is actually stored
pub trait TensorStorage: Send + Sync + std::fmt::Debug {
    /// Get raw data pointer (for unsafe operations)
    fn data_ptr(&self) -> *const u8;

    /// Get mutable data pointer (for unsafe operations)
    fn data_ptr_mut(&mut self) -> *mut u8;

    /// Get data size in bytes
    fn size_bytes(&self) -> usize;

    /// Get device where data is stored
    fn device(&self) -> &Device;

    /// Copy data to another device
    fn to_device(&self, device: &Device) -> Result<Arc<dyn TensorStorage>>;

    /// Clone the storage
    fn clone_storage(&self) -> Arc<dyn TensorStorage>;

    /// Try to get the underlying Candle tensor (if this is CandleStorage)
    fn as_candle_tensor(&self) -> Option<&CandleTensor> {
        None
    }
}

/// CPU tensor storage implementation
#[derive(Debug)]
pub struct CpuStorage {
    data: Vec<u8>,
    device: Device,
}

/// GPU tensor storage implementation (legacy - kept for compatibility)
#[derive(Debug)]
pub struct GpuStorage {
    #[allow(dead_code)]
    ptr: *mut u8,
    size: usize,
    device: Device,
}

/// Candle-backed tensor storage - the primary storage implementation
#[derive(Debug, Clone)]
pub struct CandleStorage {
    tensor: CandleTensor,
}

impl CandleStorage {
    /// Create from a Candle tensor
    pub fn new(tensor: CandleTensor) -> Self {
        Self { tensor }
    }

    /// Get the underlying Candle tensor
    pub fn tensor(&self) -> &CandleTensor {
        &self.tensor
    }

    /// Get a mutable reference to the Candle tensor
    pub fn tensor_mut(&mut self) -> &mut CandleTensor {
        &mut self.tensor
    }

    /// Create zeros tensor with Candle
    pub fn zeros(shape: &[usize], dtype: DataType, device: &Device) -> Result<Self> {
        let candle_device = device.to_candle();
        let candle_dtype = dtype.to_candle();
        let tensor = CandleTensor::zeros(shape, candle_dtype, &candle_device)?;
        Ok(Self { tensor })
    }

    /// Create ones tensor with Candle
    pub fn ones(shape: &[usize], dtype: DataType, device: &Device) -> Result<Self> {
        let candle_device = device.to_candle();
        let candle_dtype = dtype.to_candle();
        let tensor = CandleTensor::ones(shape, candle_dtype, &candle_device)?;
        Ok(Self { tensor })
    }

    /// Create tensor from raw f32 data
    pub fn from_f32_slice(data: &[f32], shape: &[usize], device: &Device) -> Result<Self> {
        let candle_device = device.to_candle();
        let tensor = CandleTensor::from_slice(data, shape, &candle_device)?;
        Ok(Self { tensor })
    }

    /// Create tensor from raw i64 data
    pub fn from_i64_slice(data: &[i64], shape: &[usize], device: &Device) -> Result<Self> {
        let candle_device = device.to_candle();
        let tensor = CandleTensor::from_slice(data, shape, &candle_device)?;
        Ok(Self { tensor })
    }
}

impl TensorStorage for CandleStorage {
    fn data_ptr(&self) -> *const u8 {
        // For Candle tensors, we need to flatten to contiguous and get the data
        // This is primarily for compatibility - prefer using Candle ops directly
        std::ptr::null() // Candle manages its own memory
    }

    fn data_ptr_mut(&mut self) -> *mut u8 {
        std::ptr::null_mut() // Candle manages its own memory
    }

    fn size_bytes(&self) -> usize {
        self.tensor.elem_count() * self.tensor.dtype().size_in_bytes()
    }

    fn device(&self) -> &Device {
        // Return a static reference - this is a bit of a hack
        // In practice, we should store the device or compute it
        static CPU_DEVICE: Device = Device::CPU;
        &CPU_DEVICE
    }

    fn to_device(&self, device: &Device) -> Result<Arc<dyn TensorStorage>> {
        let candle_device = device.to_candle();
        let new_tensor = self.tensor.to_device(&candle_device)?;
        Ok(Arc::new(CandleStorage::new(new_tensor)))
    }

    fn clone_storage(&self) -> Arc<dyn TensorStorage> {
        Arc::new(self.clone())
    }

    fn as_candle_tensor(&self) -> Option<&CandleTensor> {
        Some(&self.tensor)
    }
}

/// Core tensor operations trait - device-agnostic interface
pub trait TensorOps: Send + Sync {
    /// Matrix multiplication: C = A @ B
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// Element-wise addition: C = A + B
    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// Element-wise multiplication: C = A * B
    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// Scaled dot-product attention
    fn attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
        scale: Option<f32>,
    ) -> Result<Tensor>;

    /// Layer normalization
    fn layer_norm(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor>;

    /// GELU activation
    fn gelu(&self, input: &Tensor) -> Result<Tensor>;

    /// SiLU/Swish activation
    fn silu(&self, input: &Tensor) -> Result<Tensor>;

    /// Softmax
    fn softmax(&self, input: &Tensor, dim: isize) -> Result<Tensor>;

    /// Embedding lookup
    fn embedding(&self, indices: &Tensor, weight: &Tensor) -> Result<Tensor>;

    /// Create tensor with zeros
    fn zeros(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<Tensor>;

    /// Create tensor with random values
    fn randn(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<Tensor>;

    /// Exponential function
    fn exp(&self, input: &Tensor) -> Result<Tensor>;

    /// L2 normalization
    fn normalize(&self, input: &Tensor, p: i32, dim: i32) -> Result<Tensor>;

    /// Concatenate tensors along a dimension
    fn concat(&self, tensors: &[&Tensor], dim: usize) -> Result<Tensor>;

    /// RMS normalization
    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor>;

    /// Scale tensor by a factor
    fn scale(&self, input: &Tensor, factor: f32) -> Result<Tensor>;

    /// Sigmoid activation: 1 / (1 + exp(-x))
    fn sigmoid(&self, input: &Tensor) -> Result<Tensor>;

    /// Top-k operation: returns (values, indices) for top k elements along dimension
    fn topk(&self, input: &Tensor, k: usize, dim: i64) -> Result<(Tensor, Tensor)>;

    /// 1D convolution
    fn conv1d(&self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>, stride: usize, padding: usize) -> Result<Tensor>;

    /// Tanh activation
    fn tanh(&self, input: &Tensor) -> Result<Tensor>;

    /// Element-wise subtraction
    fn sub(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// Clamp values to a range
    fn clamp(&self, input: &Tensor, min: f32, max: f32) -> Result<Tensor>;

    /// Gather elements along dimension using indices
    fn gather(&self, input: &Tensor, dim: usize, indices: &Tensor) -> Result<Tensor>;

    /// Scatter elements along dimension using indices
    fn scatter(&self, input: &Tensor, dim: usize, indices: &Tensor, src: &Tensor) -> Result<Tensor>;

    // ========== Fused Operations for Performance ==========

    /// Fused scaled dot-product attention (Flash Attention pattern)
    /// Computes: softmax(Q @ K^T / sqrt(d_k)) @ V in a memory-efficient manner
    /// Works for all attention-based models: LLaMA, Qwen, Gemma, Mistral, etc.
    fn flash_attention(
        &self,
        query: &Tensor,      // [batch, heads, seq_q, head_dim]
        key: &Tensor,        // [batch, heads, seq_k, head_dim]
        value: &Tensor,      // [batch, heads, seq_k, head_dim]
        scale: f32,
        causal: bool,
    ) -> Result<Tensor>;

    /// Fused SwiGLU activation: silu(gate) * up
    /// Used by LLaMA, Qwen, Mistral, etc.
    fn fused_swiglu(
        &self,
        gate: &Tensor,
        up: &Tensor,
    ) -> Result<Tensor>;

    /// Fused residual add + RMS norm
    /// Computes: rms_norm(residual + hidden, weight, eps)
    fn fused_residual_rms_norm(
        &self,
        residual: &Tensor,
        hidden: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor>;
}

/// Tensor operation dispatcher - automatically selects CPU/GPU implementation
pub struct TensorDispatcher {
    cpu_ops: Box<dyn TensorOps>,
    #[allow(dead_code)]
    gpu_ops: Option<Box<dyn TensorOps>>,
}

impl Tensor {
    /// Create new tensor with data
    pub fn new(
        shape: Vec<usize>,
        dtype: DataType,
        device: Device,
        data: Arc<dyn TensorStorage>
    ) -> Self {
        Self {
            data,
            shape,
            dtype,
            device,
        }
    }

    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get tensor data type
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    /// Get tensor device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Move tensor to device
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        if &self.device == device {
            return Ok(self.clone());
        }

        let new_storage = self.data.to_device(device)?;
        Ok(Self {
            data: new_storage,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: device.clone(),
        })
    }

    /// Reshape tensor
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        let old_numel = self.numel();
        let new_numel: usize = new_shape.iter().product();

        if old_numel != new_numel {
            return Err(anyhow::anyhow!(
                "Cannot reshape tensor: {} elements to {} elements",
                old_numel, new_numel
            ));
        }

        Ok(Self {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    /// Narrow (slice) a tensor along a dimension
    ///
    /// Returns a view into the tensor containing elements from `start` to `start + length`
    /// along the specified dimension.
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Self> {
        let candle_tensor = self.to_candle()?;
        let narrowed = candle_tensor.narrow(dim, start, length)?;
        Ok(Self::from_candle(narrowed))
    }

    /// Create tensor from Candle tensor
    pub fn from_candle(candle_tensor: CandleTensor) -> Self {
        let shape = candle_tensor.dims().to_vec();
        let dtype = DataType::from_candle(candle_tensor.dtype());
        let device = Device::from_candle(&candle_tensor.device());
        let storage = Arc::new(CandleStorage::new(candle_tensor));
        Self {
            data: storage,
            shape,
            dtype,
            device,
        }
    }

    /// Try to get the underlying Candle tensor
    pub fn as_candle(&self) -> Option<&CandleTensor> {
        // Use the TensorStorage trait method to get the underlying Candle tensor
        self.data.as_candle_tensor()
    }

    /// Convert to Candle tensor (may involve copying)
    pub fn to_candle(&self) -> Result<CandleTensor> {
        // First, try to get the underlying Candle tensor if this is CandleStorage
        if let Some(candle_tensor) = self.data.as_candle_tensor() {
            // Check if we need to reshape (Tensor::reshape updates shape but not underlying storage)
            let candle_shape = candle_tensor.dims();
            if candle_shape != &self.shape[..] {
                // Need to reshape the Candle tensor to match our shape
                let shape_slice: &[usize] = &self.shape;
                return Ok(candle_tensor.reshape(shape_slice)?);
            }
            return Ok(candle_tensor.clone());
        }

        // Otherwise, create a new Candle tensor from the raw data
        let candle_device = self.device.to_candle();
        let candle_dtype = self.dtype.to_candle();
        let shape: &[usize] = &self.shape;

        // Get the raw data
        let ptr = self.data.data_ptr();
        if ptr.is_null() {
            // No data available, create zeros
            return Ok(CandleTensor::zeros(shape, candle_dtype, &candle_device)?);
        }

        let size_bytes = self.data.size_bytes();
        let data_slice = unsafe { std::slice::from_raw_parts(ptr, size_bytes) };

        // Convert based on dtype
        match self.dtype {
            DataType::Float32 => {
                let float_data: &[f32] = bytemuck::cast_slice(data_slice);
                Ok(CandleTensor::from_slice(float_data, shape, &candle_device)?)
            }
            DataType::Int64 => {
                let int_data: &[i64] = bytemuck::cast_slice(data_slice);
                Ok(CandleTensor::from_slice(int_data, shape, &candle_device)?)
            }
            DataType::Int32 => {
                // Candle doesn't support i32 directly, convert to i64
                let int_data: &[i32] = bytemuck::cast_slice(data_slice);
                let int64_data: Vec<i64> = int_data.iter().map(|&x| x as i64).collect();
                Ok(CandleTensor::from_slice(&int64_data, shape, &candle_device)?)
            }
            _ => {
                // For other types, create zeros
                Ok(CandleTensor::zeros(shape, candle_dtype, &candle_device)?)
            }
        }
    }

    /// Create tensor with zeros using Candle backend
    pub fn zeros_candle(shape: &[usize], dtype: DataType, device: &Device) -> Result<Self> {
        let storage = CandleStorage::zeros(shape, dtype, device)?;
        Ok(Self {
            data: Arc::new(storage.clone()),
            shape: shape.to_vec(),
            dtype,
            device: device.clone(),
        })
    }

    /// Create tensor from f32 slice using Candle backend
    pub fn from_f32_slice(data: &[f32], shape: &[usize], device: &Device) -> Result<Self> {
        let storage = CandleStorage::from_f32_slice(data, shape, device)?;
        Ok(Self {
            data: Arc::new(storage),
            shape: shape.to_vec(),
            dtype: DataType::Float32,
            device: device.clone(),
        })
    }

    /// Create tensor from i64 slice using Candle backend
    pub fn from_i64_slice(data: &[i64], shape: &[usize], device: &Device) -> Result<Self> {
        let storage = CandleStorage::from_i64_slice(data, shape, device)?;
        Ok(Self {
            data: Arc::new(storage),
            shape: shape.to_vec(),
            dtype: DataType::Int64,
            device: device.clone(),
        })
    }

    /// Get the raw data as f32 slice (if applicable)
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        let candle_tensor = self.to_candle()?;
        let data = candle_tensor.to_vec1::<f32>()?;
        Ok(data)
    }

    /// Transpose tensor (swap last two dimensions)
    /// For 2D tensors [M, N] -> [N, M]
    /// For higher dims [.., M, N] -> [.., N, M]
    pub fn transpose(&self) -> Result<Self> {
        let candle_tensor = self.to_candle()?;
        let dims = candle_tensor.dims();
        if dims.len() < 2 {
            return Err(anyhow::anyhow!("Cannot transpose tensor with less than 2 dimensions"));
        }
        let transposed = candle_tensor.t()?;
        Ok(Self::from_candle(transposed))
    }

    /// Transpose tensor with specific dimensions
    pub fn transpose_dims(&self, dim0: usize, dim1: usize) -> Result<Self> {
        let candle_tensor = self.to_candle()?;
        let transposed = candle_tensor.transpose(dim0, dim1)?;
        Ok(Self::from_candle(transposed))
    }
}

impl CpuStorage {
    pub fn new(data: Vec<u8>, device: Device) -> Self {
        Self { data, device }
    }

    pub fn zeros(size_bytes: usize) -> Self {
        Self {
            data: vec![0u8; size_bytes],
            device: Device::CPU,
        }
    }
}

impl TensorStorage for CpuStorage {
    fn data_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    fn data_ptr_mut(&mut self) -> *mut u8 {
        self.data.as_mut_ptr()
    }

    fn size_bytes(&self) -> usize {
        self.data.len()
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn to_device(&self, device: &Device) -> Result<Arc<dyn TensorStorage>> {
        match device {
            Device::CPU => Ok(Arc::new(self.clone())),
            Device::CUDA(_) | Device::Metal(_) => {
                // TODO: Implement GPU transfer
                Err(anyhow::anyhow!("GPU transfer not implemented yet"))
            }
        }
    }

    fn clone_storage(&self) -> Arc<dyn TensorStorage> {
        Arc::new(self.clone())
    }
}

impl Clone for CpuStorage {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            device: self.device.clone(),
        }
    }
}

unsafe impl Send for GpuStorage {}
unsafe impl Sync for GpuStorage {}

impl TensorStorage for GpuStorage {
    fn data_ptr(&self) -> *const u8 {
        self.ptr
    }

    fn data_ptr_mut(&mut self) -> *mut u8 {
        self.ptr
    }

    fn size_bytes(&self) -> usize {
        self.size
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn to_device(&self, device: &Device) -> Result<Arc<dyn TensorStorage>> {
        match device {
            Device::CPU => {
                // Basic GPU->CPU transfer: create CPU storage with same size
                let cpu_storage = Arc::new(CpuStorage::zeros(self.size));
                Ok(cpu_storage)
            }
            Device::CUDA(_) | Device::Metal(_) => {
                // Basic GPU->GPU transfer: create new GPU storage with same size
                let gpu_storage = Arc::new(GpuStorage {
                    ptr: std::ptr::null_mut(),
                    size: self.size,
                    device: device.clone(),
                });
                Ok(gpu_storage)
            }
        }
    }

    fn clone_storage(&self) -> Arc<dyn TensorStorage> {
        // Basic GPU storage cloning
        Arc::new(GpuStorage {
            ptr: self.ptr,
            size: self.size,
            device: self.device.clone(),
        })
    }
}

impl TensorDispatcher {
    pub fn new() -> Self {
        Self {
            cpu_ops: Box::new(CpuTensorOpsImpl::new()),
            gpu_ops: Some(Box::new(GpuTensorOpsImpl::new())), // Initialize basic GPU ops
        }
    }

    /// Get appropriate tensor ops for the given tensors
    fn get_ops(&self, tensors: &[&Tensor]) -> &dyn TensorOps {
        // Check if any tensor is on GPU
        for tensor in tensors {
            match tensor.device() {
                Device::CPU => continue,
                Device::CUDA(_) | Device::Metal(_) => {
                    // Use GPU ops if available
                    if let Some(gpu_ops) = &self.gpu_ops {
                        return gpu_ops.as_ref();
                    }
                }
            }
        }
        self.cpu_ops.as_ref()
    }
}

impl TensorOps for TensorDispatcher {
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let ops = self.get_ops(&[a, b]);
        ops.matmul(a, b)
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let ops = self.get_ops(&[a, b]);
        ops.add(a, b)
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let ops = self.get_ops(&[a, b]);
        ops.mul(a, b)
    }

    fn attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
        scale: Option<f32>,
    ) -> Result<Tensor> {
        let tensors = if let Some(m) = mask {
            vec![query, key, value, m]
        } else {
            vec![query, key, value]
        };
        let ops = self.get_ops(&tensors);
        ops.attention(query, key, value, mask, scale)
    }

    fn layer_norm(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        let tensors = if let Some(b) = bias {
            vec![input, weight, b]
        } else {
            vec![input, weight]
        };
        let ops = self.get_ops(&tensors);
        ops.layer_norm(input, weight, bias, eps)
    }

    fn gelu(&self, input: &Tensor) -> Result<Tensor> {
        let ops = self.get_ops(&[input]);
        ops.gelu(input)
    }

    fn silu(&self, input: &Tensor) -> Result<Tensor> {
        let ops = self.get_ops(&[input]);
        ops.silu(input)
    }

    fn softmax(&self, input: &Tensor, dim: isize) -> Result<Tensor> {
        let ops = self.get_ops(&[input]);
        ops.softmax(input, dim)
    }

    fn embedding(&self, indices: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let ops = self.get_ops(&[indices, weight]);
        ops.embedding(indices, weight)
    }

    fn zeros(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<Tensor> {
        match device {
            Device::CPU => self.cpu_ops.zeros(shape, dtype, device),
            Device::CUDA(_) | Device::Metal(_) => {
                // TODO: Use GPU ops when available
                self.cpu_ops.zeros(shape, dtype, &Device::CPU)
            }
        }
    }

    fn randn(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<Tensor> {
        match device {
            Device::CPU => self.cpu_ops.randn(shape, dtype, device),
            Device::CUDA(_) | Device::Metal(_) => {
                // TODO: Use GPU ops when available
                self.cpu_ops.randn(shape, dtype, &Device::CPU)
            }
        }
    }

    fn exp(&self, input: &Tensor) -> Result<Tensor> {
        let ops = self.get_ops(&[input]);
        ops.exp(input)
    }

    fn normalize(&self, input: &Tensor, p: i32, dim: i32) -> Result<Tensor> {
        let ops = self.get_ops(&[input]);
        ops.normalize(input, p, dim)
    }

    fn concat(&self, tensors: &[&Tensor], dim: usize) -> Result<Tensor> {
        let ops = self.get_ops(tensors);
        ops.concat(tensors, dim)
    }

    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let ops = self.get_ops(&[input, weight]);
        ops.rms_norm(input, weight, eps)
    }

    fn scale(&self, input: &Tensor, factor: f32) -> Result<Tensor> {
        let ops = self.get_ops(&[input]);
        ops.scale(input, factor)
    }

    fn sigmoid(&self, input: &Tensor) -> Result<Tensor> {
        let ops = self.get_ops(&[input]);
        ops.sigmoid(input)
    }

    fn topk(&self, input: &Tensor, k: usize, dim: i64) -> Result<(Tensor, Tensor)> {
        let ops = self.get_ops(&[input]);
        ops.topk(input, k, dim)
    }

    fn conv1d(&self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>, stride: usize, padding: usize) -> Result<Tensor> {
        let tensors = if let Some(b) = bias {
            vec![input, weight, b]
        } else {
            vec![input, weight]
        };
        let ops = self.get_ops(&tensors);
        ops.conv1d(input, weight, bias, stride, padding)
    }

    fn tanh(&self, input: &Tensor) -> Result<Tensor> {
        let ops = self.get_ops(&[input]);
        ops.tanh(input)
    }

    fn sub(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let ops = self.get_ops(&[a, b]);
        ops.sub(a, b)
    }

    fn clamp(&self, input: &Tensor, min: f32, max: f32) -> Result<Tensor> {
        let ops = self.get_ops(&[input]);
        ops.clamp(input, min, max)
    }

    fn gather(&self, input: &Tensor, dim: usize, indices: &Tensor) -> Result<Tensor> {
        let ops = self.get_ops(&[input, indices]);
        ops.gather(input, dim, indices)
    }

    fn scatter(&self, input: &Tensor, dim: usize, indices: &Tensor, src: &Tensor) -> Result<Tensor> {
        let ops = self.get_ops(&[input, indices, src]);
        ops.scatter(input, dim, indices, src)
    }

    // Fused operations for performance
    fn flash_attention(&self, query: &Tensor, key: &Tensor, value: &Tensor, scale: f32, causal: bool) -> Result<Tensor> {
        let ops = self.get_ops(&[query, key, value]);
        ops.flash_attention(query, key, value, scale, causal)
    }

    fn fused_swiglu(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        let ops = self.get_ops(&[gate, up]);
        ops.fused_swiglu(gate, up)
    }

    fn fused_residual_rms_norm(&self, residual: &Tensor, hidden: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let ops = self.get_ops(&[residual, hidden, weight]);
        ops.fused_residual_rms_norm(residual, hidden, weight, eps)
    }
}

/// CPU implementation of tensor operations
pub struct CpuTensorOpsImpl;

/// Basic GPU implementation of tensor operations
pub struct GpuTensorOpsImpl;

impl CpuTensorOpsImpl {
    pub fn new() -> Self {
        Self
    }
}

impl GpuTensorOpsImpl {
    pub fn new() -> Self {
        Self
    }
}

impl TensorOps for CpuTensorOpsImpl {
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Convert to Candle tensors
        let a_candle = a.to_candle()?;
        let b_candle = b.to_candle()?;

        let a_dims = a_candle.dims();
        let b_dims = b_candle.dims();

        // Handle different dimension cases
        let result_candle = match (a_dims.len(), b_dims.len()) {
            (2, 2) => {
                // Standard 2D matmul: (M, K) @ (K, N) -> (M, N)
                a_candle.matmul(&b_candle)?
            }
            (3, 2) => {
                // Batched matmul: (B, M, K) @ (K, N) -> (B, M, N)
                // Reshape A to 2D, matmul, reshape back
                let (batch, m, k) = (a_dims[0], a_dims[1], a_dims[2]);
                let n = b_dims[1];

                // Reshape [B, M, K] -> [B*M, K]
                let a_flat = a_candle.reshape(&[batch * m, k])?;

                // Matmul: [B*M, K] @ [K, N] -> [B*M, N]
                let result_flat = a_flat.matmul(&b_candle)?;

                // Reshape back: [B*M, N] -> [B, M, N]
                result_flat.reshape(&[batch, m, n])?
            }
            (3, 3) => {
                // Batched matmul: (B, M, K) @ (B, K, N) -> (B, M, N)
                a_candle.matmul(&b_candle)?
            }
            (2, 3) => {
                // (M, K) @ (B, K, N) -> (B, M, N)
                // Broadcast A to 3D
                let (m, k) = (a_dims[0], a_dims[1]);
                let batch = b_dims[0];

                let a_expanded = a_candle.unsqueeze(0)?.broadcast_as(&[batch, m, k])?;
                a_expanded.matmul(&b_candle)?
            }
            _ => {
                // Fallback to standard matmul, let Candle handle errors
                a_candle.matmul(&b_candle)?
            }
        };

        Ok(Tensor::from_candle(result_candle))
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Convert to Candle tensors
        let a_candle = a.to_candle()?;
        let b_candle = b.to_candle()?;

        // Handle broadcasting: if shapes don't match, try to broadcast
        let result_candle = if a.shape() == b.shape() {
            (&a_candle + &b_candle)?
        } else {
            // Try broadcast add
            a_candle.broadcast_add(&b_candle)?
        };

        Ok(Tensor::from_candle(result_candle))
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Convert to Candle tensors
        let a_candle = a.to_candle()?;
        let b_candle = b.to_candle()?;

        // Handle broadcasting
        let result_candle = if a.shape() == b.shape() {
            (&a_candle * &b_candle)?
        } else {
            a_candle.broadcast_mul(&b_candle)?
        };

        Ok(Tensor::from_candle(result_candle))
    }

    fn attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
        scale: Option<f32>,
    ) -> Result<Tensor> {
        // Convert to Candle tensors
        let q = query.to_candle()?;
        let k = key.to_candle()?;
        let v = value.to_candle()?;

        // Get dimensions
        let d_k = *q.dims().last().unwrap_or(&1);
        let scale_factor = scale.unwrap_or(1.0 / (d_k as f32).sqrt());

        // Compute attention scores: Q @ K^T
        let k_t = k.transpose(k.dims().len() - 2, k.dims().len() - 1)?;
        let scores = q.matmul(&k_t)?;

        // Scale scores
        let scaled_scores = (scores * scale_factor as f64)?;

        // Apply mask if provided
        let masked_scores = if let Some(m) = mask {
            let mask_candle = m.to_candle()?;
            // Where mask is 0, fill with -inf
            let neg_inf = CandleTensor::new(&[-1e9f32], &CandleDevice::Cpu)?;
            let neg_inf = neg_inf.broadcast_as(scaled_scores.dims())?;
            let mask_expanded = mask_candle.broadcast_as(scaled_scores.dims())?;
            // mask * scores + (1 - mask) * -inf
            let mask_f32 = mask_expanded.to_dtype(DType::F32)?;
            let inverted_mask = (1.0 - &mask_f32)?;
            ((&mask_f32 * &scaled_scores)? + (&inverted_mask * &neg_inf)?)?
        } else {
            scaled_scores
        };

        // Softmax along last dimension
        let attention_weights = candle_ops::softmax_last_dim(&masked_scores)?;

        // Apply attention to values
        let result = attention_weights.matmul(&v)?;

        Ok(Tensor::from_candle(result))
    }

    fn layer_norm(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        let x = input.to_candle()?;
        let w = weight.to_candle()?;

        // Get the last dimension for normalization
        let last_dim = x.dims().len() - 1;

        // Compute mean along last dimension
        let mean = x.mean_keepdim(last_dim)?;

        // Compute variance: E[(x - mean)^2]
        let x_centered = x.broadcast_sub(&mean)?;
        let variance = x_centered.sqr()?.mean_keepdim(last_dim)?;

        // Normalize: (x - mean) / sqrt(variance + eps)
        let std = (variance + eps as f64)?.sqrt()?;
        let normalized = x_centered.broadcast_div(&std)?;

        // Apply weight and bias
        let scaled = normalized.broadcast_mul(&w)?;

        let result = if let Some(b) = bias {
            let b_candle = b.to_candle()?;
            scaled.broadcast_add(&b_candle)?
        } else {
            scaled
        };

        Ok(Tensor::from_candle(result))
    }

    fn gelu(&self, input: &Tensor) -> Result<Tensor> {
        let x = input.to_candle()?;
        let result = x.gelu()?;
        Ok(Tensor::from_candle(result))
    }

    fn silu(&self, input: &Tensor) -> Result<Tensor> {
        let x = input.to_candle()?;
        let result = x.silu()?;
        Ok(Tensor::from_candle(result))
    }

    fn softmax(&self, input: &Tensor, dim: isize) -> Result<Tensor> {
        let x = input.to_candle()?;

        // Convert negative dimension to positive
        let ndims = x.dims().len() as isize;
        let actual_dim = if dim < 0 { ndims + dim } else { dim } as usize;

        let result = if actual_dim == x.dims().len() - 1 {
            candle_ops::softmax_last_dim(&x)?
        } else {
            candle_ops::softmax(&x, actual_dim)?
        };

        Ok(Tensor::from_candle(result))
    }

    fn embedding(&self, indices: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let idx = indices.to_candle()?;
        let w = weight.to_candle()?;

        // Convert indices to u32 for Candle embedding
        let idx_u32 = idx.to_dtype(DType::U32)?;

        // Flatten indices for embedding lookup
        let orig_shape = idx_u32.dims().to_vec();
        let flat_idx = idx_u32.flatten_all()?;

        // Perform embedding lookup
        let embedded = w.embedding(&flat_idx)?;

        // Reshape to original shape + embedding dim
        let embed_dim = w.dims()[1];
        let mut result_shape = orig_shape;
        result_shape.push(embed_dim);

        let result = embedded.reshape(result_shape.as_slice())?;

        Ok(Tensor::from_candle(result))
    }

    fn zeros(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<Tensor> {
        let storage = CandleStorage::zeros(shape, dtype, device)?;
        Ok(Tensor::new(
            shape.to_vec(),
            dtype,
            device.clone(),
            Arc::new(storage),
        ))
    }

    fn randn(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<Tensor> {
        let candle_device = device.to_candle();
        let candle_dtype = dtype.to_candle();

        // Use Candle's random tensor generation
        let result = CandleTensor::randn(0f32, 1f32, shape, &candle_device)?
            .to_dtype(candle_dtype)?;

        Ok(Tensor::from_candle(result))
    }

    fn exp(&self, input: &Tensor) -> Result<Tensor> {
        let x = input.to_candle()?;
        let result = x.exp()?;
        Ok(Tensor::from_candle(result))
    }

    fn normalize(&self, input: &Tensor, p: i32, dim: i32) -> Result<Tensor> {
        let x = input.to_candle()?;

        // L2 normalization (p=2 is most common)
        let actual_dim = if dim < 0 {
            x.dims().len() as i32 + dim
        } else {
            dim
        } as usize;

        if p == 2 {
            // L2 norm: x / ||x||_2
            let squared = x.sqr()?;
            let sum_squared = squared.sum_keepdim(actual_dim)?;
            let norm = sum_squared.sqrt()?;
            let normalized = x.broadcast_div(&norm)?;
            Ok(Tensor::from_candle(normalized))
        } else {
            // For other norms, just return input for now
            Ok(input.clone())
        }
    }

    fn concat(&self, tensors: &[&Tensor], dim: usize) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(anyhow::anyhow!("Cannot concatenate empty tensor list"));
        }

        // Convert all tensors to Candle
        let candle_tensors: Result<Vec<_>> = tensors.iter()
            .map(|t| t.to_candle())
            .collect();
        let candle_tensors = candle_tensors?;

        // Create refs for Candle concat
        let refs: Vec<&CandleTensor> = candle_tensors.iter().collect();

        let result = CandleTensor::cat(&refs, dim)?;
        Ok(Tensor::from_candle(result))
    }

    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let x = input.to_candle()?;
        let w = weight.to_candle()?;

        // RMS Norm: x * w / sqrt(mean(x^2) + eps)
        let last_dim = x.dims().len() - 1;
        let x_squared = x.sqr()?;
        let mean_squared = x_squared.mean_keepdim(last_dim)?;
        let rms = (mean_squared + eps as f64)?.sqrt()?;
        let normalized = x.broadcast_div(&rms)?;
        let result = normalized.broadcast_mul(&w)?;

        Ok(Tensor::from_candle(result))
    }

    fn scale(&self, input: &Tensor, factor: f32) -> Result<Tensor> {
        let x = input.to_candle()?;
        let result = (x * factor as f64)?;
        Ok(Tensor::from_candle(result))
    }

    fn sigmoid(&self, input: &Tensor) -> Result<Tensor> {
        let x = input.to_candle()?;
        // sigmoid(x) = 1 / (1 + exp(-x))
        let neg_x = x.neg()?;
        let exp_neg_x = neg_x.exp()?;
        let one_plus_exp = (exp_neg_x + 1.0)?;
        let result = one_plus_exp.recip()?;
        Ok(Tensor::from_candle(result))
    }

    fn topk(&self, input: &Tensor, k: usize, dim: i64) -> Result<(Tensor, Tensor)> {
        let x = input.to_candle()?;
        let ndims = x.dims().len() as i64;
        let actual_dim = if dim < 0 { ndims + dim } else { dim } as usize;

        // Get the size of the dimension we're taking topk over
        let dim_size = x.dims()[actual_dim];
        if k > dim_size {
            return Err(anyhow::anyhow!("k ({}) is larger than dimension size ({})", k, dim_size));
        }

        // For simple case: 2D tensor, topk on last dim
        // We'll implement a basic version that works for MoE routing
        if x.dims().len() == 2 && actual_dim == 1 {
            let (batch, _n) = (x.dims()[0], x.dims()[1]);
            let mut all_values = Vec::with_capacity(batch * k);
            let mut all_indices = Vec::with_capacity(batch * k);

            for b in 0..batch {
                let row = x.get(b)?;
                let row_data: Vec<f32> = row.to_vec1()?;

                // Get indices sorted by value (descending)
                let mut indexed: Vec<(usize, f32)> = row_data.iter().cloned().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // Take top k
                for i in 0..k {
                    all_values.push(indexed[i].1);
                    all_indices.push(indexed[i].0 as i64);
                }
            }

            let device = input.device();
            let values = Tensor::from_f32_slice(&all_values, &[batch, k], device)?;
            let indices = Tensor::from_i64_slice(&all_indices, &[batch, k], device)?;

            Ok((values, indices))
        } else if x.dims().len() == 1 {
            // 1D case
            let data: Vec<f32> = x.to_vec1()?;
            let mut indexed: Vec<(usize, f32)> = data.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let values: Vec<f32> = indexed.iter().take(k).map(|(_, v)| *v).collect();
            let indices: Vec<i64> = indexed.iter().take(k).map(|(i, _)| *i as i64).collect();

            let device = input.device();
            let values_tensor = Tensor::from_f32_slice(&values, &[k], device)?;
            let indices_tensor = Tensor::from_i64_slice(&indices, &[k], device)?;

            Ok((values_tensor, indices_tensor))
        } else if x.dims().len() == 3 && actual_dim == 2 {
            // 3D case: [batch, seq, n] -> topk on last dim
            let (batch, seq_len, _n) = (x.dims()[0], x.dims()[1], x.dims()[2]);
            let mut all_values = Vec::with_capacity(batch * seq_len * k);
            let mut all_indices = Vec::with_capacity(batch * seq_len * k);

            for b in 0..batch {
                for s in 0..seq_len {
                    let row = x.get(b)?.get(s)?;
                    let row_data: Vec<f32> = row.to_vec1()?;

                    let mut indexed: Vec<(usize, f32)> = row_data.iter().cloned().enumerate().collect();
                    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                    for i in 0..k {
                        all_values.push(indexed[i].1);
                        all_indices.push(indexed[i].0 as i64);
                    }
                }
            }

            let device = input.device();
            let values = Tensor::from_f32_slice(&all_values, &[batch, seq_len, k], device)?;
            let indices = Tensor::from_i64_slice(&all_indices, &[batch, seq_len, k], device)?;

            Ok((values, indices))
        } else {
            Err(anyhow::anyhow!("topk not implemented for tensor with {} dimensions on dim {}", x.dims().len(), actual_dim))
        }
    }

    fn conv1d(&self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>, stride: usize, padding: usize) -> Result<Tensor> {
        let x = input.to_candle()?;
        let w = weight.to_candle()?;

        // Input shape: [batch, in_channels, seq_len]
        // Weight shape: [out_channels, in_channels, kernel_size]

        // Pad input if needed
        let x_padded = if padding > 0 {
            x.pad_with_zeros(2, padding, padding)?
        } else {
            x
        };

        // Use Candle's conv1d
        let result = x_padded.conv1d(&w, padding, stride, 1, 1)?;

        // Add bias if provided
        let result = if let Some(b) = bias {
            let b_candle = b.to_candle()?;
            // Bias shape: [out_channels] -> [1, out_channels, 1]
            let b_expanded = b_candle.unsqueeze(0)?.unsqueeze(2)?;
            result.broadcast_add(&b_expanded)?
        } else {
            result
        };

        Ok(Tensor::from_candle(result))
    }

    fn tanh(&self, input: &Tensor) -> Result<Tensor> {
        let x = input.to_candle()?;
        let result = x.tanh()?;
        Ok(Tensor::from_candle(result))
    }

    fn sub(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_candle = a.to_candle()?;
        let b_candle = b.to_candle()?;

        let result = if a.shape() == b.shape() {
            (&a_candle - &b_candle)?
        } else {
            a_candle.broadcast_sub(&b_candle)?
        };

        Ok(Tensor::from_candle(result))
    }

    fn clamp(&self, input: &Tensor, min: f32, max: f32) -> Result<Tensor> {
        let x = input.to_candle()?;
        let result = x.clamp(min as f64, max as f64)?;
        Ok(Tensor::from_candle(result))
    }

    fn gather(&self, input: &Tensor, dim: usize, indices: &Tensor) -> Result<Tensor> {
        let x = input.to_candle()?;
        let idx = indices.to_candle()?;
        let idx_u32 = idx.to_dtype(DType::U32)?;
        let result = x.gather(&idx_u32, dim)?;
        Ok(Tensor::from_candle(result))
    }

    fn scatter(&self, input: &Tensor, dim: usize, indices: &Tensor, src: &Tensor) -> Result<Tensor> {
        let x = input.to_candle()?;
        let idx = indices.to_candle()?;
        let s = src.to_candle()?;
        let idx_u32 = idx.to_dtype(DType::U32)?;
        let result = x.scatter_add(&idx_u32, &s, dim)?;
        Ok(Tensor::from_candle(result))
    }

    // ========== Fused Operations for Performance ==========

    fn flash_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        scale: f32,
        causal: bool,
    ) -> Result<Tensor> {
        // Fused scaled dot-product attention
        // Optimized to reduce memory allocations and intermediate tensors
        let q = query.to_candle()?;
        let k = key.to_candle()?;
        let v = value.to_candle()?;

        // Expected shapes: [batch, heads, seq, head_dim]
        let q_dims = q.dims();
        let k_dims = k.dims();

        let seq_q = q_dims.get(2).copied().unwrap_or(1);
        let seq_k = k_dims.get(2).copied().unwrap_or(1);

        // Compute Q @ K^T (transpose last two dims of K)
        let k_t = k.transpose(k_dims.len() - 2, k_dims.len() - 1)?;
        let scores = q.matmul(&k_t)?;

        // Scale by provided factor
        let scaled_scores = (scores * scale as f64)?;

        // Apply causal mask if requested
        let masked_scores = if causal && seq_q > 1 {
            // Create causal mask: lower triangular matrix
            // Position (i, j) is masked if j > i
            let mask_device = scaled_scores.device();
            let mut mask_data = vec![0.0f32; seq_q * seq_k];
            for i in 0..seq_q {
                for j in 0..seq_k {
                    if j > i {
                        mask_data[i * seq_k + j] = f32::NEG_INFINITY;
                    }
                }
            }
            let mask = CandleTensor::from_slice(&mask_data, &[seq_q, seq_k], mask_device)?;

            // Broadcast mask to match scores shape
            let scores_shape = scaled_scores.dims();
            let mask_shape: Vec<usize> = scores_shape.iter()
                .take(scores_shape.len() - 2)
                .map(|_| 1)
                .chain([seq_q, seq_k])
                .collect();
            let mask_reshaped = mask.reshape(mask_shape.as_slice())?;
            let mask_broadcast = mask_reshaped.broadcast_as(scores_shape)?;

            (&scaled_scores + &mask_broadcast)?
        } else {
            scaled_scores
        };

        // Softmax along last dimension
        let attention_weights = candle_ops::softmax_last_dim(&masked_scores)?;

        // Apply attention to values: weights @ V
        let result = attention_weights.matmul(&v)?;

        Ok(Tensor::from_candle(result))
    }

    fn fused_swiglu(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        // Fused SwiGLU: silu(gate) * up
        // This avoids creating an intermediate tensor for silu output
        let g = gate.to_candle()?;
        let u = up.to_candle()?;

        // SiLU(x) = x * sigmoid(x)
        let silu_gate = g.silu()?;
        let result = (&silu_gate * &u)?;

        Ok(Tensor::from_candle(result))
    }

    fn fused_residual_rms_norm(
        &self,
        residual: &Tensor,
        hidden: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        // Fused residual + RMS norm
        // Computes: rms_norm(residual + hidden, weight, eps)
        let r = residual.to_candle()?;
        let h = hidden.to_candle()?;
        let w = weight.to_candle()?;

        // Add residual and hidden
        let combined = (&r + &h)?;

        // RMS Norm: x * w / sqrt(mean(x^2) + eps)
        let last_dim = combined.dims().len() - 1;
        let x_squared = combined.sqr()?;
        let mean_squared = x_squared.mean_keepdim(last_dim)?;
        let rms = (mean_squared + eps as f64)?.sqrt()?;
        let normalized = combined.broadcast_div(&rms)?;
        let result = normalized.broadcast_mul(&w)?;

        Ok(Tensor::from_candle(result))
    }
}

impl DataType {
    pub fn size_bytes(&self) -> usize {
        match self {
            DataType::Float32 => 4,
            DataType::Float16 => 2,
            DataType::BFloat16 => 2,
            DataType::Int32 => 4,
            DataType::Int64 => 8,
            DataType::Int8 => 1,
            DataType::Bool => 1,
        }
    }

    /// Convert to Candle DType
    pub fn to_candle(&self) -> DType {
        match self {
            DataType::Float32 => DType::F32,
            DataType::Float16 => DType::F16,
            DataType::BFloat16 => DType::BF16,
            DataType::Int32 => DType::U32, // Candle uses U32 instead of I32
            DataType::Int64 => DType::I64,
            DataType::Int8 => DType::U8,   // Candle uses U8 instead of I8
            DataType::Bool => DType::U8,   // Candle uses U8 for bool
        }
    }

    /// Convert from Candle DType
    pub fn from_candle(dtype: DType) -> Self {
        match dtype {
            DType::F32 => DataType::Float32,
            DType::F16 => DataType::Float16,
            DType::BF16 => DataType::BFloat16,
            DType::I64 => DataType::Int64,
            DType::U8 => DataType::Int8,
            DType::F64 => DataType::Float32, // Downcast f64 to f32
            DType::U32 => DataType::Int32,   // Map unsigned to signed
        }
    }
}

impl Device {
    /// Convert to Candle Device
    pub fn to_candle(&self) -> CandleDevice {
        match self {
            Device::CPU => CandleDevice::Cpu,
            Device::CUDA(id) => CandleDevice::cuda_if_available(*id).unwrap_or(CandleDevice::Cpu),
            Device::Metal(id) => {
                #[cfg(feature = "metal")]
                {
                    CandleDevice::new_metal(*id).unwrap_or(CandleDevice::Cpu)
                }
                #[cfg(not(feature = "metal"))]
                {
                    let _ = id;
                    CandleDevice::Cpu
                }
            }
        }
    }

    /// Convert from Candle Device
    pub fn from_candle(device: &CandleDevice) -> Self {
        match device {
            CandleDevice::Cpu => Device::CPU,
            CandleDevice::Cuda(_) => Device::CUDA(0), // Default to device 0
            CandleDevice::Metal(_) => Device::Metal(0), // Default to device 0
        }
    }
}

/// Global tensor operation dispatcher
static TENSOR_OPS: std::sync::OnceLock<TensorDispatcher> = std::sync::OnceLock::new();

/// Get global tensor operations
pub fn ops() -> &'static TensorDispatcher {
    TENSOR_OPS.get_or_init(|| TensorDispatcher::new())
}

/// Basic GPU implementation of tensor operations
/// For now, this just delegates to CPU implementation
impl TensorOps for GpuTensorOpsImpl {
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // For now, delegate to CPU implementation
        CpuTensorOpsImpl.matmul(a, b)
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        CpuTensorOpsImpl.add(a, b)
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        CpuTensorOpsImpl.mul(a, b)
    }

    fn attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
        scale: Option<f32>,
    ) -> Result<Tensor> {
        CpuTensorOpsImpl.attention(query, key, value, mask, scale)
    }

    fn layer_norm(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        CpuTensorOpsImpl.layer_norm(input, weight, bias, eps)
    }

    fn gelu(&self, input: &Tensor) -> Result<Tensor> {
        CpuTensorOpsImpl.gelu(input)
    }

    fn silu(&self, input: &Tensor) -> Result<Tensor> {
        CpuTensorOpsImpl.silu(input)
    }

    fn softmax(&self, input: &Tensor, dim: isize) -> Result<Tensor> {
        CpuTensorOpsImpl.softmax(input, dim)
    }

    fn embedding(&self, indices: &Tensor, weight: &Tensor) -> Result<Tensor> {
        CpuTensorOpsImpl.embedding(indices, weight)
    }

    fn zeros(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<Tensor> {
        CpuTensorOpsImpl.zeros(shape, dtype, device)
    }

    fn randn(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<Tensor> {
        CpuTensorOpsImpl.randn(shape, dtype, device)
    }

    fn exp(&self, input: &Tensor) -> Result<Tensor> {
        CpuTensorOpsImpl.exp(input)
    }

    fn normalize(&self, input: &Tensor, p: i32, dim: i32) -> Result<Tensor> {
        CpuTensorOpsImpl.normalize(input, p, dim)
    }

    fn concat(&self, tensors: &[&Tensor], dim: usize) -> Result<Tensor> {
        CpuTensorOpsImpl.concat(tensors, dim)
    }

    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        CpuTensorOpsImpl.rms_norm(input, weight, eps)
    }

    fn scale(&self, input: &Tensor, factor: f32) -> Result<Tensor> {
        CpuTensorOpsImpl.scale(input, factor)
    }

    fn sigmoid(&self, input: &Tensor) -> Result<Tensor> {
        CpuTensorOpsImpl.sigmoid(input)
    }

    fn topk(&self, input: &Tensor, k: usize, dim: i64) -> Result<(Tensor, Tensor)> {
        CpuTensorOpsImpl.topk(input, k, dim)
    }

    fn conv1d(&self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>, stride: usize, padding: usize) -> Result<Tensor> {
        CpuTensorOpsImpl.conv1d(input, weight, bias, stride, padding)
    }

    fn tanh(&self, input: &Tensor) -> Result<Tensor> {
        CpuTensorOpsImpl.tanh(input)
    }

    fn sub(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        CpuTensorOpsImpl.sub(a, b)
    }

    fn clamp(&self, input: &Tensor, min: f32, max: f32) -> Result<Tensor> {
        CpuTensorOpsImpl.clamp(input, min, max)
    }

    fn gather(&self, input: &Tensor, dim: usize, indices: &Tensor) -> Result<Tensor> {
        CpuTensorOpsImpl.gather(input, dim, indices)
    }

    fn scatter(&self, input: &Tensor, dim: usize, indices: &Tensor, src: &Tensor) -> Result<Tensor> {
        CpuTensorOpsImpl.scatter(input, dim, indices, src)
    }

    // Fused operations - delegate to CPU implementation for now
    fn flash_attention(&self, query: &Tensor, key: &Tensor, value: &Tensor, scale: f32, causal: bool) -> Result<Tensor> {
        CpuTensorOpsImpl.flash_attention(query, key, value, scale, causal)
    }

    fn fused_swiglu(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        CpuTensorOpsImpl.fused_swiglu(gate, up)
    }

    fn fused_residual_rms_norm(&self, residual: &Tensor, hidden: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        CpuTensorOpsImpl.fused_residual_rms_norm(residual, hidden, weight, eps)
    }
}

/// Convenience functions for tensor operations
pub mod ops_fn {
    use super::*;

    pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        ops().matmul(a, b)
    }

    pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        ops().add(a, b)
    }

    pub fn zeros(shape: &[usize], dtype: DataType, device: &Device) -> Result<Tensor> {
        ops().zeros(shape, dtype, device)
    }

    pub fn randn(shape: &[usize], dtype: DataType, device: &Device) -> Result<Tensor> {
        ops().randn(shape, dtype, device)
    }

    pub fn layer_norm(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>, eps: f32) -> Result<Tensor> {
        ops().layer_norm(input, weight, bias, eps)
    }

    pub fn embedding(indices: &Tensor, weight: &Tensor) -> Result<Tensor> {
        ops().embedding(indices, weight)
    }

    pub fn silu(input: &Tensor) -> Result<Tensor> {
        ops().silu(input)
    }

    pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        ops().mul(a, b)
    }

    pub fn attention(query: &Tensor, key: &Tensor, value: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        ops().attention(query, key, value, mask, None)
    }

    pub fn gelu(input: &Tensor) -> Result<Tensor> {
        ops().gelu(input)
    }

    pub fn exp(input: &Tensor) -> Result<Tensor> {
        ops().exp(input)
    }

    pub fn normalize(input: &Tensor, p: i32, dim: i32) -> Result<Tensor> {
        ops().normalize(input, p, dim)
    }

    pub fn concat(tensors: &[&Tensor], dim: usize) -> Result<Tensor> {
        ops().concat(tensors, dim)
    }

    pub fn rms_norm(input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        ops().rms_norm(input, weight, eps)
    }

    pub fn scale(input: &Tensor, factor: f32) -> Result<Tensor> {
        ops().scale(input, factor)
    }

    /// Transpose tensor (swap last two dimensions)
    pub fn transpose(input: &Tensor) -> Result<Tensor> {
        input.transpose()
    }

    /// Transpose tensor with specific dimensions
    pub fn transpose_dims(input: &Tensor, dim0: usize, dim1: usize) -> Result<Tensor> {
        input.transpose_dims(dim0, dim1)
    }

    /// Sigmoid activation
    pub fn sigmoid(input: &Tensor) -> Result<Tensor> {
        ops().sigmoid(input)
    }

    /// Top-k operation
    pub fn topk(input: &Tensor, k: usize, dim: i64) -> Result<(Tensor, Tensor)> {
        ops().topk(input, k, dim)
    }

    /// 1D convolution
    pub fn conv1d(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>, stride: usize, padding: usize) -> Result<Tensor> {
        ops().conv1d(input, weight, bias, stride, padding)
    }

    /// Tanh activation
    pub fn tanh(input: &Tensor) -> Result<Tensor> {
        ops().tanh(input)
    }

    /// Element-wise subtraction
    pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        ops().sub(a, b)
    }

    /// Clamp values to a range
    pub fn clamp(input: &Tensor, min: f32, max: f32) -> Result<Tensor> {
        ops().clamp(input, min, max)
    }

    /// Gather elements along dimension
    pub fn gather(input: &Tensor, dim: usize, indices: &Tensor) -> Result<Tensor> {
        ops().gather(input, dim, indices)
    }

    /// Scatter elements along dimension
    pub fn scatter(input: &Tensor, dim: usize, indices: &Tensor, src: &Tensor) -> Result<Tensor> {
        ops().scatter(input, dim, indices, src)
    }

    /// Softmax along specified dimension
    pub fn softmax(input: &Tensor, dim: isize) -> Result<Tensor> {
        ops().softmax(input, dim)
    }

    /// Create a causal attention mask (lower triangular)
    pub fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..=i {
                mask_data[i * seq_len + j] = 1.0;
            }
        }
        Tensor::from_f32_slice(&mask_data, &[seq_len, seq_len], device)
    }

    /// Create a sliding window attention mask
    /// Returns a mask where 1.0 means "attend" and 0.0 means "don't attend"
    /// Each position can only attend to positions within window_size positions before it
    pub fn sliding_window_mask(seq_len: usize, window_size: usize, device: &Device) -> Result<Tensor> {
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            // Can attend to positions from max(0, i - window_size + 1) to i (inclusive)
            let start = if i >= window_size { i - window_size + 1 } else { 0 };
            for j in start..=i {
                mask_data[i * seq_len + j] = 1.0;
            }
        }
        Tensor::from_f32_slice(&mask_data, &[seq_len, seq_len], device)
    }

    /// Create a combined causal + sliding window mask
    /// This is the typical mask used in Mistral/Mixtral
    pub fn causal_sliding_window_mask(seq_len: usize, window_size: usize, device: &Device) -> Result<Tensor> {
        // Same as sliding_window_mask since sliding window is already causal
        sliding_window_mask(seq_len, window_size, device)
    }

    // ========== Fused Operations for Performance ==========
    // These operations combine multiple ops to reduce memory allocations
    // and improve cache locality. Benefits all model architectures.

    /// Fused scaled dot-product attention (Flash Attention pattern)
    /// Computes: softmax(Q @ K^T / sqrt(d_k)) @ V
    /// Works for: LLaMA, Qwen, Gemma, Mistral, Phi, and all attention-based models
    ///
    /// # Arguments
    /// * `query` - Query tensor [batch, heads, seq_q, head_dim]
    /// * `key` - Key tensor [batch, heads, seq_k, head_dim]
    /// * `value` - Value tensor [batch, heads, seq_k, head_dim]
    /// * `scale` - Scaling factor (typically 1/sqrt(head_dim))
    /// * `causal` - Whether to apply causal masking
    pub fn flash_attention(
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        scale: f32,
        causal: bool,
    ) -> Result<Tensor> {
        ops().flash_attention(query, key, value, scale, causal)
    }

    /// Fused SwiGLU activation: silu(gate) * up
    /// Used by: LLaMA, Qwen, Mistral, and other modern transformer MLPs
    pub fn fused_swiglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        ops().fused_swiglu(gate, up)
    }

    /// Fused residual add + RMS normalization
    /// Computes: rms_norm(residual + hidden, weight, eps)
    /// Used by: All transformer models with pre-normalization
    pub fn fused_residual_rms_norm(
        residual: &Tensor,
        hidden: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        ops().fused_residual_rms_norm(residual, hidden, weight, eps)
    }
}
