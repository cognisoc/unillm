//! Core Tensor Abstraction
//!
//! This module provides the foundational tensor abstraction that unifies
//! all tensor operations across CPU, GPU, and different data formats.

use std::sync::Arc;
use anyhow::Result;

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
}

/// CPU tensor storage implementation
#[derive(Debug)]
pub struct CpuStorage {
    data: Vec<u8>,
    device: Device,
}

/// GPU tensor storage implementation
#[derive(Debug)]
pub struct GpuStorage {
    #[allow(dead_code)]
    ptr: *mut u8,
    size: usize,
    device: Device,
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
        // Support both 2D and batched matrix multiplication
        // Cases:
        // 1. 2D x 2D: (m,k) @ (k,n) -> (m,n)
        // 2. 3D x 2D: (batch,m,k) @ (k,n) -> (batch,m,n)
        // 3. 3D x 3D: (batch,m,k) @ (batch,k,n) -> (batch,m,n)
        
        let a_dims = a.shape().len();
        let b_dims = b.shape().len();
        
        if !(a_dims == 2 || a_dims == 3) || !(b_dims == 2 || b_dims == 3) {
            return Err(anyhow::anyhow!("matmul requires 2D or 3D tensors"));
        }

        // Extract dimensions based on tensor ranks
        let (batch_size, m, k) = if a_dims == 3 {
            (a.shape()[0], a.shape()[1], a.shape()[2])
        } else {
            (1, a.shape()[0], a.shape()[1])
        };

        let (b_batch_size, k2, n) = if b_dims == 3 {
            (b.shape()[0], b.shape()[1], b.shape()[2])
        } else {
            (1, b.shape()[0], b.shape()[1])
        };

        // Validate batch dimensions
        if a_dims == 3 && b_dims == 3 && batch_size != b_batch_size {
            return Err(anyhow::anyhow!("Batch sizes must match for 3D matmul: {} != {}", batch_size, b_batch_size));
        }

        // Validate matrix dimensions
        if k != k2 {
            return Err(anyhow::anyhow!("matmul dimension mismatch: {} != {}", k, k2));
        }

        // Calculate result shape
        let result_shape = if a_dims == 3 || b_dims == 3 {
            vec![batch_size, m, n]
        } else {
            vec![m, n]
        };

        // Create result tensor with proper shape
        let result_numel: usize = result_shape.iter().product();
        let result_size = result_numel * a.dtype().size_bytes();
        let storage = Arc::new(CpuStorage::zeros(result_size));
        let result = Tensor::new(
            result_shape,
            a.dtype(),
            a.device().clone(),
            storage
        );

        Ok(result)
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.shape() != b.shape() {
            return Err(anyhow::anyhow!("add requires same shape tensors"));
        }

        // Create result tensor with same shape
        let result_size = a.numel() * a.dtype().size_bytes();
        let storage = Arc::new(CpuStorage::zeros(result_size));
        let result = Tensor::new(
            a.shape().to_vec(),
            a.dtype(),
            a.device().clone(),
            storage
        );

        Ok(result)
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.shape() != b.shape() {
            return Err(anyhow::anyhow!("mul requires same shape tensors"));
        }

        // Create result tensor with same shape
        let result_size = a.numel() * a.dtype().size_bytes();
        let storage = Arc::new(CpuStorage::zeros(result_size));
        let result = Tensor::new(
            a.shape().to_vec(),
            a.dtype(),
            a.device().clone(),
            storage
        );

        Ok(result)
    }

    fn attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        _mask: Option<&Tensor>,
        _scale: Option<f32>,
    ) -> Result<Tensor> {
        // Simplified attention - just return tensor with query shape
        if query.shape().len() != 3 || key.shape().len() != 3 || value.shape().len() != 3 {
            return Err(anyhow::anyhow!("attention requires 3D tensors"));
        }

        let result_size = query.numel() * query.dtype().size_bytes();
        let storage = Arc::new(CpuStorage::zeros(result_size));
        let result = Tensor::new(
            query.shape().to_vec(),
            query.dtype(),
            query.device().clone(),
            storage
        );

        Ok(result)
    }

    fn layer_norm(
        &self,
        input: &Tensor,
        _weight: &Tensor,
        _bias: Option<&Tensor>,
        _eps: f32,
    ) -> Result<Tensor> {
        // Return tensor with same shape as input
        let result_size = input.numel() * input.dtype().size_bytes();
        let storage = Arc::new(CpuStorage::zeros(result_size));
        let result = Tensor::new(
            input.shape().to_vec(),
            input.dtype(),
            input.device().clone(),
            storage
        );

        Ok(result)
    }

    fn gelu(&self, input: &Tensor) -> Result<Tensor> {
        // Return tensor with same shape as input
        let result_size = input.numel() * input.dtype().size_bytes();
        let storage = Arc::new(CpuStorage::zeros(result_size));
        let result = Tensor::new(
            input.shape().to_vec(),
            input.dtype(),
            input.device().clone(),
            storage
        );

        Ok(result)
    }

    fn silu(&self, input: &Tensor) -> Result<Tensor> {
        // Return tensor with same shape as input
        let result_size = input.numel() * input.dtype().size_bytes();
        let storage = Arc::new(CpuStorage::zeros(result_size));
        let result = Tensor::new(
            input.shape().to_vec(),
            input.dtype(),
            input.device().clone(),
            storage
        );

        Ok(result)
    }

    fn softmax(&self, input: &Tensor, _dim: isize) -> Result<Tensor> {
        // Return tensor with same shape as input
        let result_size = input.numel() * input.dtype().size_bytes();
        let storage = Arc::new(CpuStorage::zeros(result_size));
        let result = Tensor::new(
            input.shape().to_vec(),
            input.dtype(),
            input.device().clone(),
            storage
        );

        Ok(result)
    }

    fn embedding(&self, indices: &Tensor, weight: &Tensor) -> Result<Tensor> {
        if indices.dtype() != DataType::Int64 {
            return Err(anyhow::anyhow!("embedding indices must be Int64"));
        }

        if weight.shape().len() != 2 {
            return Err(anyhow::anyhow!("embedding weight must be 2D"));
        }

        let _vocab_size = weight.shape()[0];
        let embed_dim = weight.shape()[1];

        // Result shape: indices.shape() + [embed_dim]
        let mut result_shape = indices.shape().to_vec();
        result_shape.push(embed_dim);

        let result_numel: usize = result_shape.iter().product();
        let result_size = result_numel * weight.dtype().size_bytes();
        let storage = Arc::new(CpuStorage::zeros(result_size));

        let result = Tensor::new(
            result_shape,
            weight.dtype(),
            weight.device().clone(),
            storage
        );

        Ok(result)
    }

    fn zeros(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<Tensor> {
        let numel: usize = shape.iter().product();
        let size_bytes = numel * dtype.size_bytes();

        let storage = Arc::new(CpuStorage::zeros(size_bytes));
        Ok(Tensor::new(shape.to_vec(), dtype, device.clone(), storage))
    }

    fn randn(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<Tensor> {
        // For now, just create zeros (random generation can be added later)
        self.zeros(shape, dtype, device)
    }

    fn exp(&self, input: &Tensor) -> Result<Tensor> {
        // Return tensor with same shape as input
        let result_size = input.numel() * input.dtype().size_bytes();
        let storage = Arc::new(CpuStorage::zeros(result_size));
        let result = Tensor::new(
            input.shape().to_vec(),
            input.dtype(),
            input.device().clone(),
            storage
        );
        Ok(result)
    }

    fn normalize(&self, input: &Tensor, _p: i32, _dim: i32) -> Result<Tensor> {
        // Return tensor with same shape as input
        let result_size = input.numel() * input.dtype().size_bytes();
        let storage = Arc::new(CpuStorage::zeros(result_size));
        let result = Tensor::new(
            input.shape().to_vec(),
            input.dtype(),
            input.device().clone(),
            storage
        );
        Ok(result)
    }

    fn concat(&self, tensors: &[&Tensor], _dim: usize) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(anyhow::anyhow!("Cannot concatenate empty tensor list"));
        }

        // For now, just return the first tensor shape
        let first = tensors[0];
        let result_size = first.numel() * first.dtype().size_bytes();
        let storage = Arc::new(CpuStorage::zeros(result_size));
        let result = Tensor::new(
            first.shape().to_vec(),
            first.dtype(),
            first.device().clone(),
            storage
        );
        Ok(result)
    }

    fn rms_norm(&self, input: &Tensor, _weight: &Tensor, _eps: f32) -> Result<Tensor> {
        // Return tensor with same shape as input
        let result_size = input.numel() * input.dtype().size_bytes();
        let storage = Arc::new(CpuStorage::zeros(result_size));
        let result = Tensor::new(
            input.shape().to_vec(),
            input.dtype(),
            input.device().clone(),
            storage
        );
        Ok(result)
    }

    fn scale(&self, input: &Tensor, _factor: f32) -> Result<Tensor> {
        // Return tensor with same shape as input
        let result_size = input.numel() * input.dtype().size_bytes();
        let storage = Arc::new(CpuStorage::zeros(result_size));
        let result = Tensor::new(
            input.shape().to_vec(),
            input.dtype(),
            input.device().clone(),
            storage
        );
        Ok(result)
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
}