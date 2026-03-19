//! GPU-accelerated tensor operations using Candle
//!
//! This module provides GPU tensor operations with automatic fallback to CPU.
//! Supports CUDA, Metal, and CPU backends through the candle library.

use crate::types::*;
use candle_core::{Device as CandleDevice, Tensor as CandleTensor, DType, Result as CandleResult};
use candle_nn;

/// GPU-capable tensor that can run on CPU, CUDA, or Metal
#[derive(Debug, Clone)]
pub struct GpuTensor {
    pub inner: CandleTensor,
    pub device: GpuDevice,
}

/// Device types for GPU acceleration
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum GpuDevice {
    Cpu,
    Cuda(usize),
    Metal(usize),
}

impl GpuDevice {
    /// Convert to candle device
    pub fn to_candle_device(&self) -> CandleResult<CandleDevice> {
        match self {
            GpuDevice::Cpu => Ok(CandleDevice::Cpu),
            GpuDevice::Cuda(id) => CandleDevice::new_cuda(*id),
            GpuDevice::Metal(id) => CandleDevice::new_metal(*id),
        }
    }

    /// Auto-detect best available device
    pub fn auto_detect() -> Self {
        // Try CUDA first
        if let Ok(_) = CandleDevice::new_cuda(0) {
            return GpuDevice::Cuda(0);
        }

        // Try Metal on macOS
        if let Ok(_) = CandleDevice::new_metal(0) {
            return GpuDevice::Metal(0);
        }

        // Fallback to CPU
        GpuDevice::Cpu
    }

    pub fn is_gpu(&self) -> bool {
        !matches!(self, GpuDevice::Cpu)
    }
}

impl GpuTensor {
    /// Create new tensor with data
    pub fn new(data: Vec<f32>, shape: Vec<usize>, device: GpuDevice) -> ModelResult<Self> {
        let candle_device = device.to_candle_device()
            .map_err(|e| ModelError::DeviceError(format!("Failed to create device: {}", e)))?;

        let inner = CandleTensor::from_vec(data, &*shape, &candle_device)
            .map_err(|e| ModelError::ComputationFailed(format!("Failed to create tensor: {}", e)))?;

        Ok(Self { inner, device })
    }

    /// Create zeros tensor
    pub fn zeros(shape: Vec<usize>, device: GpuDevice) -> ModelResult<Self> {
        let candle_device = device.to_candle_device()
            .map_err(|e| ModelError::DeviceError(format!("Failed to create device: {}", e)))?;

        let inner = CandleTensor::zeros(&*shape, DType::F32, &candle_device)
            .map_err(|e| ModelError::ComputationFailed(format!("Failed to create zeros tensor: {}", e)))?;

        Ok(Self { inner, device })
    }

    /// Create ones tensor
    pub fn ones(shape: Vec<usize>, device: GpuDevice) -> ModelResult<Self> {
        let candle_device = device.to_candle_device()
            .map_err(|e| ModelError::DeviceError(format!("Failed to create device: {}", e)))?;

        let inner = CandleTensor::ones(&*shape, DType::F32, &candle_device)
            .map_err(|e| ModelError::ComputationFailed(format!("Failed to create ones tensor: {}", e)))?;

        Ok(Self { inner, device })
    }

    /// Create random tensor for initialization
    pub fn randn(shape: Vec<usize>, device: GpuDevice) -> ModelResult<Self> {
        let candle_device = device.to_candle_device()
            .map_err(|e| ModelError::DeviceError(format!("Failed to create device: {}", e)))?;

        let inner = CandleTensor::randn(0.0f32, 1.0f32, &*shape, &candle_device)
            .map_err(|e| ModelError::ComputationFailed(format!("Failed to create random tensor: {}", e)))?;

        Ok(Self { inner, device })
    }

    /// Get tensor shape
    pub fn shape(&self) -> Vec<usize> {
        self.inner.dims().to_vec()
    }

    /// Get tensor data as Vec<f32> (copies to CPU if needed)
    pub fn to_vec(&self) -> ModelResult<Vec<f32>> {
        let flattened = self.inner.flatten_all()
            .map_err(|e| ModelError::ComputationFailed(format!("Failed to flatten tensor: {}", e)))?;
        let data = flattened.to_vec1::<f32>()
            .map_err(|e| ModelError::ComputationFailed(format!("Failed to convert to vec: {}", e)))?;
        Ok(data)
    }

    /// Get tensor data (returns owned Vec for compatibility)
    pub fn data(&self) -> Vec<f32> {
        self.to_vec().unwrap_or_default()
    }

    /// Get device
    pub fn device(&self) -> &GpuDevice {
        &self.device
    }

    /// Move tensor to specified device
    pub fn to_device(&self, target_device: GpuDevice) -> ModelResult<Self> {
        if std::mem::discriminant(&self.device) == std::mem::discriminant(&target_device) {
            return Ok(self.clone());
        }

        let candle_device = target_device.to_candle_device()
            .map_err(|e| ModelError::DeviceError(format!("Failed to create target device: {}", e)))?;

        let inner = self.inner.to_device(&candle_device)
            .map_err(|e| ModelError::DeviceError(format!("Failed to move tensor to device: {}", e)))?;

        Ok(Self { inner, device: target_device })
    }

    /// Create tensor from data vector (alias for new)
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>, device: GpuDevice) -> ModelResult<Self> {
        Self::new(data, shape, device)
    }
}

/// GPU-accelerated tensor operations
#[derive(Debug, Clone)]
pub struct GpuTensorOps {
    device: GpuDevice,
}

impl GpuTensorOps {
    /// Create new GPU tensor operations on auto-detected device
    pub fn new() -> Self {
        Self {
            device: GpuDevice::auto_detect(),
        }
    }

    /// Create with specific device
    pub fn with_device(device: GpuDevice) -> Self {
        Self { device }
    }

    /// Get current device
    pub fn device(&self) -> &GpuDevice {
        &self.device
    }

    /// Matrix multiplication: C = A @ B
    pub fn matmul(&self, a: &GpuTensor, b: &GpuTensor) -> ModelResult<GpuTensor> {
        let result = a.inner.matmul(&b.inner)
            .map_err(|e| ModelError::ComputationFailed(format!("Matrix multiplication failed: {}", e)))?;

        Ok(GpuTensor {
            inner: result,
            device: a.device.clone(),
        })
    }

    /// Matrix multiplication (alias for compatibility)
    pub fn matrix_multiply(&self, a: &GpuTensor, b: &GpuTensor) -> ModelResult<GpuTensor> {
        self.matmul(a, b)
    }

    /// Element-wise addition: C = A + B
    pub fn add(&self, a: &GpuTensor, b: &GpuTensor) -> ModelResult<GpuTensor> {
        let result = (&a.inner + &b.inner)
            .map_err(|e| ModelError::ComputationFailed(format!("Addition failed: {}", e)))?;

        Ok(GpuTensor {
            inner: result,
            device: a.device.clone(),
        })
    }

    /// Element-wise multiplication: C = A * B
    pub fn multiply(&self, a: &GpuTensor, b: &GpuTensor) -> ModelResult<GpuTensor> {
        let result = (&a.inner * &b.inner)
            .map_err(|e| ModelError::ComputationFailed(format!("Multiplication failed: {}", e)))?;

        Ok(GpuTensor {
            inner: result,
            device: a.device.clone(),
        })
    }

    /// SiLU activation: f(x) = x * sigmoid(x)
    pub fn silu(&self, input: &GpuTensor) -> ModelResult<GpuTensor> {
        let result = candle_nn::ops::silu(&input.inner)
            .map_err(|e| ModelError::ComputationFailed(format!("SiLU activation failed: {}", e)))?;

        Ok(GpuTensor {
            inner: result,
            device: input.device.clone(),
        })
    }

    /// Softmax activation
    pub fn softmax(&self, input: &GpuTensor, dim: usize) -> ModelResult<GpuTensor> {
        let result = candle_nn::ops::softmax(&input.inner, dim)
            .map_err(|e| ModelError::ComputationFailed(format!("Softmax failed: {}", e)))?;

        Ok(GpuTensor {
            inner: result,
            device: input.device.clone(),
        })
    }

    /// Layer normalization
    pub fn layer_norm(&self, input: &GpuTensor, weight: &GpuTensor, bias: Option<&GpuTensor>, eps: f64) -> ModelResult<GpuTensor> {
        let result = if let Some(bias_tensor) = bias {
            candle_nn::ops::layer_norm(&input.inner, &weight.inner, &bias_tensor.inner, eps as f32)
        } else {
            // Create zero bias if none provided
            let zeros = CandleTensor::zeros(weight.inner.shape(), DType::F32, weight.inner.device())
                .map_err(|e| ModelError::ComputationFailed(format!("Failed to create zero bias: {}", e)))?;
            candle_nn::ops::layer_norm(&input.inner, &weight.inner, &zeros, eps as f32)
        }.map_err(|e| ModelError::ComputationFailed(format!("Layer norm failed: {}", e)))?;

        Ok(GpuTensor {
            inner: result,
            device: input.device.clone(),
        })
    }

    /// RMS normalization (for Llama)
    pub fn rms_norm(&self, input: &GpuTensor, weight: &GpuTensor, eps: f64) -> ModelResult<GpuTensor> {
        // Compute RMS norm manually using candle operations
        let dims = input.inner.dims();
        let last_dim = dims.len() - 1;

        // Compute mean of squares
        let squared = (&input.inner * &input.inner)
            .map_err(|e| ModelError::ComputationFailed(format!("RMS norm square failed: {}", e)))?;

        let mean_squared = squared.mean_keepdim(last_dim)
            .map_err(|e| ModelError::ComputationFailed(format!("RMS norm mean failed: {}", e)))?;

        // Add epsilon and take reciprocal square root
        let eps_scalar = CandleTensor::new(eps as f32, input.inner.device())
            .map_err(|e| ModelError::ComputationFailed(format!("RMS norm eps scalar failed: {}", e)))?;
        let variance_eps = mean_squared.broadcast_add(&eps_scalar)
            .map_err(|e| ModelError::ComputationFailed(format!("RMS norm variance failed: {}", e)))?;

        let inv_rms = variance_eps.powf(-0.5)
            .map_err(|e| ModelError::ComputationFailed(format!("RMS norm inv failed: {}", e)))?;

        // Normalize and scale - broadcast inv_rms to match input shape
        let normalized = input.inner.broadcast_mul(&inv_rms)
            .map_err(|e| ModelError::ComputationFailed(format!("RMS norm normalize failed: {}", e)))?;

        // Broadcast weight to match normalized tensor shape
        let result = normalized.broadcast_mul(&weight.inner)
            .map_err(|e| ModelError::ComputationFailed(format!("RMS norm scale failed: {}", e)))?;

        Ok(GpuTensor {
            inner: result,
            device: input.device.clone(),
        })
    }

    /// Embedding lookup
    pub fn embedding(&self, input_ids: &GpuTensor, embedding_table: &GpuTensor) -> ModelResult<GpuTensor> {
        let result = embedding_table.inner.embedding(&input_ids.inner)
            .map_err(|e| ModelError::ComputationFailed(format!("Embedding lookup failed: {}", e)))?;

        Ok(GpuTensor {
            inner: result,
            device: embedding_table.device.clone(),
        })
    }

    /// Reshape tensor
    pub fn reshape(&self, input: &GpuTensor, new_shape: Vec<usize>) -> ModelResult<GpuTensor> {
        let result = input.inner.reshape(&*new_shape)
            .map_err(|e| ModelError::ComputationFailed(format!("Reshape failed: {}", e)))?;

        Ok(GpuTensor {
            inner: result,
            device: input.device.clone(),
        })
    }

    /// Transpose last two dimensions
    pub fn transpose(&self, input: &GpuTensor) -> ModelResult<GpuTensor> {
        let dims = input.inner.dims();
        if dims.len() < 2 {
            return Err(ModelError::ComputationFailed("Cannot transpose tensor with less than 2 dims".to_string()));
        }

        let result = input.inner.transpose(dims.len() - 2, dims.len() - 1)
            .map_err(|e| ModelError::ComputationFailed(format!("Transpose failed: {}", e)))?;

        Ok(GpuTensor {
            inner: result,
            device: input.device.clone(),
        })
    }

    /// Create tensor from token IDs
    pub fn tensor_from_ids(&self, ids: &[u32]) -> ModelResult<GpuTensor> {
        let data: Vec<f32> = ids.iter().map(|&id| id as f32).collect();
        let shape = vec![ids.len()];

        let candle_device = self.device.to_candle_device()
            .map_err(|e| ModelError::DeviceError(format!("Failed to create device: {}", e)))?;

        let inner = CandleTensor::from_vec(data, &*shape, &candle_device)
            .map_err(|e| ModelError::ComputationFailed(format!("Failed to create tensor from IDs: {}", e)))?;

        // Convert to u32 tensor for indexing
        let inner_u32 = inner.to_dtype(candle_core::DType::U32)
            .map_err(|e| ModelError::ComputationFailed(format!("Failed to convert to U32: {}", e)))?;

        Ok(GpuTensor {
            inner: inner_u32,
            device: self.device.clone(),
        })
    }

    /// Compute cosine of tensor elements
    pub fn cos(&self, input: &GpuTensor) -> ModelResult<GpuTensor> {
        let result_tensor = input.inner.cos()
            .map_err(|e| ModelError::ComputationFailed(format!("Cos failed: {}", e)))?;

        Ok(GpuTensor {
            inner: result_tensor,
            device: input.device.clone(),
        })
    }

    /// Compute sine of tensor elements
    pub fn sin(&self, input: &GpuTensor) -> ModelResult<GpuTensor> {
        let result_tensor = input.inner.sin()
            .map_err(|e| ModelError::ComputationFailed(format!("Sin failed: {}", e)))?;

        Ok(GpuTensor {
            inner: result_tensor,
            device: input.device.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_auto_detection() {
        let device = GpuDevice::auto_detect();
        // Should always succeed with at least CPU
        match device {
            GpuDevice::Cpu => println!("Using CPU device"),
            GpuDevice::Cuda(_) => println!("Using CUDA device"),
            GpuDevice::Metal(_) => println!("Using Metal device"),
        }
    }

    #[test]
    fn test_gpu_tensor_creation() {
        let device = GpuDevice::Cpu; // Use CPU for testing
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];

        let tensor = GpuTensor::new(data.clone(), shape.clone(), device).unwrap();
        assert_eq!(tensor.shape(), shape);

        let retrieved_data = tensor.to_vec().unwrap();
        assert_eq!(retrieved_data, data);
    }

    #[test]
    fn test_gpu_tensor_zeros() {
        let device = GpuDevice::Cpu;
        let shape = vec![3, 4];

        let tensor = GpuTensor::zeros(shape.clone(), device).unwrap();
        assert_eq!(tensor.shape(), shape);

        let data = tensor.to_vec().unwrap();
        assert_eq!(data.len(), 12);
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_gpu_tensor_ones() {
        let device = GpuDevice::Cpu;
        let shape = vec![2, 3];

        let tensor = GpuTensor::ones(shape.clone(), device).unwrap();
        assert_eq!(tensor.shape(), shape);

        let data = tensor.to_vec().unwrap();
        assert_eq!(data.len(), 6);
        assert!(data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_gpu_tensor_ops_creation() {
        let ops = GpuTensorOps::new();
        // Should succeed
        assert!(true);
    }

    #[test]
    fn test_matrix_multiplication() {
        let ops = GpuTensorOps::with_device(GpuDevice::Cpu);

        // A: 2x3
        let a = GpuTensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            GpuDevice::Cpu
        ).unwrap();

        // B: 3x2
        let b = GpuTensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2],
            GpuDevice::Cpu
        ).unwrap();

        let c = ops.matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), vec![2, 2]);

        let result = c.to_vec().unwrap();
        // Manual calculation:
        // C[0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
        // C[0,1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
        // C[1,0] = 4*1 + 5*3 + 6*5 = 4 + 15 + 30 = 49
        // C[1,1] = 4*2 + 5*4 + 6*6 = 8 + 20 + 36 = 64
        assert_eq!(result, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_element_wise_operations() {
        let ops = GpuTensorOps::with_device(GpuDevice::Cpu);

        let a = GpuTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], GpuDevice::Cpu).unwrap();
        let b = GpuTensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], GpuDevice::Cpu).unwrap();

        // Test addition
        let sum = ops.add(&a, &b).unwrap();
        assert_eq!(sum.to_vec().unwrap(), vec![6.0, 8.0, 10.0, 12.0]);

        // Test multiplication
        let product = ops.multiply(&a, &b).unwrap();
        assert_eq!(product.to_vec().unwrap(), vec![5.0, 12.0, 21.0, 32.0]);
    }

    #[test]
    fn test_silu_activation() {
        let ops = GpuTensorOps::with_device(GpuDevice::Cpu);

        let input = GpuTensor::new(vec![0.0, 1.0, -1.0], vec![3], GpuDevice::Cpu).unwrap();
        let output = ops.silu(&input).unwrap();

        let result = output.to_vec().unwrap();
        assert_eq!(result.len(), 3);

        // SiLU(0) should be ~0
        assert!((result[0] - 0.0).abs() < 1e-6);

        // SiLU(1) should be positive
        assert!(result[1] > 0.0);

        // SiLU(-1) should be negative
        assert!(result[2] < 0.0);
    }

    #[test]
    fn test_softmax() {
        let ops = GpuTensorOps::with_device(GpuDevice::Cpu);

        let input = GpuTensor::new(vec![1.0, 2.0, 3.0], vec![3], GpuDevice::Cpu).unwrap();
        let output = ops.softmax(&input, 0).unwrap();

        let result = output.to_vec().unwrap();
        assert_eq!(result.len(), 3);

        // Probabilities should sum to 1
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Should be in ascending order (since input was ascending)
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    #[test]
    fn test_tensor_from_ids() {
        let ops = GpuTensorOps::with_device(GpuDevice::Cpu);

        let ids = vec![1, 2, 3, 4];
        let tensor = ops.tensor_from_ids(&ids).unwrap();

        assert_eq!(tensor.shape(), vec![4]);
    }

    #[test]
    fn test_reshape() {
        let ops = GpuTensorOps::with_device(GpuDevice::Cpu);

        let tensor = GpuTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], GpuDevice::Cpu).unwrap();
        let reshaped = ops.reshape(&tensor, vec![3, 2]).unwrap();

        assert_eq!(reshaped.shape(), vec![3, 2]);
        assert_eq!(reshaped.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}