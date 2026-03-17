//! CPU Tensor operations for UniLLM
//!
//! This module provides CPU-based tensor operations with actual mathematical computation.
//! Starting with CPU-only implementations that will be expanded to GPU later.

use crate::types::*;

/// Basic CPU tensor operations
pub struct CpuTensorOps;

impl CpuTensorOps {
    pub fn new() -> Self {
        Self
    }

    /// Matrix multiplication: C = A @ B (CPU implementation)
    pub fn matmul(&self, a: &CpuTensor, b: &CpuTensor) -> ModelResult<CpuTensor> {
        self.validate_matmul_shapes(&a.shape, &b.shape)?;

        let (batch_a, m, k) = if a.shape.len() == 3 {
            (a.shape[0], a.shape[1], a.shape[2])
        } else if a.shape.len() == 2 {
            (1, a.shape[0], a.shape[1])
        } else {
            return Err(ModelError::ComputationFailed("Unsupported tensor dimensions for matmul".to_string()));
        };

        let (batch_b, k2, n) = if b.shape.len() == 3 {
            (b.shape[0], b.shape[1], b.shape[2])
        } else if b.shape.len() == 2 {
            (1, b.shape[0], b.shape[1])
        } else {
            return Err(ModelError::ComputationFailed("Unsupported tensor dimensions for matmul".to_string()));
        };

        if k != k2 {
            return Err(ModelError::ComputationFailed(format!("Matrix dimension mismatch: {} vs {}", k, k2)));
        }

        let batch_size = batch_a.max(batch_b);
        let output_shape = if a.shape.len() == 3 || b.shape.len() == 3 {
            vec![batch_size, m, n]
        } else {
            vec![m, n]
        };

        let mut output_data = vec![0.0f32; batch_size * m * n];

        // Perform matrix multiplication
        for b_idx in 0..batch_size {
            let a_batch_offset = if batch_a == 1 { 0 } else { b_idx * m * k };
            let b_batch_offset = if batch_b == 1 { 0 } else { b_idx * k * n };
            let c_batch_offset = b_idx * m * n;

            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for l in 0..k {
                        let a_val = a.data[a_batch_offset + i * k + l];
                        let b_val = b.data[b_batch_offset + l * n + j];
                        sum += a_val * b_val;
                    }
                    output_data[c_batch_offset + i * n + j] = sum;
                }
            }
        }

        Ok(CpuTensor {
            shape: output_shape,
            data: output_data,
        })
    }

    /// Element-wise addition: C = A + B
    pub fn add(&self, a: &CpuTensor, b: &CpuTensor) -> ModelResult<CpuTensor> {
        if a.shape != b.shape {
            return Err(ModelError::ComputationFailed("Shape mismatch for addition".to_string()));
        }

        let data: Vec<f32> = a.data.iter().zip(b.data.iter()).map(|(x, y)| x + y).collect();

        Ok(CpuTensor {
            shape: a.shape.clone(),
            data,
        })
    }

    /// Element-wise multiplication: C = A * B
    pub fn multiply(&self, a: &CpuTensor, b: &CpuTensor) -> ModelResult<CpuTensor> {
        if a.shape != b.shape {
            return Err(ModelError::ComputationFailed("Shape mismatch for multiplication".to_string()));
        }

        let data: Vec<f32> = a.data.iter().zip(b.data.iter()).map(|(x, y)| x * y).collect();

        Ok(CpuTensor {
            shape: a.shape.clone(),
            data,
        })
    }

    /// SiLU activation: f(x) = x * sigmoid(x)
    pub fn silu(&self, input: &CpuTensor) -> ModelResult<CpuTensor> {
        let data: Vec<f32> = input.data.iter().map(|&x| {
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            x * sigmoid
        }).collect();

        Ok(CpuTensor {
            shape: input.shape.clone(),
            data,
        })
    }

    /// Softmax operation along the last dimension
    pub fn softmax(&self, input: &CpuTensor) -> ModelResult<CpuTensor> {
        let mut output_data = input.data.clone();

        if input.shape.is_empty() {
            return Ok(CpuTensor {
                shape: input.shape.clone(),
                data: output_data,
            });
        }

        let last_dim = input.shape[input.shape.len() - 1];
        let num_sequences = input.data.len() / last_dim;

        for seq_idx in 0..num_sequences {
            let start_idx = seq_idx * last_dim;
            let end_idx = start_idx + last_dim;

            // Find max for numerical stability
            let max_val = output_data[start_idx..end_idx].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            // Compute exp(x - max) and sum
            let mut sum = 0.0f32;
            for i in start_idx..end_idx {
                output_data[i] = (output_data[i] - max_val).exp();
                sum += output_data[i];
            }

            // Normalize
            for i in start_idx..end_idx {
                output_data[i] /= sum;
            }
        }

        Ok(CpuTensor {
            shape: input.shape.clone(),
            data: output_data,
        })
    }

    fn validate_matmul_shapes(&self, shape_a: &[usize], shape_b: &[usize]) -> ModelResult<()> {
        if shape_a.len() < 2 || shape_b.len() < 2 {
            return Err(ModelError::ComputationFailed("Matmul requires at least 2D tensors".to_string()));
        }

        let k_a = shape_a[shape_a.len() - 1];
        let k_b = shape_b[shape_b.len() - 2];

        if k_a != k_b {
            return Err(ModelError::ComputationFailed(
                format!("Matrix dimension mismatch: {} vs {}", k_a, k_b)
            ));
        }

        Ok(())
    }
}

/// CPU tensor with actual data
#[derive(Debug, Clone)]
pub struct CpuTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl CpuTensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> ModelResult<Self> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(ModelError::InvalidInput(
                format!("Data size {} doesn't match shape {:?} (expected {})",
                        data.len(), shape, expected_size)
            ));
        }

        Ok(Self { shape, data })
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; size],
        }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            shape,
            data: vec![1.0; size],
        }
    }

    pub fn from_scalar(value: f32) -> Self {
        Self {
            shape: vec![],
            data: vec![value],
        }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_tensor_creation() {
        let tensor = CpuTensor::zeros(vec![2, 3]);
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.data, vec![0.0; 6]);
        assert_eq!(tensor.numel(), 6);
    }

    #[test]
    fn test_cpu_tensor_from_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = CpuTensor::new(vec![2, 2], data.clone()).unwrap();
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.data, data);
    }

    #[test]
    fn test_cpu_tensor_invalid_size() {
        let data = vec![1.0, 2.0, 3.0];
        let result = CpuTensor::new(vec![2, 2], data);
        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_2d() {
        let ops = CpuTensorOps::new();

        // A: 2x3, B: 3x2 -> C: 2x2
        let a = CpuTensor::new(vec![2, 3], vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        ]).unwrap();

        let b = CpuTensor::new(vec![3, 2], vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0
        ]).unwrap();

        let c = ops.matmul(&a, &b).unwrap();

        assert_eq!(c.shape, vec![2, 2]);
        // Manual calculation:
        // C[0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
        // C[0,1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
        // C[1,0] = 4*1 + 5*3 + 6*5 = 4 + 15 + 30 = 49
        // C[1,1] = 4*2 + 5*4 + 6*6 = 8 + 20 + 36 = 64
        assert_eq!(c.data, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_add() {
        let ops = CpuTensorOps::new();

        let a = CpuTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = CpuTensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();

        let c = ops.add(&a, &b).unwrap();

        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_multiply() {
        let ops = CpuTensorOps::new();

        let a = CpuTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = CpuTensor::new(vec![2, 2], vec![2.0, 3.0, 4.0, 5.0]).unwrap();

        let c = ops.multiply(&a, &b).unwrap();

        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    fn test_silu() {
        let ops = CpuTensorOps::new();

        let input = CpuTensor::new(vec![3], vec![0.0, 1.0, -1.0]).unwrap();
        let output = ops.silu(&input).unwrap();

        assert_eq!(output.shape, vec![3]);

        // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assert!((output.data[0] - 0.0).abs() < 1e-6);

        // SiLU(1) = 1 * sigmoid(1) ≈ 1 * 0.731 ≈ 0.731
        assert!((output.data[1] - 0.7310586).abs() < 1e-6);

        // SiLU(-1) = -1 * sigmoid(-1) ≈ -1 * 0.269 ≈ -0.269
        assert!((output.data[2] - (-0.2689414)).abs() < 1e-6);
    }

    #[test]
    fn test_softmax() {
        let ops = CpuTensorOps::new();

        let input = CpuTensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let output = ops.softmax(&input).unwrap();

        assert_eq!(output.shape, vec![3]);

        // Check that probabilities sum to 1
        let sum: f32 = output.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check that they're in ascending order (since input was ascending)
        assert!(output.data[0] < output.data[1]);
        assert!(output.data[1] < output.data[2]);
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        let ops = CpuTensorOps::new();

        let a = CpuTensor::new(vec![2, 3], vec![1.0; 6]).unwrap();
        let b = CpuTensor::new(vec![2, 2], vec![1.0; 4]).unwrap(); // Wrong dimensions

        let result = ops.matmul(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_shape_mismatch() {
        let ops = CpuTensorOps::new();

        let a = CpuTensor::new(vec![2, 2], vec![1.0; 4]).unwrap();
        let b = CpuTensor::new(vec![3, 3], vec![1.0; 9]).unwrap();

        let result = ops.add(&a, &b);
        assert!(result.is_err());
    }
}