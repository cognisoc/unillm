//! Minimal working model implementation using real tensor operations
//!
//! This module implements a basic Llama model that performs actual computation
//! using the tested tensor operations from tensor_ops module.

use crate::types::*;
use crate::tensor_ops::{CpuTensor, CpuTensorOps};
use std::collections::HashMap;

/// Basic Llama model configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_attention_heads: usize, // Alias for num_heads
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub max_seq_len: usize,
    pub eps: f32, // Layer norm epsilon
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 512,    // Small for testing
            num_layers: 4,       // Small for testing
            num_heads: 8,
            num_attention_heads: 8, // Same as num_heads
            head_dim: 64,        // hidden_size / num_heads
            intermediate_size: 1024, // ~2x hidden_size
            max_seq_len: 256,    // Small for testing
            eps: 1e-5,          // Standard epsilon
        }
    }
}

/// Embedding layer
pub struct Embedding {
    pub weight: CpuTensor, // [vocab_size, hidden_size]
}

impl Embedding {
    pub fn new(vocab_size: usize, hidden_size: usize) -> ModelResult<Self> {
        // Initialize with small random values
        let size = vocab_size * hidden_size;
        let data: Vec<f32> = (0..size).map(|i| {
            // Simple deterministic "random" for testing
            (((i * 12345) % 10007) as f32 / 10007.0 - 0.5) * 0.1
        }).collect();

        let weight = CpuTensor::new(vec![vocab_size, hidden_size], data)?;
        Ok(Self { weight })
    }

    pub fn forward(&self, input_ids: &[u32]) -> ModelResult<CpuTensor> {
        let batch_size = 1;
        let seq_len = input_ids.len();
        let hidden_size = self.weight.shape[1];

        let mut output_data = Vec::with_capacity(seq_len * hidden_size);

        for &token_id in input_ids {
            if token_id as usize >= self.weight.shape[0] {
                return Err(ModelError::InvalidInput(
                    format!("Token ID {} out of vocabulary range {}", token_id, self.weight.shape[0])
                ));
            }

            // Extract embedding for this token
            let token_start = token_id as usize * hidden_size;
            let token_end = token_start + hidden_size;
            output_data.extend_from_slice(&self.weight.data[token_start..token_end]);
        }

        Ok(CpuTensor::new(vec![batch_size, seq_len, hidden_size], output_data)?)
    }
}

/// Linear layer
pub struct Linear {
    pub weight: CpuTensor, // [out_features, in_features]
    pub bias: Option<CpuTensor>,
    pub tensor_ops: CpuTensorOps,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> ModelResult<Self> {
        // Initialize weights with small random values
        let weight_size = out_features * in_features;
        let weight_data: Vec<f32> = (0..weight_size).map(|i| {
            (((i * 23456) % 10007) as f32 / 10007.0 - 0.5) * 0.1
        }).collect();

        let weight = CpuTensor::new(vec![out_features, in_features], weight_data)?;

        let bias = if use_bias {
            let bias_data = vec![0.0; out_features];
            Some(CpuTensor::new(vec![out_features], bias_data)?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            tensor_ops: CpuTensorOps::new(),
        })
    }

    pub fn forward(&self, input: &CpuTensor) -> ModelResult<CpuTensor> {
        // input: [batch, seq_len, in_features]
        // weight: [out_features, in_features]
        // output: [batch, seq_len, out_features]

        if input.shape.len() != 3 {
            return Err(ModelError::ComputationFailed("Linear layer expects 3D input".to_string()));
        }

        let batch_size = input.shape[0];
        let seq_len = input.shape[1];
        let in_features = input.shape[2];
        let out_features = self.weight.shape[0];

        if in_features != self.weight.shape[1] {
            return Err(ModelError::ComputationFailed(
                format!("Feature dimension mismatch: {} vs {}", in_features, self.weight.shape[1])
            ));
        }

        // Reshape input to [batch*seq_len, in_features] for matrix multiplication
        let input_2d = CpuTensor::new(
            vec![batch_size * seq_len, in_features],
            input.data.clone()
        )?;

        // Perform matrix multiplication: [batch*seq_len, in_features] @ [in_features, out_features]
        // Need to transpose weight matrix for correct multiplication
        let weight_t = self.transpose_2d(&self.weight)?;
        let output_2d = self.tensor_ops.matmul(&input_2d, &weight_t)?;

        // Add bias if present
        let output_2d = if let Some(ref bias) = self.bias {
            self.add_bias(&output_2d, bias)?
        } else {
            output_2d
        };

        // Reshape back to [batch, seq_len, out_features]
        Ok(CpuTensor::new(
            vec![batch_size, seq_len, out_features],
            output_2d.data
        )?)
    }

    fn transpose_2d(&self, tensor: &CpuTensor) -> ModelResult<CpuTensor> {
        if tensor.shape.len() != 2 {
            return Err(ModelError::ComputationFailed("Transpose expects 2D tensor".to_string()));
        }

        let rows = tensor.shape[0];
        let cols = tensor.shape[1];
        let mut transposed_data = vec![0.0; rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                transposed_data[j * rows + i] = tensor.data[i * cols + j];
            }
        }

        Ok(CpuTensor::new(vec![cols, rows], transposed_data)?)
    }

    fn add_bias(&self, input: &CpuTensor, bias: &CpuTensor) -> ModelResult<CpuTensor> {
        let mut output_data = input.data.clone();
        let out_features = bias.shape[0];
        let total_elements = input.data.len();

        for i in 0..total_elements {
            let feature_idx = i % out_features;
            output_data[i] += bias.data[feature_idx];
        }

        Ok(CpuTensor::new(input.shape.clone(), output_data)?)
    }
}

/// Simple MLP block (no SwiGLU for now)
pub struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    tensor_ops: CpuTensorOps,
}

impl MLP {
    pub fn new(hidden_size: usize, intermediate_size: usize) -> ModelResult<Self> {
        Ok(Self {
            gate_proj: Linear::new(hidden_size, intermediate_size, false)?,
            up_proj: Linear::new(hidden_size, intermediate_size, false)?,
            down_proj: Linear::new(intermediate_size, hidden_size, false)?,
            tensor_ops: CpuTensorOps::new(),
        })
    }

    pub fn forward(&self, input: &CpuTensor) -> ModelResult<CpuTensor> {
        // SwiGLU: gate_proj(x) * silu(up_proj(x))
        let gate_out = self.gate_proj.forward(input)?;
        let up_out = self.up_proj.forward(input)?;

        // Apply SiLU to up projection
        let up_activated = self.tensor_ops.silu(&up_out)?;

        // Element-wise multiply gate and activated up
        let intermediate = self.tensor_ops.multiply(&gate_out, &up_activated)?;

        // Down projection
        self.down_proj.forward(&intermediate)
    }
}

/// Basic attention (no Flash Attention yet)
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    tensor_ops: CpuTensorOps,
}

impl Attention {
    pub fn new(hidden_size: usize, num_heads: usize) -> ModelResult<Self> {
        let head_dim = hidden_size / num_heads;
        if hidden_size % num_heads != 0 {
            return Err(ModelError::InvalidInput("hidden_size must be divisible by num_heads".to_string()));
        }

        Ok(Self {
            q_proj: Linear::new(hidden_size, hidden_size, false)?,
            k_proj: Linear::new(hidden_size, hidden_size, false)?,
            v_proj: Linear::new(hidden_size, hidden_size, false)?,
            o_proj: Linear::new(hidden_size, hidden_size, false)?,
            num_heads,
            head_dim,
            tensor_ops: CpuTensorOps::new(),
        })
    }

    pub fn forward(&self, input: &CpuTensor) -> ModelResult<CpuTensor> {
        // Project to Q, K, V
        let q = self.q_proj.forward(input)?;
        let k = self.k_proj.forward(input)?;
        let v = self.v_proj.forward(input)?;

        // Compute attention (simplified - no masking for now)
        let attention_output = self.compute_attention(&q, &k, &v)?;

        // Output projection
        self.o_proj.forward(&attention_output)
    }

    fn compute_attention(&self, q: &CpuTensor, k: &CpuTensor, v: &CpuTensor) -> ModelResult<CpuTensor> {
        // For simplicity, compute attention on flattened tensors
        // This is not the full multi-head attention but enough for basic testing
        let batch_size = q.shape[0];
        let seq_len = q.shape[1];
        let hidden_size = q.shape[2];

        // Simplified attention: just use values (skip Q@K^T computation for now)
        // In a full implementation, we'd compute attention weights and apply them
        Ok(v.clone())
    }
}

/// Transformer block
pub struct TransformerBlock {
    attention: Attention,
    mlp: MLP,
    tensor_ops: CpuTensorOps,
}

impl TransformerBlock {
    pub fn new(config: &ModelConfig) -> ModelResult<Self> {
        Ok(Self {
            attention: Attention::new(config.hidden_size, config.num_heads)?,
            mlp: MLP::new(config.hidden_size, config.intermediate_size)?,
            tensor_ops: CpuTensorOps::new(),
        })
    }

    pub fn forward(&self, input: &CpuTensor) -> ModelResult<CpuTensor> {
        // Attention with residual connection (skip layer norm for now)
        let attn_out = self.attention.forward(input)?;
        let after_attn = self.tensor_ops.add(input, &attn_out)?;

        // MLP with residual connection
        let mlp_out = self.mlp.forward(&after_attn)?;
        self.tensor_ops.add(&after_attn, &mlp_out)
    }
}

/// Simple Llama model
pub struct LlamaModel {
    config: ModelConfig,
    embedding: Embedding,
    layers: Vec<TransformerBlock>,
    lm_head: Linear,
}

impl LlamaModel {
    pub fn new(config: ModelConfig) -> ModelResult<Self> {
        let embedding = Embedding::new(config.vocab_size, config.hidden_size)?;

        let mut layers = Vec::new();
        for _ in 0..config.num_layers {
            layers.push(TransformerBlock::new(&config)?);
        }

        let lm_head = Linear::new(config.hidden_size, config.vocab_size, false)?;

        Ok(Self {
            config,
            embedding,
            layers,
            lm_head,
        })
    }

    pub fn forward(&self, input_ids: &[u32]) -> ModelResult<CpuTensor> {
        // Embedding lookup
        let mut hidden_states = self.embedding.forward(input_ids)?;

        // Pass through transformer layers
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }

        // Language modeling head
        self.lm_head.forward(&hidden_states)
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config() {
        let config = ModelConfig::default();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.num_layers, 4);
    }

    #[test]
    fn test_embedding_forward() {
        let embedding = Embedding::new(1000, 64).unwrap();
        let input_ids = vec![1, 2, 3];
        let output = embedding.forward(&input_ids).unwrap();

        assert_eq!(output.shape, vec![1, 3, 64]); // [batch, seq, hidden]
        assert_eq!(output.data.len(), 3 * 64);
    }

    #[test]
    fn test_embedding_out_of_bounds() {
        let embedding = Embedding::new(100, 64).unwrap();
        let input_ids = vec![1, 2, 150]; // 150 is out of bounds
        let result = embedding.forward(&input_ids);
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_forward() {
        let linear = Linear::new(128, 256, true).unwrap();
        let input = CpuTensor::new(vec![1, 4, 128], vec![1.0; 1 * 4 * 128]).unwrap();
        let output = linear.forward(&input).unwrap();

        assert_eq!(output.shape, vec![1, 4, 256]);
        assert_eq!(output.data.len(), 1 * 4 * 256);
    }

    #[test]
    fn test_linear_dimension_mismatch() {
        let linear = Linear::new(128, 256, false).unwrap();
        let input = CpuTensor::new(vec![1, 4, 64], vec![1.0; 1 * 4 * 64]).unwrap(); // Wrong size
        let result = linear.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_mlp_forward() {
        let mlp = MLP::new(256, 512).unwrap();
        let input = CpuTensor::new(vec![1, 4, 256], vec![0.1; 1 * 4 * 256]).unwrap();
        let output = mlp.forward(&input).unwrap();

        assert_eq!(output.shape, vec![1, 4, 256]);
        assert_eq!(output.data.len(), 1 * 4 * 256);
    }

    #[test]
    fn test_attention_forward() {
        let attention = Attention::new(128, 8).unwrap();
        let input = CpuTensor::new(vec![1, 4, 128], vec![0.1; 1 * 4 * 128]).unwrap();
        let output = attention.forward(&input).unwrap();

        assert_eq!(output.shape, vec![1, 4, 128]);
        assert_eq!(output.data.len(), 1 * 4 * 128);
    }

    #[test]
    fn test_transformer_block_forward() {
        let config = ModelConfig {
            hidden_size: 128,
            num_heads: 8,
            intermediate_size: 256,
            ..Default::default()
        };

        let block = TransformerBlock::new(&config).unwrap();
        let input = CpuTensor::new(vec![1, 4, 128], vec![0.1; 1 * 4 * 128]).unwrap();
        let output = block.forward(&input).unwrap();

        assert_eq!(output.shape, vec![1, 4, 128]);
        assert_eq!(output.data.len(), 1 * 4 * 128);
    }

    #[test]
    fn test_llama_model_forward() {
        let config = ModelConfig {
            vocab_size: 1000,
            hidden_size: 128,
            num_layers: 2,
            num_heads: 8,
            head_dim: 16,
            intermediate_size: 256,
            max_seq_len: 64,
        };

        let model = LlamaModel::new(config).unwrap();
        let input_ids = vec![1, 2, 3, 4];
        let output = model.forward(&input_ids).unwrap();

        assert_eq!(output.shape, vec![1, 4, 1000]); // [batch, seq, vocab]
        assert_eq!(output.data.len(), 1 * 4 * 1000);
    }

    #[test]
    fn test_llama_model_different_sequence_lengths() {
        let config = ModelConfig {
            vocab_size: 1000,
            hidden_size: 64,
            num_layers: 1,
            num_heads: 4,
            head_dim: 16,
            intermediate_size: 128,
            max_seq_len: 64,
        };

        let model = LlamaModel::new(config).unwrap();

        // Test different sequence lengths
        let short_input = vec![1, 2];
        let output1 = model.forward(&short_input).unwrap();
        assert_eq!(output1.shape, vec![1, 2, 1000]);

        let long_input = vec![1, 2, 3, 4, 5, 6];
        let output2 = model.forward(&long_input).unwrap();
        assert_eq!(output2.shape, vec![1, 6, 1000]);
    }
}