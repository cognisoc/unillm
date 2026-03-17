//! GPU-accelerated Llama model implementation
//!
//! This module provides a GPU-accelerated version of the Llama model
//! that can run on CPU, CUDA, or Metal devices through the candle library.

use crate::types::*;
use crate::gpu_tensor_ops::{GpuTensor, GpuTensorOps, GpuDevice};

/// GPU-accelerated embedding layer
#[derive(Debug, Clone)]
pub struct GpuEmbedding {
    vocab_size: usize,
    embedding_dim: usize,
    weights: GpuTensor,
    tensor_ops: GpuTensorOps,
}

impl GpuEmbedding {
    pub fn new(vocab_size: usize, embedding_dim: usize, device: GpuDevice) -> ModelResult<Self> {
        let tensor_ops = GpuTensorOps::with_device(device.clone());

        // Initialize embedding weights with random values
        let weights = GpuTensor::randn(vec![vocab_size, embedding_dim], device)?;

        Ok(Self {
            vocab_size,
            embedding_dim,
            weights,
            tensor_ops,
        })
    }

    pub fn forward(&self, input_ids: &GpuTensor) -> ModelResult<GpuTensor> {
        // Use embedding lookup from tensor ops
        self.tensor_ops.embedding(input_ids, &self.weights)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

/// GPU-accelerated linear layer
#[derive(Debug, Clone)]
pub struct GpuLinear {
    input_dim: usize,
    output_dim: usize,
    weights: GpuTensor,
    bias: Option<GpuTensor>,
    tensor_ops: GpuTensorOps,
}

impl GpuLinear {
    pub fn new(input_dim: usize, output_dim: usize, use_bias: bool, device: GpuDevice) -> ModelResult<Self> {
        let tensor_ops = GpuTensorOps::with_device(device.clone());

        // Initialize weights with Xavier/Glorot initialization
        let weights = GpuTensor::randn(vec![input_dim, output_dim], device.clone())?;

        let bias = if use_bias {
            Some(GpuTensor::zeros(vec![output_dim], device)?)
        } else {
            None
        };

        Ok(Self {
            input_dim,
            output_dim,
            weights,
            bias,
            tensor_ops,
        })
    }

    pub fn forward(&self, input: &GpuTensor) -> ModelResult<GpuTensor> {
        // Matrix multiplication: input * weights
        let mut output = self.tensor_ops.matmul(input, &self.weights)?;

        // Add bias if present
        if let Some(ref bias) = self.bias {
            output = self.tensor_ops.add(&output, bias)?;
        }

        Ok(output)
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }
}

/// GPU-accelerated multi-head attention layer
#[derive(Debug, Clone)]
pub struct GpuAttention {
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
    q_proj: GpuLinear,
    k_proj: GpuLinear,
    v_proj: GpuLinear,
    o_proj: GpuLinear,
    tensor_ops: GpuTensorOps,
}

impl GpuAttention {
    pub fn new(hidden_size: usize, num_heads: usize, device: GpuDevice) -> ModelResult<Self> {
        let head_dim = hidden_size / num_heads;
        if hidden_size % num_heads != 0 {
            return Err(ModelError::InitializationFailed(
                format!("Hidden size {} must be divisible by num_heads {}", hidden_size, num_heads)
            ));
        }

        let tensor_ops = GpuTensorOps::with_device(device.clone());

        let q_proj = GpuLinear::new(hidden_size, hidden_size, false, device.clone())?;
        let k_proj = GpuLinear::new(hidden_size, hidden_size, false, device.clone())?;
        let v_proj = GpuLinear::new(hidden_size, hidden_size, false, device.clone())?;
        let o_proj = GpuLinear::new(hidden_size, hidden_size, false, device)?;

        Ok(Self {
            hidden_size,
            num_heads,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            tensor_ops,
        })
    }

    pub fn forward(&self, hidden_states: &GpuTensor) -> ModelResult<GpuTensor> {
        let batch_size = hidden_states.shape()[0];
        let seq_len = hidden_states.shape()[1];

        // Project to Q, K, V
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape for multi-head attention: [batch, seq, heads, head_dim]
        let q = self.tensor_ops.reshape(&q, vec![batch_size, seq_len, self.num_heads, self.head_dim])?;
        let k = self.tensor_ops.reshape(&k, vec![batch_size, seq_len, self.num_heads, self.head_dim])?;
        let v = self.tensor_ops.reshape(&v, vec![batch_size, seq_len, self.num_heads, self.head_dim])?;

        // For now, use a simplified attention computation
        // In a full implementation, we would transpose and compute scaled dot-product attention
        let attended = self.compute_attention(&q, &k, &v)?;

        // Reshape back to [batch, seq, hidden_size]
        let attended = self.tensor_ops.reshape(&attended, vec![batch_size, seq_len, self.hidden_size])?;

        // Output projection
        self.o_proj.forward(&attended)
    }

    fn compute_attention(&self, q: &GpuTensor, _k: &GpuTensor, v: &GpuTensor) -> ModelResult<GpuTensor> {
        // Simplified attention: just return V for now
        // TODO: Implement proper scaled dot-product attention with Flash Attention
        Ok(v.clone())
    }
}

/// GPU-accelerated MLP (Feed-Forward Network)
#[derive(Debug, Clone)]
pub struct GpuMLP {
    hidden_size: usize,
    intermediate_size: usize,
    gate_proj: GpuLinear,
    up_proj: GpuLinear,
    down_proj: GpuLinear,
    tensor_ops: GpuTensorOps,
}

impl GpuMLP {
    pub fn new(hidden_size: usize, intermediate_size: usize, device: GpuDevice) -> ModelResult<Self> {
        let tensor_ops = GpuTensorOps::with_device(device.clone());

        let gate_proj = GpuLinear::new(hidden_size, intermediate_size, false, device.clone())?;
        let up_proj = GpuLinear::new(hidden_size, intermediate_size, false, device.clone())?;
        let down_proj = GpuLinear::new(intermediate_size, hidden_size, false, device)?;

        Ok(Self {
            hidden_size,
            intermediate_size,
            gate_proj,
            up_proj,
            down_proj,
            tensor_ops,
        })
    }

    pub fn forward(&self, hidden_states: &GpuTensor) -> ModelResult<GpuTensor> {
        // Llama uses SwiGLU activation: gate_proj(x) * silu(up_proj(x))
        let gate = self.gate_proj.forward(hidden_states)?;
        let up = self.up_proj.forward(hidden_states)?;

        // Apply SiLU activation to gate
        let gate_activated = self.tensor_ops.silu(&gate)?;

        // Element-wise multiplication
        let intermediate = self.tensor_ops.multiply(&gate_activated, &up)?;

        // Down projection
        self.down_proj.forward(&intermediate)
    }
}

/// GPU-accelerated transformer block
#[derive(Debug, Clone)]
pub struct GpuTransformerBlock {
    hidden_size: usize,
    attention: GpuAttention,
    mlp: GpuMLP,
    input_layernorm: GpuTensor,  // Will be replaced with proper normalization
    post_attention_layernorm: GpuTensor,
    tensor_ops: GpuTensorOps,
}

impl GpuTransformerBlock {
    pub fn new(config: &ModelConfig, device: GpuDevice) -> ModelResult<Self> {
        let tensor_ops = GpuTensorOps::with_device(device.clone());

        let attention = GpuAttention::new(config.hidden_size, config.num_heads, device.clone())?;
        let mlp = GpuMLP::new(config.hidden_size, config.intermediate_size, device.clone())?;

        // Placeholder normalization weights (should be learned parameters)
        let input_layernorm = GpuTensor::ones(vec![config.hidden_size], device.clone())?;
        let post_attention_layernorm = GpuTensor::ones(vec![config.hidden_size], device)?;

        Ok(Self {
            hidden_size: config.hidden_size,
            attention,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            tensor_ops,
        })
    }

    pub fn forward(&self, hidden_states: &GpuTensor) -> ModelResult<GpuTensor> {
        // Pre-attention normalization
        let normed_input = self.tensor_ops.rms_norm(hidden_states, &self.input_layernorm, 1e-6)?;

        // Self-attention with residual connection
        let attention_output = self.attention.forward(&normed_input)?;
        let hidden_states = self.tensor_ops.add(hidden_states, &attention_output)?;

        // Pre-MLP normalization
        let normed_hidden = self.tensor_ops.rms_norm(&hidden_states, &self.post_attention_layernorm, 1e-6)?;

        // MLP with residual connection
        let mlp_output = self.mlp.forward(&normed_hidden)?;
        self.tensor_ops.add(&hidden_states, &mlp_output)
    }
}

/// GPU-accelerated Llama model
#[derive(Debug, Clone)]
pub struct GpuLlamaModel {
    config: ModelConfig,
    embedding: GpuEmbedding,
    layers: Vec<GpuTransformerBlock>,
    norm: GpuTensor,
    lm_head: GpuLinear,
    tensor_ops: GpuTensorOps,
    device: GpuDevice,
}

impl GpuLlamaModel {
    pub fn new(config: ModelConfig, device: GpuDevice) -> ModelResult<Self> {
        let tensor_ops = GpuTensorOps::with_device(device.clone());

        let embedding = GpuEmbedding::new(config.vocabulary_size, config.hidden_size, device.clone())?;

        let mut layers = Vec::new();
        for _ in 0..config.num_layers {
            layers.push(GpuTransformerBlock::new(&config, device.clone())?);
        }

        let norm = GpuTensor::ones(vec![config.hidden_size], device.clone())?;
        let lm_head = GpuLinear::new(config.hidden_size, config.vocabulary_size, false, device.clone())?;

        Ok(Self {
            config: config.clone(),
            embedding,
            layers,
            norm,
            lm_head,
            tensor_ops,
            device,
        })
    }

    pub fn forward(&self, input_ids: &[u32]) -> ModelResult<GpuTensor> {
        // Convert input IDs to GPU tensor
        let input_tensor = self.tensor_ops.tensor_from_ids(input_ids)?;

        // Embedding lookup
        let mut hidden_states = self.embedding.forward(&input_tensor)?;

        // Apply transformer layers
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }

        // Final normalization
        hidden_states = self.tensor_ops.rms_norm(&hidden_states, &self.norm, 1e-6)?;

        // Language modeling head
        self.lm_head.forward(&hidden_states)
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn device(&self) -> &GpuDevice {
        &self.device
    }

    /// Move model to a different device
    pub fn to_device(&self, target_device: GpuDevice) -> ModelResult<Self> {
        Self::new(self.config.clone(), target_device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_embedding_creation() {
        let device = GpuDevice::auto_detect();
        let embedding = GpuEmbedding::new(1000, 128, device);
        assert!(embedding.is_ok());

        let embedding = embedding.unwrap();
        assert_eq!(embedding.vocab_size(), 1000);
        assert_eq!(embedding.embedding_dim(), 128);
    }

    #[test]
    fn test_gpu_linear_creation() {
        let device = GpuDevice::auto_detect();
        let linear = GpuLinear::new(256, 128, true, device);
        assert!(linear.is_ok());

        let linear = linear.unwrap();
        assert_eq!(linear.input_dim(), 256);
        assert_eq!(linear.output_dim(), 128);
    }

    #[test]
    fn test_gpu_linear_forward() {
        let device = GpuDevice::auto_detect();
        let linear = GpuLinear::new(4, 2, false, device.clone()).unwrap();

        let input = GpuTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4], device).unwrap();
        let output = linear.forward(&input);
        if let Err(ref e) = output {
            println!("Linear forward error: {:?}", e);
        }
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.shape()[0], 1); // batch size
        assert_eq!(output.shape()[1], 2); // output dim
    }

    #[test]
    fn test_gpu_attention_creation() {
        let device = GpuDevice::auto_detect();
        let attention = GpuAttention::new(512, 8, device);
        assert!(attention.is_ok());
    }

    #[test]
    fn test_gpu_mlp_creation() {
        let device = GpuDevice::auto_detect();
        let mlp = GpuMLP::new(512, 2048, device);
        assert!(mlp.is_ok());
    }

    #[test]
    fn test_gpu_transformer_block_creation() {
        let config = ModelConfig {
            model_name: "test".to_string(),
            model_path: "/tmp/test".to_string(),
            max_sequence_length: 128,
            vocabulary_size: 1000,
            num_layers: 1,
            num_heads: 8,
            head_dim: 32,
            hidden_size: 256,
            intermediate_size: 1024,
            dtype: crate::types::DataType::Float32,
        };

        let device = GpuDevice::auto_detect();
        let block = GpuTransformerBlock::new(&config, device);
        assert!(block.is_ok());
    }

    #[test]
    fn test_gpu_llama_model_creation() {
        let config = ModelConfig {
            model_name: "test".to_string(),
            model_path: "/tmp/test".to_string(),
            max_sequence_length: 64,
            vocabulary_size: 1000,
            num_layers: 2,
            num_heads: 4,
            head_dim: 32,
            hidden_size: 128,
            intermediate_size: 256,
            dtype: crate::types::DataType::Float32,
        };

        let device = GpuDevice::auto_detect();
        let model = GpuLlamaModel::new(config, device);
        assert!(model.is_ok());
    }

    #[test]
    fn test_gpu_llama_model_forward() {
        let config = ModelConfig {
            model_name: "test".to_string(),
            model_path: "/tmp/test".to_string(),
            max_sequence_length: 32,
            vocabulary_size: 500,
            num_layers: 1,
            num_heads: 4,
            head_dim: 16,
            hidden_size: 64,
            intermediate_size: 128,
            dtype: crate::types::DataType::Float32,
        };

        let device = GpuDevice::auto_detect();
        let model = GpuLlamaModel::new(config, device).unwrap();

        let input_ids = vec![1, 5, 10, 15];
        let output = model.forward(&input_ids);
        if let Err(ref e) = output {
            println!("Model forward error: {:?}", e);
        }
        assert!(output.is_ok());

        let output = output.unwrap();
        let output_shape = output.shape();
        assert_eq!(output_shape.len(), 2); // Should be 2D tensor
        assert_eq!(output_shape[0], input_ids.len()); // Sequence length
        assert_eq!(output_shape[1], model.config().vocabulary_size); // Vocab size
    }

    #[test]
    fn test_device_movement() {
        let config = ModelConfig {
            model_name: "test".to_string(),
            model_path: "/tmp/test".to_string(),
            max_sequence_length: 16,
            vocabulary_size: 100,
            num_layers: 1,
            num_heads: 2,
            head_dim: 16,
            hidden_size: 32,
            intermediate_size: 64,
            dtype: crate::types::DataType::Float32,
        };

        let device1 = GpuDevice::Cpu;
        let model = GpuLlamaModel::new(config.clone(), device1).unwrap();

        let device2 = GpuDevice::auto_detect();
        let moved_model = model.to_device(device2);
        assert!(moved_model.is_ok());
    }
}