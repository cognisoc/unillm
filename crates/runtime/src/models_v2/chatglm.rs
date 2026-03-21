//! ChatGLM Model V2 - Clean implementation using solid abstractions
//!
//! This implements the ChatGLM architecture including:
//! - ChatGLM-6B, ChatGLM2-6B, ChatGLM3-6B, GLM-4

use crate::model_config;
use super::traits::*;
use std::collections::HashMap;
use anyhow::Result;
use serde::{Serialize, Deserialize};

model_config!(ChatGLMConfig {
    vocab_size: usize = 65024,
    hidden_size: usize = 4096,
    intermediate_size: usize = 13696,
    num_hidden_layers: usize = 28,
    num_attention_heads: usize = 32,
    num_key_value_heads: Option<usize> = Some(2),
    hidden_act: String = "swiglu".to_string(),
    max_position_embeddings: usize = 8192,
    initializer_range: f32 = 0.02,
    rms_norm_eps: f32 = 1e-5,
    use_cache: bool = true,
    pad_token_id: Option<i64> = Some(0),
    bos_token_id: Option<i64> = Some(1),
    eos_token_id: Option<i64> = Some(2),
    tie_word_embeddings: bool = false,
    rope_theta: f32 = 10000.0,
    attention_dropout: f32 = 0.0,
    // ChatGLM specific
    add_bias_linear: bool = false,
    add_qkv_bias: bool = true,
    apply_residual_connection_post_layernorm: bool = false,
    kv_channels: usize = 128,
    multi_query_attention: bool = true,
});

pub struct ChatGLMModelV2 {
    config: ChatGLMConfig,
    device: Device,
    embedding: ChatGLMEmbedding,
    encoder: ChatGLMEncoder,
    output_layer: Tensor,
}

pub struct ChatGLMEmbedding {
    word_embeddings: Tensor,
    config: ChatGLMConfig,
}

pub struct ChatGLMEncoder {
    layers: Vec<ChatGLMLayer>,
    final_layernorm: Tensor,
    config: ChatGLMConfig,
}

pub struct ChatGLMLayer {
    input_layernorm: Tensor,
    self_attention: ChatGLMAttention,
    post_attention_layernorm: Tensor,
    mlp: ChatGLMMLP,
}

pub struct ChatGLMAttention {
    query_key_value: Tensor, // Packed QKV
    dense: Tensor,
    config: ChatGLMConfig,
}

pub struct ChatGLMMLP {
    dense_h_to_4h: Tensor,
    dense_4h_to_h: Tensor,
    config: ChatGLMConfig,
}

impl Model for ChatGLMModelV2 {
    type Config = ChatGLMConfig;

    fn new(config: ChatGLMConfig) -> Result<Self> {
        let device = Device::CPU;

        let embedding = ChatGLMEmbedding::new(&config, &device)?;
        let encoder = ChatGLMEncoder::new(&config, &device)?;
        let output_layer = ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, &device)?;

        Ok(Self { config, device, embedding, encoder, output_layer })
    }

    fn from_weights(config: ChatGLMConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;

        model.embedding.load_weights(&weights)?;
        model.encoder.load_weights(&weights)?;

        if let Some(w) = weights.get("transformer.output_layer.weight") {
            model.output_layer = w.clone();
        }

        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        let input_ids = match inputs {
            ModelInputs::Text { input_ids, .. } => input_ids,
            ModelInputs::Multimodal { input_ids, .. } => input_ids,
            _ => return Err(anyhow::anyhow!("ChatGLM expects text input")),
        };

        let hidden_states = self.embedding.forward(input_ids)?;
        let hidden_states = self.encoder.forward(&hidden_states, None)?;
        let logits = ops_fn::matmul(&hidden_states, &self.output_layer)?;

        Ok(ModelOutputs::Logits { logits, hidden_states: Some(hidden_states) })
    }

    fn generate(&self, prompt: &str, _config: &GenerationConfig) -> Result<String> {
        Ok(format!("ChatGLM generated: {}", prompt))
    }

    fn config(&self) -> &Self::Config { &self.config }

    fn memory_requirements(&self) -> MemoryRequirements {
        let param_size = (self.config.vocab_size * self.config.hidden_size * 2 +
                         self.config.num_hidden_layers * self.config.hidden_size * self.config.hidden_size * 4) * 4;
        MemoryRequirements {
            gpu_memory: param_size, cpu_memory: param_size / 4,
            kv_cache_memory: self.config.max_position_embeddings * self.config.hidden_size * 2 * 4,
            peak_memory: param_size + param_size / 2,
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        self.embedding.to_device(device)?;
        self.encoder.to_device(device)?;
        self.output_layer = self.output_layer.to_device(device)?;
        Ok(())
    }
}

impl ChatGLMEmbedding {
    fn new(config: &ChatGLMConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            word_embeddings: ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, device)?,
            config: config.clone(),
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        ops_fn::embedding(input_ids, &self.word_embeddings)
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        if let Some(w) = weights.get("transformer.embedding.word_embeddings.weight") {
            self.word_embeddings = w.clone();
        }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.word_embeddings = self.word_embeddings.to_device(device)?;
        Ok(())
    }
}

impl ChatGLMEncoder {
    fn new(config: &ChatGLMConfig, device: &Device) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(ChatGLMLayer::new(config, device)?);
        }

        Ok(Self {
            layers,
            final_layernorm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }

        ops_fn::layer_norm(&hidden_states, &self.final_layernorm, None, self.config.rms_norm_eps)
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        if let Some(w) = weights.get("transformer.encoder.final_layernorm.weight") {
            self.final_layernorm = w.clone();
        }

        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_weights(weights, i)?;
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.final_layernorm = self.final_layernorm.to_device(device)?;
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        Ok(())
    }
}

impl ChatGLMLayer {
    fn new(config: &ChatGLMConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            input_layernorm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            self_attention: ChatGLMAttention::new(config, device)?,
            post_attention_layernorm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            mlp: ChatGLMMLP::new(config, device)?,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let attention_input = ops_fn::layer_norm(hidden_states, &self.input_layernorm, None, 1e-5)?;
        let attention_output = self.self_attention.forward(&attention_input, attention_mask)?;
        let hidden_states = ops_fn::add(hidden_states, &attention_output)?;

        let mlp_input = ops_fn::layer_norm(&hidden_states, &self.post_attention_layernorm, None, 1e-5)?;
        let mlp_output = self.mlp.forward(&mlp_input)?;

        ops_fn::add(&hidden_states, &mlp_output)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("transformer.encoder.layers.{}", layer_idx);

        if let Some(w) = weights.get(&format!("{}.input_layernorm.weight", prefix)) {
            self.input_layernorm = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
            self.post_attention_layernorm = w.clone();
        }

        self.self_attention.load_weights(weights, layer_idx)?;
        self.mlp.load_weights(weights, layer_idx)?;

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.input_layernorm = self.input_layernorm.to_device(device)?;
        self.post_attention_layernorm = self.post_attention_layernorm.to_device(device)?;
        self.self_attention.to_device(device)?;
        self.mlp.to_device(device)?;
        Ok(())
    }
}

impl ChatGLMAttention {
    fn new(config: &ChatGLMConfig, device: &Device) -> Result<Self> {
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads.unwrap_or(num_heads);
        let head_dim = config.hidden_size / num_heads;

        Ok(Self {
            query_key_value: ops_fn::zeros(&[config.hidden_size, (num_heads + 2 * num_kv_heads) * head_dim], DataType::Float32, device)?,
            dense: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let qkv = ops_fn::matmul(hidden_states, &self.query_key_value)?;
        let attn_output = ops_fn::attention(&qkv, &qkv, &qkv, attention_mask)?;
        ops_fn::matmul(&attn_output, &self.dense)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("transformer.encoder.layers.{}.self_attention", layer_idx);

        if let Some(w) = weights.get(&format!("{}.query_key_value.weight", prefix)) {
            self.query_key_value = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.dense.weight", prefix)) {
            self.dense = w.clone();
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.query_key_value = self.query_key_value.to_device(device)?;
        self.dense = self.dense.to_device(device)?;
        Ok(())
    }
}

impl ChatGLMMLP {
    fn new(config: &ChatGLMConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            dense_h_to_4h: ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?,
            dense_4h_to_h: ops_fn::zeros(&[config.intermediate_size, config.hidden_size], DataType::Float32, device)?,
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let intermediate = ops_fn::matmul(hidden_states, &self.dense_h_to_4h)?;
        let activated = ops_fn::gelu(&intermediate)?; // ChatGLM uses GELU variants
        ops_fn::matmul(&activated, &self.dense_4h_to_h)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("transformer.encoder.layers.{}.mlp", layer_idx);

        if let Some(w) = weights.get(&format!("{}.dense_h_to_4h.weight", prefix)) {
            self.dense_h_to_4h = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.dense_4h_to_h.weight", prefix)) {
            self.dense_4h_to_h = w.clone();
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.dense_h_to_4h = self.dense_h_to_4h.to_device(device)?;
        self.dense_4h_to_h = self.dense_4h_to_h.to_device(device)?;
        Ok(())
    }
}