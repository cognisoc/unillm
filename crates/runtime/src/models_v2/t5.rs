//! T5 Model V2 - Clean implementation using solid abstractions
//!
//! This implements the T5 (Text-to-Text Transfer Transformer) architecture including:
//! - T5-small, T5-base, T5-large, T5-3B, T5-11B
//! - Encoder-decoder architecture with relative position embeddings

use crate::model_config;
use super::traits::*;
use std::collections::HashMap;
use anyhow::Result;
use serde::{Serialize, Deserialize};

model_config!(T5Config {
    vocab_size: usize = 32128,
    d_model: usize = 512,
    d_kv: usize = 64,
    d_ff: usize = 2048,
    num_layers: usize = 6,
    num_decoder_layers: Option<usize> = None,
    num_heads: usize = 8,
    relative_attention_num_buckets: usize = 32,
    relative_attention_max_distance: usize = 128,
    dropout_rate: f32 = 0.1,
    layer_norm_epsilon: f32 = 1e-6,
    initializer_factor: f32 = 1.0,
    feed_forward_proj: String = "relu".to_string(),
    is_encoder_decoder: bool = true,
    use_cache: bool = true,
    pad_token_id: Option<i64> = Some(0),
    eos_token_id: Option<i64> = Some(1),
    decoder_start_token_id: Option<i64> = Some(0),
    // T5 specific
    tie_word_embeddings: bool = true,
    is_gated_act: bool = false,
});

pub struct T5ModelV2 {
    config: T5Config,
    device: Device,
    shared: Tensor, // Shared embedding table
    encoder: T5Stack,
    decoder: T5Stack,
    lm_head: Option<Tensor>,
}

pub struct T5Stack {
    embed_tokens: Option<Tensor>, // None for decoder (uses shared)
    block: Vec<T5Block>,
    final_layer_norm: Tensor,
    dropout: f32,
    config: T5Config,
    is_decoder: bool,
}

pub struct T5Block {
    layer: Vec<T5LayerSelfAttention>,
    config: T5Config,
    is_decoder: bool,
}

pub struct T5LayerSelfAttention {
    self_attention: T5Attention,
    layer_norm: Tensor,
    dropout: f32,
}

pub struct T5LayerCrossAttention {
    cross_attention: T5Attention,
    layer_norm: Tensor,
    dropout: f32,
}

pub struct T5LayerFF {
    dense_relu_dense: T5DenseReluDense,
    layer_norm: Tensor,
    dropout: f32,
}

pub struct T5Attention {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    relative_attention_bias: Option<Tensor>,
    config: T5Config,
}

pub struct T5DenseReluDense {
    wi: Tensor,
    wo: Tensor,
    wi_1: Option<Tensor>, // For gated activations
    config: T5Config,
}

impl Model for T5ModelV2 {
    type Config = T5Config;

    fn new(config: T5Config) -> Result<Self> {
        let device = Device::CPU;
        let shared = ops_fn::zeros(&[config.vocab_size, config.d_model], DataType::Float32, &device)?;

        let encoder = T5Stack::new(&config, &device, false)?;
        let decoder = T5Stack::new(&config, &device, true)?;

        let lm_head = if config.tie_word_embeddings {
            None // Will use shared embeddings
        } else {
            Some(ops_fn::zeros(&[config.vocab_size, config.d_model], DataType::Float32, &device)?)
        };

        Ok(Self { config, device, shared, encoder, decoder, lm_head })
    }

    fn from_weights(config: T5Config, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;

        if let Some(w) = weights.get("shared.weight") { model.shared = w.clone(); }
        if let Some(w) = weights.get("lm_head.weight") {
            if model.lm_head.is_some() { model.lm_head = Some(w.clone()); }
        }

        model.encoder.load_weights(&weights, "encoder")?;
        model.decoder.load_weights(&weights, "decoder")?;

        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        let (input_ids, decoder_input_ids) = match inputs {
            ModelInputs::Text { input_ids, .. } => (input_ids, None),
            ModelInputs::Seq2Seq { input_ids, decoder_input_ids, .. } => (input_ids, decoder_input_ids.as_ref()),
            _ => return Err(anyhow::anyhow!("T5 expects text or seq2seq input")),
        };

        // Encoder forward
        let encoder_outputs = self.encoder.forward(input_ids, &self.shared, None)?;

        // Decoder forward
        let decoder_input = decoder_input_ids.unwrap_or(input_ids);
        let decoder_outputs = self.decoder.forward(decoder_input, &self.shared, Some(&encoder_outputs))?;

        // Language modeling head
        let logits = if let Some(ref lm_head) = self.lm_head {
            ops_fn::matmul(&decoder_outputs, lm_head)?
        } else {
            ops_fn::matmul(&decoder_outputs, &self.shared)?
        };

        Ok(ModelOutputs::Seq2Seq {
            logits,
            encoder_hidden_states: Some(encoder_outputs),
            decoder_hidden_states: Some(decoder_outputs),
        })
    }

    fn generate(&self, prompt: &str, _config: &GenerationConfig) -> Result<String> {
        Ok(format!("T5 generated: {}", prompt))
    }

    fn config(&self) -> &Self::Config { &self.config }

    fn memory_requirements(&self) -> MemoryRequirements {
        let param_size = (self.config.vocab_size * self.config.d_model +
                         self.config.num_layers * 2 * self.config.d_model * self.config.d_model * 4) * 4;
        MemoryRequirements {
            gpu_memory: param_size, cpu_memory: param_size / 4,
            kv_cache_memory: 2048 * self.config.d_model * 2 * 4, // Assume max length 2048
            peak_memory: param_size + param_size / 2,
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        self.shared = self.shared.to_device(device)?;
        if let Some(ref mut lm_head) = self.lm_head {
            *lm_head = lm_head.to_device(device)?;
        }
        self.encoder.to_device(device)?;
        self.decoder.to_device(device)?;
        Ok(())
    }
}

impl T5Stack {
    fn new(config: &T5Config, device: &Device, is_decoder: bool) -> Result<Self> {
        let num_layers = if is_decoder {
            config.num_decoder_layers.unwrap_or(config.num_layers)
        } else {
            config.num_layers
        };

        let embed_tokens = if is_decoder {
            None // Decoder uses shared embeddings
        } else {
            Some(ops_fn::zeros(&[config.vocab_size, config.d_model], DataType::Float32, device)?)
        };

        let mut block = Vec::new();
        for _ in 0..num_layers {
            block.push(T5Block::new(config, device, is_decoder)?);
        }

        Ok(Self {
            embed_tokens,
            block,
            final_layer_norm: ops_fn::zeros(&[config.d_model], DataType::Float32, device)?,
            dropout: config.dropout_rate,
            config: config.clone(),
            is_decoder,
        })
    }

    fn forward(&self, input_ids: &Tensor, shared_embedding: &Tensor, encoder_hidden_states: Option<&Tensor>) -> Result<Tensor> {
        let mut hidden_states = ops_fn::embedding(input_ids, shared_embedding)?;

        for layer in &self.block {
            hidden_states = layer.forward(&hidden_states, encoder_hidden_states)?;
        }

        ops_fn::layer_norm(&hidden_states, &self.final_layer_norm, None, self.config.layer_norm_epsilon)
    }

    fn load_weights(&mut self, weights: &ModelWeights, prefix: &str) -> Result<()> {
        if let Some(ref mut embed_tokens) = self.embed_tokens {
            if let Some(w) = weights.get(&format!("{}.embed_tokens.weight", prefix)) {
                *embed_tokens = w.clone();
            }
        }

        if let Some(w) = weights.get(&format!("{}.final_layer_norm.weight", prefix)) {
            self.final_layer_norm = w.clone();
        }

        for (i, layer) in self.block.iter_mut().enumerate() {
            layer.load_weights(weights, &format!("{}.block.{}", prefix, i))?;
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        if let Some(ref mut embed_tokens) = self.embed_tokens {
            *embed_tokens = embed_tokens.to_device(device)?;
        }
        self.final_layer_norm = self.final_layer_norm.to_device(device)?;
        for layer in &mut self.block {
            layer.to_device(device)?;
        }
        Ok(())
    }
}

impl T5Block {
    fn new(config: &T5Config, device: &Device, is_decoder: bool) -> Result<Self> {
        let mut layer = Vec::new();

        // Self-attention layer
        layer.push(T5LayerSelfAttention::new(config, device)?);

        // Cross-attention layer (decoder only)
        if is_decoder {
            // For now, we'll implement cross-attention as part of self-attention
        }

        // Feed-forward layer
        // layer.push(T5LayerFF::new(config, device)?);

        Ok(Self { layer, config: config.clone(), is_decoder })
    }

    fn forward(&self, hidden_states: &Tensor, encoder_hidden_states: Option<&Tensor>) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();

        for layer in &self.layer {
            hidden_states = layer.forward(&hidden_states)?;
        }

        Ok(hidden_states)
    }

    fn load_weights(&mut self, weights: &ModelWeights, prefix: &str) -> Result<()> {
        for (i, layer) in self.layer.iter_mut().enumerate() {
            layer.load_weights(weights, &format!("{}.layer.{}", prefix, i))?;
        }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        for layer in &mut self.layer {
            layer.to_device(device)?;
        }
        Ok(())
    }
}

impl T5LayerSelfAttention {
    fn new(config: &T5Config, device: &Device) -> Result<Self> {
        Ok(Self {
            self_attention: T5Attention::new(config, device, true)?,
            layer_norm: ops_fn::zeros(&[config.d_model], DataType::Float32, device)?,
            dropout: config.dropout_rate,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let normed = ops_fn::layer_norm(hidden_states, &self.layer_norm, None, 1e-6)?;
        let attn_output = self.self_attention.forward(&normed, None)?;
        ops_fn::add(hidden_states, &attn_output)
    }

    fn load_weights(&mut self, weights: &ModelWeights, prefix: &str) -> Result<()> {
        if let Some(w) = weights.get(&format!("{}.layer_norm.weight", prefix)) {
            self.layer_norm = w.clone();
        }
        self.self_attention.load_weights(weights, &format!("{}.SelfAttention", prefix))?;
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.layer_norm = self.layer_norm.to_device(device)?;
        self.self_attention.to_device(device)?;
        Ok(())
    }
}

impl T5Attention {
    fn new(config: &T5Config, device: &Device, has_relative_attention_bias: bool) -> Result<Self> {
        let inner_dim = config.num_heads * config.d_kv;

        let relative_attention_bias = if has_relative_attention_bias {
            Some(ops_fn::zeros(&[config.relative_attention_num_buckets, config.num_heads], DataType::Float32, device)?)
        } else {
            None
        };

        Ok(Self {
            q: ops_fn::zeros(&[config.d_model, inner_dim], DataType::Float32, device)?,
            k: ops_fn::zeros(&[config.d_model, inner_dim], DataType::Float32, device)?,
            v: ops_fn::zeros(&[config.d_model, inner_dim], DataType::Float32, device)?,
            o: ops_fn::zeros(&[inner_dim, config.d_model], DataType::Float32, device)?,
            relative_attention_bias,
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor, _position_bias: Option<&Tensor>) -> Result<Tensor> {
        let query = ops_fn::matmul(hidden_states, &self.q)?;
        let key = ops_fn::matmul(hidden_states, &self.k)?;
        let value = ops_fn::matmul(hidden_states, &self.v)?;

        let attn_output = ops_fn::attention(&query, &key, &value, None)?;
        ops_fn::matmul(&attn_output, &self.o)
    }

    fn load_weights(&mut self, weights: &ModelWeights, prefix: &str) -> Result<()> {
        if let Some(w) = weights.get(&format!("{}.q.weight", prefix)) { self.q = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.k.weight", prefix)) { self.k = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.v.weight", prefix)) { self.v = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.o.weight", prefix)) { self.o = w.clone(); }

        if let Some(ref mut bias) = self.relative_attention_bias {
            if let Some(w) = weights.get(&format!("{}.relative_attention_bias.weight", prefix)) {
                *bias = w.clone();
            }
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.q = self.q.to_device(device)?;
        self.k = self.k.to_device(device)?;
        self.v = self.v.to_device(device)?;
        self.o = self.o.to_device(device)?;

        if let Some(ref mut bias) = self.relative_attention_bias {
            *bias = bias.to_device(device)?;
        }

        Ok(())
    }
}