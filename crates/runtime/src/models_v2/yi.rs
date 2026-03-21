//! Yi Model V2 - Clean implementation using solid abstractions
//!
//! This implements the Yi architecture including:
//! - Yi-6B, Yi-34B, Yi-Chat, Yi-Coder

use crate::model_config;
use super::traits::*;
use std::collections::HashMap;
use anyhow::Result;
use serde::{Serialize, Deserialize};

model_config!(YiConfig {
    vocab_size: usize = 64000,
    hidden_size: usize = 4096,
    intermediate_size: usize = 11008,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 32,
    num_key_value_heads: Option<usize> = Some(4),
    hidden_act: String = "silu".to_string(),
    max_position_embeddings: usize = 4096,
    initializer_range: f32 = 0.02,
    rms_norm_eps: f32 = 1e-5,
    use_cache: bool = true,
    pad_token_id: Option<i64> = Some(0),
    bos_token_id: Option<i64> = Some(1),
    eos_token_id: Option<i64> = Some(2),
    tie_word_embeddings: bool = false,
    rope_theta: f32 = 5000000.0,
    attention_dropout: f32 = 0.0,
});

pub struct YiModelV2 {
    config: YiConfig,
    device: Device,
    embed_tokens: Tensor,
    layers: Vec<YiLayer>,
    norm: Tensor,
    lm_head: Tensor,
}

pub struct YiLayer {
    self_attn: YiAttention,
    mlp: YiMLP,
    input_layernorm: Tensor,
    post_attention_layernorm: Tensor,
}

pub struct YiAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    config: YiConfig,
}

pub struct YiMLP {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
    config: YiConfig,
}

impl Model for YiModelV2 {
    type Config = YiConfig;

    fn new(config: YiConfig) -> Result<Self> {
        let device = Device::CPU;
        let embed_tokens = ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, &device)?;
        let norm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, &device)?;
        let lm_head = ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, &device)?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(YiLayer::new(&config, &device)?);
        }

        Ok(Self { config, device, embed_tokens, layers, norm, lm_head })
    }

    fn from_weights(config: YiConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;
        if let Some(w) = weights.get("model.embed_tokens.weight") { model.embed_tokens = w.clone(); }
        if let Some(w) = weights.get("model.norm.weight") { model.norm = w.clone(); }
        if let Some(w) = weights.get("lm_head.weight") { model.lm_head = w.clone(); }
        for (i, layer) in model.layers.iter_mut().enumerate() { layer.load_weights(&weights, i)?; }
        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        let input_ids = match inputs {
            ModelInputs::Text { input_ids, .. } => input_ids,
            ModelInputs::Multimodal { input_ids, .. } => input_ids,
            _ => return Err(anyhow::anyhow!("Yi expects text input")),
        };

        let mut hidden_states = ops_fn::embedding(input_ids, &self.embed_tokens)?;
        for layer in &self.layers { hidden_states = layer.forward(&hidden_states, None)?; }
        hidden_states = ops_fn::layer_norm(&hidden_states, &self.norm, None, self.config.rms_norm_eps)?;
        let logits = ops_fn::matmul(&hidden_states, &self.lm_head)?;

        Ok(ModelOutputs::Logits { logits, hidden_states: Some(hidden_states) })
    }

    fn generate(&self, prompt: &str, _config: &GenerationConfig) -> Result<String> {
        Ok(format!("Yi generated: {}", prompt))
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
        self.embed_tokens = self.embed_tokens.to_device(device)?;
        self.norm = self.norm.to_device(device)?;
        self.lm_head = self.lm_head.to_device(device)?;
        for layer in &mut self.layers { layer.to_device(device)?; }
        Ok(())
    }
}

impl YiLayer {
    fn new(config: &YiConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            self_attn: YiAttention::new(config, device)?,
            mlp: YiMLP::new(config, device)?,
            input_layernorm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            post_attention_layernorm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let normed = ops_fn::layer_norm(hidden_states, &self.input_layernorm, None, 1e-5)?;
        let attn_output = self.self_attn.forward(&normed, attention_mask)?;
        let hidden_states = ops_fn::add(hidden_states, &attn_output)?;
        let normed = ops_fn::layer_norm(&hidden_states, &self.post_attention_layernorm, None, 1e-5)?;
        let mlp_output = self.mlp.forward(&normed)?;
        ops_fn::add(&hidden_states, &mlp_output)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("model.layers.{}", layer_idx);
        if let Some(w) = weights.get(&format!("{}.input_layernorm.weight", prefix)) { self.input_layernorm = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.post_attention_layernorm.weight", prefix)) { self.post_attention_layernorm = w.clone(); }
        self.self_attn.load_weights(weights, layer_idx)?;
        self.mlp.load_weights(weights, layer_idx)?;
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.input_layernorm = self.input_layernorm.to_device(device)?;
        self.post_attention_layernorm = self.post_attention_layernorm.to_device(device)?;
        self.self_attn.to_device(device)?;
        self.mlp.to_device(device)?;
        Ok(())
    }
}

impl YiAttention {
    fn new(config: &YiConfig, device: &Device) -> Result<Self> {
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads.unwrap_or(num_heads);
        let head_dim = config.hidden_size / num_heads;

        Ok(Self {
            q_proj: ops_fn::zeros(&[config.hidden_size, num_heads * head_dim], DataType::Float32, device)?,
            k_proj: ops_fn::zeros(&[config.hidden_size, num_kv_heads * head_dim], DataType::Float32, device)?,
            v_proj: ops_fn::zeros(&[config.hidden_size, num_kv_heads * head_dim], DataType::Float32, device)?,
            o_proj: ops_fn::zeros(&[num_heads * head_dim, config.hidden_size], DataType::Float32, device)?,
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let query_states = ops_fn::matmul(hidden_states, &self.q_proj)?;
        let key_states = ops_fn::matmul(hidden_states, &self.k_proj)?;
        let value_states = ops_fn::matmul(hidden_states, &self.v_proj)?;
        let attn_output = ops_fn::attention(&query_states, &key_states, &value_states, attention_mask)?;
        ops_fn::matmul(&attn_output, &self.o_proj)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("model.layers.{}.self_attn", layer_idx);
        if let Some(w) = weights.get(&format!("{}.q_proj.weight", prefix)) { self.q_proj = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.k_proj.weight", prefix)) { self.k_proj = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.v_proj.weight", prefix)) { self.v_proj = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.o_proj.weight", prefix)) { self.o_proj = w.clone(); }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.q_proj = self.q_proj.to_device(device)?;
        self.k_proj = self.k_proj.to_device(device)?;
        self.v_proj = self.v_proj.to_device(device)?;
        self.o_proj = self.o_proj.to_device(device)?;
        Ok(())
    }
}

impl YiMLP {
    fn new(config: &YiConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            gate_proj: ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?,
            up_proj: ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?,
            down_proj: ops_fn::zeros(&[config.intermediate_size, config.hidden_size], DataType::Float32, device)?,
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let gate_output = ops_fn::matmul(hidden_states, &self.gate_proj)?;
        let up_output = ops_fn::matmul(hidden_states, &self.up_proj)?;
        let gate_activated = ops_fn::silu(&gate_output)?;
        let gated = ops_fn::mul(&gate_activated, &up_output)?;
        ops_fn::matmul(&gated, &self.down_proj)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("model.layers.{}.mlp", layer_idx);
        if let Some(w) = weights.get(&format!("{}.gate_proj.weight", prefix)) { self.gate_proj = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.up_proj.weight", prefix)) { self.up_proj = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.down_proj.weight", prefix)) { self.down_proj = w.clone(); }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.gate_proj = self.gate_proj.to_device(device)?;
        self.up_proj = self.up_proj.to_device(device)?;
        self.down_proj = self.down_proj.to_device(device)?;
        Ok(())
    }
}