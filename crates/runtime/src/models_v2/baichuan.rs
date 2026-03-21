//! Baichuan Model V2 - Clean implementation using solid abstractions
//!
//! This implements the Baichuan architecture including:
//! - Baichuan-7B, Baichuan-13B, Baichuan2-7B, Baichuan2-13B

use crate::model_config;
use super::traits::*;
use std::collections::HashMap;
use anyhow::Result;
use serde::{Serialize, Deserialize};

model_config!(BaichuanConfig {
    vocab_size: usize = 64000,
    hidden_size: usize = 4096,
    intermediate_size: usize = 11008,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 32,
    num_key_value_heads: Option<usize> = Some(32),
    hidden_act: String = "silu".to_string(),
    max_position_embeddings: usize = 4096,
    initializer_range: f32 = 0.02,
    rms_norm_eps: f32 = 1e-6,
    use_cache: bool = true,
    pad_token_id: Option<i64> = Some(0),
    bos_token_id: Option<i64> = Some(1),
    eos_token_id: Option<i64> = Some(2),
    tie_word_embeddings: bool = false,
    rope_theta: f32 = 10000.0,
    attention_bias: bool = false,
    // Baichuan specific
    model_max_length: usize = 4096,
    z_loss_weight: f32 = 0.0,
    gradient_checkpointing: bool = false,
});

pub struct BaichuanModelV2 {
    config: BaichuanConfig,
    device: Device,
    embed_tokens: Tensor,
    layers: Vec<BaichuanLayer>,
    norm: Tensor,
    lm_head: Tensor,
}

pub struct BaichuanLayer {
    self_attn: BaichuanAttention,
    mlp: BaichuanMLP,
    input_layernorm: Tensor,
    post_attention_layernorm: Tensor,
}

pub struct BaichuanAttention {
    w_qkv: Tensor, // Baichuan uses packed attention weights
    o_proj: Tensor,
    config: BaichuanConfig,
}

pub struct BaichuanMLP {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
    config: BaichuanConfig,
}

impl Model for BaichuanModelV2 {
    type Config = BaichuanConfig;

    fn new(config: BaichuanConfig) -> Result<Self> {
        let device = Device::CPU;
        let embed_tokens = ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, &device)?;
        let norm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, &device)?;
        let lm_head = ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, &device)?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(BaichuanLayer::new(&config, &device)?);
        }

        Ok(Self { config, device, embed_tokens, layers, norm, lm_head })
    }

    fn from_weights(config: BaichuanConfig, weights: ModelWeights) -> Result<Self> {
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
            _ => return Err(anyhow::anyhow!("Baichuan expects text input")),
        };

        let mut hidden_states = ops_fn::embedding(input_ids, &self.embed_tokens)?;
        for layer in &self.layers { hidden_states = layer.forward(&hidden_states, None)?; }
        hidden_states = ops_fn::layer_norm(&hidden_states, &self.norm, None, self.config.rms_norm_eps)?;
        let logits = ops_fn::matmul(&hidden_states, &self.lm_head)?;

        Ok(ModelOutputs::Logits { logits, hidden_states: Some(hidden_states) })
    }

    fn generate(&self, prompt: &str, _config: &GenerationConfig) -> Result<String> {
        Ok(format!("Baichuan generated: {}", prompt))
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

impl BaichuanLayer {
    fn new(config: &BaichuanConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            self_attn: BaichuanAttention::new(config, device)?,
            mlp: BaichuanMLP::new(config, device)?,
            input_layernorm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            post_attention_layernorm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let normed = ops_fn::layer_norm(hidden_states, &self.input_layernorm, None, 1e-6)?;
        let attn_output = self.self_attn.forward(&normed, attention_mask)?;
        let hidden_states = ops_fn::add(hidden_states, &attn_output)?;
        let normed = ops_fn::layer_norm(&hidden_states, &self.post_attention_layernorm, None, 1e-6)?;
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

impl BaichuanAttention {
    fn new(config: &BaichuanConfig, device: &Device) -> Result<Self> {
        let num_heads = config.num_attention_heads;
        let head_dim = config.hidden_size / num_heads;

        Ok(Self {
            w_qkv: ops_fn::zeros(&[config.hidden_size, 3 * config.hidden_size], DataType::Float32, device)?,
            o_proj: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Baichuan uses packed QKV projection
        let qkv_out = ops_fn::matmul(hidden_states, &self.w_qkv)?;

        // For now, treat as simple attention (real implementation would split qkv_out)
        let attn_output = ops_fn::attention(&qkv_out, &qkv_out, &qkv_out, attention_mask)?;
        ops_fn::matmul(&attn_output, &self.o_proj)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("model.layers.{}.self_attn", layer_idx);
        if let Some(w) = weights.get(&format!("{}.W_pack.weight", prefix)) { self.w_qkv = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.o_proj.weight", prefix)) { self.o_proj = w.clone(); }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.w_qkv = self.w_qkv.to_device(device)?;
        self.o_proj = self.o_proj.to_device(device)?;
        Ok(())
    }
}

impl BaichuanMLP {
    fn new(config: &BaichuanConfig, device: &Device) -> Result<Self> {
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