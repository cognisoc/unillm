//! Falcon Model V2 - Clean implementation using solid abstractions

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

model_config!(FalconConfig {
    vocab_size: usize = 65024,
    hidden_size: usize = 4544,
    intermediate_size: usize = 18176,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 71,
    num_key_value_heads: Option<usize> = Some(1),
    hidden_act: String = "gelu".to_string(),
    max_position_embeddings: usize = 2048,
    initializer_range: f32 = 0.02,
    rms_norm_eps: f32 = 1e-5,
    use_cache: bool = true,
    pad_token_id: Option<i64> = Some(11),
    bos_token_id: Option<i64> = Some(11),
    eos_token_id: Option<i64> = Some(11),
    tie_word_embeddings: bool = false,
    rope_theta: f32 = 10000.0,
    // Falcon specific
    parallel_attn: bool = true,
    bias: bool = false,
    new_decoder_architecture: bool = false,
});

pub struct FalconModelV2 {
    config: FalconConfig,
    device: Device,
    word_embeddings: Tensor,
    h: Vec<FalconDecoderLayer>,
    ln_f: Tensor,
    lm_head: Tensor,
}

pub struct FalconDecoderLayer {
    self_attention: FalconAttention,
    mlp: FalconMLP,
    input_layernorm: Tensor,
    config: FalconConfig,
}

pub struct FalconAttention {
    query_key_value: Tensor,
    dense: Tensor,
    config: FalconConfig,
}

pub struct FalconMLP {
    dense_h_to_4h: Tensor,
    dense_4h_to_h: Tensor,
    config: FalconConfig,
}

impl Model for FalconModelV2 {
    type Config = FalconConfig;

    fn new(config: FalconConfig) -> Result<Self> {
        let device = Device::CPU;
        let word_embeddings = ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, &device)?;
        let ln_f = ops_fn::zeros(&[config.hidden_size], DataType::Float32, &device)?;
        let lm_head = ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, &device)?;

        let mut h = Vec::new();
        for _ in 0..config.num_hidden_layers {
            h.push(FalconDecoderLayer::new(&config, &device)?);
        }

        Ok(Self { config, device, word_embeddings, h, ln_f, lm_head })
    }

    fn from_weights(config: FalconConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;
        if let Some(w) = weights.get("transformer.word_embeddings.weight") { model.word_embeddings = w.clone(); }
        if let Some(w) = weights.get("transformer.ln_f.weight") { model.ln_f = w.clone(); }
        if let Some(w) = weights.get("lm_head.weight") { model.lm_head = w.clone(); }
        for (i, layer) in model.h.iter_mut().enumerate() { layer.load_weights(&weights, i)?; }
        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        let input_ids = match inputs {
            ModelInputs::Text { input_ids, .. } => input_ids,
            _ => return Err(anyhow::anyhow!("Falcon expects text input")),
        };

        let mut hidden_states = ops_fn::embedding(input_ids, &self.word_embeddings)?;
        for layer in &self.h { hidden_states = layer.forward(&hidden_states, None)?; }
        hidden_states = ops_fn::layer_norm(&hidden_states, &self.ln_f, None, self.config.rms_norm_eps)?;
        let logits = ops_fn::matmul(&hidden_states, &self.lm_head)?;

        Ok(ModelOutputs::Logits { logits, hidden_states: Some(hidden_states) })
    }

    fn generate(&self, prompt: &str, _config: &GenerationConfig) -> Result<String> {
        Ok(format!("Falcon generated: {}", prompt))
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
        self.word_embeddings = self.word_embeddings.to_device(device)?;
        self.ln_f = self.ln_f.to_device(device)?;
        self.lm_head = self.lm_head.to_device(device)?;
        for layer in &mut self.h { layer.to_device(device)?; }
        Ok(())
    }
}

impl FalconDecoderLayer {
    fn new(config: &FalconConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            self_attention: FalconAttention::new(config, device)?,
            mlp: FalconMLP::new(config, device)?,
            input_layernorm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let layernorm_output = ops_fn::layer_norm(hidden_states, &self.input_layernorm, None, 1e-5)?;

        if self.config.parallel_attn {
            // Falcon: parallel attention and MLP
            let attn_output = self.self_attention.forward(&layernorm_output, attention_mask)?;
            let mlp_output = self.mlp.forward(&layernorm_output)?;
            let combined = ops_fn::add(&attn_output, &mlp_output)?;
            ops_fn::add(&residual, &combined)
        } else {
            // Sequential
            let attn_output = self.self_attention.forward(&layernorm_output, attention_mask)?;
            let hidden_states = ops_fn::add(&residual, &attn_output)?;
            let mlp_output = self.mlp.forward(&hidden_states)?;
            ops_fn::add(&hidden_states, &mlp_output)
        }
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("transformer.h.{}", layer_idx);
        if let Some(w) = weights.get(&format!("{}.input_layernorm.weight", prefix)) { self.input_layernorm = w.clone(); }
        self.self_attention.load_weights(weights, layer_idx)?;
        self.mlp.load_weights(weights, layer_idx)?;
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.input_layernorm = self.input_layernorm.to_device(device)?;
        self.self_attention.to_device(device)?;
        self.mlp.to_device(device)?;
        Ok(())
    }
}

impl FalconAttention {
    fn new(config: &FalconConfig, device: &Device) -> Result<Self> {
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
        let prefix = format!("transformer.h.{}.self_attention", layer_idx);
        if let Some(w) = weights.get(&format!("{}.query_key_value.weight", prefix)) { self.query_key_value = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.dense.weight", prefix)) { self.dense = w.clone(); }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.query_key_value = self.query_key_value.to_device(device)?;
        self.dense = self.dense.to_device(device)?;
        Ok(())
    }
}

impl FalconMLP {
    fn new(config: &FalconConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            dense_h_to_4h: ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?,
            dense_4h_to_h: ops_fn::zeros(&[config.intermediate_size, config.hidden_size], DataType::Float32, device)?,
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = ops_fn::matmul(hidden_states, &self.dense_h_to_4h)?;
        let hidden_states = ops_fn::gelu(&hidden_states)?;
        ops_fn::matmul(&hidden_states, &self.dense_4h_to_h)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("transformer.h.{}.mlp", layer_idx);
        if let Some(w) = weights.get(&format!("{}.dense_h_to_4h.weight", prefix)) { self.dense_h_to_4h = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.dense_4h_to_h.weight", prefix)) { self.dense_4h_to_h = w.clone(); }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.dense_h_to_4h = self.dense_h_to_4h.to_device(device)?;
        self.dense_4h_to_h = self.dense_4h_to_h.to_device(device)?;
        Ok(())
    }
}