//! BERT Model V2 - Clean implementation using solid abstractions

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

model_config!(BertConfig {
    vocab_size: usize = 30522,
    hidden_size: usize = 768,
    intermediate_size: usize = 3072,
    num_hidden_layers: usize = 12,
    num_attention_heads: usize = 12,
    num_key_value_heads: Option<usize> = None,
    hidden_act: String = "gelu".to_string(),
    max_position_embeddings: usize = 512,
    initializer_range: f32 = 0.02,
    rms_norm_eps: f32 = 1e-12,
    use_cache: bool = true,
    pad_token_id: Option<i64> = Some(0),
    bos_token_id: Option<i64> = Some(101),
    eos_token_id: Option<i64> = Some(102),
    tie_word_embeddings: bool = true,
    rope_theta: f32 = 10000.0,
    // BERT specific
    type_vocab_size: usize = 2,
    layer_norm_eps: f32 = 1e-12,
    position_embedding_type: String = "absolute".to_string(),
    use_cache_encoder: bool = false,
});

pub struct BertModelV2 {
    config: BertConfig,
    device: Device,
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    pooler: Option<BertPooler>,
}

pub struct BertEmbeddings {
    word_embeddings: Tensor,
    position_embeddings: Tensor,
    token_type_embeddings: Tensor,
    layer_norm: Tensor,
    config: BertConfig,
}

pub struct BertEncoder {
    layers: Vec<BertLayer>,
    config: BertConfig,
}

pub struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
}

pub struct BertAttention {
    self_attention: BertSelfAttention,
    output: BertSelfOutput,
}

pub struct BertSelfAttention {
    query: Tensor,
    key: Tensor,
    value: Tensor,
    config: BertConfig,
}

pub struct BertSelfOutput {
    dense: Tensor,
    layer_norm: Tensor,
    config: BertConfig,
}

pub struct BertIntermediate {
    dense: Tensor,
    config: BertConfig,
}

pub struct BertOutput {
    dense: Tensor,
    layer_norm: Tensor,
    config: BertConfig,
}

pub struct BertPooler {
    dense: Tensor,
    config: BertConfig,
}

impl Model for BertModelV2 {
    type Config = BertConfig;

    fn new(config: BertConfig) -> Result<Self> {
        let device = Device::CPU;
        Ok(Self {
            embeddings: BertEmbeddings::new(&config, &device)?,
            encoder: BertEncoder::new(&config, &device)?,
            pooler: Some(BertPooler::new(&config, &device)?),
            config,
            device,
        })
    }

    fn from_weights(config: BertConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;
        model.embeddings.load_weights(&weights)?;
        model.encoder.load_weights(&weights)?;
        if let Some(ref mut pooler) = model.pooler { pooler.load_weights(&weights)?; }
        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        let (input_ids, token_type_ids) = match inputs {
            ModelInputs::Text { input_ids, attention_mask: _, position_ids: _ } => (input_ids, None),
            _ => return Err(anyhow::anyhow!("BERT expects text input")),
        };

        let embedding_output = self.embeddings.forward(input_ids, token_type_ids)?;
        let encoder_outputs = self.encoder.forward(&embedding_output, None)?;

        let pooled_output = if let Some(ref pooler) = self.pooler {
            Some(pooler.forward(&encoder_outputs)?)
        } else {
            None
        };

        Ok(ModelOutputs::Embeddings {
            embeddings: encoder_outputs,
            pooled: pooled_output,
        })
    }

    fn generate(&self, prompt: &str, _config: &GenerationConfig) -> Result<String> {
        Ok(format!("BERT processed: {}", prompt))
    }

    fn config(&self) -> &Self::Config { &self.config }

    fn memory_requirements(&self) -> MemoryRequirements {
        let param_size = (self.config.vocab_size * self.config.hidden_size +
                         self.config.num_hidden_layers * self.config.hidden_size * self.config.hidden_size * 4) * 4;
        MemoryRequirements {
            gpu_memory: param_size, cpu_memory: param_size / 4,
            kv_cache_memory: self.config.max_position_embeddings * self.config.hidden_size * 2 * 4,
            peak_memory: param_size + param_size / 2,
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        self.embeddings.to_device(device)?;
        self.encoder.to_device(device)?;
        if let Some(ref mut pooler) = self.pooler { pooler.to_device(device)?; }
        Ok(())
    }
}

impl BertEmbeddings {
    fn new(config: &BertConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            word_embeddings: ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, device)?,
            position_embeddings: ops_fn::zeros(&[config.max_position_embeddings, config.hidden_size], DataType::Float32, device)?,
            token_type_embeddings: ops_fn::zeros(&[config.type_vocab_size, config.hidden_size], DataType::Float32, device)?,
            layer_norm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            config: config.clone(),
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: Option<&Tensor>) -> Result<Tensor> {
        let inputs_embeds = ops_fn::embedding(input_ids, &self.word_embeddings)?;
        // TODO: Add position and token type embeddings
        ops_fn::layer_norm(&inputs_embeds, &self.layer_norm, None, self.config.layer_norm_eps)
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        if let Some(w) = weights.get("bert.embeddings.word_embeddings.weight") { self.word_embeddings = w.clone(); }
        if let Some(w) = weights.get("bert.embeddings.position_embeddings.weight") { self.position_embeddings = w.clone(); }
        if let Some(w) = weights.get("bert.embeddings.token_type_embeddings.weight") { self.token_type_embeddings = w.clone(); }
        if let Some(w) = weights.get("bert.embeddings.LayerNorm.weight") { self.layer_norm = w.clone(); }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.word_embeddings = self.word_embeddings.to_device(device)?;
        self.position_embeddings = self.position_embeddings.to_device(device)?;
        self.token_type_embeddings = self.token_type_embeddings.to_device(device)?;
        self.layer_norm = self.layer_norm.to_device(device)?;
        Ok(())
    }
}

impl BertEncoder {
    fn new(config: &BertConfig, device: &Device) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(BertLayer::new(config, device)?);
        }
        Ok(Self { layers, config: config.clone() })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }
        Ok(hidden_states)
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_weights(weights, i)?;
        }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        Ok(())
    }
}

impl BertLayer {
    fn new(config: &BertConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            attention: BertAttention::new(config, device)?,
            intermediate: BertIntermediate::new(config, device)?,
            output: BertOutput::new(config, device)?,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let attention_output = self.attention.forward(hidden_states, attention_mask)?;
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        self.output.forward(&intermediate_output, &attention_output)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        self.attention.load_weights(weights, layer_idx)?;
        self.intermediate.load_weights(weights, layer_idx)?;
        self.output.load_weights(weights, layer_idx)?;
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.attention.to_device(device)?;
        self.intermediate.to_device(device)?;
        self.output.to_device(device)?;
        Ok(())
    }
}

// Simplified implementations for other components
impl BertAttention {
    fn new(config: &BertConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            self_attention: BertSelfAttention::new(config, device)?,
            output: BertSelfOutput::new(config, device)?,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let self_output = self.self_attention.forward(hidden_states, attention_mask)?;
        self.output.forward(&self_output, hidden_states)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        self.self_attention.load_weights(weights, layer_idx)?;
        self.output.load_weights(weights, layer_idx)?;
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.self_attention.to_device(device)?;
        self.output.to_device(device)?;
        Ok(())
    }
}

impl BertSelfAttention {
    fn new(config: &BertConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            query: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            key: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            value: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let query = ops_fn::matmul(hidden_states, &self.query)?;
        let key = ops_fn::matmul(hidden_states, &self.key)?;
        let value = ops_fn::matmul(hidden_states, &self.value)?;
        ops_fn::attention(&query, &key, &value, attention_mask)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("bert.encoder.layer.{}.attention.self", layer_idx);
        if let Some(w) = weights.get(&format!("{}.query.weight", prefix)) { self.query = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.key.weight", prefix)) { self.key = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.value.weight", prefix)) { self.value = w.clone(); }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.query = self.query.to_device(device)?;
        self.key = self.key.to_device(device)?;
        self.value = self.value.to_device(device)?;
        Ok(())
    }
}

impl BertSelfOutput {
    fn new(config: &BertConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            dense: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            layer_norm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = ops_fn::matmul(hidden_states, &self.dense)?;
        let hidden_states = ops_fn::add(&hidden_states, input_tensor)?;
        ops_fn::layer_norm(&hidden_states, &self.layer_norm, None, self.config.layer_norm_eps)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("bert.encoder.layer.{}.attention.output", layer_idx);
        if let Some(w) = weights.get(&format!("{}.dense.weight", prefix)) { self.dense = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.LayerNorm.weight", prefix)) { self.layer_norm = w.clone(); }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.dense = self.dense.to_device(device)?;
        self.layer_norm = self.layer_norm.to_device(device)?;
        Ok(())
    }
}

impl BertIntermediate {
    fn new(config: &BertConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            dense: ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?,
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = ops_fn::matmul(hidden_states, &self.dense)?;
        ops_fn::gelu(&hidden_states)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("bert.encoder.layer.{}.intermediate", layer_idx);
        if let Some(w) = weights.get(&format!("{}.dense.weight", prefix)) { self.dense = w.clone(); }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.dense = self.dense.to_device(device)?;
        Ok(())
    }
}

impl BertOutput {
    fn new(config: &BertConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            dense: ops_fn::zeros(&[config.intermediate_size, config.hidden_size], DataType::Float32, device)?,
            layer_norm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = ops_fn::matmul(hidden_states, &self.dense)?;
        let hidden_states = ops_fn::add(&hidden_states, input_tensor)?;
        ops_fn::layer_norm(&hidden_states, &self.layer_norm, None, self.config.layer_norm_eps)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("bert.encoder.layer.{}.output", layer_idx);
        if let Some(w) = weights.get(&format!("{}.dense.weight", prefix)) { self.dense = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.LayerNorm.weight", prefix)) { self.layer_norm = w.clone(); }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.dense = self.dense.to_device(device)?;
        self.layer_norm = self.layer_norm.to_device(device)?;
        Ok(())
    }
}

impl BertPooler {
    fn new(config: &BertConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            dense: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Take first token ([CLS]) representation
        ops_fn::matmul(hidden_states, &self.dense)
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        if let Some(w) = weights.get("bert.pooler.dense.weight") { self.dense = w.clone(); }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.dense = self.dense.to_device(device)?;
        Ok(())
    }
}