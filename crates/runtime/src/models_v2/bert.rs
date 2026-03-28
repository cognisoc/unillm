//! BERT Model V2 - Clean implementation using solid abstractions
//!
//! BERT is an encoder-only model with key differences from decoder-only models:
//! - Returns embeddings (ModelOutputs::Embeddings) instead of logits
//! - Bidirectional attention (NO causal masking)
//! - Uses position embeddings (learned) instead of RoPE
//! - Uses token type embeddings for segment distinction
//! - Uses LayerNorm (not RMSNorm)
//! - Has a pooler for [CLS] token representation

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
    hidden_act: String = "gelu".to_string(),
    max_position_embeddings: usize = 512,
    initializer_range: f32 = 0.02,
    layer_norm_eps: f32 = 1e-12,
    pad_token_id: i64 = 0,
    // BERT specific
    type_vocab_size: usize = 2,
    // For ModelConfig trait compatibility
    rms_norm_eps: f32 = 1e-12,
});

impl BertConfig {
    /// Create BertConfig from GGUF model configuration
    pub fn from_gguf_config(gguf: &crate::weight_loader_core::GGUFModelConfig) -> Self {
        Self {
            vocab_size: gguf.vocab_size,
            hidden_size: gguf.hidden_size,
            intermediate_size: gguf.intermediate_size,
            num_hidden_layers: gguf.num_hidden_layers,
            num_attention_heads: gguf.num_attention_heads,
            max_position_embeddings: gguf.max_position_embeddings,
            layer_norm_eps: gguf.rms_norm_eps,  // GGUF uses rms_norm_eps field
            ..Default::default()
        }
    }
}

/// Main BERT model implementation
pub struct BertModelV2 {
    config: BertConfig,
    device: Device,
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    pooler: Option<BertPooler>,
}

/// BERT embeddings: word + position + token_type embeddings, then LayerNorm
pub struct BertEmbeddings {
    word_embeddings: Tensor,
    position_embeddings: Tensor,
    token_type_embeddings: Tensor,
    layer_norm_weight: Tensor,
    layer_norm_bias: Tensor,
    config: BertConfig,
}

/// BERT encoder: stack of BertLayer
pub struct BertEncoder {
    layers: Vec<BertLayer>,
    #[allow(dead_code)]
    config: BertConfig,
}

/// BERT transformer layer
pub struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
}

/// BERT attention: self-attention + output projection with residual
pub struct BertAttention {
    self_attention: BertSelfAttention,
    output: BertSelfOutput,
}

/// BERT self-attention (bidirectional, no causal mask)
pub struct BertSelfAttention {
    query: Tensor,
    query_bias: Tensor,
    key: Tensor,
    key_bias: Tensor,
    value: Tensor,
    value_bias: Tensor,
    num_attention_heads: usize,
    head_dim: usize,
    scale: f32,
}

/// BERT self-attention output projection
pub struct BertSelfOutput {
    dense: Tensor,
    dense_bias: Tensor,
    layer_norm_weight: Tensor,
    layer_norm_bias: Tensor,
    layer_norm_eps: f32,
}

/// BERT intermediate (first FFN layer with activation)
pub struct BertIntermediate {
    dense: Tensor,
    dense_bias: Tensor,
    hidden_act: String,
}

/// BERT output (second FFN layer with residual and LayerNorm)
pub struct BertOutput {
    dense: Tensor,
    dense_bias: Tensor,
    layer_norm_weight: Tensor,
    layer_norm_bias: Tensor,
    layer_norm_eps: f32,
}

/// BERT pooler: takes [CLS] token and projects through dense + tanh
pub struct BertPooler {
    dense: Tensor,
    dense_bias: Tensor,
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
        if let Some(ref mut pooler) = model.pooler {
            pooler.load_weights(&weights)?;
        }
        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        let (input_ids, attention_mask, token_type_ids) = match inputs {
            ModelInputs::Text { input_ids, attention_mask, position_ids: _ } => {
                (input_ids, attention_mask.as_ref(), None)
            }
            _ => return Err(anyhow::anyhow!("BERT expects text input")),
        };

        // 1. Embeddings: word + position + token_type, then LayerNorm
        let embedding_output = self.embeddings.forward(input_ids, token_type_ids)?;

        // 2. Encoder: stack of transformer layers with bidirectional attention
        let encoder_outputs = self.encoder.forward(&embedding_output, attention_mask)?;

        // 3. Pooler: extract [CLS] token representation
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

    fn generate(&self, _prompt: &str, _config: &GenerationConfig) -> Result<String> {
        Err(anyhow::anyhow!(
            "BERT is an encoder-only model and cannot generate text. \
             Use forward() to get embeddings instead."
        ))
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn memory_requirements(&self) -> MemoryRequirements {
        let param_size = (self.config.vocab_size * self.config.hidden_size
            + self.config.max_position_embeddings * self.config.hidden_size
            + self.config.type_vocab_size * self.config.hidden_size
            + self.config.num_hidden_layers * self.config.hidden_size * self.config.hidden_size * 4
            + self.config.num_hidden_layers * self.config.hidden_size * self.config.intermediate_size * 2)
            * 4;
        MemoryRequirements {
            gpu_memory: param_size,
            cpu_memory: param_size / 4,
            kv_cache_memory: 0,  // BERT doesn't use KV cache (no autoregressive generation)
            peak_memory: param_size + param_size / 2,
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        self.embeddings.to_device(device)?;
        self.encoder.to_device(device)?;
        if let Some(ref mut pooler) = self.pooler {
            pooler.to_device(device)?;
        }
        Ok(())
    }
}

impl BertEmbeddings {
    fn new(config: &BertConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            word_embeddings: ops_fn::zeros(
                &[config.vocab_size, config.hidden_size],
                DataType::Float32,
                device,
            )?,
            position_embeddings: ops_fn::zeros(
                &[config.max_position_embeddings, config.hidden_size],
                DataType::Float32,
                device,
            )?,
            token_type_embeddings: ops_fn::zeros(
                &[config.type_vocab_size, config.hidden_size],
                DataType::Float32,
                device,
            )?,
            layer_norm_weight: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            layer_norm_bias: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            config: config.clone(),
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: Option<&Tensor>) -> Result<Tensor> {
        let shape = input_ids.shape();
        let seq_len = if shape.len() == 2 { shape[1] } else { shape[0] };

        // 1. Word embeddings
        let inputs_embeds = ops_fn::embedding(input_ids, &self.word_embeddings)?;

        // 2. Position embeddings (learned, absolute)
        // Create position_ids: [0, 1, 2, ..., seq_len-1]
        let position_ids: Vec<i64> = (0..seq_len as i64).collect();
        let batch_size = if shape.len() == 2 { shape[0] } else { 1 };
        let position_ids_expanded: Vec<i64> = (0..batch_size)
            .flat_map(|_| position_ids.iter().cloned())
            .collect();
        let position_ids_tensor = Tensor::from_i64_slice(
            &position_ids_expanded,
            &[batch_size, seq_len],
            input_ids.device(),
        )?;
        let position_embeds = ops_fn::embedding(&position_ids_tensor, &self.position_embeddings)?;

        // 3. Token type embeddings (segment embeddings)
        let token_type_embeds = if let Some(tt_ids) = token_type_ids {
            ops_fn::embedding(tt_ids, &self.token_type_embeddings)?
        } else {
            // Default to all zeros (segment A)
            let zeros: Vec<i64> = vec![0; batch_size * seq_len];
            let tt_tensor = Tensor::from_i64_slice(&zeros, &[batch_size, seq_len], input_ids.device())?;
            ops_fn::embedding(&tt_tensor, &self.token_type_embeddings)?
        };

        // 4. Sum all embeddings
        let embeddings = ops_fn::add(&inputs_embeds, &position_embeds)?;
        let embeddings = ops_fn::add(&embeddings, &token_type_embeds)?;

        // 5. Layer normalization with bias
        self.layer_norm(&embeddings)
    }

    fn layer_norm(&self, input: &Tensor) -> Result<Tensor> {
        let x = input.to_candle()?;
        let w = self.layer_norm_weight.to_candle()?;
        let b = self.layer_norm_bias.to_candle()?;

        let last_dim = x.dims().len() - 1;
        let mean = x.mean_keepdim(last_dim)?;
        let x_centered = x.broadcast_sub(&mean)?;
        let variance = x_centered.sqr()?.mean_keepdim(last_dim)?;
        let std = (variance + self.config.layer_norm_eps as f64)?.sqrt()?;
        let normalized = x_centered.broadcast_div(&std)?;
        let scaled = normalized.broadcast_mul(&w)?;
        let result = scaled.broadcast_add(&b)?;

        Ok(Tensor::from_candle(result))
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        // Try different weight naming conventions
        let word_keys = [
            "bert.embeddings.word_embeddings.weight",
            "embeddings.word_embeddings.weight",
        ];
        for key in word_keys {
            if let Some(w) = weights.get(key) {
                self.word_embeddings = w.clone();
                break;
            }
        }

        let pos_keys = [
            "bert.embeddings.position_embeddings.weight",
            "embeddings.position_embeddings.weight",
        ];
        for key in pos_keys {
            if let Some(w) = weights.get(key) {
                self.position_embeddings = w.clone();
                break;
            }
        }

        let tt_keys = [
            "bert.embeddings.token_type_embeddings.weight",
            "embeddings.token_type_embeddings.weight",
        ];
        for key in tt_keys {
            if let Some(w) = weights.get(key) {
                self.token_type_embeddings = w.clone();
                break;
            }
        }

        let ln_weight_keys = [
            "bert.embeddings.LayerNorm.weight",
            "embeddings.LayerNorm.weight",
            "bert.embeddings.LayerNorm.gamma",
        ];
        for key in ln_weight_keys {
            if let Some(w) = weights.get(key) {
                self.layer_norm_weight = w.clone();
                break;
            }
        }

        let ln_bias_keys = [
            "bert.embeddings.LayerNorm.bias",
            "embeddings.LayerNorm.bias",
            "bert.embeddings.LayerNorm.beta",
        ];
        for key in ln_bias_keys {
            if let Some(w) = weights.get(key) {
                self.layer_norm_bias = w.clone();
                break;
            }
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.word_embeddings = self.word_embeddings.to_device(device)?;
        self.position_embeddings = self.position_embeddings.to_device(device)?;
        self.token_type_embeddings = self.token_type_embeddings.to_device(device)?;
        self.layer_norm_weight = self.layer_norm_weight.to_device(device)?;
        self.layer_norm_bias = self.layer_norm_bias.to_device(device)?;
        Ok(())
    }
}

impl BertEncoder {
    fn new(config: &BertConfig, device: &Device) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(BertLayer::new(config, device)?);
        }
        Ok(Self {
            layers,
            config: config.clone(),
        })
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
        // Self-attention with residual
        let attention_output = self.attention.forward(hidden_states, attention_mask)?;
        // FFN with residual
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
        let head_dim = config.hidden_size / config.num_attention_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        Ok(Self {
            query: ops_fn::zeros(
                &[config.hidden_size, config.hidden_size],
                DataType::Float32,
                device,
            )?,
            query_bias: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            key: ops_fn::zeros(
                &[config.hidden_size, config.hidden_size],
                DataType::Float32,
                device,
            )?,
            key_bias: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            value: ops_fn::zeros(
                &[config.hidden_size, config.hidden_size],
                DataType::Float32,
                device,
            )?,
            value_bias: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            num_attention_heads: config.num_attention_heads,
            head_dim,
            scale,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch_size, seq_len, _hidden_size) = if shape.len() == 3 {
            (shape[0], shape[1], shape[2])
        } else if shape.len() == 2 {
            (1, shape[0], shape[1])
        } else {
            return Err(anyhow::anyhow!("Invalid hidden_states shape: {:?}", shape));
        };

        // 1. Project to Q, K, V with bias
        let query_states = self.linear_with_bias(hidden_states, &self.query, &self.query_bias)?;
        let key_states = self.linear_with_bias(hidden_states, &self.key, &self.key_bias)?;
        let value_states = self.linear_with_bias(hidden_states, &self.value, &self.value_bias)?;

        // 2. Reshape for multi-head attention
        let q_candle = query_states.to_candle()?;
        let k_candle = key_states.to_candle()?;
        let v_candle = value_states.to_candle()?;

        // [batch, seq, hidden] -> [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        let q_reshaped = q_candle
            .reshape(&[batch_size, seq_len, self.num_attention_heads, self.head_dim])?
            .transpose(1, 2)?;

        let k_reshaped = k_candle
            .reshape(&[batch_size, seq_len, self.num_attention_heads, self.head_dim])?
            .transpose(1, 2)?;

        let v_reshaped = v_candle
            .reshape(&[batch_size, seq_len, self.num_attention_heads, self.head_dim])?
            .transpose(1, 2)?;

        // 3. Scaled dot-product attention (BIDIRECTIONAL - no causal mask!)
        let k_t = k_reshaped.transpose(2, 3)?;
        let q_contiguous = q_reshaped.contiguous()?;
        let k_contiguous = k_t.contiguous()?;

        let scores = q_contiguous.matmul(&k_contiguous)?;
        let scaled_scores = (scores * (self.scale as f64))?;

        // Apply attention mask if provided (for padding tokens)
        let masked_scores = if let Some(mask) = attention_mask {
            let mask_candle = mask.to_candle()?;
            // Expand mask: [batch, seq] -> [batch, 1, 1, seq] for broadcasting
            let mask_expanded = if mask_candle.dims().len() == 2 {
                mask_candle.unsqueeze(1)?.unsqueeze(1)?
            } else {
                mask_candle
            };
            // Convert mask: 1 -> 0, 0 -> -inf
            let mask_f32 = mask_expanded.to_dtype(candle_core::DType::F32)?;
            let inverted_mask = ((1.0 - &mask_f32)? * f32::NEG_INFINITY as f64)?;
            scaled_scores.broadcast_add(&inverted_mask)?
        } else {
            scaled_scores
        };

        // Softmax
        let attention_weights = candle_nn::ops::softmax_last_dim(&masked_scores)?;

        // Apply attention to values
        let v_contiguous = v_reshaped.contiguous()?;
        let attn_output = attention_weights.matmul(&v_contiguous)?;

        // 4. Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape(&[batch_size, seq_len, self.num_attention_heads * self.head_dim])?;

        Ok(Tensor::from_candle(attn_output))
    }

    fn linear_with_bias(&self, input: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let output = ops_fn::matmul(input, weight)?;
        ops_fn::add(&output, bias)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefixes = [
            format!("bert.encoder.layer.{}.attention.self", layer_idx),
            format!("encoder.layer.{}.attention.self", layer_idx),
        ];

        for prefix in &prefixes {
            // Transpose weights: [out, in] -> [in, out] for matmul
            if let Some(w) = weights.get(&format!("{}.query.weight", prefix)) {
                self.query = ops_fn::transpose(w)?;
            }
            if let Some(b) = weights.get(&format!("{}.query.bias", prefix)) {
                self.query_bias = b.clone();
            }
            if let Some(w) = weights.get(&format!("{}.key.weight", prefix)) {
                self.key = ops_fn::transpose(w)?;
            }
            if let Some(b) = weights.get(&format!("{}.key.bias", prefix)) {
                self.key_bias = b.clone();
            }
            if let Some(w) = weights.get(&format!("{}.value.weight", prefix)) {
                self.value = ops_fn::transpose(w)?;
            }
            if let Some(b) = weights.get(&format!("{}.value.bias", prefix)) {
                self.value_bias = b.clone();
            }
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.query = self.query.to_device(device)?;
        self.query_bias = self.query_bias.to_device(device)?;
        self.key = self.key.to_device(device)?;
        self.key_bias = self.key_bias.to_device(device)?;
        self.value = self.value.to_device(device)?;
        self.value_bias = self.value_bias.to_device(device)?;
        Ok(())
    }
}

impl BertSelfOutput {
    fn new(config: &BertConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            dense: ops_fn::zeros(
                &[config.hidden_size, config.hidden_size],
                DataType::Float32,
                device,
            )?,
            dense_bias: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            layer_norm_weight: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            layer_norm_bias: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            layer_norm_eps: config.layer_norm_eps,
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        // Dense projection with bias
        let hidden_states = ops_fn::matmul(hidden_states, &self.dense)?;
        let hidden_states = ops_fn::add(&hidden_states, &self.dense_bias)?;
        // Residual connection
        let hidden_states = ops_fn::add(&hidden_states, input_tensor)?;
        // Layer normalization with bias
        self.layer_norm(&hidden_states)
    }

    fn layer_norm(&self, input: &Tensor) -> Result<Tensor> {
        let x = input.to_candle()?;
        let w = self.layer_norm_weight.to_candle()?;
        let b = self.layer_norm_bias.to_candle()?;

        let last_dim = x.dims().len() - 1;
        let mean = x.mean_keepdim(last_dim)?;
        let x_centered = x.broadcast_sub(&mean)?;
        let variance = x_centered.sqr()?.mean_keepdim(last_dim)?;
        let std = (variance + self.layer_norm_eps as f64)?.sqrt()?;
        let normalized = x_centered.broadcast_div(&std)?;
        let scaled = normalized.broadcast_mul(&w)?;
        let result = scaled.broadcast_add(&b)?;

        Ok(Tensor::from_candle(result))
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefixes = [
            format!("bert.encoder.layer.{}.attention.output", layer_idx),
            format!("encoder.layer.{}.attention.output", layer_idx),
        ];

        for prefix in &prefixes {
            if let Some(w) = weights.get(&format!("{}.dense.weight", prefix)) {
                self.dense = ops_fn::transpose(w)?;
            }
            if let Some(b) = weights.get(&format!("{}.dense.bias", prefix)) {
                self.dense_bias = b.clone();
            }
            if let Some(w) = weights.get(&format!("{}.LayerNorm.weight", prefix)) {
                self.layer_norm_weight = w.clone();
            }
            if let Some(b) = weights.get(&format!("{}.LayerNorm.bias", prefix)) {
                self.layer_norm_bias = b.clone();
            }
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.dense = self.dense.to_device(device)?;
        self.dense_bias = self.dense_bias.to_device(device)?;
        self.layer_norm_weight = self.layer_norm_weight.to_device(device)?;
        self.layer_norm_bias = self.layer_norm_bias.to_device(device)?;
        Ok(())
    }
}

impl BertIntermediate {
    fn new(config: &BertConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            dense: ops_fn::zeros(
                &[config.hidden_size, config.intermediate_size],
                DataType::Float32,
                device,
            )?,
            dense_bias: ops_fn::zeros(&[config.intermediate_size], DataType::Float32, device)?,
            hidden_act: config.hidden_act.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = ops_fn::matmul(hidden_states, &self.dense)?;
        let hidden_states = ops_fn::add(&hidden_states, &self.dense_bias)?;
        // BERT uses GELU activation
        match self.hidden_act.as_str() {
            "gelu" | "gelu_new" => ops_fn::gelu(&hidden_states),
            "relu" => {
                let x = hidden_states.to_candle()?;
                let result = x.relu()?;
                Ok(Tensor::from_candle(result))
            }
            _ => ops_fn::gelu(&hidden_states),  // Default to GELU
        }
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefixes = [
            format!("bert.encoder.layer.{}.intermediate", layer_idx),
            format!("encoder.layer.{}.intermediate", layer_idx),
        ];

        for prefix in &prefixes {
            if let Some(w) = weights.get(&format!("{}.dense.weight", prefix)) {
                self.dense = ops_fn::transpose(w)?;
            }
            if let Some(b) = weights.get(&format!("{}.dense.bias", prefix)) {
                self.dense_bias = b.clone();
            }
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.dense = self.dense.to_device(device)?;
        self.dense_bias = self.dense_bias.to_device(device)?;
        Ok(())
    }
}

impl BertOutput {
    fn new(config: &BertConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            dense: ops_fn::zeros(
                &[config.intermediate_size, config.hidden_size],
                DataType::Float32,
                device,
            )?,
            dense_bias: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            layer_norm_weight: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            layer_norm_bias: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            layer_norm_eps: config.layer_norm_eps,
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        // Dense projection with bias
        let hidden_states = ops_fn::matmul(hidden_states, &self.dense)?;
        let hidden_states = ops_fn::add(&hidden_states, &self.dense_bias)?;
        // Residual connection
        let hidden_states = ops_fn::add(&hidden_states, input_tensor)?;
        // Layer normalization with bias
        self.layer_norm(&hidden_states)
    }

    fn layer_norm(&self, input: &Tensor) -> Result<Tensor> {
        let x = input.to_candle()?;
        let w = self.layer_norm_weight.to_candle()?;
        let b = self.layer_norm_bias.to_candle()?;

        let last_dim = x.dims().len() - 1;
        let mean = x.mean_keepdim(last_dim)?;
        let x_centered = x.broadcast_sub(&mean)?;
        let variance = x_centered.sqr()?.mean_keepdim(last_dim)?;
        let std = (variance + self.layer_norm_eps as f64)?.sqrt()?;
        let normalized = x_centered.broadcast_div(&std)?;
        let scaled = normalized.broadcast_mul(&w)?;
        let result = scaled.broadcast_add(&b)?;

        Ok(Tensor::from_candle(result))
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefixes = [
            format!("bert.encoder.layer.{}.output", layer_idx),
            format!("encoder.layer.{}.output", layer_idx),
        ];

        for prefix in &prefixes {
            if let Some(w) = weights.get(&format!("{}.dense.weight", prefix)) {
                self.dense = ops_fn::transpose(w)?;
            }
            if let Some(b) = weights.get(&format!("{}.dense.bias", prefix)) {
                self.dense_bias = b.clone();
            }
            if let Some(w) = weights.get(&format!("{}.LayerNorm.weight", prefix)) {
                self.layer_norm_weight = w.clone();
            }
            if let Some(b) = weights.get(&format!("{}.LayerNorm.bias", prefix)) {
                self.layer_norm_bias = b.clone();
            }
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.dense = self.dense.to_device(device)?;
        self.dense_bias = self.dense_bias.to_device(device)?;
        self.layer_norm_weight = self.layer_norm_weight.to_device(device)?;
        self.layer_norm_bias = self.layer_norm_bias.to_device(device)?;
        Ok(())
    }
}

impl BertPooler {
    fn new(config: &BertConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            dense: ops_fn::zeros(
                &[config.hidden_size, config.hidden_size],
                DataType::Float32,
                device,
            )?,
            dense_bias: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Take the first token ([CLS]) representation
        let candle_tensor = hidden_states.to_candle()?;
        let shape = candle_tensor.dims();

        // Extract [CLS] token: [batch, seq, hidden] -> [batch, hidden]
        let first_token = if shape.len() == 3 {
            candle_tensor.narrow(1, 0, 1)?.squeeze(1)?
        } else {
            candle_tensor.narrow(0, 0, 1)?.squeeze(0)?
        };

        let first_token = Tensor::from_candle(first_token);

        // Dense projection with bias
        let pooled = ops_fn::matmul(&first_token, &self.dense)?;
        let pooled = ops_fn::add(&pooled, &self.dense_bias)?;

        // Tanh activation
        let pooled_candle = pooled.to_candle()?;
        let result = pooled_candle.tanh()?;

        Ok(Tensor::from_candle(result))
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        let weight_keys = ["bert.pooler.dense.weight", "pooler.dense.weight"];
        let bias_keys = ["bert.pooler.dense.bias", "pooler.dense.bias"];

        for key in weight_keys {
            if let Some(w) = weights.get(key) {
                self.dense = ops_fn::transpose(w)?;
                break;
            }
        }

        for key in bias_keys {
            if let Some(b) = weights.get(key) {
                self.dense_bias = b.clone();
                break;
            }
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.dense = self.dense.to_device(device)?;
        self.dense_bias = self.dense_bias.to_device(device)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bert_model_creation() {
        let config = BertConfig {
            vocab_size: 1000,
            hidden_size: 128,
            intermediate_size: 512,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            ..Default::default()
        };

        let model = BertModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 1000);
        assert_eq!(model.config().hidden_size(), 128);
        assert_eq!(model.config().num_layers(), 2);
    }

    #[test]
    fn test_bert_forward_pass() {
        let config = BertConfig {
            vocab_size: 100,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            ..Default::default()
        };

        let model = BertModelV2::new(config).unwrap();
        let input_ids = ops_fn::zeros(&[2, 8], DataType::Int64, &Device::CPU).unwrap();
        let inputs = ModelInputs::text(input_ids);

        let outputs = model.forward(&inputs).unwrap();
        match outputs {
            ModelOutputs::Embeddings { embeddings, pooled } => {
                assert_eq!(embeddings.shape(), &[2, 8, 64]); // batch, seq, hidden
                assert!(pooled.is_some());
                let pooled = pooled.unwrap();
                assert_eq!(pooled.shape(), &[2, 64]); // batch, hidden
            }
            _ => panic!("Expected embeddings output"),
        }
    }

    #[test]
    fn test_bert_generate_returns_error() {
        let config = BertConfig {
            vocab_size: 100,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            ..Default::default()
        };
        let model = BertModelV2::new(config).unwrap();
        let gen_config = GenerationConfig::default();

        let result = model.generate("Hello", &gen_config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("encoder-only"));
    }

    #[test]
    fn test_bert_bidirectional_attention() {
        // BERT should allow attending to all positions (no causal mask)
        let config = BertConfig {
            vocab_size: 100,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            ..Default::default()
        };

        let model = BertModelV2::new(config).unwrap();

        // Create input with some tokens
        let input_data: Vec<i64> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let input_ids = Tensor::from_i64_slice(&input_data, &[1, 8], &Device::CPU).unwrap();
        let inputs = ModelInputs::text(input_ids);

        // Forward pass should work with bidirectional attention
        let outputs = model.forward(&inputs);
        assert!(outputs.is_ok());
    }
}
