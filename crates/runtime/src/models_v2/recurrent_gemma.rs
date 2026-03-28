//! RecurrentGemma Model V2 - Griffin Architecture with Linear Recurrence
//!
//! This implements the RecurrentGemma (Griffin) architecture which features:
//! - Interleaved local attention and linear recurrence layers
//! - Real-gated Linear Recurrent Unit (RG-LRU)
//! - Local sliding window attention for global context
//! - Efficient O(1) state per token during generation
//!
//! Supports: RecurrentGemma-2B, RecurrentGemma-9B

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

/// RecurrentGemma model configuration
model_config!(RecurrentGemmaConfig {
    vocab_size: usize = 256000,
    hidden_size: usize = 2560,
    num_hidden_layers: usize = 26,
    intermediate_size: usize = 7680,
    num_attention_heads: usize = 10,
    num_key_value_heads: usize = 1,
    head_dim: usize = 256,
    max_position_embeddings: usize = 8192,
    rms_norm_eps: f32 = 1e-6,
    rope_theta: f32 = 10000.0,
    attention_window_size: usize = 2048,  // Local attention window
    lru_width: usize = 0,                 // 0 = auto: hidden_size
    recurrent_block_ratio: usize = 2,     // 1 attention per N blocks
    tie_word_embeddings: bool = true,
    pad_token_id: i64 = 0,
    bos_token_id: i64 = 2,
    eos_token_id: i64 = 1,
});

impl RecurrentGemmaConfig {
    pub fn from_gguf_config(gguf: &crate::weight_loader_core::GGUFModelConfig) -> Self {
        Self {
            vocab_size: gguf.vocab_size,
            hidden_size: gguf.hidden_size,
            num_hidden_layers: gguf.num_hidden_layers,
            intermediate_size: gguf.intermediate_size,
            num_attention_heads: gguf.num_attention_heads,
            num_key_value_heads: gguf.num_key_value_heads,
            head_dim: gguf.hidden_size / gguf.num_attention_heads,
            rms_norm_eps: gguf.rms_norm_eps,
            rope_theta: gguf.rope_theta,
            max_position_embeddings: gguf.max_position_embeddings,
            ..Default::default()
        }
    }

    pub fn effective_lru_width(&self) -> usize {
        if self.lru_width > 0 {
            self.lru_width
        } else {
            self.hidden_size
        }
    }

    pub fn is_recurrent_layer(&self, layer_idx: usize) -> bool {
        // Recurrent layers are all except those at positions divisible by recurrent_block_ratio
        layer_idx % self.recurrent_block_ratio != 0
    }
}

/// Main RecurrentGemma model
pub struct RecurrentGemmaModelV2 {
    config: RecurrentGemmaConfig,
    device: Device,
    embed_tokens: Tensor,
    layers: Vec<RecurrentGemmaLayer>,
    norm: Tensor,
    lm_head: Tensor,
}

/// Layer type for RecurrentGemma
pub enum RecurrentGemmaLayerType {
    Attention(GriffinAttention),
    Recurrent(GriffinRecurrent),
}

/// RecurrentGemma layer
pub struct RecurrentGemmaLayer {
    layer_type: RecurrentGemmaLayerType,
    mlp: GriffinMLP,
    input_layernorm: Tensor,
    pre_feedforward_layernorm: Tensor,
    post_attention_layernorm: Tensor,
    post_feedforward_layernorm: Tensor,
    config: RecurrentGemmaConfig,
}

/// Griffin local attention block
pub struct GriffinAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    num_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    scale: f32,
    window_size: usize,
}

/// Griffin Real-Gated Linear Recurrent Unit (RG-LRU)
pub struct GriffinRecurrent {
    // Linear projections
    linear_x: Tensor,    // [hidden_size, lru_width]
    linear_y: Tensor,    // [hidden_size, lru_width]

    // Recurrence parameters
    a_param: Tensor,     // [lru_width] - learnable decay parameter
    input_gate: Tensor,  // [hidden_size, lru_width]
    output_proj: Tensor, // [lru_width, hidden_size]

    lru_width: usize,
}

/// Griffin MLP
pub struct GriffinMLP {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
}

/// RecurrentGemma state for generation
#[derive(Clone)]
pub struct RecurrentGemmaState {
    /// LRU state [batch, lru_width]
    pub lru_state: Tensor,
    /// Conv state for temporal convolution [batch, lru_width, conv_len]
    pub conv_state: Option<Tensor>,
}

impl Model for RecurrentGemmaModelV2 {
    type Config = RecurrentGemmaConfig;

    fn new(config: RecurrentGemmaConfig) -> Result<Self> {
        let device = Device::CPU;

        let embed_tokens = ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, &device)?;
        let norm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, &device)?;

        let lm_head = if config.tie_word_embeddings {
            embed_tokens.clone()
        } else {
            ops_fn::zeros(&[config.hidden_size, config.vocab_size], DataType::Float32, &device)?
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(RecurrentGemmaLayer::new(&config, i, &device)?);
        }

        Ok(Self {
            config,
            device,
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    fn from_weights(config: RecurrentGemmaConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;

        if let Some(w) = weights.get("model.embed_tokens.weight") {
            model.embed_tokens = w.clone();
        }

        if let Some(w) = weights.get("model.norm.weight") {
            model.norm = w.clone();
        }

        if !model.config.tie_word_embeddings {
            if let Some(w) = weights.get("lm_head.weight") {
                model.lm_head = ops_fn::transpose(w)?;
            }
        }

        for (i, layer) in model.layers.iter_mut().enumerate() {
            layer.load_weights(&weights, i)?;
        }

        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        match inputs {
            ModelInputs::Text { input_ids, .. } => {
                let mut hidden_states = ops_fn::embedding(input_ids, &self.embed_tokens)?;

                // Gemma-style: multiply embeddings by sqrt(hidden_size)
                let scale = (self.config.hidden_size as f32).sqrt();
                hidden_states = ops_fn::scale(&hidden_states, scale)?;

                for layer in &self.layers {
                    hidden_states = layer.forward(&hidden_states)?;
                }

                hidden_states = ops_fn::rms_norm(&hidden_states, &self.norm, self.config.rms_norm_eps)?;

                let logits = if self.config.tie_word_embeddings {
                    let embed_t = ops_fn::transpose(&self.embed_tokens)?;
                    ops_fn::matmul(&hidden_states, &embed_t)?
                } else {
                    ops_fn::matmul(&hidden_states, &self.lm_head)?
                };

                Ok(ModelOutputs::Logits {
                    logits,
                    hidden_states: None,
                })
            }
            _ => Err(anyhow::anyhow!("RecurrentGemma only supports text inputs")),
        }
    }

    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        use crate::tokenizer::Tokenizer;
        use rand::Rng;

        let tokenizer = Tokenizer::new();
        let mut tokens: Vec<u32> = tokenizer.encode(prompt);

        // Initialize states for recurrent layers
        let batch_size = 1;
        let lru_width = self.config.effective_lru_width();
        let mut layer_states: Vec<Option<RecurrentGemmaState>> = Vec::new();

        for i in 0..self.config.num_hidden_layers {
            if self.config.is_recurrent_layer(i) {
                layer_states.push(Some(RecurrentGemmaState {
                    lru_state: ops_fn::zeros(&[batch_size, lru_width], DataType::Float32, &self.device)?,
                    conv_state: None,
                }));
            } else {
                layer_states.push(None);
            }
        }

        // Process with full forward for prompt
        let input_ids = Tensor::from_i64_slice(
            &tokens.iter().map(|&t| t as i64).collect::<Vec<_>>(),
            &[1, tokens.len()],
            &self.device
        )?;
        let inputs = ModelInputs::text(input_ids);
        let _ = self.forward(&inputs)?;

        // Generation loop
        for _ in 0..config.max_new_tokens {
            let last_token = *tokens.last().unwrap_or(&0);
            let input_tensor = Tensor::from_i64_slice(&[last_token as i64], &[1, 1], &self.device)?;

            let mut hidden = ops_fn::embedding(&input_tensor, &self.embed_tokens)?;
            let scale = (self.config.hidden_size as f32).sqrt();
            hidden = ops_fn::scale(&hidden, scale)?;

            for (i, layer) in self.layers.iter().enumerate() {
                hidden = layer.forward_with_state(&hidden, layer_states[i].as_mut())?;
            }

            hidden = ops_fn::rms_norm(&hidden, &self.norm, self.config.rms_norm_eps)?;

            let logits = if self.config.tie_word_embeddings {
                let embed_t = ops_fn::transpose(&self.embed_tokens)?;
                ops_fn::matmul(&hidden, &embed_t)?
            } else {
                ops_fn::matmul(&hidden, &self.lm_head)?
            };

            let logits_candle = logits.to_candle()?;
            let last_logits = logits_candle.squeeze(0)?.squeeze(0)?;
            let logits_vec: Vec<f32> = last_logits.to_vec1()?;

            let next_token = if config.do_sample && config.temperature > 0.0 {
                let scaled: Vec<f32> = logits_vec.iter().map(|&x| x / config.temperature).collect();
                let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scaled.iter().map(|&x| (x - max_val).exp()).sum();
                let probs: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp() / exp_sum).collect();

                let mut rng = rand::thread_rng();
                let random_val: f32 = rng.gen();
                let mut cumulative = 0.0;
                let mut sampled = 0u32;

                for (idx, &prob) in probs.iter().enumerate() {
                    cumulative += prob;
                    if random_val <= cumulative {
                        sampled = idx as u32;
                        break;
                    }
                }
                sampled
            } else {
                logits_vec.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(0)
            };

            if next_token == config.eos_token_id {
                break;
            }

            tokens.push(next_token);
        }

        Ok(tokenizer.decode(&tokens))
    }

    fn config(&self) -> &Self::Config { &self.config }

    fn memory_requirements(&self) -> MemoryRequirements {
        let param_size = (
            self.config.vocab_size * self.config.hidden_size +
            self.config.num_hidden_layers * (
                4 * self.config.hidden_size * self.config.hidden_size +
                3 * self.config.hidden_size * self.config.intermediate_size
            )
        ) * 4;

        let lru_width = self.config.effective_lru_width();
        let num_recurrent = self.config.num_hidden_layers * (self.config.recurrent_block_ratio - 1) / self.config.recurrent_block_ratio;
        let state_size = num_recurrent * lru_width * 4;

        MemoryRequirements {
            gpu_memory: param_size,
            cpu_memory: param_size / 4,
            kv_cache_memory: state_size,
            peak_memory: param_size + param_size / 2,
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        self.embed_tokens = self.embed_tokens.to_device(device)?;
        self.norm = self.norm.to_device(device)?;
        if !self.config.tie_word_embeddings {
            self.lm_head = self.lm_head.to_device(device)?;
        }
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        Ok(())
    }
}

impl RecurrentGemmaLayer {
    fn new(config: &RecurrentGemmaConfig, layer_idx: usize, device: &Device) -> Result<Self> {
        let layer_type = if config.is_recurrent_layer(layer_idx) {
            RecurrentGemmaLayerType::Recurrent(GriffinRecurrent::new(config, device)?)
        } else {
            RecurrentGemmaLayerType::Attention(GriffinAttention::new(config, device)?)
        };

        let mlp = GriffinMLP::new(config, device)?;

        let input_layernorm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;
        let pre_feedforward_layernorm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;
        let post_attention_layernorm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;
        let post_feedforward_layernorm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;

        Ok(Self {
            layer_type,
            mlp,
            input_layernorm,
            pre_feedforward_layernorm,
            post_attention_layernorm,
            post_feedforward_layernorm,
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let residual = hidden_states.clone();

        // Pre-norm for temporal block
        let normed = ops_fn::rms_norm(hidden_states, &self.input_layernorm, self.config.rms_norm_eps)?;

        // Apply temporal mixing (attention or recurrent)
        let temporal_out = match &self.layer_type {
            RecurrentGemmaLayerType::Attention(attn) => attn.forward(&normed)?,
            RecurrentGemmaLayerType::Recurrent(rec) => rec.forward(&normed)?,
        };

        // Post-norm and residual
        let temporal_out = ops_fn::rms_norm(&temporal_out, &self.post_attention_layernorm, self.config.rms_norm_eps)?;
        let hidden_states = ops_fn::add(&residual, &temporal_out)?;

        // MLP block
        let residual = hidden_states.clone();
        let normed = ops_fn::rms_norm(&hidden_states, &self.pre_feedforward_layernorm, self.config.rms_norm_eps)?;
        let mlp_out = self.mlp.forward(&normed)?;
        let mlp_out = ops_fn::rms_norm(&mlp_out, &self.post_feedforward_layernorm, self.config.rms_norm_eps)?;

        ops_fn::add(&residual, &mlp_out)
    }

    fn forward_with_state(&self, hidden_states: &Tensor, state: Option<&mut RecurrentGemmaState>) -> Result<Tensor> {
        let residual = hidden_states.clone();

        let normed = ops_fn::rms_norm(hidden_states, &self.input_layernorm, self.config.rms_norm_eps)?;

        let temporal_out = match (&self.layer_type, state) {
            (RecurrentGemmaLayerType::Attention(attn), _) => attn.forward(&normed)?,
            (RecurrentGemmaLayerType::Recurrent(rec), Some(s)) => rec.forward_with_state(&normed, s)?,
            (RecurrentGemmaLayerType::Recurrent(rec), None) => rec.forward(&normed)?,
        };

        let temporal_out = ops_fn::rms_norm(&temporal_out, &self.post_attention_layernorm, self.config.rms_norm_eps)?;
        let hidden_states = ops_fn::add(&residual, &temporal_out)?;

        let residual = hidden_states.clone();
        let normed = ops_fn::rms_norm(&hidden_states, &self.pre_feedforward_layernorm, self.config.rms_norm_eps)?;
        let mlp_out = self.mlp.forward(&normed)?;
        let mlp_out = ops_fn::rms_norm(&mlp_out, &self.post_feedforward_layernorm, self.config.rms_norm_eps)?;

        ops_fn::add(&residual, &mlp_out)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("model.layers.{}", layer_idx);

        if let Some(w) = weights.get(&format!("{}.input_layernorm.weight", prefix)) {
            self.input_layernorm = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.pre_feedforward_layernorm.weight", prefix)) {
            self.pre_feedforward_layernorm = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
            self.post_attention_layernorm = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.post_feedforward_layernorm.weight", prefix)) {
            self.post_feedforward_layernorm = w.clone();
        }

        match &mut self.layer_type {
            RecurrentGemmaLayerType::Attention(attn) => attn.load_weights(weights, layer_idx)?,
            RecurrentGemmaLayerType::Recurrent(rec) => rec.load_weights(weights, layer_idx)?,
        }

        self.mlp.load_weights(weights, layer_idx)?;

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.input_layernorm = self.input_layernorm.to_device(device)?;
        self.pre_feedforward_layernorm = self.pre_feedforward_layernorm.to_device(device)?;
        self.post_attention_layernorm = self.post_attention_layernorm.to_device(device)?;
        self.post_feedforward_layernorm = self.post_feedforward_layernorm.to_device(device)?;

        match &mut self.layer_type {
            RecurrentGemmaLayerType::Attention(attn) => attn.to_device(device)?,
            RecurrentGemmaLayerType::Recurrent(rec) => rec.to_device(device)?,
        }

        self.mlp.to_device(device)?;
        Ok(())
    }
}

impl GriffinAttention {
    fn new(config: &RecurrentGemmaConfig, device: &Device) -> Result<Self> {
        let num_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;

        let q_proj = ops_fn::zeros(&[config.hidden_size, num_heads * head_dim], DataType::Float32, device)?;
        let k_proj = ops_fn::zeros(&[config.hidden_size, num_key_value_heads * head_dim], DataType::Float32, device)?;
        let v_proj = ops_fn::zeros(&[config.hidden_size, num_key_value_heads * head_dim], DataType::Float32, device)?;
        let o_proj = ops_fn::zeros(&[num_heads * head_dim, config.hidden_size], DataType::Float32, device)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_key_value_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            window_size: config.attention_window_size,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch_size, seq_len, _) = (shape[0], shape[1], shape[2]);

        // Project Q, K, V
        let q = ops_fn::matmul(hidden_states, &self.q_proj)?;
        let k = ops_fn::matmul(hidden_states, &self.k_proj)?;
        let v = ops_fn::matmul(hidden_states, &self.v_proj)?;

        let q_candle = q.to_candle()?;
        let k_candle = k.to_candle()?;
        let v_candle = v.to_candle()?;

        // Reshape for attention
        let q_reshaped = q_candle
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k_reshaped = k_candle
            .reshape(&[batch_size, seq_len, self.num_key_value_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v_reshaped = v_candle
            .reshape(&[batch_size, seq_len, self.num_key_value_heads, self.head_dim])?
            .transpose(1, 2)?;

        // GQA expansion
        let num_groups = self.num_heads / self.num_key_value_heads;
        let (k_expanded, v_expanded) = if num_groups > 1 {
            let k_rep = k_reshaped
                .unsqueeze(2)?
                .broadcast_as(&[batch_size, self.num_key_value_heads, num_groups, seq_len, self.head_dim])?
                .reshape(&[batch_size, self.num_heads, seq_len, self.head_dim])?;
            let v_rep = v_reshaped
                .unsqueeze(2)?
                .broadcast_as(&[batch_size, self.num_key_value_heads, num_groups, seq_len, self.head_dim])?
                .reshape(&[batch_size, self.num_heads, seq_len, self.head_dim])?;
            (k_rep, v_rep)
        } else {
            (k_reshaped, v_reshaped)
        };

        // Attention scores
        let k_t = k_expanded.transpose(2, 3)?;
        let q_cont = q_reshaped.contiguous()?;
        let k_cont = k_t.contiguous()?;

        let scores = q_cont.matmul(&k_cont)?;
        let scaled_scores = (scores * (self.scale as f64))?;

        // Apply local + causal mask
        let device = scaled_scores.device();
        let mask = {
            let mut mask_data = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    // Causal: can't see future
                    // Local: can only see within window
                    let is_causal_ok = j <= i;
                    let is_local_ok = i.saturating_sub(self.window_size) <= j;
                    if !is_causal_ok || !is_local_ok {
                        mask_data[i * seq_len + j] = f32::NEG_INFINITY;
                    }
                }
            }
            candle_core::Tensor::from_vec(mask_data, &[1, 1, seq_len, seq_len], device)?
        };

        let masked_scores = scaled_scores.broadcast_add(&mask)?;
        let attention_weights = candle_nn::ops::softmax_last_dim(&masked_scores)?;

        let v_cont = v_expanded.contiguous()?;
        let attn_output = attention_weights.matmul(&v_cont)?;

        // Reshape back
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape(&[batch_size, seq_len, self.num_heads * self.head_dim])?;

        let attn_output = Tensor::from_candle(attn_output);
        ops_fn::matmul(&attn_output, &self.o_proj)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("model.layers.{}.temporal_block", layer_idx);

        if let Some(w) = weights.get(&format!("{}.q_proj.weight", prefix)) {
            self.q_proj = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.k_proj.weight", prefix)) {
            self.k_proj = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.v_proj.weight", prefix)) {
            self.v_proj = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.o_proj.weight", prefix)) {
            self.o_proj = ops_fn::transpose(w)?;
        }

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

impl GriffinRecurrent {
    fn new(config: &RecurrentGemmaConfig, device: &Device) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let lru_width = config.effective_lru_width();

        let linear_x = ops_fn::zeros(&[hidden_size, lru_width], DataType::Float32, device)?;
        let linear_y = ops_fn::zeros(&[hidden_size, lru_width], DataType::Float32, device)?;
        let a_param = ops_fn::zeros(&[lru_width], DataType::Float32, device)?;
        let input_gate = ops_fn::zeros(&[hidden_size, lru_width], DataType::Float32, device)?;
        let output_proj = ops_fn::zeros(&[lru_width, hidden_size], DataType::Float32, device)?;

        Ok(Self {
            linear_x,
            linear_y,
            a_param,
            input_gate,
            output_proj,
            lru_width,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch_size, seq_len, _) = (shape[0], shape[1], shape[2]);

        // Project x and y
        let x = ops_fn::matmul(hidden_states, &self.linear_x)?;
        let y = ops_fn::matmul(hidden_states, &self.linear_y)?;

        // Compute recurrence
        let x_candle = x.to_candle()?;
        let y_candle = y.to_candle()?;
        let a = self.a_param.to_candle()?.neg()?.exp()?; // Decay factor

        // Process sequence
        let mut h = candle_core::Tensor::zeros(&[batch_size, self.lru_width], candle_core::DType::F32, x_candle.device())?;
        let mut outputs = Vec::new();

        for t in 0..seq_len {
            let x_t = x_candle.narrow(1, t, 1)?.squeeze(1)?;
            let y_t = y_candle.narrow(1, t, 1)?.squeeze(1)?;

            // h = a * h + (1 - a) * x
            let one_minus_a = candle_core::Tensor::ones_like(&a)?.sub(&a)?;
            h = a.broadcast_mul(&h)?.add(&one_minus_a.broadcast_mul(&x_t)?)?;

            // output = y * h (gating)
            let out_t = y_t.broadcast_mul(&h)?;
            outputs.push(out_t);
        }

        let output = candle_core::Tensor::stack(&outputs, 1)?;
        let output = Tensor::from_candle(output);

        ops_fn::matmul(&output, &self.output_proj)
    }

    fn forward_with_state(&self, hidden_states: &Tensor, state: &mut RecurrentGemmaState) -> Result<Tensor> {
        // hidden_states: [batch, 1, hidden_size] or [batch, hidden_size]
        let x_candle = hidden_states.to_candle()?;
        let x = if x_candle.dims().len() == 3 {
            x_candle.squeeze(1)?
        } else {
            x_candle.clone()
        };

        // Project
        let linear_x = self.linear_x.to_candle()?;
        let linear_y = self.linear_y.to_candle()?;

        let x_proj = x.matmul(&linear_x)?;
        let y_proj = x.matmul(&linear_y)?;

        // Recurrence update
        let a = self.a_param.to_candle()?.neg()?.exp()?;
        let one_minus_a = candle_core::Tensor::ones_like(&a)?.sub(&a)?;

        let h_prev = state.lru_state.to_candle()?;
        let h_new = a.broadcast_mul(&h_prev)?.add(&one_minus_a.broadcast_mul(&x_proj)?)?;

        state.lru_state = Tensor::from_candle(h_new.clone());

        // Output
        let out = y_proj.broadcast_mul(&h_new)?;
        let output_proj = self.output_proj.to_candle()?;
        let output = out.matmul(&output_proj)?;

        // Ensure output is 3D
        let output = if output.dims().len() == 2 {
            output.unsqueeze(1)?
        } else {
            output
        };

        Ok(Tensor::from_candle(output))
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("model.layers.{}.temporal_block", layer_idx);

        if let Some(w) = weights.get(&format!("{}.linear_x.weight", prefix)) {
            self.linear_x = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.linear_y.weight", prefix)) {
            self.linear_y = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.a_param", prefix)) {
            self.a_param = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.input_gate.weight", prefix)) {
            self.input_gate = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.output_proj.weight", prefix)) {
            self.output_proj = ops_fn::transpose(w)?;
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.linear_x = self.linear_x.to_device(device)?;
        self.linear_y = self.linear_y.to_device(device)?;
        self.a_param = self.a_param.to_device(device)?;
        self.input_gate = self.input_gate.to_device(device)?;
        self.output_proj = self.output_proj.to_device(device)?;
        Ok(())
    }
}

impl GriffinMLP {
    fn new(config: &RecurrentGemmaConfig, device: &Device) -> Result<Self> {
        let gate_proj = ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?;
        let up_proj = ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?;
        let down_proj = ops_fn::zeros(&[config.intermediate_size, config.hidden_size], DataType::Float32, device)?;

        Ok(Self { gate_proj, up_proj, down_proj })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let gate = ops_fn::matmul(hidden_states, &self.gate_proj)?;
        let gate = ops_fn::gelu(&gate)?;
        let up = ops_fn::matmul(hidden_states, &self.up_proj)?;
        let hidden = ops_fn::mul(&gate, &up)?;
        ops_fn::matmul(&hidden, &self.down_proj)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("model.layers.{}.mlp", layer_idx);

        if let Some(w) = weights.get(&format!("{}.gate_proj.weight", prefix)) {
            self.gate_proj = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.up_proj.weight", prefix)) {
            self.up_proj = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.down_proj.weight", prefix)) {
            self.down_proj = ops_fn::transpose(w)?;
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.gate_proj = self.gate_proj.to_device(device)?;
        self.up_proj = self.up_proj.to_device(device)?;
        self.down_proj = self.down_proj.to_device(device)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recurrent_gemma_config() {
        let config = RecurrentGemmaConfig::default();
        assert_eq!(config.vocab_size, 256000);
        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.recurrent_block_ratio, 2);
    }

    #[test]
    fn test_layer_type_selection() {
        let config = RecurrentGemmaConfig {
            recurrent_block_ratio: 3,
            ..Default::default()
        };

        // Layer 0: 0 % 3 == 0 -> Attention
        // Layer 1: 1 % 3 != 0 -> Recurrent
        // Layer 2: 2 % 3 != 0 -> Recurrent
        // Layer 3: 3 % 3 == 0 -> Attention
        assert!(!config.is_recurrent_layer(0));
        assert!(config.is_recurrent_layer(1));
        assert!(config.is_recurrent_layer(2));
        assert!(!config.is_recurrent_layer(3));
    }

    #[test]
    fn test_recurrent_gemma_model_creation() {
        let config = RecurrentGemmaConfig {
            vocab_size: 1000,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            recurrent_block_ratio: 2,
            ..Default::default()
        };

        let model = RecurrentGemmaModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 1000);
        assert_eq!(model.config().hidden_size(), 64);
        assert_eq!(model.config().num_layers(), 4);
    }

    #[test]
    fn test_recurrent_gemma_forward_pass() {
        let config = RecurrentGemmaConfig {
            vocab_size: 100,
            hidden_size: 32,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 16,
            recurrent_block_ratio: 2,
            ..Default::default()
        };

        let model = RecurrentGemmaModelV2::new(config).unwrap();
        let input_ids = ops_fn::zeros(&[1, 4], DataType::Int64, &Device::CPU).unwrap();
        let inputs = ModelInputs::text(input_ids);

        let outputs = model.forward(&inputs).unwrap();
        match outputs {
            ModelOutputs::Logits { logits, .. } => {
                assert_eq!(logits.shape(), &[1, 4, 100]);
            }
            _ => panic!("Expected logits output"),
        }
    }
}
