//! Jamba Model V2 - Hybrid Mamba-Transformer with MoE
//!
//! This implements the Jamba architecture which features:
//! - Interleaved Mamba and Attention layers
//! - Mixture of Experts (MoE) in the MLP layers
//! - State-space layers for efficient long-context processing
//! - Attention layers for high-quality modeling
//!
//! Supports: AI21's Jamba models

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

/// Jamba model configuration using the model_config macro
model_config!(JambaConfig {
    vocab_size: usize = 65536,
    hidden_size: usize = 4096,
    intermediate_size: usize = 14336,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 32,
    num_key_value_heads: usize = 8,
    hidden_act: String = "silu".to_string(),
    max_position_embeddings: usize = 262144,
    initializer_range: f32 = 0.02,
    rms_norm_eps: f32 = 1e-6,
    use_cache: bool = true,
    pad_token_id: i64 = 0,
    bos_token_id: i64 = 1,
    eos_token_id: i64 = 2,
    tie_word_embeddings: bool = false,
    rope_theta: f32 = 10000.0,
    attention_dropout: f32 = 0.0,
    // Mamba parameters
    d_state: usize = 16,
    d_conv: usize = 4,
    expand: usize = 2,
    // MoE parameters
    num_experts: usize = 16,
    num_experts_per_tok: usize = 2,
    // Hybrid architecture parameters
    attn_layer_period: usize = 8,       // Attention layer every N layers
    attn_layer_offset: usize = 4,       // Offset for first attention layer
    use_mamba_preprocessing: bool = true,
    mamba_d_inner: usize = 0,           // 0 = auto: hidden_size * expand
    mamba_dt_rank: usize = 0,           // 0 = auto: ceil(hidden_size/16)
});

impl JambaConfig {
    /// Create JambaConfig from GGUF model configuration
    pub fn from_gguf_config(gguf: &crate::weight_loader_core::GGUFModelConfig) -> Self {
        Self {
            vocab_size: gguf.vocab_size,
            hidden_size: gguf.hidden_size,
            intermediate_size: gguf.intermediate_size,
            num_hidden_layers: gguf.num_hidden_layers,
            num_attention_heads: gguf.num_attention_heads,
            num_key_value_heads: gguf.num_key_value_heads,
            rms_norm_eps: gguf.rms_norm_eps,
            rope_theta: gguf.rope_theta,
            max_position_embeddings: gguf.max_position_embeddings,
            ..Default::default()
        }
    }

    /// Check if layer at index should use attention (vs Mamba)
    pub fn is_attention_layer(&self, layer_idx: usize) -> bool {
        (layer_idx + self.attn_layer_offset) % self.attn_layer_period == 0
    }

    /// Get the effective Mamba d_inner dimension
    pub fn effective_mamba_d_inner(&self) -> usize {
        if self.mamba_d_inner > 0 {
            self.mamba_d_inner
        } else {
            self.hidden_size * self.expand
        }
    }

    /// Get the effective Mamba dt_rank
    pub fn effective_mamba_dt_rank(&self) -> usize {
        if self.mamba_dt_rank > 0 {
            self.mamba_dt_rank
        } else {
            ((self.hidden_size as f32 / 16.0).ceil() as usize).max(1)
        }
    }
}

/// Main Jamba model implementation
pub struct JambaModelV2 {
    config: JambaConfig,
    device: Device,
    embed_tokens: Tensor,
    layers: Vec<JambaLayer>,
    norm: Tensor,
    lm_head: Tensor,
}

/// Layer type enum for Jamba hybrid architecture
pub enum JambaLayerType {
    Mamba(JambaMambaBlock),
    Attention(JambaAttentionBlock),
}

/// Jamba layer - can be either Mamba or Attention based
pub struct JambaLayer {
    layer_type: JambaLayerType,
    moe: JambaMoE,
    input_layernorm: Tensor,
    post_layernorm: Tensor,
    config: JambaConfig,
}

/// Jamba Mamba block (selective state space)
pub struct JambaMambaBlock {
    in_proj: Tensor,
    conv1d_weight: Tensor,
    conv1d_bias: Option<Tensor>,
    x_proj: Tensor,
    dt_proj: Tensor,
    dt_proj_bias: Option<Tensor>,
    a_log: Tensor,
    d: Tensor,
    out_proj: Tensor,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    dt_rank: usize,
}

/// Jamba attention block
pub struct JambaAttentionBlock {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    num_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    scale: f32,
}

/// Jamba MoE layer
pub struct JambaMoE {
    router: Tensor,
    experts: Vec<JambaExpert>,
    num_experts: usize,
    num_experts_per_tok: usize,
}

/// Single expert in Jamba MoE
pub struct JambaExpert {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
}

impl Model for JambaModelV2 {
    type Config = JambaConfig;

    fn new(config: JambaConfig) -> Result<Self> {
        let device = Device::CPU;

        let embed_tokens = ops_fn::zeros(
            &[config.vocab_size, config.hidden_size],
            DataType::Float32,
            &device
        )?;

        let norm = ops_fn::zeros(
            &[config.hidden_size],
            DataType::Float32,
            &device
        )?;

        let lm_head = if config.tie_word_embeddings {
            embed_tokens.clone()
        } else {
            ops_fn::zeros(
                &[config.hidden_size, config.vocab_size],
                DataType::Float32,
                &device
            )?
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(JambaLayer::new(&config, i, &device)?);
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

    fn from_weights(config: JambaConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;

        if let Some(w) = weights.get("model.embed_tokens.weight") {
            model.embed_tokens = w.clone();
        }

        if let Some(w) = weights.get("model.norm.weight") {
            model.norm = w.clone();
        }

        if let Some(w) = weights.get("lm_head.weight") {
            model.lm_head = ops_fn::transpose(w)?;
        }

        for (i, layer) in model.layers.iter_mut().enumerate() {
            layer.load_weights(&weights, i)?;
        }

        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        match inputs {
            ModelInputs::Text { input_ids, attention_mask, .. } => {
                let mut hidden_states = ops_fn::embedding(input_ids, &self.embed_tokens)?;

                for layer in &self.layers {
                    hidden_states = layer.forward(
                        &hidden_states,
                        attention_mask.as_ref(),
                    )?;
                }

                hidden_states = ops_fn::rms_norm(&hidden_states, &self.norm, self.config.rms_norm_eps)?;
                let logits = ops_fn::matmul(&hidden_states, &self.lm_head)?;

                Ok(ModelOutputs::Logits {
                    logits,
                    hidden_states: None,
                })
            }
            _ => Err(anyhow::anyhow!("Jamba model only supports text inputs")),
        }
    }

    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        use crate::tokenizer::Tokenizer;
        use rand::Rng;

        let tokenizer = Tokenizer::new();
        let mut tokens: Vec<u32> = tokenizer.encode(prompt);

        for _ in 0..config.max_new_tokens {
            let input_ids = Tensor::from_i64_slice(
                &tokens.iter().map(|&t| t as i64).collect::<Vec<_>>(),
                &[1, tokens.len()],
                &self.device
            )?;

            let inputs = ModelInputs::text(input_ids);
            let outputs = self.forward(&inputs)?;

            let logits = match outputs {
                ModelOutputs::Logits { logits, .. } => logits,
                _ => return Err(anyhow::anyhow!("Expected logits output")),
            };

            let logits_candle = logits.to_candle()?;
            let shape = logits_candle.dims();
            let last_logits = logits_candle
                .narrow(1, shape[1] - 1, 1)?
                .squeeze(0)?
                .squeeze(0)?;

            let logits_vec: Vec<f32> = last_logits.to_vec1()?;

            let next_token = if config.do_sample && config.temperature > 0.0 {
                let scaled: Vec<f32> = logits_vec.iter()
                    .map(|&x| x / config.temperature)
                    .collect();

                let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scaled.iter().map(|&x| (x - max_val).exp()).sum();
                let probs: Vec<f32> = scaled.iter()
                    .map(|&x| (x - max_val).exp() / exp_sum)
                    .collect();

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
                let mut max_idx = 0;
                let mut max_val = logits_vec[0];
                for (idx, &val) in logits_vec.iter().enumerate() {
                    if val > max_val {
                        max_val = val;
                        max_idx = idx;
                    }
                }
                max_idx as u32
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
        let param_size = self.config.vocab_size * self.config.hidden_size +
            self.config.num_hidden_layers * (
                // Mamba/Attention parameters
                4 * self.config.hidden_size * self.config.hidden_size +
                // MoE parameters
                self.config.num_experts * 3 * self.config.hidden_size * self.config.intermediate_size
            );
        let param_size = param_size * 4;

        MemoryRequirements {
            gpu_memory: param_size,
            cpu_memory: param_size / 4,
            kv_cache_memory: param_size / 8,
            peak_memory: param_size + param_size / 2,
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        self.embed_tokens = self.embed_tokens.to_device(device)?;
        self.norm = self.norm.to_device(device)?;
        self.lm_head = self.lm_head.to_device(device)?;
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        Ok(())
    }
}

impl JambaLayer {
    fn new(config: &JambaConfig, layer_idx: usize, device: &Device) -> Result<Self> {
        let layer_type = if config.is_attention_layer(layer_idx) {
            JambaLayerType::Attention(JambaAttentionBlock::new(config, device)?)
        } else {
            JambaLayerType::Mamba(JambaMambaBlock::new(config, device)?)
        };

        let moe = JambaMoE::new(config, device)?;

        let input_layernorm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;
        let post_layernorm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;

        Ok(Self {
            layer_type,
            moe,
            input_layernorm,
            post_layernorm,
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = hidden_states.clone();

        // Pre-norm for mixer (Mamba or Attention)
        let normed = ops_fn::rms_norm(hidden_states, &self.input_layernorm, self.config.rms_norm_eps)?;

        // Apply mixer layer (Mamba or Attention)
        let mixed = match &self.layer_type {
            JambaLayerType::Mamba(mamba) => mamba.forward(&normed)?,
            JambaLayerType::Attention(attn) => attn.forward(&normed, attention_mask, self.config.rope_theta)?,
        };

        // Residual connection
        let hidden_states = ops_fn::add(&residual, &mixed)?;

        // Pre-norm for MoE
        let residual = hidden_states.clone();
        let normed = ops_fn::rms_norm(&hidden_states, &self.post_layernorm, self.config.rms_norm_eps)?;

        // Apply MoE
        let moe_out = self.moe.forward(&normed)?;

        // Residual connection
        ops_fn::add(&residual, &moe_out)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("model.layers.{}", layer_idx);

        if let Some(w) = weights.get(&format!("{}.input_layernorm.weight", prefix)) {
            self.input_layernorm = w.clone();
        }

        if let Some(w) = weights.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
            self.post_layernorm = w.clone();
        }

        // Load mixer weights
        match &mut self.layer_type {
            JambaLayerType::Mamba(mamba) => mamba.load_weights(weights, layer_idx)?,
            JambaLayerType::Attention(attn) => attn.load_weights(weights, layer_idx)?,
        }

        // Load MoE weights
        self.moe.load_weights(weights, layer_idx)?;

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.input_layernorm = self.input_layernorm.to_device(device)?;
        self.post_layernorm = self.post_layernorm.to_device(device)?;

        match &mut self.layer_type {
            JambaLayerType::Mamba(mamba) => mamba.to_device(device)?,
            JambaLayerType::Attention(attn) => attn.to_device(device)?,
        }

        self.moe.to_device(device)?;
        Ok(())
    }
}

impl JambaMambaBlock {
    fn new(config: &JambaConfig, device: &Device) -> Result<Self> {
        let d_inner = config.effective_mamba_d_inner();
        let dt_rank = config.effective_mamba_dt_rank();
        let d_state = config.d_state;
        let d_conv = config.d_conv;

        let in_proj = ops_fn::zeros(&[config.hidden_size, d_inner * 2], DataType::Float32, device)?;
        let conv1d_weight = ops_fn::zeros(&[d_inner, d_conv], DataType::Float32, device)?;
        let conv1d_bias = Some(ops_fn::zeros(&[d_inner], DataType::Float32, device)?);
        let x_proj = ops_fn::zeros(&[d_inner, dt_rank + d_state * 2], DataType::Float32, device)?;
        let dt_proj = ops_fn::zeros(&[dt_rank, d_inner], DataType::Float32, device)?;
        let dt_proj_bias = Some(ops_fn::zeros(&[d_inner], DataType::Float32, device)?);
        let a_log = ops_fn::zeros(&[d_inner, d_state], DataType::Float32, device)?;
        let d = ops_fn::zeros(&[d_inner], DataType::Float32, device)?;
        let out_proj = ops_fn::zeros(&[d_inner, config.hidden_size], DataType::Float32, device)?;

        Ok(Self {
            in_proj,
            conv1d_weight,
            conv1d_bias,
            x_proj,
            dt_proj,
            dt_proj_bias,
            a_log,
            d,
            out_proj,
            d_inner,
            d_state,
            d_conv,
            dt_rank,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch_size, seq_len, _) = (shape[0], shape[1], shape[2]);

        // 1. Input projection: [B, L, D] -> [B, L, 2*d_inner]
        let projected = ops_fn::matmul(hidden_states, &self.in_proj)?;

        // 2. Split into x and z
        let proj_candle = projected.to_candle()?;
        let x = proj_candle.narrow(2, 0, self.d_inner)?;
        let z = proj_candle.narrow(2, self.d_inner, self.d_inner)?;

        // 3. Apply causal Conv1D to x
        let x_conv = self.apply_conv1d(&Tensor::from_candle(x.clone()), batch_size, seq_len)?;

        // 4. Apply SiLU activation
        let x_act = ops_fn::silu(&x_conv)?;

        // 5. Selective scan (SSM)
        let y = self.selective_scan(&x_act, batch_size, seq_len)?;

        // 6. Gate with z (SiLU(z) * y)
        let z_act = ops_fn::silu(&Tensor::from_candle(z))?;
        let gated = ops_fn::mul(&y, &z_act)?;

        // 7. Output projection
        ops_fn::matmul(&gated, &self.out_proj)
    }

    fn apply_conv1d(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        let x_candle = x.to_candle()?;
        let w_candle = self.conv1d_weight.to_candle()?;

        // Pad for causal convolution
        let pad_len = self.d_conv - 1;
        let zeros = candle_core::Tensor::zeros(
            &[batch_size, pad_len, self.d_inner],
            x_candle.dtype(),
            x_candle.device()
        )?;

        let x_padded = candle_core::Tensor::cat(&[&zeros, &x_candle], 1)?;

        let mut outputs = Vec::new();
        for i in 0..seq_len {
            let window = x_padded.narrow(1, i, self.d_conv)?;
            let window_t = window.transpose(1, 2)?;
            let conv_out = window_t.broadcast_mul(&w_candle)?;
            let summed = conv_out.sum(2)?;
            outputs.push(summed);
        }

        let result = candle_core::Tensor::stack(&outputs, 1)?;

        let result = if let Some(ref bias) = self.conv1d_bias {
            let b_candle = bias.to_candle()?;
            result.broadcast_add(&b_candle)?
        } else {
            result
        };

        Ok(Tensor::from_candle(result))
    }

    fn selective_scan(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        // Project to get delta, B, C
        let dbc = ops_fn::matmul(x, &self.x_proj)?;
        let dbc_candle = dbc.to_candle()?;

        let dt_raw = dbc_candle.narrow(2, 0, self.dt_rank)?;
        let b = dbc_candle.narrow(2, self.dt_rank, self.d_state)?;
        let c = dbc_candle.narrow(2, self.dt_rank + self.d_state, self.d_state)?;

        // Project dt
        let dt_proj_candle = self.dt_proj.to_candle()?;
        let dt = dt_raw.broadcast_matmul(&dt_proj_candle)?;

        let dt = if let Some(ref bias) = self.dt_proj_bias {
            let b_candle = bias.to_candle()?;
            dt.broadcast_add(&b_candle)?
        } else {
            dt
        };

        let dt = softplus(&dt)?;

        // Get A from A_log
        let a_log_candle = self.a_log.to_candle()?;
        let a = a_log_candle.exp()?.neg()?;

        // Selective scan loop
        let x_candle = x.to_candle()?;
        let d_candle = self.d.to_candle()?;

        let mut h = candle_core::Tensor::zeros(
            &[batch_size, self.d_inner, self.d_state],
            candle_core::DType::F32,
            x_candle.device()
        )?;

        let mut outputs = Vec::new();

        for t in 0..seq_len {
            let x_t = x_candle.narrow(1, t, 1)?.squeeze(1)?;
            let dt_t = dt.narrow(1, t, 1)?.squeeze(1)?;
            let b_t = b.narrow(1, t, 1)?.squeeze(1)?;
            let c_t = c.narrow(1, t, 1)?.squeeze(1)?;

            let dt_expanded = dt_t.unsqueeze(2)?;
            let dt_a = dt_expanded.broadcast_mul(&a)?;
            let a_bar = dt_a.exp()?;

            let b_expanded = b_t.unsqueeze(1)?;
            let dt_b = dt_expanded.broadcast_mul(&b_expanded)?;

            let x_expanded = x_t.unsqueeze(2)?;

            let ah = a_bar.mul(&h)?;
            let bx = dt_b.mul(&x_expanded.broadcast_as(dt_b.dims())?)?;
            h = ah.add(&bx)?;

            let c_expanded = c_t.unsqueeze(1)?;
            let y_state = h.mul(&c_expanded.broadcast_as(h.dims())?)?.sum(2)?;

            let y_skip = x_t.broadcast_mul(&d_candle)?;
            let y_t = y_state.add(&y_skip)?;

            outputs.push(y_t);
        }

        let result = candle_core::Tensor::stack(&outputs, 1)?;
        Ok(Tensor::from_candle(result))
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("model.layers.{}.mamba", layer_idx);

        if let Some(w) = weights.get(&format!("{}.in_proj.weight", prefix)) {
            self.in_proj = ops_fn::transpose(w)?;
        }

        if let Some(w) = weights.get(&format!("{}.conv1d.weight", prefix)) {
            let w_candle = w.to_candle()?;
            let dims = w_candle.dims();
            if dims.len() == 3 && dims[1] == 1 {
                let reshaped = w_candle.squeeze(1)?;
                self.conv1d_weight = Tensor::from_candle(reshaped);
            } else {
                self.conv1d_weight = w.clone();
            }
        }
        if let Some(w) = weights.get(&format!("{}.conv1d.bias", prefix)) {
            self.conv1d_bias = Some(w.clone());
        }

        if let Some(w) = weights.get(&format!("{}.x_proj.weight", prefix)) {
            self.x_proj = ops_fn::transpose(w)?;
        }

        if let Some(w) = weights.get(&format!("{}.dt_proj.weight", prefix)) {
            self.dt_proj = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.dt_proj.bias", prefix)) {
            self.dt_proj_bias = Some(w.clone());
        }

        if let Some(w) = weights.get(&format!("{}.A_log", prefix)) {
            self.a_log = w.clone();
        }

        if let Some(w) = weights.get(&format!("{}.D", prefix)) {
            self.d = w.clone();
        }

        if let Some(w) = weights.get(&format!("{}.out_proj.weight", prefix)) {
            self.out_proj = ops_fn::transpose(w)?;
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.in_proj = self.in_proj.to_device(device)?;
        self.conv1d_weight = self.conv1d_weight.to_device(device)?;
        if let Some(ref mut bias) = self.conv1d_bias {
            *bias = bias.to_device(device)?;
        }
        self.x_proj = self.x_proj.to_device(device)?;
        self.dt_proj = self.dt_proj.to_device(device)?;
        if let Some(ref mut bias) = self.dt_proj_bias {
            *bias = bias.to_device(device)?;
        }
        self.a_log = self.a_log.to_device(device)?;
        self.d = self.d.to_device(device)?;
        self.out_proj = self.out_proj.to_device(device)?;
        Ok(())
    }
}

impl JambaAttentionBlock {
    fn new(config: &JambaConfig, device: &Device) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let num_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads;

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
        })
    }

    fn forward(&self, hidden_states: &Tensor, _attention_mask: Option<&Tensor>, rope_theta: f32) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch_size, seq_len, _) = (shape[0], shape[1], shape[2]);

        // Project Q, K, V
        let q = ops_fn::matmul(hidden_states, &self.q_proj)?;
        let k = ops_fn::matmul(hidden_states, &self.k_proj)?;
        let v = ops_fn::matmul(hidden_states, &self.v_proj)?;

        // Convert to candle tensors for attention operations
        let q_candle = q.to_candle()?;
        let k_candle = k.to_candle()?;
        let v_candle = v.to_candle()?;

        // Reshape: [batch, seq, heads*head_dim] -> [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        let q_reshaped = q_candle
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;

        let k_reshaped = k_candle
            .reshape(&[batch_size, seq_len, self.num_key_value_heads, self.head_dim])?
            .transpose(1, 2)?;

        let v_reshaped = v_candle
            .reshape(&[batch_size, seq_len, self.num_key_value_heads, self.head_dim])?
            .transpose(1, 2)?;

        // Apply RoPE
        let (q_with_rope, k_with_rope) = apply_rope(&q_reshaped, &k_reshaped, seq_len, self.head_dim, rope_theta)?;

        // Handle GQA - repeat K/V heads to match Q heads
        let num_groups = self.num_heads / self.num_key_value_heads;
        let (k_expanded, v_expanded) = if num_groups > 1 {
            let k_rep = k_with_rope
                .unsqueeze(2)?
                .broadcast_as(&[batch_size, self.num_key_value_heads, num_groups, seq_len, self.head_dim])?
                .reshape(&[batch_size, self.num_heads, seq_len, self.head_dim])?;
            let v_rep = v_reshaped
                .unsqueeze(2)?
                .broadcast_as(&[batch_size, self.num_key_value_heads, num_groups, seq_len, self.head_dim])?
                .reshape(&[batch_size, self.num_heads, seq_len, self.head_dim])?;
            (k_rep, v_rep)
        } else {
            (k_with_rope, v_reshaped)
        };

        // Scaled dot-product attention
        let k_t = k_expanded.transpose(2, 3)?;
        let q_contiguous = q_with_rope.contiguous()?;
        let k_contiguous = k_t.contiguous()?;

        let scores = q_contiguous.matmul(&k_contiguous)?;
        let scaled_scores = (scores * (self.scale as f64))?;

        // Apply causal mask
        let device = scaled_scores.device();
        let causal_mask = {
            let mut mask_data = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    if j > i {
                        mask_data[i * seq_len + j] = f32::NEG_INFINITY;
                    }
                }
            }
            candle_core::Tensor::from_vec(mask_data, &[1, 1, seq_len, seq_len], device)?
        };

        let masked_scores = scaled_scores.broadcast_add(&causal_mask)?;
        let attention_weights = candle_nn::ops::softmax_last_dim(&masked_scores)?;

        // Apply attention to values
        let v_contiguous = v_expanded.contiguous()?;
        let attn_output = attention_weights.matmul(&v_contiguous)?;

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, heads * head_dim]
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape(&[batch_size, seq_len, self.num_heads * self.head_dim])?;

        let attn_output = Tensor::from_candle(attn_output);

        // Output projection
        ops_fn::matmul(&attn_output, &self.o_proj)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("model.layers.{}.self_attn", layer_idx);

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

impl JambaMoE {
    fn new(config: &JambaConfig, device: &Device) -> Result<Self> {
        let router = ops_fn::zeros(
            &[config.hidden_size, config.num_experts],
            DataType::Float32,
            device
        )?;

        let mut experts = Vec::with_capacity(config.num_experts);
        for _ in 0..config.num_experts {
            experts.push(JambaExpert::new(config, device)?);
        }

        Ok(Self {
            router,
            experts,
            num_experts: config.num_experts,
            num_experts_per_tok: config.num_experts_per_tok,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch_size, seq_len, hidden_size) = (shape[0], shape[1], shape[2]);

        // Compute router logits
        let router_logits = ops_fn::matmul(hidden_states, &self.router)?;

        // Get top-k experts along last dimension
        let (topk_weights, topk_indices) = ops_fn::topk(&router_logits, self.num_experts_per_tok, -1)?;
        let routing_weights = ops_fn::softmax(&topk_weights, -1)?;

        // Flatten for processing
        let hidden_flat = hidden_states.reshape(&[batch_size * seq_len, hidden_size])?;

        // Extract routing data
        let all_indices: Vec<i64> = topk_indices.to_candle()?.flatten_all()?.to_vec1()?;
        let all_weights: Vec<f32> = routing_weights.to_candle()?.flatten_all()?.to_vec1()?;
        let num_tokens = batch_size * seq_len;
        let k = self.num_experts_per_tok;

        // Initialize output
        let mut output_candle = candle_core::Tensor::zeros(
            &[num_tokens, hidden_size],
            candle_core::DType::F32,
            &candle_core::Device::Cpu
        )?;

        // Process each token
        for tok_idx in 0..num_tokens {
            let tok_hidden = hidden_flat.to_candle()?.narrow(0, tok_idx, 1)?;
            let mut tok_output = candle_core::Tensor::zeros(
                &[1, hidden_size],
                candle_core::DType::F32,
                &candle_core::Device::Cpu
            )?;

            let start = tok_idx * k;
            for j in 0..k {
                let expert_idx = all_indices[start + j] as usize;
                let weight = all_weights[start + j];

                if expert_idx < self.experts.len() {
                    let expert_out = self.experts[expert_idx].forward(&Tensor::from_candle(tok_hidden.clone()))?;
                    let weighted = expert_out.to_candle()?.affine(weight as f64, 0.0)?;
                    tok_output = tok_output.add(&weighted)?;
                }
            }

            output_candle = output_candle.slice_assign(&[tok_idx..tok_idx+1, 0..hidden_size], &tok_output)?;
        }

        // Reshape back
        let output = Tensor::from_candle(output_candle);
        output.reshape(&[batch_size, seq_len, hidden_size])
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("model.layers.{}.block_sparse_moe", layer_idx);

        if let Some(w) = weights.get(&format!("{}.gate.weight", prefix)) {
            self.router = ops_fn::transpose(w)?;
        }

        for (i, expert) in self.experts.iter_mut().enumerate() {
            expert.load_weights(weights, layer_idx, i)?;
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.router = self.router.to_device(device)?;
        for expert in &mut self.experts {
            expert.to_device(device)?;
        }
        Ok(())
    }
}

impl JambaExpert {
    fn new(config: &JambaConfig, device: &Device) -> Result<Self> {
        let gate_proj = ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?;
        let up_proj = ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?;
        let down_proj = ops_fn::zeros(&[config.intermediate_size, config.hidden_size], DataType::Float32, device)?;

        Ok(Self { gate_proj, up_proj, down_proj })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let gate = ops_fn::matmul(hidden_states, &self.gate_proj)?;
        let gate = ops_fn::silu(&gate)?;
        let up = ops_fn::matmul(hidden_states, &self.up_proj)?;
        let hidden = ops_fn::mul(&gate, &up)?;
        ops_fn::matmul(&hidden, &self.down_proj)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize, expert_idx: usize) -> Result<()> {
        let prefix = format!("model.layers.{}.block_sparse_moe.experts.{}", layer_idx, expert_idx);

        if let Some(w) = weights.get(&format!("{}.w1.weight", prefix)) {
            self.gate_proj = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.w3.weight", prefix)) {
            self.up_proj = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.w2.weight", prefix)) {
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

/// Softplus activation: log(1 + exp(x))
fn softplus(x: &candle_core::Tensor) -> Result<candle_core::Tensor> {
    let one = candle_core::Tensor::ones(x.dims(), x.dtype(), x.device())?;
    let exp_x = x.exp()?;
    let one_plus_exp = one.add(&exp_x)?;
    Ok(one_plus_exp.log()?)
}

/// Apply Rotary Position Embedding (RoPE) to Q and K tensors
fn apply_rope(
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    seq_len: usize,
    head_dim: usize,
    rope_theta: f32,
) -> Result<(candle_core::Tensor, candle_core::Tensor)> {
    let device = q.device();
    let half_dim = head_dim / 2;

    // Build position indices [0, 1, 2, ..., seq_len-1]
    let positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
    let positions = candle_core::Tensor::from_vec(positions, &[seq_len, 1], device)?;

    // Build frequency indices [0, 1, 2, ..., half_dim-1]
    let freq_indices: Vec<f32> = (0..half_dim).map(|i| i as f32).collect();
    let freq_indices = candle_core::Tensor::from_vec(freq_indices, &[1, half_dim], device)?;

    // Compute frequencies: 1 / (theta^(2i/head_dim))
    let power = (freq_indices * (2.0 / head_dim as f64))?;
    let theta_tensor = candle_core::Tensor::from_vec(vec![rope_theta], &[1], device)?
        .broadcast_as(&[1, half_dim])?;
    let freqs = theta_tensor.pow(&power)?;
    let inv_freqs = (freqs.recip())?;

    // Compute angles: positions * inv_freqs -> [seq_len, half_dim]
    let angles = positions.broadcast_mul(&inv_freqs)?;

    // Compute cos and sin
    let cos = angles.cos()?;
    let sin = angles.sin()?;

    // Reshape for broadcasting: [1, 1, seq_len, half_dim]
    let cos = cos.reshape(&[1, 1, seq_len, half_dim])?;
    let sin = sin.reshape(&[1, 1, seq_len, half_dim])?;

    // Apply rotary embedding to Q
    let q_shape = q.dims();
    let q_first_half = q.narrow(3, 0, half_dim)?;
    let q_second_half = q.narrow(3, half_dim, half_dim)?;

    let cos_q = cos.broadcast_as(&[q_shape[0], q_shape[1], seq_len, half_dim])?;
    let sin_q = sin.broadcast_as(&[q_shape[0], q_shape[1], seq_len, half_dim])?;

    let q_rotated_first = (q_first_half.broadcast_mul(&cos_q)?
        .sub(&q_second_half.broadcast_mul(&sin_q)?))?;
    let q_rotated_second = (q_first_half.broadcast_mul(&sin_q)?
        .add(&q_second_half.broadcast_mul(&cos_q)?))?;

    let q_rotated = candle_core::Tensor::cat(&[&q_rotated_first, &q_rotated_second], 3)?;

    // Apply rotary embedding to K
    let k_shape = k.dims();
    let k_first_half = k.narrow(3, 0, half_dim)?;
    let k_second_half = k.narrow(3, half_dim, half_dim)?;

    let cos_k = cos.broadcast_as(&[k_shape[0], k_shape[1], seq_len, half_dim])?;
    let sin_k = sin.broadcast_as(&[k_shape[0], k_shape[1], seq_len, half_dim])?;

    let k_rotated_first = (k_first_half.broadcast_mul(&cos_k)?
        .sub(&k_second_half.broadcast_mul(&sin_k)?))?;
    let k_rotated_second = (k_first_half.broadcast_mul(&sin_k)?
        .add(&k_second_half.broadcast_mul(&cos_k)?))?;

    let k_rotated = candle_core::Tensor::cat(&[&k_rotated_first, &k_rotated_second], 3)?;

    Ok((q_rotated, k_rotated))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jamba_config() {
        let config = JambaConfig::default();
        assert_eq!(config.vocab_size, 65536);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_experts, 16);
        assert_eq!(config.attn_layer_period, 8);
    }

    #[test]
    fn test_jamba_layer_type_selection() {
        let config = JambaConfig {
            attn_layer_period: 4,
            attn_layer_offset: 2,
            ..Default::default()
        };

        // With offset=2 and period=4:
        // Layer 2: (2+2) % 4 = 0 -> Attention
        // Layer 3: (3+2) % 4 = 1 -> Mamba
        // Layer 6: (6+2) % 4 = 0 -> Attention
        assert!(config.is_attention_layer(2));
        assert!(!config.is_attention_layer(3));
        assert!(config.is_attention_layer(6));
    }

    #[test]
    fn test_jamba_model_creation() {
        let config = JambaConfig {
            vocab_size: 1000,
            hidden_size: 128,
            intermediate_size: 512,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_experts: 4,
            num_experts_per_tok: 2,
            d_state: 8,
            d_conv: 4,
            attn_layer_period: 2,
            attn_layer_offset: 1,
            ..Default::default()
        };

        let model = JambaModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 1000);
        assert_eq!(model.config().hidden_size(), 128);
        assert_eq!(model.config().num_layers(), 4);
    }

    #[test]
    fn test_jamba_forward_pass() {
        let config = JambaConfig {
            vocab_size: 100,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_experts: 2,
            num_experts_per_tok: 1,
            d_state: 8,
            d_conv: 4,
            expand: 2,
            attn_layer_period: 2,
            attn_layer_offset: 1,
            ..Default::default()
        };

        let model = JambaModelV2::new(config).unwrap();
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
