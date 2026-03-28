//! RWKV-4 Model V2 - Linear Attention with Time and Channel Mixing
//!
//! This implements the RWKV-4 architecture which features:
//! - Time mixing: Linear attention with exponential decay (WKV computation)
//! - Channel mixing: FFN with receptance, key, value gates
//! - Running state instead of attention/KV cache
//! - O(1) memory per token during generation
//!
//! Supports: RWKV-4 models (169M to 14B)

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

/// RWKV-4 model configuration
model_config!(Rwkv4Config {
    vocab_size: usize = 50277,
    hidden_size: usize = 768,      // d_model
    num_hidden_layers: usize = 12,
    intermediate_size: usize = 0,  // 0 = auto: hidden_size * 4
    layer_norm_epsilon: f32 = 1e-5,
    rescale_every: usize = 6,      // Rescale every N layers for numerical stability
    tie_word_embeddings: bool = false,
    pad_token_id: i64 = 0,
    bos_token_id: i64 = 0,
    eos_token_id: i64 = 0,
});

impl Rwkv4Config {
    pub fn from_gguf_config(gguf: &crate::weight_loader_core::GGUFModelConfig) -> Self {
        Self {
            vocab_size: gguf.vocab_size,
            hidden_size: gguf.hidden_size,
            num_hidden_layers: gguf.num_hidden_layers,
            intermediate_size: gguf.intermediate_size,
            layer_norm_epsilon: gguf.rms_norm_eps,
            ..Default::default()
        }
    }

    pub fn effective_intermediate_size(&self) -> usize {
        if self.intermediate_size > 0 {
            self.intermediate_size
        } else {
            self.hidden_size * 4
        }
    }
}

/// Main RWKV-4 model
pub struct Rwkv4ModelV2 {
    config: Rwkv4Config,
    device: Device,
    embeddings: Tensor,
    blocks: Vec<Rwkv4Block>,
    ln_out: Tensor,
    head: Tensor,
}

/// RWKV-4 block with time mixing and channel mixing
pub struct Rwkv4Block {
    ln1: Tensor,
    ln2: Tensor,
    time_mixing: Rwkv4TimeMixing,
    channel_mixing: Rwkv4ChannelMixing,
    layer_idx: usize,
    rescale_every: usize,
}

/// RWKV-4 Time Mixing (attention-like with linear complexity)
pub struct Rwkv4TimeMixing {
    // Learnable parameters
    time_decay: Tensor,      // w [hidden_size]
    time_first: Tensor,      // u [hidden_size]
    time_mix_k: Tensor,      // [hidden_size]
    time_mix_v: Tensor,      // [hidden_size]
    time_mix_r: Tensor,      // [hidden_size]

    // Projections
    key: Tensor,             // W_k [hidden_size, hidden_size]
    value: Tensor,           // W_v [hidden_size, hidden_size]
    receptance: Tensor,      // W_r [hidden_size, hidden_size]
    output: Tensor,          // W_o [hidden_size, hidden_size]

    hidden_size: usize,
}

/// RWKV-4 Channel Mixing (FFN-like)
pub struct Rwkv4ChannelMixing {
    time_mix_k: Tensor,      // [hidden_size]
    time_mix_r: Tensor,      // [hidden_size]

    key: Tensor,             // [hidden_size, intermediate_size]
    value: Tensor,           // [intermediate_size, hidden_size]
    receptance: Tensor,      // [hidden_size, hidden_size]

    hidden_size: usize,
    intermediate_size: usize,
}

/// RWKV state for generation
#[derive(Clone)]
pub struct Rwkv4State {
    /// Numerator state [batch, hidden_size]
    pub num: Tensor,
    /// Denominator state [batch, hidden_size]
    pub den: Tensor,
    /// Previous x for time mixing [batch, hidden_size]
    pub prev_x_tm: Tensor,
    /// Previous x for channel mixing [batch, hidden_size]
    pub prev_x_cm: Tensor,
}

impl Model for Rwkv4ModelV2 {
    type Config = Rwkv4Config;

    fn new(config: Rwkv4Config) -> Result<Self> {
        let device = Device::CPU;

        let embeddings = ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, &device)?;
        let ln_out = ops_fn::zeros(&[config.hidden_size], DataType::Float32, &device)?;
        let head = ops_fn::zeros(&[config.hidden_size, config.vocab_size], DataType::Float32, &device)?;

        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            blocks.push(Rwkv4Block::new(&config, i, &device)?);
        }

        Ok(Self {
            config,
            device,
            embeddings,
            blocks,
            ln_out,
            head,
        })
    }

    fn from_weights(config: Rwkv4Config, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;

        if let Some(w) = weights.get("emb.weight").or_else(|| weights.get("rwkv.embeddings.weight")) {
            model.embeddings = w.clone();
        }

        if let Some(w) = weights.get("ln_out.weight").or_else(|| weights.get("rwkv.ln_out.weight")) {
            model.ln_out = w.clone();
        }

        if let Some(w) = weights.get("head.weight") {
            model.head = ops_fn::transpose(w)?;
        }

        for (i, block) in model.blocks.iter_mut().enumerate() {
            block.load_weights(&weights, i)?;
        }

        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        match inputs {
            ModelInputs::Text { input_ids, .. } => {
                let mut hidden_states = ops_fn::embedding(input_ids, &self.embeddings)?;

                for block in &self.blocks {
                    hidden_states = block.forward(&hidden_states)?;
                }

                hidden_states = ops_fn::layer_norm(&hidden_states, &self.ln_out, None, self.config.layer_norm_epsilon)?;
                let logits = ops_fn::matmul(&hidden_states, &self.head)?;

                Ok(ModelOutputs::Logits {
                    logits,
                    hidden_states: None,
                })
            }
            _ => Err(anyhow::anyhow!("RWKV-4 only supports text inputs")),
        }
    }

    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        use crate::tokenizer::Tokenizer;
        use rand::Rng;

        let tokenizer = Tokenizer::new();
        let mut tokens: Vec<u32> = tokenizer.encode(prompt);

        // Initialize states for all layers
        let batch_size = 1;
        let mut layer_states: Vec<Rwkv4State> = Vec::new();
        for _ in 0..self.config.num_hidden_layers {
            layer_states.push(Rwkv4State {
                num: ops_fn::zeros(&[batch_size, self.config.hidden_size], DataType::Float32, &self.device)?,
                den: ops_fn::zeros(&[batch_size, self.config.hidden_size], DataType::Float32, &self.device)?,
                prev_x_tm: ops_fn::zeros(&[batch_size, self.config.hidden_size], DataType::Float32, &self.device)?,
                prev_x_cm: ops_fn::zeros(&[batch_size, self.config.hidden_size], DataType::Float32, &self.device)?,
            });
        }

        // Process prompt
        for &token in &tokens[..tokens.len().saturating_sub(1)] {
            let input_tensor = Tensor::from_i64_slice(&[token as i64], &[1, 1], &self.device)?;
            let mut hidden = ops_fn::embedding(&input_tensor, &self.embeddings)?;
            hidden = hidden.reshape(&[1, self.config.hidden_size])?;

            for (i, block) in self.blocks.iter().enumerate() {
                hidden = block.forward_with_state(&hidden, &mut layer_states[i])?;
            }
        }

        // Generation loop
        for _ in 0..config.max_new_tokens {
            let last_token = *tokens.last().unwrap_or(&0);
            let input_tensor = Tensor::from_i64_slice(&[last_token as i64], &[1, 1], &self.device)?;
            let mut hidden = ops_fn::embedding(&input_tensor, &self.embeddings)?;
            hidden = hidden.reshape(&[1, self.config.hidden_size])?;

            for (i, block) in self.blocks.iter().enumerate() {
                hidden = block.forward_with_state(&hidden, &mut layer_states[i])?;
            }

            hidden = ops_fn::layer_norm(&hidden, &self.ln_out, None, self.config.layer_norm_epsilon)?;
            let logits = ops_fn::matmul(&hidden, &self.head)?;

            let logits_vec: Vec<f32> = logits.to_candle()?.flatten_all()?.to_vec1()?;

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
        let inter_size = self.config.effective_intermediate_size();
        let param_size = (
            self.config.vocab_size * self.config.hidden_size +
            self.config.num_hidden_layers * (
                4 * self.config.hidden_size * self.config.hidden_size +  // time mixing projections
                self.config.hidden_size * inter_size * 2 +                // channel mixing
                self.config.hidden_size * 10                              // various parameters
            )
        ) * 4;

        let state_size = self.config.num_hidden_layers * self.config.hidden_size * 4 * 4;

        MemoryRequirements {
            gpu_memory: param_size,
            cpu_memory: param_size / 4,
            kv_cache_memory: state_size,
            peak_memory: param_size + param_size / 2,
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        self.embeddings = self.embeddings.to_device(device)?;
        self.ln_out = self.ln_out.to_device(device)?;
        self.head = self.head.to_device(device)?;
        for block in &mut self.blocks {
            block.to_device(device)?;
        }
        Ok(())
    }
}

impl Rwkv4Block {
    fn new(config: &Rwkv4Config, layer_idx: usize, device: &Device) -> Result<Self> {
        let ln1 = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;
        let ln2 = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;
        let time_mixing = Rwkv4TimeMixing::new(config, device)?;
        let channel_mixing = Rwkv4ChannelMixing::new(config, device)?;

        Ok(Self {
            ln1,
            ln2,
            time_mixing,
            channel_mixing,
            layer_idx,
            rescale_every: config.rescale_every,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch_size, seq_len, _) = (shape[0], shape[1], shape[2]);

        let mut output = hidden_states.clone();

        // Process each position sequentially (RWKV is autoregressive by nature)
        for t in 0..seq_len {
            let x = hidden_states.to_candle()?.narrow(1, t, 1)?.squeeze(1)?;
            let x = Tensor::from_candle(x);

            // Time mixing
            let ln_x = ops_fn::layer_norm(&x, &self.ln1, None, 1e-5)?;
            let tm_out = self.time_mixing.forward(&ln_x, t)?;
            let x = ops_fn::add(&x, &tm_out)?;

            // Channel mixing
            let ln_x = ops_fn::layer_norm(&x, &self.ln2, None, 1e-5)?;
            let cm_out = self.channel_mixing.forward(&ln_x, t)?;
            let x = ops_fn::add(&x, &cm_out)?;

            // Rescale for numerical stability
            let x = if self.rescale_every > 0 && (self.layer_idx + 1) % self.rescale_every == 0 {
                ops_fn::scale(&x, 0.5)?
            } else {
                x
            };

            // Update output at position t
            let x_expanded = x.to_candle()?.unsqueeze(1)?;
            let output_candle = output.to_candle()?;
            output = Tensor::from_candle(output_candle.slice_assign(&[0..batch_size, t..t+1, 0..shape[2]], &x_expanded)?);
        }

        Ok(output)
    }

    fn forward_with_state(&self, hidden_states: &Tensor, state: &mut Rwkv4State) -> Result<Tensor> {
        // hidden_states: [batch, hidden_size]

        // Time mixing
        let ln_x = ops_fn::layer_norm(hidden_states, &self.ln1, None, 1e-5)?;
        let tm_out = self.time_mixing.forward_with_state(&ln_x, state)?;
        let x = ops_fn::add(hidden_states, &tm_out)?;

        // Channel mixing
        let ln_x = ops_fn::layer_norm(&x, &self.ln2, None, 1e-5)?;
        let cm_out = self.channel_mixing.forward_with_state(&ln_x, state)?;
        let x = ops_fn::add(&x, &cm_out)?;

        // Rescale
        if self.rescale_every > 0 && (self.layer_idx + 1) % self.rescale_every == 0 {
            ops_fn::scale(&x, 0.5)
        } else {
            Ok(x)
        }
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("blocks.{}", layer_idx);

        if let Some(w) = weights.get(&format!("{}.ln1.weight", prefix)) {
            self.ln1 = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.ln2.weight", prefix)) {
            self.ln2 = w.clone();
        }

        self.time_mixing.load_weights(weights, layer_idx)?;
        self.channel_mixing.load_weights(weights, layer_idx)?;

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.ln1 = self.ln1.to_device(device)?;
        self.ln2 = self.ln2.to_device(device)?;
        self.time_mixing.to_device(device)?;
        self.channel_mixing.to_device(device)?;
        Ok(())
    }
}

impl Rwkv4TimeMixing {
    fn new(config: &Rwkv4Config, device: &Device) -> Result<Self> {
        let hidden_size = config.hidden_size;

        Ok(Self {
            time_decay: ops_fn::zeros(&[hidden_size], DataType::Float32, device)?,
            time_first: ops_fn::zeros(&[hidden_size], DataType::Float32, device)?,
            time_mix_k: ops_fn::zeros(&[hidden_size], DataType::Float32, device)?,
            time_mix_v: ops_fn::zeros(&[hidden_size], DataType::Float32, device)?,
            time_mix_r: ops_fn::zeros(&[hidden_size], DataType::Float32, device)?,
            key: ops_fn::zeros(&[hidden_size, hidden_size], DataType::Float32, device)?,
            value: ops_fn::zeros(&[hidden_size, hidden_size], DataType::Float32, device)?,
            receptance: ops_fn::zeros(&[hidden_size, hidden_size], DataType::Float32, device)?,
            output: ops_fn::zeros(&[hidden_size, hidden_size], DataType::Float32, device)?,
            hidden_size,
        })
    }

    fn forward(&self, x: &Tensor, _position: usize) -> Result<Tensor> {
        // Simplified forward without state (uses zeros for prev_x)
        let x_candle = x.to_candle()?;
        let zeros = candle_core::Tensor::zeros(x_candle.dims(), x_candle.dtype(), x_candle.device())?;

        // Time mixing
        let mix_k = self.time_mix_k.to_candle()?;
        let mix_v = self.time_mix_v.to_candle()?;
        let mix_r = self.time_mix_r.to_candle()?;

        // xk = x * time_mix_k + prev_x * (1 - time_mix_k)
        let one_minus_k = candle_core::Tensor::ones_like(&mix_k)?.sub(&mix_k)?;
        let one_minus_v = candle_core::Tensor::ones_like(&mix_v)?.sub(&mix_v)?;
        let one_minus_r = candle_core::Tensor::ones_like(&mix_r)?.sub(&mix_r)?;
        let xk = x_candle.broadcast_mul(&mix_k)?.add(&zeros.broadcast_mul(&one_minus_k)?)?;
        let xv = x_candle.broadcast_mul(&mix_v)?.add(&zeros.broadcast_mul(&one_minus_v)?)?;
        let xr = x_candle.broadcast_mul(&mix_r)?.add(&zeros.broadcast_mul(&one_minus_r)?)?;

        // Project
        let k_proj = self.key.to_candle()?;
        let v_proj = self.value.to_candle()?;
        let r_proj = self.receptance.to_candle()?;
        let o_proj = self.output.to_candle()?;

        let k = xk.matmul(&k_proj)?;
        let v = xv.matmul(&v_proj)?;
        let r = xr.matmul(&r_proj)?;

        // Receptance gating
        let r_sigmoid = candle_nn::ops::sigmoid(&r)?;

        // WKV computation (simplified - no state)
        let w = self.time_decay.to_candle()?.neg()?.exp()?;
        let u = self.time_first.to_candle()?;

        let wkv = {
            let ek = (u.broadcast_add(&k)?).exp()?;
            let numerator = ek.broadcast_mul(&v)?;
            let denominator = ek;
            numerator.broadcast_div(&denominator.broadcast_add(&candle_core::Tensor::ones_like(&denominator)?)?)?
        };

        // Output
        let output = r_sigmoid.broadcast_mul(&wkv)?;
        let output = output.matmul(&o_proj)?;

        Ok(Tensor::from_candle(output))
    }

    fn forward_with_state(&self, x: &Tensor, state: &mut Rwkv4State) -> Result<Tensor> {
        let x_candle = x.to_candle()?;
        let prev_x = state.prev_x_tm.to_candle()?;

        // Time mixing with previous x
        let mix_k = self.time_mix_k.to_candle()?;
        let mix_v = self.time_mix_v.to_candle()?;
        let mix_r = self.time_mix_r.to_candle()?;

        let one_minus_k = candle_core::Tensor::ones_like(&mix_k)?.sub(&mix_k)?;
        let one_minus_v = candle_core::Tensor::ones_like(&mix_v)?.sub(&mix_v)?;
        let one_minus_r = candle_core::Tensor::ones_like(&mix_r)?.sub(&mix_r)?;

        let xk = x_candle.broadcast_mul(&mix_k)?.add(&prev_x.broadcast_mul(&one_minus_k)?)?;
        let xv = x_candle.broadcast_mul(&mix_v)?.add(&prev_x.broadcast_mul(&one_minus_v)?)?;
        let xr = x_candle.broadcast_mul(&mix_r)?.add(&prev_x.broadcast_mul(&one_minus_r)?)?;

        // Update state
        state.prev_x_tm = Tensor::from_candle(x_candle.clone());

        // Project
        let k_proj = self.key.to_candle()?;
        let v_proj = self.value.to_candle()?;
        let r_proj = self.receptance.to_candle()?;
        let o_proj = self.output.to_candle()?;

        let k = xk.matmul(&k_proj)?;
        let v = xv.matmul(&v_proj)?;
        let r = xr.matmul(&r_proj)?;

        // Receptance gating
        let r_sigmoid = candle_nn::ops::sigmoid(&r)?;

        // WKV computation with state
        let w = self.time_decay.to_candle()?.neg()?.exp()?;
        let u = self.time_first.to_candle()?;

        let num_prev = state.num.to_candle()?;
        let den_prev = state.den.to_candle()?;

        // e^(u + k)
        let ek = (u.broadcast_add(&k)?).exp()?;

        // numerator = e^(u+k) * v + w * num_prev
        let numerator = ek.broadcast_mul(&v)?.add(&w.broadcast_mul(&num_prev)?)?;

        // denominator = e^(u+k) + w * den_prev
        let denominator = ek.add(&w.broadcast_mul(&den_prev)?)?;

        let wkv = numerator.broadcast_div(&denominator.broadcast_add(&candle_core::Tensor::full(1e-8f32, denominator.dims(), denominator.device())?)?)?;

        // Update WKV state
        let ek_simple = k.exp()?;
        state.num = Tensor::from_candle(w.broadcast_mul(&num_prev)?.add(&ek_simple.broadcast_mul(&v)?)?);
        state.den = Tensor::from_candle(w.broadcast_mul(&den_prev)?.add(&ek_simple)?);

        // Output
        let output = r_sigmoid.broadcast_mul(&wkv)?;
        let output = output.matmul(&o_proj)?;

        Ok(Tensor::from_candle(output))
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("blocks.{}.att", layer_idx);

        if let Some(w) = weights.get(&format!("{}.time_decay", prefix)) {
            self.time_decay = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.time_first", prefix)) {
            self.time_first = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.time_mix_k", prefix)) {
            self.time_mix_k = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.time_mix_v", prefix)) {
            self.time_mix_v = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.time_mix_r", prefix)) {
            self.time_mix_r = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.key.weight", prefix)) {
            self.key = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.value.weight", prefix)) {
            self.value = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.receptance.weight", prefix)) {
            self.receptance = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.output.weight", prefix)) {
            self.output = ops_fn::transpose(w)?;
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.time_decay = self.time_decay.to_device(device)?;
        self.time_first = self.time_first.to_device(device)?;
        self.time_mix_k = self.time_mix_k.to_device(device)?;
        self.time_mix_v = self.time_mix_v.to_device(device)?;
        self.time_mix_r = self.time_mix_r.to_device(device)?;
        self.key = self.key.to_device(device)?;
        self.value = self.value.to_device(device)?;
        self.receptance = self.receptance.to_device(device)?;
        self.output = self.output.to_device(device)?;
        Ok(())
    }
}

impl Rwkv4ChannelMixing {
    fn new(config: &Rwkv4Config, device: &Device) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.effective_intermediate_size();

        Ok(Self {
            time_mix_k: ops_fn::zeros(&[hidden_size], DataType::Float32, device)?,
            time_mix_r: ops_fn::zeros(&[hidden_size], DataType::Float32, device)?,
            key: ops_fn::zeros(&[hidden_size, intermediate_size], DataType::Float32, device)?,
            value: ops_fn::zeros(&[intermediate_size, hidden_size], DataType::Float32, device)?,
            receptance: ops_fn::zeros(&[hidden_size, hidden_size], DataType::Float32, device)?,
            hidden_size,
            intermediate_size,
        })
    }

    fn forward(&self, x: &Tensor, _position: usize) -> Result<Tensor> {
        let x_candle = x.to_candle()?;
        let zeros = candle_core::Tensor::zeros(x_candle.dims(), x_candle.dtype(), x_candle.device())?;

        // Channel mixing with zeros for prev_x
        let mix_k = self.time_mix_k.to_candle()?;
        let mix_r = self.time_mix_r.to_candle()?;

        let one_minus_k = candle_core::Tensor::ones_like(&mix_k)?.sub(&mix_k)?;
        let one_minus_r = candle_core::Tensor::ones_like(&mix_r)?.sub(&mix_r)?;
        let xk = x_candle.broadcast_mul(&mix_k)?.add(&zeros.broadcast_mul(&one_minus_k)?)?;
        let xr = x_candle.broadcast_mul(&mix_r)?.add(&zeros.broadcast_mul(&one_minus_r)?)?;

        // Project
        let k_proj = self.key.to_candle()?;
        let v_proj = self.value.to_candle()?;
        let r_proj = self.receptance.to_candle()?;

        let k = xk.matmul(&k_proj)?;
        let r = xr.matmul(&r_proj)?;

        // Squared ReLU for k
        let k_relu = k.relu()?;
        let k_squared = k_relu.sqr()?;

        // Value projection
        let v = k_squared.matmul(&v_proj)?;

        // Receptance gating
        let r_sigmoid = candle_nn::ops::sigmoid(&r)?;
        let output = r_sigmoid.broadcast_mul(&v)?;

        Ok(Tensor::from_candle(output))
    }

    fn forward_with_state(&self, x: &Tensor, state: &mut Rwkv4State) -> Result<Tensor> {
        let x_candle = x.to_candle()?;
        let prev_x = state.prev_x_cm.to_candle()?;

        // Channel mixing with state
        let mix_k = self.time_mix_k.to_candle()?;
        let mix_r = self.time_mix_r.to_candle()?;

        let one_minus_k = candle_core::Tensor::ones_like(&mix_k)?.sub(&mix_k)?;
        let one_minus_r = candle_core::Tensor::ones_like(&mix_r)?.sub(&mix_r)?;

        let xk = x_candle.broadcast_mul(&mix_k)?.add(&prev_x.broadcast_mul(&one_minus_k)?)?;
        let xr = x_candle.broadcast_mul(&mix_r)?.add(&prev_x.broadcast_mul(&one_minus_r)?)?;

        // Update state
        state.prev_x_cm = Tensor::from_candle(x_candle.clone());

        // Project
        let k_proj = self.key.to_candle()?;
        let v_proj = self.value.to_candle()?;
        let r_proj = self.receptance.to_candle()?;

        let k = xk.matmul(&k_proj)?;
        let r = xr.matmul(&r_proj)?;

        // Squared ReLU
        let k_relu = k.relu()?;
        let k_squared = k_relu.sqr()?;

        // Value projection
        let v = k_squared.matmul(&v_proj)?;

        // Receptance gating
        let r_sigmoid = candle_nn::ops::sigmoid(&r)?;
        let output = r_sigmoid.broadcast_mul(&v)?;

        Ok(Tensor::from_candle(output))
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("blocks.{}.ffn", layer_idx);

        if let Some(w) = weights.get(&format!("{}.time_mix_k", prefix)) {
            self.time_mix_k = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.time_mix_r", prefix)) {
            self.time_mix_r = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.key.weight", prefix)) {
            self.key = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.value.weight", prefix)) {
            self.value = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.receptance.weight", prefix)) {
            self.receptance = ops_fn::transpose(w)?;
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.time_mix_k = self.time_mix_k.to_device(device)?;
        self.time_mix_r = self.time_mix_r.to_device(device)?;
        self.key = self.key.to_device(device)?;
        self.value = self.value.to_device(device)?;
        self.receptance = self.receptance.to_device(device)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rwkv4_config() {
        let config = Rwkv4Config::default();
        assert_eq!(config.vocab_size, 50277);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.effective_intermediate_size(), 768 * 4);
    }

    #[test]
    fn test_rwkv4_model_creation() {
        let config = Rwkv4Config {
            vocab_size: 1000,
            hidden_size: 64,
            num_hidden_layers: 2,
            ..Default::default()
        };

        let model = Rwkv4ModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 1000);
        assert_eq!(model.config().hidden_size(), 64);
        assert_eq!(model.config().num_layers(), 2);
    }

    #[test]
    fn test_rwkv4_forward_pass() {
        let config = Rwkv4Config {
            vocab_size: 100,
            hidden_size: 32,
            num_hidden_layers: 1,
            ..Default::default()
        };

        let model = Rwkv4ModelV2::new(config).unwrap();
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
