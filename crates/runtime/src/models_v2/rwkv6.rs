//! RWKV-6 Model V2 - Linear Attention with Matrix-Valued States
//!
//! This implements the RWKV-6 architecture which extends RWKV-4 with:
//! - Matrix-valued states (upgraded from vector)
//! - Data-dependent decay rates
//! - Improved time mixing with bonus term
//! - Better numerical stability
//!
//! Supports: RWKV-6 models

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

/// RWKV-6 model configuration
model_config!(Rwkv6Config {
    vocab_size: usize = 65536,
    hidden_size: usize = 2048,
    num_hidden_layers: usize = 24,
    intermediate_size: usize = 0,  // 0 = auto: hidden_size * 3.5
    head_size: usize = 64,         // Size of each attention head
    num_heads: usize = 0,          // 0 = auto: hidden_size / head_size
    layer_norm_epsilon: f32 = 1e-5,
    rescale_every: usize = 6,
    tie_word_embeddings: bool = false,
    pad_token_id: i64 = 0,
    bos_token_id: i64 = 1,
    eos_token_id: i64 = 2,
});

impl Rwkv6Config {
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
            ((self.hidden_size as f32 * 3.5) as usize / 32) * 32
        }
    }

    pub fn effective_num_heads(&self) -> usize {
        if self.num_heads > 0 {
            self.num_heads
        } else {
            self.hidden_size / self.head_size
        }
    }
}

/// Main RWKV-6 model
pub struct Rwkv6ModelV2 {
    config: Rwkv6Config,
    device: Device,
    embeddings: Tensor,
    blocks: Vec<Rwkv6Block>,
    ln_out: Tensor,
    head: Tensor,
}

/// RWKV-6 block
pub struct Rwkv6Block {
    ln1: Tensor,
    ln2: Tensor,
    time_mixing: Rwkv6TimeMixing,
    channel_mixing: Rwkv6ChannelMixing,
    layer_idx: usize,
    rescale_every: usize,
}

/// RWKV-6 Time Mixing with matrix-valued states
pub struct Rwkv6TimeMixing {
    // Learnable decay parameters (data-dependent)
    time_maa_x: Tensor,      // [hidden_size]
    time_maa_w: Tensor,      // [hidden_size]
    time_maa_k: Tensor,      // [hidden_size]
    time_maa_v: Tensor,      // [hidden_size]
    time_maa_r: Tensor,      // [hidden_size]
    time_maa_g: Tensor,      // [hidden_size] - gate for RWKV-6

    // Decay projections
    time_decay: Tensor,      // w [hidden_size]
    time_decay_w1: Tensor,   // [hidden_size, lora_rank]
    time_decay_w2: Tensor,   // [lora_rank, hidden_size]
    time_first: Tensor,      // u [num_heads, head_size]

    // Projections
    receptance: Tensor,
    key: Tensor,
    value: Tensor,
    output: Tensor,
    gate: Tensor,            // RWKV-6 adds a gate

    // Group norm for output
    ln_x: Tensor,

    hidden_size: usize,
    head_size: usize,
    num_heads: usize,
}

/// RWKV-6 Channel Mixing
pub struct Rwkv6ChannelMixing {
    time_maa_k: Tensor,
    time_maa_r: Tensor,

    key: Tensor,
    value: Tensor,
    receptance: Tensor,

    hidden_size: usize,
    intermediate_size: usize,
}

/// RWKV-6 state with matrix-valued WKV state
#[derive(Clone)]
pub struct Rwkv6State {
    /// WKV state [batch, num_heads, head_size, head_size]
    pub wkv_state: Tensor,
    /// Previous x for time mixing [batch, hidden_size]
    pub prev_x_tm: Tensor,
    /// Previous x for channel mixing [batch, hidden_size]
    pub prev_x_cm: Tensor,
}

impl Model for Rwkv6ModelV2 {
    type Config = Rwkv6Config;

    fn new(config: Rwkv6Config) -> Result<Self> {
        let device = Device::CPU;

        let embeddings = ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, &device)?;
        let ln_out = ops_fn::zeros(&[config.hidden_size], DataType::Float32, &device)?;
        let head = ops_fn::zeros(&[config.hidden_size, config.vocab_size], DataType::Float32, &device)?;

        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            blocks.push(Rwkv6Block::new(&config, i, &device)?);
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

    fn from_weights(config: Rwkv6Config, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;

        if let Some(w) = weights.get("emb.weight").or_else(|| weights.get("rwkv.embeddings.weight")) {
            model.embeddings = w.clone();
        }

        if let Some(w) = weights.get("ln_out.weight") {
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
            _ => Err(anyhow::anyhow!("RWKV-6 only supports text inputs")),
        }
    }

    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        use crate::tokenizer::Tokenizer;
        use rand::Rng;

        let tokenizer = Tokenizer::new();
        let mut tokens: Vec<u32> = tokenizer.encode(prompt);

        let batch_size = 1;
        let num_heads = self.config.effective_num_heads();
        let head_size = self.config.head_size;

        // Initialize states
        let mut layer_states: Vec<Rwkv6State> = Vec::new();
        for _ in 0..self.config.num_hidden_layers {
            layer_states.push(Rwkv6State {
                wkv_state: ops_fn::zeros(&[batch_size, num_heads, head_size, head_size], DataType::Float32, &self.device)?,
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
        let num_heads = self.config.effective_num_heads();
        let head_size = self.config.head_size;

        let param_size = (
            self.config.vocab_size * self.config.hidden_size +
            self.config.num_hidden_layers * (
                4 * self.config.hidden_size * self.config.hidden_size +
                self.config.hidden_size * inter_size * 2 +
                self.config.hidden_size * 15
            )
        ) * 4;

        // Matrix state: [num_heads, head_size, head_size] per layer
        let state_size = self.config.num_hidden_layers * (num_heads * head_size * head_size + self.config.hidden_size * 2) * 4;

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

impl Rwkv6Block {
    fn new(config: &Rwkv6Config, layer_idx: usize, device: &Device) -> Result<Self> {
        let ln1 = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;
        let ln2 = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;
        let time_mixing = Rwkv6TimeMixing::new(config, device)?;
        let channel_mixing = Rwkv6ChannelMixing::new(config, device)?;

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
        let (batch_size, seq_len, hidden_size) = (shape[0], shape[1], shape[2]);

        let mut output = hidden_states.clone();

        for t in 0..seq_len {
            let x = hidden_states.to_candle()?.narrow(1, t, 1)?.squeeze(1)?;
            let x = Tensor::from_candle(x);

            let ln_x = ops_fn::layer_norm(&x, &self.ln1, None, 1e-5)?;
            let tm_out = self.time_mixing.forward(&ln_x)?;
            let x = ops_fn::add(&x, &tm_out)?;

            let ln_x = ops_fn::layer_norm(&x, &self.ln2, None, 1e-5)?;
            let cm_out = self.channel_mixing.forward(&ln_x)?;
            let x = ops_fn::add(&x, &cm_out)?;

            let x = if self.rescale_every > 0 && (self.layer_idx + 1) % self.rescale_every == 0 {
                ops_fn::scale(&x, 0.5)?
            } else {
                x
            };

            let x_expanded = x.to_candle()?.unsqueeze(1)?;
            let output_candle = output.to_candle()?;
            output = Tensor::from_candle(output_candle.slice_assign(&[0..batch_size, t..t+1, 0..hidden_size], &x_expanded)?);
        }

        Ok(output)
    }

    fn forward_with_state(&self, hidden_states: &Tensor, state: &mut Rwkv6State) -> Result<Tensor> {
        let ln_x = ops_fn::layer_norm(hidden_states, &self.ln1, None, 1e-5)?;
        let tm_out = self.time_mixing.forward_with_state(&ln_x, state)?;
        let x = ops_fn::add(hidden_states, &tm_out)?;

        let ln_x = ops_fn::layer_norm(&x, &self.ln2, None, 1e-5)?;
        let cm_out = self.channel_mixing.forward_with_state(&ln_x, state)?;
        let x = ops_fn::add(&x, &cm_out)?;

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

impl Rwkv6TimeMixing {
    fn new(config: &Rwkv6Config, device: &Device) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let head_size = config.head_size;
        let num_heads = config.effective_num_heads();
        let lora_rank = 32; // Typical LoRA rank for RWKV-6

        Ok(Self {
            time_maa_x: ops_fn::zeros(&[hidden_size], DataType::Float32, device)?,
            time_maa_w: ops_fn::zeros(&[hidden_size], DataType::Float32, device)?,
            time_maa_k: ops_fn::zeros(&[hidden_size], DataType::Float32, device)?,
            time_maa_v: ops_fn::zeros(&[hidden_size], DataType::Float32, device)?,
            time_maa_r: ops_fn::zeros(&[hidden_size], DataType::Float32, device)?,
            time_maa_g: ops_fn::zeros(&[hidden_size], DataType::Float32, device)?,
            time_decay: ops_fn::zeros(&[hidden_size], DataType::Float32, device)?,
            time_decay_w1: ops_fn::zeros(&[hidden_size, lora_rank], DataType::Float32, device)?,
            time_decay_w2: ops_fn::zeros(&[lora_rank, hidden_size], DataType::Float32, device)?,
            time_first: ops_fn::zeros(&[num_heads, head_size], DataType::Float32, device)?,
            receptance: ops_fn::zeros(&[hidden_size, hidden_size], DataType::Float32, device)?,
            key: ops_fn::zeros(&[hidden_size, hidden_size], DataType::Float32, device)?,
            value: ops_fn::zeros(&[hidden_size, hidden_size], DataType::Float32, device)?,
            output: ops_fn::zeros(&[hidden_size, hidden_size], DataType::Float32, device)?,
            gate: ops_fn::zeros(&[hidden_size, hidden_size], DataType::Float32, device)?,
            ln_x: ops_fn::zeros(&[hidden_size], DataType::Float32, device)?,
            hidden_size,
            head_size,
            num_heads,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Simplified forward without state
        let x_candle = x.to_candle()?;

        // Apply projections
        let r_proj = self.receptance.to_candle()?;
        let k_proj = self.key.to_candle()?;
        let v_proj = self.value.to_candle()?;
        let o_proj = self.output.to_candle()?;
        let g_proj = self.gate.to_candle()?;

        let r = x_candle.matmul(&r_proj)?;
        let k = x_candle.matmul(&k_proj)?;
        let v = x_candle.matmul(&v_proj)?;
        let g = x_candle.matmul(&g_proj)?;

        // Receptance and gate
        let r_sigmoid = candle_nn::ops::sigmoid(&r)?;
        let g_silu = candle_nn::ops::silu(&g)?;

        // Simplified WKV
        let u = self.time_first.to_candle()?;
        let w = self.time_decay.to_candle()?.neg()?.exp()?;

        let ek = k.exp()?;
        let wkv = ek.broadcast_mul(&v)?.broadcast_div(&ek.broadcast_add(&candle_core::Tensor::ones_like(&ek)?)?)?;

        // Apply gating
        let output = r_sigmoid.broadcast_mul(&wkv)?;
        let output = output.broadcast_mul(&g_silu)?;
        let output = output.matmul(&o_proj)?;

        Ok(Tensor::from_candle(output))
    }

    fn forward_with_state(&self, x: &Tensor, state: &mut Rwkv6State) -> Result<Tensor> {
        let x_candle = x.to_candle()?;
        let prev_x = state.prev_x_tm.to_candle()?;

        // Data-dependent mixing
        let maa_x = self.time_maa_x.to_candle()?;
        let one_minus_maa = candle_core::Tensor::ones_like(&maa_x)?.sub(&maa_x)?;
        let sx = x_candle.broadcast_mul(&maa_x)?.add(&prev_x.broadcast_mul(&one_minus_maa)?)?;

        state.prev_x_tm = Tensor::from_candle(x_candle.clone());

        // Project
        let r_proj = self.receptance.to_candle()?;
        let k_proj = self.key.to_candle()?;
        let v_proj = self.value.to_candle()?;
        let o_proj = self.output.to_candle()?;
        let g_proj = self.gate.to_candle()?;

        let r = sx.matmul(&r_proj)?;
        let k = sx.matmul(&k_proj)?;
        let v = sx.matmul(&v_proj)?;
        let g = sx.matmul(&g_proj)?;

        // Gates
        let r_sigmoid = candle_nn::ops::sigmoid(&r)?;
        let g_silu = candle_nn::ops::silu(&g)?;

        // Data-dependent decay
        let w_base = self.time_decay.to_candle()?.neg()?.exp()?;
        let w1 = self.time_decay_w1.to_candle()?;
        let w2 = self.time_decay_w2.to_candle()?;
        let w_delta = sx.matmul(&w1)?.tanh()?.matmul(&w2)?;
        let w = w_base.broadcast_mul(&(w_delta.exp())?)?;

        // Matrix-valued WKV state update
        let batch_size = x_candle.dims()[0];
        let wkv_state = state.wkv_state.to_candle()?;
        let u = self.time_first.to_candle()?;

        // Reshape k, v for multi-head
        let k_heads = k.reshape(&[batch_size, self.num_heads, self.head_size])?;
        let v_heads = v.reshape(&[batch_size, self.num_heads, self.head_size])?;

        // State update: S_new = w * S + e^k * v^T
        let ek = k_heads.exp()?;
        let kv_outer = ek.unsqueeze(3)?.matmul(&v_heads.unsqueeze(2)?)?; // [B, H, head, head]

        let w_expanded = w.reshape(&[batch_size, self.num_heads, self.head_size, 1])?;
        let new_state = wkv_state.broadcast_mul(&w_expanded)?.add(&kv_outer)?;

        // Output: y = r * (u * e^k * v + S @ e^k)
        let state_contrib = wkv_state.matmul(&ek.unsqueeze(3)?)?.squeeze(3)?;
        let direct_contrib = ek.broadcast_mul(&v_heads)?;
        let u_expanded = u.unsqueeze(0)?;
        let wkv_out = u_expanded.broadcast_mul(&direct_contrib)?.add(&state_contrib)?;

        state.wkv_state = Tensor::from_candle(new_state);

        // Reshape back
        let wkv_flat = wkv_out.reshape(&[batch_size, self.hidden_size])?;

        // Group norm
        let ln_x = self.ln_x.to_candle()?;
        let wkv_normed = wkv_flat.broadcast_mul(&ln_x)?;

        // Apply gating and output
        let output = r_sigmoid.broadcast_mul(&wkv_normed)?;
        let output = output.broadcast_mul(&g_silu)?;
        let output = output.matmul(&o_proj)?;

        Ok(Tensor::from_candle(output))
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("blocks.{}.att", layer_idx);

        if let Some(w) = weights.get(&format!("{}.time_maa_x", prefix)) {
            self.time_maa_x = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.time_maa_w", prefix)) {
            self.time_maa_w = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.time_maa_k", prefix)) {
            self.time_maa_k = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.time_maa_v", prefix)) {
            self.time_maa_v = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.time_maa_r", prefix)) {
            self.time_maa_r = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.time_maa_g", prefix)) {
            self.time_maa_g = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.time_decay", prefix)) {
            self.time_decay = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.time_decay_w1", prefix)) {
            self.time_decay_w1 = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.time_decay_w2", prefix)) {
            self.time_decay_w2 = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.time_first", prefix)) {
            self.time_first = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.receptance.weight", prefix)) {
            self.receptance = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.key.weight", prefix)) {
            self.key = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.value.weight", prefix)) {
            self.value = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.output.weight", prefix)) {
            self.output = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.gate.weight", prefix)) {
            self.gate = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.ln_x.weight", prefix)) {
            self.ln_x = w.clone();
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.time_maa_x = self.time_maa_x.to_device(device)?;
        self.time_maa_w = self.time_maa_w.to_device(device)?;
        self.time_maa_k = self.time_maa_k.to_device(device)?;
        self.time_maa_v = self.time_maa_v.to_device(device)?;
        self.time_maa_r = self.time_maa_r.to_device(device)?;
        self.time_maa_g = self.time_maa_g.to_device(device)?;
        self.time_decay = self.time_decay.to_device(device)?;
        self.time_decay_w1 = self.time_decay_w1.to_device(device)?;
        self.time_decay_w2 = self.time_decay_w2.to_device(device)?;
        self.time_first = self.time_first.to_device(device)?;
        self.receptance = self.receptance.to_device(device)?;
        self.key = self.key.to_device(device)?;
        self.value = self.value.to_device(device)?;
        self.output = self.output.to_device(device)?;
        self.gate = self.gate.to_device(device)?;
        self.ln_x = self.ln_x.to_device(device)?;
        Ok(())
    }
}

impl Rwkv6ChannelMixing {
    fn new(config: &Rwkv6Config, device: &Device) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.effective_intermediate_size();

        Ok(Self {
            time_maa_k: ops_fn::zeros(&[hidden_size], DataType::Float32, device)?,
            time_maa_r: ops_fn::zeros(&[hidden_size], DataType::Float32, device)?,
            key: ops_fn::zeros(&[hidden_size, intermediate_size], DataType::Float32, device)?,
            value: ops_fn::zeros(&[intermediate_size, hidden_size], DataType::Float32, device)?,
            receptance: ops_fn::zeros(&[hidden_size, hidden_size], DataType::Float32, device)?,
            hidden_size,
            intermediate_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_candle = x.to_candle()?;

        let k_proj = self.key.to_candle()?;
        let v_proj = self.value.to_candle()?;
        let r_proj = self.receptance.to_candle()?;

        let k = x_candle.matmul(&k_proj)?;
        let r = x_candle.matmul(&r_proj)?;

        let k_relu = k.relu()?;
        let k_squared = k_relu.sqr()?;
        let v = k_squared.matmul(&v_proj)?;

        let r_sigmoid = candle_nn::ops::sigmoid(&r)?;
        let output = r_sigmoid.broadcast_mul(&v)?;

        Ok(Tensor::from_candle(output))
    }

    fn forward_with_state(&self, x: &Tensor, state: &mut Rwkv6State) -> Result<Tensor> {
        let x_candle = x.to_candle()?;
        let prev_x = state.prev_x_cm.to_candle()?;

        let maa_k = self.time_maa_k.to_candle()?;
        let maa_r = self.time_maa_r.to_candle()?;

        let one_minus_k = candle_core::Tensor::ones_like(&maa_k)?.sub(&maa_k)?;
        let one_minus_r = candle_core::Tensor::ones_like(&maa_r)?.sub(&maa_r)?;

        let xk = x_candle.broadcast_mul(&maa_k)?.add(&prev_x.broadcast_mul(&one_minus_k)?)?;
        let xr = x_candle.broadcast_mul(&maa_r)?.add(&prev_x.broadcast_mul(&one_minus_r)?)?;

        state.prev_x_cm = Tensor::from_candle(x_candle.clone());

        let k_proj = self.key.to_candle()?;
        let v_proj = self.value.to_candle()?;
        let r_proj = self.receptance.to_candle()?;

        let k = xk.matmul(&k_proj)?;
        let r = xr.matmul(&r_proj)?;

        let k_relu = k.relu()?;
        let k_squared = k_relu.sqr()?;
        let v = k_squared.matmul(&v_proj)?;

        let r_sigmoid = candle_nn::ops::sigmoid(&r)?;
        let output = r_sigmoid.broadcast_mul(&v)?;

        Ok(Tensor::from_candle(output))
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("blocks.{}.ffn", layer_idx);

        if let Some(w) = weights.get(&format!("{}.time_maa_k", prefix)) {
            self.time_maa_k = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.time_maa_r", prefix)) {
            self.time_maa_r = w.clone();
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
        self.time_maa_k = self.time_maa_k.to_device(device)?;
        self.time_maa_r = self.time_maa_r.to_device(device)?;
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
    fn test_rwkv6_config() {
        let config = Rwkv6Config::default();
        assert_eq!(config.vocab_size, 65536);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.head_size, 64);
        assert_eq!(config.effective_num_heads(), 32);
    }

    #[test]
    fn test_rwkv6_model_creation() {
        let config = Rwkv6Config {
            vocab_size: 1000,
            hidden_size: 64,
            num_hidden_layers: 2,
            head_size: 16,
            ..Default::default()
        };

        let model = Rwkv6ModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 1000);
        assert_eq!(model.config().hidden_size(), 64);
        assert_eq!(model.config().num_layers(), 2);
    }

    #[test]
    fn test_rwkv6_forward_pass() {
        let config = Rwkv6Config {
            vocab_size: 100,
            hidden_size: 32,
            num_hidden_layers: 1,
            head_size: 16,
            ..Default::default()
        };

        let model = Rwkv6ModelV2::new(config).unwrap();
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
