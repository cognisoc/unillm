//! Mixtral Model V2 - Production-quality implementation
//!
//! Implements the Mixtral 8x7B architecture with:
//! - KV caching for efficient autoregressive generation
//! - Quantized weight support (QMatMul)
//! - Grouped Query Attention (GQA)
//! - Sliding window attention
//! - RoPE (Rotary Position Embedding)
//! - SwiGLU activation
//! - RMS normalization
//! - Mixture of Experts (MoE) with top-2 routing
//! - ModelRunner trait for streaming generation

use super::traits::*;
use crate::kv_cache::{KVCache, LayerKVCache};
use crate::model_config;
use anyhow::Result;
use candle_core::quantized::QMatMul;
use candle_nn::Module;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

model_config!(MixtralConfig {
    vocab_size: usize = 32000,
    hidden_size: usize = 4096,
    intermediate_size: usize = 14336,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 32,
    num_key_value_heads: usize = 8,
    hidden_act: String = "silu".to_string(),
    max_position_embeddings: usize = 32768,
    initializer_range: f32 = 0.02,
    rms_norm_eps: f32 = 1e-5,
    use_cache: bool = true,
    pad_token_id: i64 = 0,
    bos_token_id: i64 = 1,
    eos_token_id: i64 = 2,
    tie_word_embeddings: bool = false,
    rope_theta: f32 = 1000000.0,
    sliding_window: usize = 4096,
    attention_dropout: f32 = 0.0,
    num_experts: usize = 8,
    num_experts_per_tok: usize = 2,
    router_aux_loss_coef: f32 = 0.02,
});

impl MixtralConfig {
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
}

pub struct MixtralModelV2 {
    config: MixtralConfig,
    device: Device,
    embed_tokens: Tensor,
    layers: Vec<MixtralLayer>,
    norm: Tensor,
    lm_head: Tensor,
    lm_head_q: Option<Arc<QMatMul>>,
}

pub struct MixtralLayer {
    self_attn: MixtralAttention,
    moe: MixtralMoE,
    input_layernorm: Tensor,
    post_attention_layernorm: Tensor,
}

pub struct MixtralAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    q_proj_q: Option<Arc<QMatMul>>,
    k_proj_q: Option<Arc<QMatMul>>,
    v_proj_q: Option<Arc<QMatMul>>,
    o_proj_q: Option<Arc<QMatMul>>,
    num_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    scale: f32,
    sliding_window: usize,
}

pub struct MixtralMoE {
    router: Tensor,
    router_q: Option<Arc<QMatMul>>,
    experts: Vec<MixtralExpert>,
    num_experts: usize,
    num_experts_per_tok: usize,
}

pub struct MixtralExpert {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
    gate_proj_q: Option<Arc<QMatMul>>,
    up_proj_q: Option<Arc<QMatMul>>,
    down_proj_q: Option<Arc<QMatMul>>,
}

impl Model for MixtralModelV2 {
    type Config = MixtralConfig;

    fn new(config: MixtralConfig) -> Result<Self> {
        let device = Device::CPU;

        let embed_tokens = ops_fn::zeros(
            &[config.vocab_size, config.hidden_size],
            DataType::Float32,
            &device,
        )?;

        let norm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, &device)?;

        let lm_head = if config.tie_word_embeddings {
            embed_tokens.clone()
        } else {
            ops_fn::zeros(
                &[config.hidden_size, config.vocab_size],
                DataType::Float32,
                &device,
            )?
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(MixtralLayer::new(&config, &device)?);
        }

        Ok(Self {
            config,
            device,
            embed_tokens,
            layers,
            norm,
            lm_head,
            lm_head_q: None,
        })
    }

    fn from_weights(config: MixtralConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;

        if let Some(embed_weights) = weights.get("model.embed_tokens.weight") {
            model.embed_tokens = embed_weights.clone();
        }

        if let Some(norm_weights) = weights.get("model.norm.weight") {
            model.norm = norm_weights.clone();
        }

        if let Some(lm_head_weights) = weights.get("lm_head.weight") {
            model.lm_head = ops_fn::transpose(lm_head_weights)?;
        }

        model.lm_head_q = weights.get_quantized("lm_head.weight");

        for (i, layer) in model.layers.iter_mut().enumerate() {
            layer.load_weights(&weights, i)?;
        }

        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        match inputs {
            ModelInputs::Text {
                input_ids,
                attention_mask,
                ..
            } => {
                let mut hidden_states = ops_fn::embedding(input_ids, &self.embed_tokens)?;

                for layer in &self.layers {
                    hidden_states = layer.forward(
                        &hidden_states,
                        attention_mask.as_ref(),
                        self.config.rope_theta,
                        self.config.sliding_window,
                    )?;
                }

                hidden_states =
                    ops_fn::rms_norm(&hidden_states, &self.norm, self.config.rms_norm_eps)?;
                let logits = self.lm_head_forward(&hidden_states)?;

                Ok(ModelOutputs::Logits {
                    logits,
                    hidden_states: None,
                })
            }
            _ => Err(anyhow::anyhow!("Mixtral model only supports text inputs")),
        }
    }

    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        use crate::tokenizer::Tokenizer;
        use rand::Rng;

        let tokenizer = Tokenizer::new();
        let mut tokens: Vec<u32> = tokenizer.encode(prompt);

        for _ in 0..config.max_new_tokens {
            let tokens_i64: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
            let input_tensor =
                Tensor::from_i64_slice(&tokens_i64, &[1, tokens.len()], &self.device)?;

            let inputs = ModelInputs::Text {
                input_ids: input_tensor,
                attention_mask: None,
                position_ids: None,
            };

            let outputs = self.forward(&inputs)?;

            let logits = match outputs {
                ModelOutputs::Logits { logits, .. } => logits,
                _ => return Err(anyhow::anyhow!("Expected logits output")),
            };

            let logits_candle = logits.to_candle()?;
            let shape = logits_candle.dims();
            let last_logits = if shape.len() == 3 {
                let seq_len = shape[1];
                logits_candle
                    .narrow(1, seq_len - 1, 1)?
                    .squeeze(1)?
                    .squeeze(0)?
            } else {
                let seq_len = shape[0];
                logits_candle.narrow(0, seq_len - 1, 1)?.squeeze(0)?
            };

            let logits_vec: Vec<f32> = last_logits.to_vec1()?;

            let next_token = if config.do_sample && config.temperature > 0.0 {
                let scaled: Vec<f32> = logits_vec.iter().map(|&x| x / config.temperature).collect();
                let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scaled.iter().map(|&x| (x - max_val).exp()).sum();
                let probs: Vec<f32> = scaled
                    .iter()
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

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn memory_requirements(&self) -> MemoryRequirements {
        let attn_params = 4 * self.config.hidden_size * self.config.hidden_size;
        let expert_params = 3 * self.config.hidden_size * self.config.intermediate_size;
        let moe_params = self.config.num_experts * expert_params
            + self.config.hidden_size * self.config.num_experts;

        let param_size = self.config.vocab_size * self.config.hidden_size
            + self.config.num_hidden_layers * (attn_params + moe_params);

        let param_bytes = param_size * 4;
        let kv_cache_bytes = 2
            * self.config.num_hidden_layers
            * self.config.sliding_window
            * self.config.hidden_size
            * 4;

        MemoryRequirements {
            gpu_memory: param_bytes,
            cpu_memory: param_bytes / 4,
            kv_cache_memory: kv_cache_bytes,
            peak_memory: param_bytes + kv_cache_bytes,
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.embed_tokens = self.embed_tokens.to_device(device)?;
        self.norm = self.norm.to_device(device)?;
        self.lm_head = self.lm_head.to_device(device)?;

        for layer in &mut self.layers {
            layer.to_device(device)?;
        }

        self.device = device.clone();
        Ok(())
    }
}

impl MixtralModelV2 {
    fn lm_head_forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        if let Some(ref qmatmul) = self.lm_head_q {
            let input_candle = hidden_states.to_candle()?;
            let output = qmatmul
                .forward(&input_candle)
                .map_err(|e| anyhow::anyhow!("QMatMul lm_head forward failed: {}", e))?;
            Ok(Tensor::from_candle(output))
        } else {
            ops_fn::matmul(hidden_states, &self.lm_head)
        }
    }

    pub fn forward_with_cache(
        &self,
        inputs: &ModelInputs,
        mut cache: Option<&mut KVCache>,
    ) -> Result<ModelOutputs> {
        match inputs {
            ModelInputs::Text { input_ids, .. } => {
                let position_offset = cache.as_ref().map(|c| c.seq_len()).unwrap_or(0);
                let mut hidden_states = ops_fn::embedding(input_ids, &self.embed_tokens)?;

                for (layer_idx, layer) in self.layers.iter().enumerate() {
                    let layer_cache = cache.as_mut().map(|c| c.layer_mut(layer_idx));
                    hidden_states = layer.forward_with_cache(
                        &hidden_states,
                        layer_cache,
                        position_offset,
                        self.config.rope_theta,
                        self.config.rms_norm_eps,
                        self.config.sliding_window,
                    )?;
                }

                hidden_states =
                    ops_fn::rms_norm(&hidden_states, &self.norm, self.config.rms_norm_eps)?;
                let logits = self.lm_head_forward(&hidden_states)?;

                if let Some(cache) = cache {
                    let new_tokens = input_ids.shape().get(1).copied().unwrap_or(1);
                    cache.set_seq_len(position_offset + new_tokens);
                }

                Ok(ModelOutputs::Logits {
                    logits,
                    hidden_states: None,
                })
            }
            _ => Err(anyhow::anyhow!(
                "forward_with_cache only supports text inputs"
            )),
        }
    }

    pub fn generate_with_cache(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        use crate::tokenizer::Tokenizer;

        let tokenizer = Tokenizer::new();
        let prompt_tokens: Vec<u32> = tokenizer.encode(prompt);
        let mut tokens = prompt_tokens.clone();

        let mut cache = KVCache::new(self.config.num_hidden_layers);

        let prompt_i64: Vec<i64> = prompt_tokens.iter().map(|&t| t as i64).collect();
        let prompt_tensor =
            Tensor::from_i64_slice(&prompt_i64, &[1, prompt_tokens.len()], &self.device)?;
        let prompt_inputs = ModelInputs::Text {
            input_ids: prompt_tensor,
            attention_mask: None,
            position_ids: None,
        };

        let outputs = self.forward_with_cache(&prompt_inputs, Some(&mut cache))?;
        let logits = match outputs {
            ModelOutputs::Logits { logits, .. } => logits,
            _ => return Err(anyhow::anyhow!("Expected logits output")),
        };

        let logits_candle = logits.to_candle()?;
        let shape = logits_candle.dims();
        let seq_len = if shape.len() == 3 { shape[1] } else { shape[0] };
        let last_logits = if shape.len() == 3 {
            logits_candle
                .narrow(1, seq_len - 1, 1)?
                .squeeze(1)?
                .squeeze(0)?
        } else {
            logits_candle.narrow(0, seq_len - 1, 1)?.squeeze(0)?
        };

        let sampler = crate::inference::Sampler::new();
        let mut next_token =
            sampler.sample_with_config(&last_logits.to_vec1()?, config, &tokens)?;

        if next_token == config.eos_token_id {
            return Ok(tokenizer.decode(&tokens));
        }
        tokens.push(next_token);

        for _ in 1..config.max_new_tokens {
            let input_tensor = Tensor::from_i64_slice(&[next_token as i64], &[1, 1], &self.device)?;

            let inputs = ModelInputs::Text {
                input_ids: input_tensor,
                attention_mask: None,
                position_ids: None,
            };

            let outputs = self.forward_with_cache(&inputs, Some(&mut cache))?;
            let logits = match outputs {
                ModelOutputs::Logits { logits, .. } => logits,
                _ => return Err(anyhow::anyhow!("Expected logits output")),
            };

            let logits_candle = logits.to_candle()?;
            let last_logits = logits_candle.squeeze(0)?.squeeze(0)?;

            next_token = sampler.sample_with_config(&last_logits.to_vec1()?, config, &tokens)?;

            if next_token == config.eos_token_id {
                break;
            }
            tokens.push(next_token);
        }

        Ok(tokenizer.decode(&tokens))
    }

    pub fn generate_streaming_with_tokenizer(
        &self,
        tokenizer: &crate::tokenizer::Tokenizer,
        prompt_tokens: &[u32],
        config: &GenerationConfig,
        callback: &mut dyn FnMut(u32, &str) -> bool,
    ) -> Result<Vec<u32>> {
        let sampler = crate::inference::Sampler::new();
        let mut generated = Vec::new();

        let mut cache = KVCache::new(self.config.num_hidden_layers);

        let prompt_i64: Vec<i64> = prompt_tokens.iter().map(|&t| t as i64).collect();
        let prompt_tensor =
            Tensor::from_i64_slice(&prompt_i64, &[1, prompt_tokens.len()], &self.device)?;
        let prompt_inputs = ModelInputs::Text {
            input_ids: prompt_tensor,
            attention_mask: None,
            position_ids: None,
        };

        let outputs = self.forward_with_cache(&prompt_inputs, Some(&mut cache))?;
        let logits = match outputs {
            ModelOutputs::Logits { logits, .. } => logits,
            _ => return Err(anyhow::anyhow!("Expected logits output")),
        };

        let logits_candle = logits.to_candle()?;
        let shape = logits_candle.dims();
        let seq_len = if shape.len() == 3 { shape[1] } else { shape[0] };
        let last_logits = if shape.len() == 3 {
            logits_candle
                .narrow(1, seq_len - 1, 1)?
                .squeeze(1)?
                .squeeze(0)?
        } else {
            logits_candle.narrow(0, seq_len - 1, 1)?.squeeze(0)?
        };

        let mut next_token =
            sampler.sample_with_config(&last_logits.to_vec1()?, config, &generated)?;

        if next_token == config.eos_token_id {
            return Ok(generated);
        }

        generated.push(next_token);
        let text = tokenizer.decode_single(next_token);
        if !callback(next_token, &text) {
            return Ok(generated);
        }

        for _ in 1..config.max_new_tokens {
            let input_tensor = Tensor::from_i64_slice(&[next_token as i64], &[1, 1], &self.device)?;

            let inputs = ModelInputs::Text {
                input_ids: input_tensor,
                attention_mask: None,
                position_ids: None,
            };

            let outputs = self.forward_with_cache(&inputs, Some(&mut cache))?;
            let logits = match outputs {
                ModelOutputs::Logits { logits, .. } => logits,
                _ => return Err(anyhow::anyhow!("Expected logits output")),
            };

            let logits_candle = logits.to_candle()?;
            let last_logits = logits_candle.squeeze(0)?.squeeze(0)?;

            next_token = sampler.sample_with_config(&last_logits.to_vec1()?, config, &generated)?;

            if next_token == config.eos_token_id {
                break;
            }

            generated.push(next_token);
            let text = tokenizer.decode_single(next_token);
            if !callback(next_token, &text) {
                break;
            }
        }

        Ok(generated)
    }
}

impl ModelRunner for MixtralModelV2 {
    fn generate_streaming(
        &self,
        tokens: &[u32],
        config: &GenerationConfig,
        callback: &mut dyn FnMut(u32, &str) -> bool,
    ) -> Result<Vec<u32>> {
        let tokenizer = crate::tokenizer::Tokenizer::new();
        self.generate_streaming_with_tokenizer(&tokenizer, tokens, config, callback)
    }

    fn eos_token_id(&self) -> u32 {
        self.config.eos_token_id as u32
    }

    fn architecture(&self) -> &str {
        "mixtral"
    }

    fn context_size(&self) -> usize {
        self.config.sliding_window
    }
}

// ============================================================================
// Layer
// ============================================================================

impl MixtralLayer {
    fn new(config: &MixtralConfig, device: &Device) -> Result<Self> {
        let self_attn = MixtralAttention::new(config, device)?;
        let moe = MixtralMoE::new(config, device)?;

        let input_layernorm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;
        let post_attention_layernorm =
            ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;

        Ok(Self {
            self_attn,
            moe,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        rope_theta: f32,
        sliding_window: usize,
    ) -> Result<Tensor> {
        let normed = ops_fn::rms_norm(hidden_states, &self.input_layernorm, 1e-5)?;
        let attn_output =
            self.self_attn
                .forward(&normed, attention_mask, rope_theta, sliding_window)?;
        let hidden_states = ops_fn::add(hidden_states, &attn_output)?;
        let normed = ops_fn::rms_norm(&hidden_states, &self.post_attention_layernorm, 1e-5)?;
        let moe_output = self.moe.forward(&normed)?;
        let output = ops_fn::add(&hidden_states, &moe_output)?;
        Ok(output)
    }

    fn forward_with_cache(
        &self,
        hidden_states: &Tensor,
        cache: Option<&mut LayerKVCache>,
        position_offset: usize,
        rope_theta: f32,
        rms_norm_eps: f32,
        sliding_window: usize,
    ) -> Result<Tensor> {
        let normed = ops_fn::rms_norm(hidden_states, &self.input_layernorm, rms_norm_eps)?;
        let attn_output = self.self_attn.forward_with_cache(
            &normed,
            cache,
            position_offset,
            rope_theta,
            sliding_window,
        )?;
        let hidden_states = ops_fn::add(hidden_states, &attn_output)?;
        let normed =
            ops_fn::rms_norm(&hidden_states, &self.post_attention_layernorm, rms_norm_eps)?;
        let moe_output = self.moe.forward(&normed)?;
        let output = ops_fn::add(&hidden_states, &moe_output)?;
        Ok(output)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("model.layers.{}", layer_idx);

        if let Some(q_proj) = weights.get(&format!("{}.self_attn.q_proj.weight", prefix)) {
            self.self_attn.q_proj = ops_fn::transpose(q_proj)?;
        }
        if let Some(k_proj) = weights.get(&format!("{}.self_attn.k_proj.weight", prefix)) {
            self.self_attn.k_proj = ops_fn::transpose(k_proj)?;
        }
        if let Some(v_proj) = weights.get(&format!("{}.self_attn.v_proj.weight", prefix)) {
            self.self_attn.v_proj = ops_fn::transpose(v_proj)?;
        }
        if let Some(o_proj) = weights.get(&format!("{}.self_attn.o_proj.weight", prefix)) {
            self.self_attn.o_proj = ops_fn::transpose(o_proj)?;
        }

        self.self_attn.q_proj_q =
            weights.get_quantized(&format!("{}.self_attn.q_proj.weight", prefix));
        self.self_attn.k_proj_q =
            weights.get_quantized(&format!("{}.self_attn.k_proj.weight", prefix));
        self.self_attn.v_proj_q =
            weights.get_quantized(&format!("{}.self_attn.v_proj.weight", prefix));
        self.self_attn.o_proj_q =
            weights.get_quantized(&format!("{}.self_attn.o_proj.weight", prefix));

        if let Some(router) = weights.get(&format!("{}.block_sparse_moe.gate.weight", prefix)) {
            self.moe.router = ops_fn::transpose(router)?;
        }
        self.moe.router_q =
            weights.get_quantized(&format!("{}.block_sparse_moe.gate.weight", prefix));

        for (expert_idx, expert) in self.moe.experts.iter_mut().enumerate() {
            let expert_prefix = format!("{}.block_sparse_moe.experts.{}", prefix, expert_idx);

            if let Some(gate_proj) = weights.get(&format!("{}.w1.weight", expert_prefix)) {
                expert.gate_proj = ops_fn::transpose(gate_proj)?;
            }
            if let Some(up_proj) = weights.get(&format!("{}.w3.weight", expert_prefix)) {
                expert.up_proj = ops_fn::transpose(up_proj)?;
            }
            if let Some(down_proj) = weights.get(&format!("{}.w2.weight", expert_prefix)) {
                expert.down_proj = ops_fn::transpose(down_proj)?;
            }

            expert.gate_proj_q = weights.get_quantized(&format!("{}.w1.weight", expert_prefix));
            expert.up_proj_q = weights.get_quantized(&format!("{}.w3.weight", expert_prefix));
            expert.down_proj_q = weights.get_quantized(&format!("{}.w2.weight", expert_prefix));
        }

        if let Some(input_ln) = weights.get(&format!("{}.input_layernorm.weight", prefix)) {
            self.input_layernorm = input_ln.clone();
        }
        if let Some(post_ln) = weights.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
            self.post_attention_layernorm = post_ln.clone();
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.self_attn.to_device(device)?;
        self.moe.to_device(device)?;
        self.input_layernorm = self.input_layernorm.to_device(device)?;
        self.post_attention_layernorm = self.post_attention_layernorm.to_device(device)?;
        Ok(())
    }
}

// ============================================================================
// RoPE
// ============================================================================

fn apply_rope_with_offset(
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    seq_len: usize,
    head_dim: usize,
    rope_theta: f32,
    position_offset: usize,
) -> Result<(candle_core::Tensor, candle_core::Tensor)> {
    let device = q.device();
    let half_dim = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / rope_theta.powf((2 * i) as f32 / head_dim as f32))
        .collect();

    let positions: Vec<f32> = (0..seq_len).map(|p| (p + position_offset) as f32).collect();

    let mut angles = Vec::with_capacity(seq_len * half_dim);
    for pos in &positions {
        for freq in &inv_freq {
            angles.push(pos * freq);
        }
    }

    let angles_tensor = candle_core::Tensor::from_vec(angles, &[seq_len, half_dim], device)?;
    let cos = angles_tensor.cos()?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = angles_tensor.sin()?.unsqueeze(0)?.unsqueeze(0)?;

    let q_half1 = q.narrow(3, 0, half_dim)?;
    let q_half2 = q.narrow(3, half_dim, half_dim)?;
    let k_half1 = k.narrow(3, 0, half_dim)?;
    let k_half2 = k.narrow(3, half_dim, half_dim)?;

    let q_rot1 = (q_half1.broadcast_mul(&cos)? - q_half2.broadcast_mul(&sin)?)?;
    let q_rot2 = (q_half1.broadcast_mul(&sin)? + q_half2.broadcast_mul(&cos)?)?;
    let k_rot1 = (k_half1.broadcast_mul(&cos)? - k_half2.broadcast_mul(&sin)?)?;
    let k_rot2 = (k_half1.broadcast_mul(&sin)? + k_half2.broadcast_mul(&cos)?)?;

    let q_rotated = candle_core::Tensor::cat(&[&q_rot1, &q_rot2], 3)?;
    let k_rotated = candle_core::Tensor::cat(&[&k_rot1, &k_rot2], 3)?;

    Ok((q_rotated, k_rotated))
}

// ============================================================================
// Attention
// ============================================================================

impl MixtralAttention {
    fn new(config: &MixtralConfig, device: &Device) -> Result<Self> {
        let num_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads;
        let head_dim = config.hidden_size / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q_proj = ops_fn::zeros(
            &[config.hidden_size, num_heads * head_dim],
            DataType::Float32,
            device,
        )?;
        let k_proj = ops_fn::zeros(
            &[config.hidden_size, num_key_value_heads * head_dim],
            DataType::Float32,
            device,
        )?;
        let v_proj = ops_fn::zeros(
            &[config.hidden_size, num_key_value_heads * head_dim],
            DataType::Float32,
            device,
        )?;
        let o_proj = ops_fn::zeros(
            &[num_heads * head_dim, config.hidden_size],
            DataType::Float32,
            device,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_proj_q: None,
            k_proj_q: None,
            v_proj_q: None,
            o_proj_q: None,
            num_heads,
            num_key_value_heads,
            head_dim,
            scale,
            sliding_window: config.sliding_window,
        })
    }

    fn quantized_matmul(
        &self,
        input: &Tensor,
        weight: &Tensor,
        quantized: &Option<Arc<QMatMul>>,
    ) -> Result<Tensor> {
        if let Some(ref qmatmul) = quantized {
            let input_candle = input.to_candle()?;
            let output = qmatmul
                .forward(&input_candle)
                .map_err(|e| anyhow::anyhow!("QMatMul forward failed: {}", e))?;
            Ok(Tensor::from_candle(output))
        } else {
            ops_fn::matmul(input, weight)
        }
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        _attention_mask: Option<&Tensor>,
        rope_theta: f32,
        sliding_window: usize,
    ) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch_size, seq_len, _hidden_size) = if shape.len() == 3 {
            (shape[0], shape[1], shape[2])
        } else if shape.len() == 2 {
            (1, shape[0], shape[1])
        } else {
            return Err(anyhow::anyhow!("Invalid hidden_states shape: {:?}", shape));
        };

        let query_states = self.quantized_matmul(hidden_states, &self.q_proj, &self.q_proj_q)?;
        let key_states = self.quantized_matmul(hidden_states, &self.k_proj, &self.k_proj_q)?;
        let value_states = self.quantized_matmul(hidden_states, &self.v_proj, &self.v_proj_q)?;

        let q_candle = query_states.to_candle()?;
        let k_candle = key_states.to_candle()?;
        let v_candle = value_states.to_candle()?;

        let q_reshaped = q_candle
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k_reshaped = k_candle
            .reshape(&[batch_size, seq_len, self.num_key_value_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v_reshaped = v_candle
            .reshape(&[batch_size, seq_len, self.num_key_value_heads, self.head_dim])?
            .transpose(1, 2)?;

        let (q_with_rope, k_with_rope) = apply_rope_with_offset(
            &q_reshaped,
            &k_reshaped,
            seq_len,
            self.head_dim,
            rope_theta,
            0,
        )?;

        let num_groups = self.num_heads / self.num_key_value_heads;
        let (k_expanded, v_expanded) = if num_groups > 1 {
            let k_rep = k_with_rope
                .unsqueeze(2)?
                .broadcast_as(&[
                    batch_size,
                    self.num_key_value_heads,
                    num_groups,
                    seq_len,
                    self.head_dim,
                ])?
                .reshape(&[batch_size, self.num_heads, seq_len, self.head_dim])?;
            let v_rep = v_reshaped
                .unsqueeze(2)?
                .broadcast_as(&[
                    batch_size,
                    self.num_key_value_heads,
                    num_groups,
                    seq_len,
                    self.head_dim,
                ])?
                .reshape(&[batch_size, self.num_heads, seq_len, self.head_dim])?;
            (k_rep, v_rep)
        } else {
            (k_with_rope, v_reshaped)
        };

        let k_t = k_expanded.transpose(2, 3)?;
        let q_contiguous = q_with_rope.contiguous()?;
        let k_contiguous = k_t.contiguous()?;
        let scores = q_contiguous.matmul(&k_contiguous)?;
        let scaled_scores = (scores * (self.scale as f64))?;

        let device = scaled_scores.device();
        let causal_mask = {
            let mut mask_data = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                let window_start = if i > sliding_window {
                    i - sliding_window
                } else {
                    0
                };
                for j in 0..seq_len {
                    if j > i || j < window_start {
                        mask_data[i * seq_len + j] = f32::NEG_INFINITY;
                    }
                }
            }
            candle_core::Tensor::from_vec(mask_data, &[1, 1, seq_len, seq_len], device)?
        };

        let masked_scores = scaled_scores.broadcast_add(&causal_mask)?;
        let attention_weights = candle_nn::ops::softmax_last_dim(&masked_scores)?;

        let v_contiguous = v_expanded.contiguous()?;
        let attn_output = attention_weights.matmul(&v_contiguous)?;

        let attn_output = attn_output.transpose(1, 2)?.reshape(&[
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ])?;

        let attn_output = Tensor::from_candle(attn_output);
        let output = self.quantized_matmul(&attn_output, &self.o_proj, &self.o_proj_q)?;

        Ok(output)
    }

    fn forward_with_cache(
        &self,
        hidden_states: &Tensor,
        cache: Option<&mut LayerKVCache>,
        position_offset: usize,
        rope_theta: f32,
        sliding_window: usize,
    ) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch_size, new_seq_len, _hidden_size) = if shape.len() == 3 {
            (shape[0], shape[1], shape[2])
        } else if shape.len() == 2 {
            (1, shape[0], shape[1])
        } else {
            return Err(anyhow::anyhow!("Invalid hidden_states shape: {:?}", shape));
        };

        let query_states = self.quantized_matmul(hidden_states, &self.q_proj, &self.q_proj_q)?;
        let key_states = self.quantized_matmul(hidden_states, &self.k_proj, &self.k_proj_q)?;
        let value_states = self.quantized_matmul(hidden_states, &self.v_proj, &self.v_proj_q)?;

        let q_candle = query_states.to_candle()?;
        let k_candle = key_states.to_candle()?;
        let v_candle = value_states.to_candle()?;

        let q_reshaped = q_candle
            .reshape(&[batch_size, new_seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k_reshaped = k_candle
            .reshape(&[
                batch_size,
                new_seq_len,
                self.num_key_value_heads,
                self.head_dim,
            ])?
            .transpose(1, 2)?;
        let v_reshaped = v_candle
            .reshape(&[
                batch_size,
                new_seq_len,
                self.num_key_value_heads,
                self.head_dim,
            ])?
            .transpose(1, 2)?;

        let (q_with_rope, k_with_rope) = apply_rope_with_offset(
            &q_reshaped,
            &k_reshaped,
            new_seq_len,
            self.head_dim,
            rope_theta,
            position_offset,
        )?;

        let num_groups = self.num_heads / self.num_key_value_heads;
        let (k_expanded, v_expanded) = if num_groups > 1 {
            let k_rep = k_with_rope
                .unsqueeze(2)?
                .broadcast_as(&[
                    batch_size,
                    self.num_key_value_heads,
                    num_groups,
                    new_seq_len,
                    self.head_dim,
                ])?
                .contiguous()?
                .reshape(&[batch_size, self.num_heads, new_seq_len, self.head_dim])?;
            let v_rep = v_reshaped
                .unsqueeze(2)?
                .broadcast_as(&[
                    batch_size,
                    self.num_key_value_heads,
                    num_groups,
                    new_seq_len,
                    self.head_dim,
                ])?
                .contiguous()?
                .reshape(&[batch_size, self.num_heads, new_seq_len, self.head_dim])?;
            (k_rep, v_rep)
        } else {
            (k_with_rope.contiguous()?, v_reshaped.contiguous()?)
        };

        let (full_k, full_v, total_seq_len) = if let Some(cache) = cache {
            if let Err(e) = cache.append(&k_expanded, &v_expanded) {
                return Err(anyhow::anyhow!("KV cache append failed: {}", e));
            }
            match cache.get_kv() {
                Some((k, v)) => {
                    let total_len = k.dims()[2];
                    (k, v, total_len)
                }
                None => return Err(anyhow::anyhow!("Cache should not be empty after append")),
            }
        } else {
            (k_expanded, v_expanded, new_seq_len)
        };

        let k_t = full_k.transpose(2, 3)?;
        let q_contiguous = q_with_rope.contiguous()?;
        let k_contiguous = k_t.contiguous()?;
        let scores = q_contiguous.matmul(&k_contiguous)?;
        let scaled_scores = (scores * (self.scale as f64))?;

        let device = scaled_scores.device();
        let causal_mask = {
            let mut mask_data = vec![0.0f32; new_seq_len * total_seq_len];
            for i in 0..new_seq_len {
                let query_pos = position_offset + i;
                let window_start = if query_pos > sliding_window {
                    query_pos - sliding_window
                } else {
                    0
                };
                for j in 0..total_seq_len {
                    if j > query_pos || j < window_start {
                        mask_data[i * total_seq_len + j] = f32::NEG_INFINITY;
                    }
                }
            }
            candle_core::Tensor::from_vec(mask_data, &[1, 1, new_seq_len, total_seq_len], device)?
        };

        let masked_scores = scaled_scores.broadcast_add(&causal_mask)?;
        let attention_weights = candle_nn::ops::softmax_last_dim(&masked_scores)?;

        let v_contiguous = full_v.contiguous()?;
        let attn_output = attention_weights.matmul(&v_contiguous)?;

        let attn_output = attn_output.transpose(1, 2)?.reshape(&[
            batch_size,
            new_seq_len,
            self.num_heads * self.head_dim,
        ])?;

        let attn_output = Tensor::from_candle(attn_output);
        let output = self.quantized_matmul(&attn_output, &self.o_proj, &self.o_proj_q)?;

        Ok(output)
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.q_proj = self.q_proj.to_device(device)?;
        self.k_proj = self.k_proj.to_device(device)?;
        self.v_proj = self.v_proj.to_device(device)?;
        self.o_proj = self.o_proj.to_device(device)?;
        Ok(())
    }
}

// ============================================================================
// MoE
// ============================================================================

impl MixtralMoE {
    fn new(config: &MixtralConfig, device: &Device) -> Result<Self> {
        let router = ops_fn::zeros(
            &[config.hidden_size, config.num_experts],
            DataType::Float32,
            device,
        )?;

        let mut experts = Vec::with_capacity(config.num_experts);
        for _ in 0..config.num_experts {
            experts.push(MixtralExpert::new(config, device)?);
        }

        Ok(Self {
            router,
            router_q: None,
            experts,
            num_experts: config.num_experts,
            num_experts_per_tok: config.num_experts_per_tok,
        })
    }

    fn router_forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        if let Some(ref qmatmul) = self.router_q {
            let input_candle = hidden_states.to_candle()?;
            let output = qmatmul
                .forward(&input_candle)
                .map_err(|e| anyhow::anyhow!("QMatMul router forward failed: {}", e))?;
            Ok(Tensor::from_candle(output))
        } else {
            ops_fn::matmul(hidden_states, &self.router)
        }
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch_size, seq_len, hidden_size) = if shape.len() == 3 {
            (shape[0], shape[1], shape[2])
        } else if shape.len() == 2 {
            (1, shape[0], shape[1])
        } else {
            return Err(anyhow::anyhow!(
                "Invalid hidden_states shape for MoE: {:?}",
                shape
            ));
        };

        let num_tokens = batch_size * seq_len;
        let k = self.num_experts_per_tok;

        let flat_hidden = hidden_states.reshape(&[num_tokens, hidden_size])?;
        let router_logits = self.router_forward(&flat_hidden)?;

        let (topk_weights, topk_indices) = ops_fn::topk(&router_logits, k, -1)?;
        let routing_weights = ops_fn::softmax(&topk_weights, -1)?;

        let all_indices: Vec<i64> = topk_indices.to_candle()?.flatten_all()?.to_vec1()?;
        let all_weights: Vec<f32> = routing_weights.to_candle()?.flatten_all()?.to_vec1()?;
        let flat_hidden_candle = flat_hidden.to_candle()?;

        let mut output_data = vec![0.0f32; num_tokens * hidden_size];

        for tok_idx in 0..num_tokens {
            let token_hidden = flat_hidden_candle.get(tok_idx)?;
            let token_tensor = Tensor::from_candle(token_hidden.unsqueeze(0)?);

            let start = tok_idx * k;
            let indices = &all_indices[start..start + k];
            let weights = &all_weights[start..start + k];

            let mut token_output = ops_fn::zeros(
                &[1, hidden_size],
                hidden_states.dtype(),
                hidden_states.device(),
            )?;

            for (i, &expert_idx) in indices.iter().enumerate() {
                let expert = &self.experts[expert_idx as usize];
                let expert_output = expert.forward(&token_tensor)?;
                let scaled_output = ops_fn::scale(&expert_output, weights[i])?;
                token_output = ops_fn::add(&token_output, &scaled_output)?;
            }

            let token_data: Vec<f32> = token_output.to_candle()?.flatten_all()?.to_vec1()?;
            for (i, &v) in token_data.iter().enumerate() {
                output_data[tok_idx * hidden_size + i] = v;
            }
        }

        let output = Tensor::from_f32_slice(
            &output_data,
            &[num_tokens, hidden_size],
            hidden_states.device(),
        )?;
        output.reshape(&[batch_size, seq_len, hidden_size])
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.router = self.router.to_device(device)?;
        for expert in &mut self.experts {
            expert.to_device(device)?;
        }
        Ok(())
    }
}

impl MixtralExpert {
    fn new(config: &MixtralConfig, device: &Device) -> Result<Self> {
        let gate_proj = ops_fn::zeros(
            &[config.hidden_size, config.intermediate_size],
            DataType::Float32,
            device,
        )?;
        let up_proj = ops_fn::zeros(
            &[config.hidden_size, config.intermediate_size],
            DataType::Float32,
            device,
        )?;
        let down_proj = ops_fn::zeros(
            &[config.intermediate_size, config.hidden_size],
            DataType::Float32,
            device,
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            gate_proj_q: None,
            up_proj_q: None,
            down_proj_q: None,
        })
    }

    fn quantized_matmul(
        &self,
        input: &Tensor,
        weight: &Tensor,
        quantized: &Option<Arc<QMatMul>>,
    ) -> Result<Tensor> {
        if let Some(ref qmatmul) = quantized {
            let input_candle = input.to_candle()?;
            let output = qmatmul
                .forward(&input_candle)
                .map_err(|e| anyhow::anyhow!("QMatMul forward failed: {}", e))?;
            Ok(Tensor::from_candle(output))
        } else {
            ops_fn::matmul(input, weight)
        }
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let gate_output =
            self.quantized_matmul(hidden_states, &self.gate_proj, &self.gate_proj_q)?;
        let up_output = self.quantized_matmul(hidden_states, &self.up_proj, &self.up_proj_q)?;

        let gated = ops_fn::fused_swiglu(&gate_output, &up_output)?;

        let output = self.quantized_matmul(&gated, &self.down_proj, &self.down_proj_q)?;
        Ok(output)
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.gate_proj = self.gate_proj.to_device(device)?;
        self.up_proj = self.up_proj.to_device(device)?;
        self.down_proj = self.down_proj.to_device(device)?;
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixtral_model_creation() {
        let config = MixtralConfig {
            vocab_size: 1000,
            hidden_size: 128,
            intermediate_size: 512,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            num_experts: 4,
            num_experts_per_tok: 2,
            sliding_window: 256,
            ..Default::default()
        };

        let model = MixtralModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 1000);
        assert_eq!(model.config().hidden_size(), 128);
        assert_eq!(model.config().num_layers(), 2);
    }

    #[test]
    fn test_mixtral_forward_pass() {
        let config = MixtralConfig {
            vocab_size: 100,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_experts: 4,
            num_experts_per_tok: 2,
            sliding_window: 32,
            ..Default::default()
        };

        let model = MixtralModelV2::new(config).unwrap();
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

    #[test]
    fn test_mixtral_kv_cache_generation() {
        let config = MixtralConfig {
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_experts: 4,
            num_experts_per_tok: 2,
            sliding_window: 512,
            ..Default::default()
        };
        let model = MixtralModelV2::new(config).unwrap();
        let gen_config = GenerationConfig {
            max_new_tokens: 5,
            ..Default::default()
        };

        let output = model.generate_with_cache("Hello", &gen_config).unwrap();
        assert!(!output.is_empty());
    }
}
