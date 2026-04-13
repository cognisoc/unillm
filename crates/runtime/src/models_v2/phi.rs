//! Phi Model V2 - Production-quality implementation
//!
//! Implements the Phi architecture with:
//! - KV caching for efficient autoregressive generation
//! - Quantized weight support (QMatMul)
//! - Grouped Query Attention (GQA)
//! - Partial RoPE (Rotary Position Embedding)
//! - GELU activation (standard 2-layer MLP, not SwiGLU)
//! - Layer normalization
//! - Parallel attention + MLP per layer
//! - ModelRunner trait for streaming generation

use super::traits::*;
use crate::kv_cache::{KVCache, LayerKVCache};
use crate::model_config;
use anyhow::Result;
use candle_core::quantized::QMatMul;
use candle_nn::Module;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

model_config!(PhiConfig {
    vocab_size: usize = 51200,
    hidden_size: usize = 2560,
    intermediate_size: usize = 10240,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 32,
    num_key_value_heads: usize = 32,
    hidden_act: String = "gelu_new".to_string(),
    max_position_embeddings: usize = 2048,
    initializer_range: f32 = 0.02,
    rms_norm_eps: f32 = 1e-5,
    use_cache: bool = true,
    pad_token_id: i64 = 0,
    bos_token_id: i64 = 1,
    eos_token_id: i64 = 2,
    tie_word_embeddings: bool = false,
    rope_theta: f32 = 10000.0,
    attention_dropout: f32 = 0.0,
    partial_rotary_factor: f32 = 0.5,
});

impl PhiConfig {
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

pub struct PhiModelV2 {
    config: PhiConfig,
    device: Device,
    embed_tokens: Tensor,
    layers: Vec<PhiLayer>,
    norm: Tensor,
    lm_head: Tensor,
    lm_head_q: Option<Arc<QMatMul>>,
}

pub struct PhiLayer {
    self_attn: PhiAttention,
    mlp: PhiMLP,
    input_layernorm: Tensor,
}

pub struct PhiAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    dense: Tensor,
    q_proj_q: Option<Arc<QMatMul>>,
    k_proj_q: Option<Arc<QMatMul>>,
    v_proj_q: Option<Arc<QMatMul>>,
    dense_q: Option<Arc<QMatMul>>,
    num_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    scale: f32,
    partial_rotary_factor: f32,
}

pub struct PhiMLP {
    fc1: Tensor,
    fc2: Tensor,
    fc1_q: Option<Arc<QMatMul>>,
    fc2_q: Option<Arc<QMatMul>>,
    hidden_act: String,
}

impl Model for PhiModelV2 {
    type Config = PhiConfig;

    fn new(config: PhiConfig) -> Result<Self> {
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
            layers.push(PhiLayer::new(&config, &device)?);
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

    fn from_weights(config: PhiConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;

        if let Some(embed_weights) = weights.get("model.embed_tokens.weight") {
            model.embed_tokens = embed_weights.clone();
        }

        if let Some(norm_weights) = weights.get("model.final_layernorm.weight") {
            model.norm = norm_weights.clone();
        } else if let Some(norm_weights) = weights.get("model.norm.weight") {
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
                        self.config.partial_rotary_factor,
                    )?;
                }

                hidden_states =
                    ops_fn::layer_norm(&hidden_states, &self.norm, None, self.config.rms_norm_eps)?;

                let logits = self.lm_head_forward(&hidden_states)?;

                Ok(ModelOutputs::Logits {
                    logits,
                    hidden_states: None,
                })
            }
            _ => Err(anyhow::anyhow!("Phi model only supports text inputs")),
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
        let param_size = self.config.vocab_size * self.config.hidden_size
            + self.config.num_hidden_layers
                * (4 * self.config.hidden_size * self.config.hidden_size
                    + 2 * self.config.hidden_size * self.config.intermediate_size);
        let param_bytes = param_size * 4;
        let kv_cache_bytes = 2
            * self.config.num_hidden_layers
            * self.config.max_position_embeddings
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

impl PhiModelV2 {
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
                        self.config.partial_rotary_factor,
                        self.config.rms_norm_eps,
                    )?;
                }

                hidden_states =
                    ops_fn::layer_norm(&hidden_states, &self.norm, None, self.config.rms_norm_eps)?;
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

impl ModelRunner for PhiModelV2 {
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
        "phi"
    }

    fn context_size(&self) -> usize {
        self.config.max_position_embeddings
    }
}

impl PhiLayer {
    fn new(config: &PhiConfig, device: &Device) -> Result<Self> {
        let self_attn = PhiAttention::new(config, device)?;
        let mlp = PhiMLP::new(config, device)?;

        let input_layernorm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        rope_theta: f32,
        partial_rotary_factor: f32,
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();

        let normed = ops_fn::layer_norm(hidden_states, &self.input_layernorm, None, 1e-5)?;

        let attn_output =
            self.self_attn
                .forward(&normed, attention_mask, rope_theta, partial_rotary_factor)?;
        let mlp_output = self.mlp.forward(&normed)?;

        let combined = ops_fn::add(&attn_output, &mlp_output)?;
        let output = ops_fn::add(&residual, &combined)?;
        Ok(output)
    }

    fn forward_with_cache(
        &self,
        hidden_states: &Tensor,
        cache: Option<&mut LayerKVCache>,
        position_offset: usize,
        rope_theta: f32,
        partial_rotary_factor: f32,
        layer_norm_eps: f32,
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();

        let normed =
            ops_fn::layer_norm(hidden_states, &self.input_layernorm, None, layer_norm_eps)?;

        let attn_output = self.self_attn.forward_with_cache(
            &normed,
            cache,
            position_offset,
            rope_theta,
            partial_rotary_factor,
        )?;
        let mlp_output = self.mlp.forward(&normed)?;

        let combined = ops_fn::add(&attn_output, &mlp_output)?;
        let output = ops_fn::add(&residual, &combined)?;
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
        if let Some(dense) = weights.get(&format!("{}.self_attn.dense.weight", prefix)) {
            self.self_attn.dense = ops_fn::transpose(dense)?;
        } else if let Some(o_proj) = weights.get(&format!("{}.self_attn.o_proj.weight", prefix)) {
            self.self_attn.dense = ops_fn::transpose(o_proj)?;
        }

        self.self_attn.q_proj_q =
            weights.get_quantized(&format!("{}.self_attn.q_proj.weight", prefix));
        self.self_attn.k_proj_q =
            weights.get_quantized(&format!("{}.self_attn.k_proj.weight", prefix));
        self.self_attn.v_proj_q =
            weights.get_quantized(&format!("{}.self_attn.v_proj.weight", prefix));
        if let Some(dense_q) = weights.get_quantized(&format!("{}.self_attn.dense.weight", prefix))
        {
            self.self_attn.dense_q = Some(dense_q);
        } else {
            self.self_attn.dense_q =
                weights.get_quantized(&format!("{}.self_attn.o_proj.weight", prefix));
        }

        if let Some(fc1) = weights.get(&format!("{}.mlp.fc1.weight", prefix)) {
            self.mlp.fc1 = ops_fn::transpose(fc1)?;
        }
        if let Some(fc2) = weights.get(&format!("{}.mlp.fc2.weight", prefix)) {
            self.mlp.fc2 = ops_fn::transpose(fc2)?;
        }

        self.mlp.fc1_q = weights.get_quantized(&format!("{}.mlp.fc1.weight", prefix));
        self.mlp.fc2_q = weights.get_quantized(&format!("{}.mlp.fc2.weight", prefix));

        if let Some(input_ln) = weights.get(&format!("{}.input_layernorm.weight", prefix)) {
            self.input_layernorm = input_ln.clone();
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.self_attn.to_device(device)?;
        self.mlp.to_device(device)?;
        self.input_layernorm = self.input_layernorm.to_device(device)?;
        Ok(())
    }
}

fn apply_partial_rope_with_offset(
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    seq_len: usize,
    head_dim: usize,
    rope_theta: f32,
    partial_rotary_factor: f32,
    position_offset: usize,
) -> Result<(candle_core::Tensor, candle_core::Tensor)> {
    let device = q.device();

    let rotary_dim = ((head_dim as f32 * partial_rotary_factor) as usize / 2) * 2;
    let half_rotary_dim = rotary_dim / 2;

    if rotary_dim == 0 {
        return Ok((q.clone(), k.clone()));
    }

    let inv_freq: Vec<f32> = (0..half_rotary_dim)
        .map(|i| 1.0 / rope_theta.powf((2 * i) as f32 / rotary_dim as f32))
        .collect();

    let positions: Vec<f32> = (0..seq_len).map(|p| (p + position_offset) as f32).collect();

    let mut angles = Vec::with_capacity(seq_len * half_rotary_dim);
    for pos in &positions {
        for freq in &inv_freq {
            angles.push(pos * freq);
        }
    }

    let angles_tensor = candle_core::Tensor::from_vec(angles, &[seq_len, half_rotary_dim], device)?;
    let cos = angles_tensor.cos()?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = angles_tensor.sin()?.unsqueeze(0)?.unsqueeze(0)?;

    let q_rot_part = q.narrow(3, 0, rotary_dim)?;
    let q_pass_part = q.narrow(3, rotary_dim, head_dim - rotary_dim)?;
    let k_rot_part = k.narrow(3, 0, rotary_dim)?;
    let k_pass_part = k.narrow(3, rotary_dim, head_dim - rotary_dim)?;

    let q_half1 = q_rot_part.narrow(3, 0, half_rotary_dim)?;
    let q_half2 = q_rot_part.narrow(3, half_rotary_dim, half_rotary_dim)?;
    let k_half1 = k_rot_part.narrow(3, 0, half_rotary_dim)?;
    let k_half2 = k_rot_part.narrow(3, half_rotary_dim, half_rotary_dim)?;

    let q_rot1 = (q_half1.broadcast_mul(&cos)? - q_half2.broadcast_mul(&sin)?)?;
    let q_rot2 = (q_half1.broadcast_mul(&sin)? + q_half2.broadcast_mul(&cos)?)?;
    let k_rot1 = (k_half1.broadcast_mul(&cos)? - k_half2.broadcast_mul(&sin)?)?;
    let k_rot2 = (k_half1.broadcast_mul(&sin)? + k_half2.broadcast_mul(&cos)?)?;

    let q_rotated_part = candle_core::Tensor::cat(&[&q_rot1, &q_rot2], 3)?;
    let k_rotated_part = candle_core::Tensor::cat(&[&k_rot1, &k_rot2], 3)?;

    let q_rotated = candle_core::Tensor::cat(&[&q_rotated_part, &q_pass_part], 3)?;
    let k_rotated = candle_core::Tensor::cat(&[&k_rotated_part, &k_pass_part], 3)?;

    Ok((q_rotated, k_rotated))
}

impl PhiAttention {
    fn new(config: &PhiConfig, device: &Device) -> Result<Self> {
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
        let dense = ops_fn::zeros(
            &[num_heads * head_dim, config.hidden_size],
            DataType::Float32,
            device,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            dense,
            q_proj_q: None,
            k_proj_q: None,
            v_proj_q: None,
            dense_q: None,
            num_heads,
            num_key_value_heads,
            head_dim,
            scale,
            partial_rotary_factor: config.partial_rotary_factor,
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
        partial_rotary_factor: f32,
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

        let (q_with_rope, k_with_rope) = apply_partial_rope_with_offset(
            &q_reshaped,
            &k_reshaped,
            seq_len,
            self.head_dim,
            rope_theta,
            partial_rotary_factor,
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

        let v_contiguous = v_expanded.contiguous()?;
        let attn_output = attention_weights.matmul(&v_contiguous)?;

        let attn_output = attn_output.transpose(1, 2)?.reshape(&[
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ])?;

        let attn_output = Tensor::from_candle(attn_output);
        let output = self.quantized_matmul(&attn_output, &self.dense, &self.dense_q)?;

        Ok(output)
    }

    fn forward_with_cache(
        &self,
        hidden_states: &Tensor,
        cache: Option<&mut LayerKVCache>,
        position_offset: usize,
        rope_theta: f32,
        partial_rotary_factor: f32,
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

        let (q_with_rope, k_with_rope) = apply_partial_rope_with_offset(
            &q_reshaped,
            &k_reshaped,
            new_seq_len,
            self.head_dim,
            rope_theta,
            partial_rotary_factor,
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
                .reshape(&[batch_size, self.num_heads, new_seq_len, self.head_dim])?;
            (k_rep, v_rep)
        } else {
            (k_with_rope, v_reshaped)
        };

        let k_for_cache = k_expanded.contiguous()?;
        let v_for_cache = v_expanded.contiguous()?;

        let (full_k, full_v, total_seq_len) = if let Some(cache) = cache {
            if let Err(e) = cache.append(&k_for_cache, &v_for_cache) {
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
                for j in 0..total_seq_len {
                    if j > query_pos {
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
        let output = self.quantized_matmul(&attn_output, &self.dense, &self.dense_q)?;

        Ok(output)
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.q_proj = self.q_proj.to_device(device)?;
        self.k_proj = self.k_proj.to_device(device)?;
        self.v_proj = self.v_proj.to_device(device)?;
        self.dense = self.dense.to_device(device)?;
        Ok(())
    }
}

impl PhiMLP {
    fn new(config: &PhiConfig, device: &Device) -> Result<Self> {
        let fc1 = ops_fn::zeros(
            &[config.hidden_size, config.intermediate_size],
            DataType::Float32,
            device,
        )?;
        let fc2 = ops_fn::zeros(
            &[config.intermediate_size, config.hidden_size],
            DataType::Float32,
            device,
        )?;

        Ok(Self {
            fc1,
            fc2,
            fc1_q: None,
            fc2_q: None,
            hidden_act: config.hidden_act.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let intermediate = self.quantized_matmul(hidden_states, &self.fc1, &self.fc1_q)?;

        let activated = match self.hidden_act.as_str() {
            "gelu" | "gelu_new" => ops_fn::gelu(&intermediate)?,
            "silu" | "swish" => ops_fn::silu(&intermediate)?,
            _ => {
                return Err(anyhow::anyhow!(
                    "Unsupported activation: {}",
                    self.hidden_act
                ))
            }
        };

        let output = self.quantized_matmul(&activated, &self.fc2, &self.fc2_q)?;
        Ok(output)
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

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.fc1 = self.fc1.to_device(device)?;
        self.fc2 = self.fc2.to_device(device)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_model_creation() {
        let config = PhiConfig {
            vocab_size: 1000,
            hidden_size: 128,
            intermediate_size: 512,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            num_key_value_heads: 8,
            ..Default::default()
        };

        let model = PhiModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 1000);
        assert_eq!(model.config().hidden_size(), 128);
        assert_eq!(model.config().num_layers(), 2);
    }

    #[test]
    fn test_phi_forward_pass() {
        let config = PhiConfig {
            vocab_size: 100,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            ..Default::default()
        };

        let model = PhiModelV2::new(config).unwrap();
        let input_ids = ops_fn::zeros(&[2, 8], DataType::Int64, &Device::CPU).unwrap();
        let inputs = ModelInputs::text(input_ids);

        let outputs = model.forward(&inputs).unwrap();
        match outputs {
            ModelOutputs::Logits { logits, .. } => {
                assert_eq!(logits.shape(), &[2, 8, 100]);
            }
            _ => panic!("Expected logits output"),
        }
    }

    #[test]
    fn test_phi_kv_cache_generation() {
        let config = PhiConfig {
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            ..Default::default()
        };
        let model = PhiModelV2::new(config).unwrap();
        let gen_config = GenerationConfig {
            max_new_tokens: 5,
            ..Default::default()
        };

        let output = model.generate_with_cache("Hello", &gen_config).unwrap();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_partial_rope() {
        let config = PhiConfig {
            vocab_size: 100,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            partial_rotary_factor: 0.5,
            ..Default::default()
        };

        let model = PhiModelV2::new(config).unwrap();
        assert_eq!(model.config().partial_rotary_factor, 0.5);
    }

    #[test]
    fn test_phi_gqa() {
        let config = PhiConfig {
            vocab_size: 100,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            ..Default::default()
        };

        let model = PhiModelV2::new(config).unwrap();
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
