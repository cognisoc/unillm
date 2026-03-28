//! Arctic Model V2 - Clean implementation
//!
//! Arctic (Snowflake) architecture features:
//! - Dense + MoE hybrid architecture
//! - Dense MLP runs in parallel with MoE
//! - RoPE embeddings

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

model_config!(ArcticConfig {
    vocab_size: usize = 32000,
    hidden_size: usize = 4096,
    intermediate_size: usize = 11008,
    moe_intermediate_size: usize = 2816,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 32,
    num_key_value_heads: usize = 8,
    hidden_act: String = "silu".to_string(),
    max_position_embeddings: usize = 4096,
    rms_norm_eps: f32 = 1e-5,
    use_cache: bool = true,
    pad_token_id: i64 = 0,
    bos_token_id: i64 = 1,
    eos_token_id: i64 = 2,
    tie_word_embeddings: bool = false,
    rope_theta: f32 = 10000.0,
    num_experts: usize = 128,
    num_experts_per_tok: usize = 2,
});

impl ArcticConfig {
    pub fn from_gguf_config(gguf: &crate::weight_loader_core::GGUFModelConfig) -> Self {
        Self {
            vocab_size: gguf.vocab_size,
            hidden_size: gguf.hidden_size,
            intermediate_size: gguf.intermediate_size,
            num_hidden_layers: gguf.num_hidden_layers,
            num_attention_heads: gguf.num_attention_heads,
            num_key_value_heads: gguf.num_key_value_heads,
            max_position_embeddings: gguf.max_position_embeddings,
            rope_theta: gguf.rope_theta,
            ..Default::default()
        }
    }
}

pub struct ArcticModelV2 {
    config: ArcticConfig,
    device: Device,
    embed_tokens: Tensor,
    layers: Vec<ArcticLayer>,
    norm: Tensor,
    lm_head: Tensor,
}

pub struct ArcticLayer {
    self_attn: ArcticAttention,
    dense_mlp: ArcticMLP,
    moe: ArcticMoE,
    input_layernorm: Tensor,
    post_attention_layernorm: Tensor,
}

pub struct ArcticAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
}

pub struct ArcticMLP {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
}

pub struct ArcticMoE {
    router: Tensor,
    experts: Vec<ArcticExpert>,
    num_experts_per_tok: usize,
}

pub struct ArcticExpert {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
}

fn apply_rope_arctic(
    q: &candle_core::Tensor, k: &candle_core::Tensor, seq_len: usize, head_dim: usize, rope_theta: f32,
) -> Result<(candle_core::Tensor, candle_core::Tensor)> {
    let device = q.device();
    let half_dim = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim).map(|i| 1.0 / rope_theta.powf((2 * i) as f32 / head_dim as f32)).collect();
    let positions: Vec<f32> = (0..seq_len).map(|p| p as f32).collect();
    let mut angles = Vec::with_capacity(seq_len * half_dim);
    for pos in &positions { for freq in &inv_freq { angles.push(pos * freq); } }
    let angles_tensor = candle_core::Tensor::from_vec(angles, &[seq_len, half_dim], device)?;
    let cos = angles_tensor.cos()?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = angles_tensor.sin()?.unsqueeze(0)?.unsqueeze(0)?;
    let (q_half1, q_half2) = (q.narrow(3, 0, half_dim)?, q.narrow(3, half_dim, half_dim)?);
    let (k_half1, k_half2) = (k.narrow(3, 0, half_dim)?, k.narrow(3, half_dim, half_dim)?);
    Ok((
        candle_core::Tensor::cat(&[&(q_half1.broadcast_mul(&cos)? - q_half2.broadcast_mul(&sin)?)?, &(q_half1.broadcast_mul(&sin)? + q_half2.broadcast_mul(&cos)?)?], 3)?,
        candle_core::Tensor::cat(&[&(k_half1.broadcast_mul(&cos)? - k_half2.broadcast_mul(&sin)?)?, &(k_half1.broadcast_mul(&sin)? + k_half2.broadcast_mul(&cos)?)?], 3)?
    ))
}

impl Model for ArcticModelV2 {
    type Config = ArcticConfig;

    fn new(config: ArcticConfig) -> Result<Self> {
        let device = Device::CPU;
        let embed_tokens = ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, &device)?;
        let norm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, &device)?;
        let lm_head = ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, &device)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers { layers.push(ArcticLayer::new(&config, &device)?); }
        Ok(Self { config, device, embed_tokens, layers, norm, lm_head })
    }

    fn from_weights(config: ArcticConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;
        if let Some(w) = weights.get("model.embed_tokens.weight") { model.embed_tokens = w.clone(); }
        if let Some(w) = weights.get("model.norm.weight") { model.norm = w.clone(); }
        if let Some(w) = weights.get("lm_head.weight") { model.lm_head = w.clone(); }
        for (i, layer) in model.layers.iter_mut().enumerate() { layer.load_weights(&weights, i)?; }
        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        match inputs {
            ModelInputs::Text { input_ids, .. } => {
                let seq_len = input_ids.shape()[1];
                let mut hidden = ops_fn::embedding(input_ids, &self.embed_tokens)?;
                for layer in &self.layers { hidden = layer.forward(&hidden, seq_len, self.config.rope_theta)?; }
                hidden = ops_fn::rms_norm(&hidden, &self.norm, self.config.rms_norm_eps)?;
                let logits = ops_fn::matmul(&hidden, &ops_fn::transpose(&self.lm_head)?)?;
                Ok(ModelOutputs::Logits { logits, hidden_states: None })
            }
            _ => Err(anyhow::anyhow!("Arctic only supports text inputs")),
        }
    }

    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        use crate::tokenizer::Tokenizer;
        use rand::Rng;
        let tokenizer = Tokenizer::new();
        let mut tokens: Vec<u32> = tokenizer.encode(prompt);
        for _ in 0..config.max_new_tokens {
            let tokens_i64: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
            let input = Tensor::from_i64_slice(&tokens_i64, &[1, tokens.len()], &self.device)?;
            let outputs = self.forward(&ModelInputs::text(input))?;
            let logits = match outputs { ModelOutputs::Logits { logits, .. } => logits, _ => return Err(anyhow::anyhow!("Expected logits")) };
            let logits_candle = logits.to_candle()?;
            let last = logits_candle.narrow(1, logits_candle.dims()[1] - 1, 1)?.squeeze(1)?.squeeze(0)?;
            let logits_vec: Vec<f32> = last.to_vec1()?;
            let next = if config.do_sample && config.temperature > 0.0 {
                let scaled: Vec<f32> = logits_vec.iter().map(|&x| x / config.temperature).collect();
                let max_v = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scaled.iter().map(|&x| (x - max_v).exp()).sum();
                let probs: Vec<f32> = scaled.iter().map(|&x| (x - max_v).exp() / exp_sum).collect();
                let mut rng = rand::thread_rng();
                let r: f32 = rng.gen();
                let mut cum = 0.0;
                let mut s = 0u32;
                for (i, &p) in probs.iter().enumerate() { cum += p; if r <= cum { s = i as u32; break; } }
                s
            } else {
                logits_vec.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as u32).unwrap_or(0)
            };
            if next == config.eos_token_id { break; }
            tokens.push(next);
        }
        Ok(tokenizer.decode(&tokens))
    }

    fn config(&self) -> &Self::Config { &self.config }
    fn memory_requirements(&self) -> MemoryRequirements {
        let p = self.config.vocab_size * self.config.hidden_size + self.config.num_hidden_layers * 8 * self.config.hidden_size.pow(2);
        MemoryRequirements { gpu_memory: p * 4, cpu_memory: p, kv_cache_memory: 2 * self.config.num_hidden_layers * self.config.max_position_embeddings * self.config.hidden_size * 4, peak_memory: p * 5 }
    }
    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.embed_tokens = self.embed_tokens.to_device(device)?;
        self.norm = self.norm.to_device(device)?;
        self.lm_head = self.lm_head.to_device(device)?;
        for layer in &mut self.layers { layer.to_device(device)?; }
        self.device = device.clone();
        Ok(())
    }
}

impl ArcticLayer {
    fn new(config: &ArcticConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            self_attn: ArcticAttention::new(config, device)?,
            dense_mlp: ArcticMLP::new(config.hidden_size, config.intermediate_size, device)?,
            moe: ArcticMoE::new(config, device)?,
            input_layernorm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            post_attention_layernorm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
        })
    }

    fn forward(&self, hidden_states: &Tensor, seq_len: usize, rope_theta: f32) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let h = ops_fn::rms_norm(hidden_states, &self.input_layernorm, 1e-5)?;
        let attn_out = self.self_attn.forward(&h, seq_len, rope_theta)?;
        let h = ops_fn::add(&residual, &attn_out)?;
        let residual = h.clone();
        let h = ops_fn::rms_norm(&h, &self.post_attention_layernorm, 1e-5)?;
        // Dense + MoE hybrid: run both and combine
        let dense_out = self.dense_mlp.forward(&h)?;
        let moe_out = self.moe.forward(&h)?;
        let mlp_out = ops_fn::add(&dense_out, &moe_out)?;
        ops_fn::add(&residual, &mlp_out)
    }

    fn load_weights(&mut self, weights: &ModelWeights, idx: usize) -> Result<()> {
        let p = format!("model.layers.{}", idx);
        if let Some(w) = weights.get(&format!("{}.self_attn.q_proj.weight", p)) { self.self_attn.q_proj = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.self_attn.k_proj.weight", p)) { self.self_attn.k_proj = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.self_attn.v_proj.weight", p)) { self.self_attn.v_proj = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.self_attn.o_proj.weight", p)) { self.self_attn.o_proj = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.input_layernorm.weight", p)) { self.input_layernorm = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.post_attention_layernorm.weight", p)) { self.post_attention_layernorm = w.clone(); }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.self_attn.to_device(device)?;
        self.dense_mlp.to_device(device)?;
        self.moe.to_device(device)?;
        self.input_layernorm = self.input_layernorm.to_device(device)?;
        self.post_attention_layernorm = self.post_attention_layernorm.to_device(device)?;
        Ok(())
    }
}

impl ArcticAttention {
    fn new(config: &ArcticConfig, device: &Device) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        Ok(Self {
            q_proj: ops_fn::zeros(&[config.hidden_size, config.num_attention_heads * head_dim], DataType::Float32, device)?,
            k_proj: ops_fn::zeros(&[config.hidden_size, config.num_key_value_heads * head_dim], DataType::Float32, device)?,
            v_proj: ops_fn::zeros(&[config.hidden_size, config.num_key_value_heads * head_dim], DataType::Float32, device)?,
            o_proj: ops_fn::zeros(&[config.num_attention_heads * head_dim, config.hidden_size], DataType::Float32, device)?,
            num_heads: config.num_attention_heads, num_kv_heads: config.num_key_value_heads, head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    fn forward(&self, hidden_states: &Tensor, seq_len: usize, rope_theta: f32) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let batch = shape[0];
        let q = ops_fn::matmul(hidden_states, &self.q_proj)?.to_candle()?.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?.transpose(1, 2)?;
        let k = ops_fn::matmul(hidden_states, &self.k_proj)?.to_candle()?.reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?.transpose(1, 2)?;
        let v = ops_fn::matmul(hidden_states, &self.v_proj)?.to_candle()?.reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?.transpose(1, 2)?;
        let (q, k) = apply_rope_arctic(&q, &k, seq_len, self.head_dim, rope_theta)?;
        let num_groups = self.num_heads / self.num_kv_heads;
        let (k, v) = if num_groups > 1 {
            (k.unsqueeze(2)?.broadcast_as(&[batch, self.num_kv_heads, num_groups, seq_len, self.head_dim])?.reshape(&[batch, self.num_heads, seq_len, self.head_dim])?,
             v.unsqueeze(2)?.broadcast_as(&[batch, self.num_kv_heads, num_groups, seq_len, self.head_dim])?.reshape(&[batch, self.num_heads, seq_len, self.head_dim])?)
        } else { (k, v) };
        let q = q.contiguous()?;
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scores = (q.matmul(&k_t)? * (self.scale as f64))?;
        let device = scores.device();
        let mut m = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len { for j in (i+1)..seq_len { m[i*seq_len+j] = f32::NEG_INFINITY; } }
        let mask = candle_core::Tensor::from_vec(m, &[1, 1, seq_len, seq_len], device)?;
        let scores = scores.broadcast_add(&mask)?;
        let v = v.contiguous()?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?;
        let out = attn.transpose(1, 2)?.reshape(&[batch, seq_len, self.num_heads * self.head_dim])?;
        ops_fn::matmul(&Tensor::from_candle(out), &self.o_proj)
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.q_proj = self.q_proj.to_device(device)?; self.k_proj = self.k_proj.to_device(device)?;
        self.v_proj = self.v_proj.to_device(device)?; self.o_proj = self.o_proj.to_device(device)?;
        Ok(())
    }
}

impl ArcticMLP {
    fn new(hidden_size: usize, intermediate_size: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            gate_proj: ops_fn::zeros(&[hidden_size, intermediate_size], DataType::Float32, device)?,
            up_proj: ops_fn::zeros(&[hidden_size, intermediate_size], DataType::Float32, device)?,
            down_proj: ops_fn::zeros(&[intermediate_size, hidden_size], DataType::Float32, device)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = ops_fn::matmul(x, &self.gate_proj)?;
        let up = ops_fn::matmul(x, &self.up_proj)?;
        let h = ops_fn::mul(&ops_fn::silu(&gate)?, &up)?;
        ops_fn::matmul(&h, &self.down_proj)
    }
    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.gate_proj = self.gate_proj.to_device(device)?;
        self.up_proj = self.up_proj.to_device(device)?;
        self.down_proj = self.down_proj.to_device(device)?;
        Ok(())
    }
}

impl ArcticMoE {
    fn new(config: &ArcticConfig, device: &Device) -> Result<Self> {
        let router = ops_fn::zeros(&[config.hidden_size, config.num_experts], DataType::Float32, device)?;
        let mut experts = Vec::with_capacity(config.num_experts);
        for _ in 0..config.num_experts { experts.push(ArcticExpert::new(config.hidden_size, config.moe_intermediate_size, device)?); }
        Ok(Self { router, experts, num_experts_per_tok: config.num_experts_per_tok })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch_size, seq_len, hidden_size) = (shape[0], shape[1], shape[2]);
        let num_tokens = batch_size * seq_len;
        let k = self.num_experts_per_tok;
        let flat_hidden = hidden_states.reshape(&[num_tokens, hidden_size])?;
        let router_logits = ops_fn::matmul(&flat_hidden, &self.router)?;
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
            let mut token_output = ops_fn::zeros(&[1, hidden_size], hidden_states.dtype(), hidden_states.device())?;
            for (i, &expert_idx) in indices.iter().enumerate() {
                if (expert_idx as usize) < self.experts.len() {
                    let expert = &self.experts[expert_idx as usize];
                    let expert_output = expert.forward(&token_tensor)?;
                    let scaled_output = ops_fn::scale(&expert_output, weights[i])?;
                    token_output = ops_fn::add(&token_output, &scaled_output)?;
                }
            }
            let token_data: Vec<f32> = token_output.to_candle()?.flatten_all()?.to_vec1()?;
            for (i, &v) in token_data.iter().enumerate() { output_data[tok_idx * hidden_size + i] = v; }
        }
        let output = Tensor::from_f32_slice(&output_data, &[num_tokens, hidden_size], hidden_states.device())?;
        output.reshape(&[batch_size, seq_len, hidden_size])
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.router = self.router.to_device(device)?;
        for expert in &mut self.experts { expert.to_device(device)?; }
        Ok(())
    }
}

impl ArcticExpert {
    fn new(hidden_size: usize, intermediate_size: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            gate_proj: ops_fn::zeros(&[hidden_size, intermediate_size], DataType::Float32, device)?,
            up_proj: ops_fn::zeros(&[hidden_size, intermediate_size], DataType::Float32, device)?,
            down_proj: ops_fn::zeros(&[intermediate_size, hidden_size], DataType::Float32, device)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = ops_fn::matmul(x, &self.gate_proj)?;
        let up = ops_fn::matmul(x, &self.up_proj)?;
        let h = ops_fn::mul(&ops_fn::silu(&gate)?, &up)?;
        ops_fn::matmul(&h, &self.down_proj)
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
    fn test_arctic_creation() {
        let config = ArcticConfig { vocab_size: 1000, hidden_size: 64, intermediate_size: 256, moe_intermediate_size: 64, num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 2, num_experts: 4, num_experts_per_tok: 2, ..Default::default() };
        let model = ArcticModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 1000);
    }
    #[test]
    fn test_arctic_forward() {
        let config = ArcticConfig { vocab_size: 100, hidden_size: 64, intermediate_size: 256, moe_intermediate_size: 64, num_hidden_layers: 1, num_attention_heads: 4, num_key_value_heads: 2, num_experts: 4, num_experts_per_tok: 2, ..Default::default() };
        let model = ArcticModelV2::new(config).unwrap();
        let inputs = ModelInputs::text(ops_fn::zeros(&[1, 4], DataType::Int64, &Device::CPU).unwrap());
        match model.forward(&inputs).unwrap() { ModelOutputs::Logits { logits, .. } => assert_eq!(logits.shape(), &[1, 4, 100]), _ => panic!() }
    }
}
