//! MPT Model V2 - Clean implementation
//!
//! MPT (MosaicML Pretrained Transformer) architecture features:
//! - ALiBi (Attention with Linear Biases) position embeddings
//! - Low-rank attention options
//! - Flash attention compatibility

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

model_config!(MPTConfig {
    vocab_size: usize = 50432,
    hidden_size: usize = 4096,
    intermediate_size: usize = 16384,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 32,
    num_key_value_heads: usize = 32,
    hidden_act: String = "gelu".to_string(),
    max_position_embeddings: usize = 2048,
    initializer_range: f32 = 0.02,
    layer_norm_eps: f32 = 1e-5,
    use_cache: bool = true,
    pad_token_id: i64 = 0,
    bos_token_id: i64 = 0,
    eos_token_id: i64 = 0,
    tie_word_embeddings: bool = true,
    alibi: bool = true,
    no_bias: bool = true,
});

impl MPTConfig {
    pub fn from_gguf_config(gguf: &crate::weight_loader_core::GGUFModelConfig) -> Self {
        Self {
            vocab_size: gguf.vocab_size,
            hidden_size: gguf.hidden_size,
            intermediate_size: gguf.intermediate_size,
            num_hidden_layers: gguf.num_hidden_layers,
            num_attention_heads: gguf.num_attention_heads,
            num_key_value_heads: gguf.num_key_value_heads,
            max_position_embeddings: gguf.max_position_embeddings,
            ..Default::default()
        }
    }
}

pub struct MPTModelV2 {
    config: MPTConfig,
    device: Device,
    wte: Tensor,
    blocks: Vec<MPTBlock>,
    norm_f: Tensor,
}

pub struct MPTBlock {
    attn: MPTAttention,
    ffn: MPTFFN,
    norm_1: Tensor,
    norm_2: Tensor,
}

pub struct MPTAttention {
    wqkv: Tensor,
    out_proj: Tensor,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    alibi: bool,
}

pub struct MPTFFN {
    up_proj: Tensor,
    down_proj: Tensor,
}

impl Model for MPTModelV2 {
    type Config = MPTConfig;

    fn new(config: MPTConfig) -> Result<Self> {
        let device = Device::CPU;
        let wte = ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, &device)?;
        let norm_f = ops_fn::zeros(&[config.hidden_size], DataType::Float32, &device)?;

        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            blocks.push(MPTBlock::new(&config, &device)?);
        }

        Ok(Self { config, device, wte, blocks, norm_f })
    }

    fn from_weights(config: MPTConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;
        if let Some(w) = weights.get("transformer.wte.weight") { model.wte = w.clone(); }
        if let Some(w) = weights.get("transformer.norm_f.weight") { model.norm_f = w.clone(); }
        for (i, block) in model.blocks.iter_mut().enumerate() { block.load_weights(&weights, i)?; }
        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        match inputs {
            ModelInputs::Text { input_ids, .. } => {
                let mut hidden = ops_fn::embedding(input_ids, &self.wte)?;
                let seq_len = input_ids.shape()[1];

                for block in &self.blocks {
                    hidden = block.forward(&hidden, seq_len)?;
                }

                hidden = ops_fn::layer_norm(&hidden, &self.norm_f, None, self.config.layer_norm_eps)?;

                // Tied embeddings - flatten to 2D for matmul, then reshape back
                let wte_candle = self.wte.to_candle()?;
                let hidden_candle = hidden.to_candle()?.contiguous()?;
                let batch = hidden_candle.dims()[0];
                let seq = hidden_candle.dims()[1];
                let hidden_size = hidden_candle.dims()[2];
                let flat = hidden_candle.reshape(&[batch * seq, hidden_size])?;
                let logits_flat = flat.matmul(&wte_candle.t()?)?;
                let logits_candle = logits_flat.reshape(&[batch, seq, self.config.vocab_size])?;
                let logits = Tensor::from_candle(logits_candle);

                Ok(ModelOutputs::Logits { logits, hidden_states: None })
            }
            _ => Err(anyhow::anyhow!("MPT only supports text inputs")),
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
        self.wte = self.wte.to_device(device)?;
        self.norm_f = self.norm_f.to_device(device)?;
        for b in &mut self.blocks { b.to_device(device)?; }
        self.device = device.clone();
        Ok(())
    }
}

impl MPTBlock {
    fn new(config: &MPTConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            attn: MPTAttention::new(config, device)?,
            ffn: MPTFFN::new(config, device)?,
            norm_1: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            norm_2: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
        })
    }

    fn forward(&self, hidden_states: &Tensor, seq_len: usize) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let h = ops_fn::layer_norm(hidden_states, &self.norm_1, None, 1e-5)?;
        let attn_out = self.attn.forward(&h, seq_len)?;
        let h = ops_fn::add(&residual, &attn_out)?;

        let residual = h.clone();
        let h = ops_fn::layer_norm(&h, &self.norm_2, None, 1e-5)?;
        let ffn_out = self.ffn.forward(&h)?;
        ops_fn::add(&residual, &ffn_out)
    }

    fn load_weights(&mut self, weights: &ModelWeights, idx: usize) -> Result<()> {
        let p = format!("transformer.blocks.{}", idx);
        if let Some(w) = weights.get(&format!("{}.attn.Wqkv.weight", p)) { self.attn.wqkv = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.attn.out_proj.weight", p)) { self.attn.out_proj = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.ffn.up_proj.weight", p)) { self.ffn.up_proj = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.ffn.down_proj.weight", p)) { self.ffn.down_proj = ops_fn::transpose(w)?; }
        if let Some(w) = weights.get(&format!("{}.norm_1.weight", p)) { self.norm_1 = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.norm_2.weight", p)) { self.norm_2 = w.clone(); }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.attn.to_device(device)?;
        self.ffn.to_device(device)?;
        self.norm_1 = self.norm_1.to_device(device)?;
        self.norm_2 = self.norm_2.to_device(device)?;
        Ok(())
    }
}

fn build_alibi_bias(num_heads: usize, seq_len: usize, device: &candle_core::Device) -> Result<candle_core::Tensor> {
    let closest_power_of_2 = 2usize.pow((num_heads as f64).log2().floor() as u32);
    let base = 2f32.powf(-(2f32.powf(-((closest_power_of_2 as f32).log2() - 3.0))));
    let mut slopes = Vec::with_capacity(num_heads);
    for i in 0..num_heads { slopes.push(base.powf((i + 1) as f32)); }

    let mut bias_data = vec![0.0f32; num_heads * seq_len * seq_len];
    for h in 0..num_heads {
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j <= i {
                    bias_data[h * seq_len * seq_len + i * seq_len + j] = slopes[h] * (j as i32 - i as i32) as f32;
                } else {
                    bias_data[h * seq_len * seq_len + i * seq_len + j] = f32::NEG_INFINITY;
                }
            }
        }
    }
    Ok(candle_core::Tensor::from_vec(bias_data, &[1, num_heads, seq_len, seq_len], device)?)
}

impl MPTAttention {
    fn new(config: &MPTConfig, device: &Device) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        Ok(Self {
            wqkv: ops_fn::zeros(&[config.hidden_size, 3 * config.hidden_size], DataType::Float32, device)?,
            out_proj: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            num_heads: config.num_attention_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            alibi: config.alibi,
        })
    }

    fn forward(&self, hidden_states: &Tensor, seq_len: usize) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch, seq, hidden_size) = (shape[0], shape[1], shape[2]);

        let qkv = ops_fn::matmul(hidden_states, &self.wqkv)?.to_candle()?;
        let q = qkv.narrow(2, 0, hidden_size)?.reshape(&[batch, seq, self.num_heads, self.head_dim])?.transpose(1, 2)?;
        let k = qkv.narrow(2, hidden_size, hidden_size)?.reshape(&[batch, seq, self.num_heads, self.head_dim])?.transpose(1, 2)?;
        let v = qkv.narrow(2, 2 * hidden_size, hidden_size)?.reshape(&[batch, seq, self.num_heads, self.head_dim])?.transpose(1, 2)?;

        let q = q.contiguous()?;
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scores = (q.matmul(&k_t)? * (self.scale as f64))?;

        let device = scores.device();
        let scores = if self.alibi {
            let alibi_bias = build_alibi_bias(self.num_heads, seq_len, device)?;
            scores.broadcast_add(&alibi_bias)?
        } else {
            let mask = {
                let mut m = vec![0.0f32; seq * seq];
                for i in 0..seq { for j in (i+1)..seq { m[i*seq+j] = f32::NEG_INFINITY; } }
                candle_core::Tensor::from_vec(m, &[1, 1, seq, seq], device)?
            };
            scores.broadcast_add(&mask)?
        };

        let v = v.contiguous()?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?;
        let out = attn.transpose(1, 2)?.reshape(&[batch, seq, hidden_size])?;
        ops_fn::matmul(&Tensor::from_candle(out), &self.out_proj)
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.wqkv = self.wqkv.to_device(device)?;
        self.out_proj = self.out_proj.to_device(device)?;
        Ok(())
    }
}

impl MPTFFN {
    fn new(config: &MPTConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            up_proj: ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?,
            down_proj: ops_fn::zeros(&[config.intermediate_size, config.hidden_size], DataType::Float32, device)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = ops_fn::matmul(x, &self.up_proj)?;
        let h = ops_fn::gelu(&h)?;
        ops_fn::matmul(&h, &self.down_proj)
    }
    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.up_proj = self.up_proj.to_device(device)?;
        self.down_proj = self.down_proj.to_device(device)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_mpt_creation() {
        let config = MPTConfig { vocab_size: 1000, hidden_size: 128, intermediate_size: 512, num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 4, ..Default::default() };
        let model = MPTModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 1000);
    }
    #[test]
    fn test_mpt_forward() {
        let config = MPTConfig { vocab_size: 100, hidden_size: 64, intermediate_size: 256, num_hidden_layers: 1, num_attention_heads: 4, num_key_value_heads: 4, ..Default::default() };
        let model = MPTModelV2::new(config).unwrap();
        let inputs = ModelInputs::text(ops_fn::zeros(&[2, 8], DataType::Int64, &Device::CPU).unwrap());
        match model.forward(&inputs).unwrap() { ModelOutputs::Logits { logits, .. } => assert_eq!(logits.shape(), &[2, 8, 100]), _ => panic!() }
    }
}
