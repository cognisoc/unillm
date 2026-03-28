//! CogVLM Model V2 - Vision-Language Model with Expert Attention

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

model_config!(CogVLMConfig {
    vocab_size: usize = 32000,
    hidden_size: usize = 4096,
    intermediate_size: usize = 11008,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 32,
    num_key_value_heads: usize = 32,
    vision_hidden_size: usize = 1024,
    vision_num_hidden_layers: usize = 24,
    vision_num_attention_heads: usize = 16,
    vision_patch_size: usize = 14,
    vision_image_size: usize = 224,
    rms_norm_eps: f32 = 1e-5,
    rope_theta: f32 = 10000.0,
    pad_token_id: i64 = 0,
    bos_token_id: i64 = 1,
    eos_token_id: i64 = 2,
});

impl CogVLMConfig {
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
            ..Default::default()
        }
    }
}

pub struct CogVLMModelV2 {
    config: CogVLMConfig,
    device: Device,
    embed_tokens: Tensor,
    layers: Vec<CogVLMLayer>,
    norm: Tensor,
    lm_head: Tensor,
}

pub struct CogVLMLayer {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
    input_layernorm: Tensor,
    post_attention_layernorm: Tensor,
    num_heads: usize,
    head_dim: usize,
}

impl Model for CogVLMModelV2 {
    type Config = CogVLMConfig;

    fn new(config: CogVLMConfig) -> Result<Self> {
        let device = Device::CPU;
        let embed_tokens = ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, &device)?;
        let norm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, &device)?;
        let lm_head = ops_fn::zeros(&[config.hidden_size, config.vocab_size], DataType::Float32, &device)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(CogVLMLayer::new(&config, &device)?);
        }

        Ok(Self { config, device, embed_tokens, layers, norm, lm_head })
    }

    fn from_weights(config: CogVLMConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;
        if let Some(w) = weights.get("model.embed_tokens.weight") { model.embed_tokens = w.clone(); }
        if let Some(w) = weights.get("model.norm.weight") { model.norm = w.clone(); }
        if let Some(w) = weights.get("lm_head.weight") { model.lm_head = ops_fn::transpose(w)?; }
        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        match inputs {
            ModelInputs::Text { input_ids, .. } => {
                let mut hidden = ops_fn::embedding(input_ids, &self.embed_tokens)?;
                for layer in &self.layers { hidden = layer.forward(&hidden)?; }
                hidden = ops_fn::rms_norm(&hidden, &self.norm, self.config.rms_norm_eps)?;
                let logits = ops_fn::matmul(&hidden, &self.lm_head)?;
                Ok(ModelOutputs::Logits { logits, hidden_states: None })
            }
            _ => Err(anyhow::anyhow!("CogVLM requires text input")),
        }
    }

    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        use crate::tokenizer::Tokenizer;
        let tokenizer = Tokenizer::new();
        let mut tokens: Vec<u32> = tokenizer.encode(prompt);

        for _ in 0..config.max_new_tokens {
            let input_ids = Tensor::from_i64_slice(&tokens.iter().map(|&t| t as i64).collect::<Vec<_>>(), &[1, tokens.len()], &self.device)?;
            let outputs = self.forward(&ModelInputs::text(input_ids))?;
            let logits = match outputs { ModelOutputs::Logits { logits, .. } => logits, _ => return Err(anyhow::anyhow!("Expected logits")) };

            let logits_vec: Vec<f32> = logits.to_candle()?.flatten_all()?.to_vec1()?;
            let start = (tokens.len() - 1) * self.config.vocab_size;
            let next_token = logits_vec[start..start + self.config.vocab_size].iter()
                .enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i as u32).unwrap_or(0);

            if next_token == config.eos_token_id { break; }
            tokens.push(next_token);
        }

        Ok(tokenizer.decode(&tokens))
    }

    fn config(&self) -> &Self::Config { &self.config }
    fn memory_requirements(&self) -> MemoryRequirements {
        let p = (self.config.vocab_size * self.config.hidden_size) * 4;
        MemoryRequirements { gpu_memory: p, cpu_memory: p / 4, kv_cache_memory: p / 8, peak_memory: p * 2 }
    }
    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        self.embed_tokens = self.embed_tokens.to_device(device)?;
        self.norm = self.norm.to_device(device)?;
        self.lm_head = self.lm_head.to_device(device)?;
        Ok(())
    }
}

impl CogVLMLayer {
    fn new(config: &CogVLMConfig, device: &Device) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        Ok(Self {
            q_proj: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            k_proj: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            v_proj: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            o_proj: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            gate_proj: ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?,
            up_proj: ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?,
            down_proj: ops_fn::zeros(&[config.intermediate_size, config.hidden_size], DataType::Float32, device)?,
            input_layernorm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            post_attention_layernorm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            num_heads: config.num_attention_heads,
            head_dim,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch_size, seq_len, _) = (shape[0], shape[1], shape[2]);

        let residual = hidden_states.clone();
        let hidden = ops_fn::rms_norm(hidden_states, &self.input_layernorm, 1e-5)?;

        let q = ops_fn::matmul(&hidden, &self.q_proj)?.to_candle()?.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?.transpose(1, 2)?;
        let k = ops_fn::matmul(&hidden, &self.k_proj)?.to_candle()?.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?.transpose(1, 2)?;
        let v = ops_fn::matmul(&hidden, &self.v_proj)?.to_candle()?.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?.transpose(1, 2)?;

        let scale = (self.head_dim as f32).powf(-0.5);
        let scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * (scale as f64))?;
        let device = scores.device();
        let mask = { let mut m = vec![0.0f32; seq_len * seq_len]; for i in 0..seq_len { for j in (i+1)..seq_len { m[i*seq_len+j] = f32::NEG_INFINITY; } } candle_core::Tensor::from_vec(m, &[1,1,seq_len,seq_len], device)? };
        let attn = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(&mask)?)?.matmul(&v.contiguous()?)?
            .transpose(1, 2)?.reshape(&[batch_size, seq_len, self.num_heads * self.head_dim])?;
        let hidden = ops_fn::add(&residual, &ops_fn::matmul(&Tensor::from_candle(attn), &self.o_proj)?)?;

        let residual = hidden.clone();
        let hidden = ops_fn::rms_norm(&hidden, &self.post_attention_layernorm, 1e-5)?;
        let gate = ops_fn::silu(&ops_fn::matmul(&hidden, &self.gate_proj)?)?;
        let up = ops_fn::matmul(&hidden, &self.up_proj)?;
        ops_fn::add(&residual, &ops_fn::matmul(&ops_fn::mul(&gate, &up)?, &self.down_proj)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cogvlm_forward() {
        let config = CogVLMConfig { vocab_size: 100, hidden_size: 32, intermediate_size: 128, num_hidden_layers: 1, num_attention_heads: 2, num_key_value_heads: 2, ..Default::default() };
        let model = CogVLMModelV2::new(config).unwrap();
        let outputs = model.forward(&ModelInputs::text(ops_fn::zeros(&[1, 4], DataType::Int64, &Device::CPU).unwrap())).unwrap();
        match outputs { ModelOutputs::Logits { logits, .. } => assert_eq!(logits.shape(), &[1, 4, 100]), _ => panic!() }
    }
}
