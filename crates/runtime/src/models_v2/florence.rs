//! Florence Model V2 - Microsoft Vision-Language Foundation Model

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

model_config!(FlorenceConfig {
    vocab_size: usize = 51289,
    hidden_size: usize = 768,
    intermediate_size: usize = 3072,
    num_hidden_layers: usize = 12,
    num_attention_heads: usize = 12,
    vision_hidden_size: usize = 768,
    vision_num_hidden_layers: usize = 12,
    vision_num_attention_heads: usize = 12,
    vision_patch_size: usize = 16,
    vision_image_size: usize = 384,
    layer_norm_eps: f32 = 1e-6,
    pad_token_id: i64 = 1,
    bos_token_id: i64 = 0,
    eos_token_id: i64 = 2,
});

impl FlorenceConfig {
    pub fn from_gguf_config(gguf: &crate::weight_loader_core::GGUFModelConfig) -> Self {
        Self {
            vocab_size: gguf.vocab_size,
            hidden_size: gguf.hidden_size,
            num_hidden_layers: gguf.num_hidden_layers,
            num_attention_heads: gguf.num_attention_heads,
            ..Default::default()
        }
    }
}

pub struct FlorenceModelV2 {
    config: FlorenceConfig,
    device: Device,
    vision_tower: FlorenceVisionTower,
    language_model: FlorenceLanguageModel,
    projector: Tensor,
}

pub struct FlorenceVisionTower {
    patch_embed: Tensor,
    pos_embed: Tensor,
    blocks: Vec<FlorenceVisionBlock>,
    norm: Tensor,
    hidden_size: usize,
}

pub struct FlorenceVisionBlock {
    norm1: Tensor,
    attn_qkv: Tensor,
    attn_proj: Tensor,
    norm2: Tensor,
    mlp_fc1: Tensor,
    mlp_fc2: Tensor,
    num_heads: usize,
}

pub struct FlorenceLanguageModel {
    embed_tokens: Tensor,
    layers: Vec<FlorenceDecoderLayer>,
    norm: Tensor,
    lm_head: Tensor,
}

pub struct FlorenceDecoderLayer {
    self_attn_q: Tensor,
    self_attn_k: Tensor,
    self_attn_v: Tensor,
    self_attn_o: Tensor,
    mlp_fc1: Tensor,
    mlp_fc2: Tensor,
    norm1: Tensor,
    norm2: Tensor,
    num_heads: usize,
    head_dim: usize,
}

impl Model for FlorenceModelV2 {
    type Config = FlorenceConfig;

    fn new(config: FlorenceConfig) -> Result<Self> {
        let device = Device::CPU;
        let vision_tower = FlorenceVisionTower::new(&config, &device)?;
        let language_model = FlorenceLanguageModel::new(&config, &device)?;
        let projector = ops_fn::zeros(&[config.vision_hidden_size, config.hidden_size], DataType::Float32, &device)?;

        Ok(Self { config, device, vision_tower, language_model, projector })
    }

    fn from_weights(config: FlorenceConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;
        model.language_model.load_weights(&weights)?;
        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        match inputs {
            ModelInputs::Text { input_ids, .. } => {
                let hidden = ops_fn::embedding(input_ids, &self.language_model.embed_tokens)?;
                let logits = self.language_model.forward(&hidden)?;
                Ok(ModelOutputs::Logits { logits, hidden_states: None })
            }
            _ => Err(anyhow::anyhow!("Florence requires text input")),
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
        self.projector = self.projector.to_device(device)?;
        Ok(())
    }
}

impl FlorenceVisionTower {
    fn new(config: &FlorenceConfig, device: &Device) -> Result<Self> {
        let patch_dim = 3 * config.vision_patch_size * config.vision_patch_size;
        let num_patches = (config.vision_image_size / config.vision_patch_size).pow(2);

        let patch_embed = ops_fn::zeros(&[patch_dim, config.vision_hidden_size], DataType::Float32, device)?;
        let pos_embed = ops_fn::zeros(&[1, num_patches + 1, config.vision_hidden_size], DataType::Float32, device)?;
        let norm = ops_fn::zeros(&[config.vision_hidden_size], DataType::Float32, device)?;

        let mut blocks = Vec::with_capacity(config.vision_num_hidden_layers);
        for _ in 0..config.vision_num_hidden_layers {
            blocks.push(FlorenceVisionBlock::new(config, device)?);
        }

        Ok(Self { patch_embed, pos_embed, blocks, norm, hidden_size: config.vision_hidden_size })
    }
}

impl FlorenceVisionBlock {
    fn new(config: &FlorenceConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            norm1: ops_fn::zeros(&[config.vision_hidden_size], DataType::Float32, device)?,
            attn_qkv: ops_fn::zeros(&[config.vision_hidden_size, config.vision_hidden_size * 3], DataType::Float32, device)?,
            attn_proj: ops_fn::zeros(&[config.vision_hidden_size, config.vision_hidden_size], DataType::Float32, device)?,
            norm2: ops_fn::zeros(&[config.vision_hidden_size], DataType::Float32, device)?,
            mlp_fc1: ops_fn::zeros(&[config.vision_hidden_size, config.vision_hidden_size * 4], DataType::Float32, device)?,
            mlp_fc2: ops_fn::zeros(&[config.vision_hidden_size * 4, config.vision_hidden_size], DataType::Float32, device)?,
            num_heads: config.vision_num_attention_heads,
        })
    }
}

impl FlorenceLanguageModel {
    fn new(config: &FlorenceConfig, device: &Device) -> Result<Self> {
        let embed_tokens = ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, device)?;
        let norm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;
        let lm_head = ops_fn::zeros(&[config.hidden_size, config.vocab_size], DataType::Float32, device)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(FlorenceDecoderLayer::new(config, device)?);
        }

        Ok(Self { embed_tokens, layers, norm, lm_head })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut hidden = hidden_states.clone();
        for layer in &self.layers { hidden = layer.forward(&hidden)?; }
        hidden = ops_fn::layer_norm(&hidden, &self.norm, None, 1e-6)?;
        ops_fn::matmul(&hidden, &self.lm_head)
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        if let Some(w) = weights.get("model.embed_tokens.weight") { self.embed_tokens = w.clone(); }
        if let Some(w) = weights.get("model.norm.weight") { self.norm = w.clone(); }
        if let Some(w) = weights.get("lm_head.weight") { self.lm_head = ops_fn::transpose(w)?; }
        Ok(())
    }
}

impl FlorenceDecoderLayer {
    fn new(config: &FlorenceConfig, device: &Device) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        Ok(Self {
            self_attn_q: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            self_attn_k: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            self_attn_v: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            self_attn_o: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            mlp_fc1: ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?,
            mlp_fc2: ops_fn::zeros(&[config.intermediate_size, config.hidden_size], DataType::Float32, device)?,
            norm1: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            norm2: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            num_heads: config.num_attention_heads,
            head_dim,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch_size, seq_len, _) = (shape[0], shape[1], shape[2]);

        let residual = hidden_states.clone();
        let hidden = ops_fn::layer_norm(hidden_states, &self.norm1, None, 1e-6)?;

        let q = ops_fn::matmul(&hidden, &self.self_attn_q)?.to_candle()?.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?.transpose(1, 2)?;
        let k = ops_fn::matmul(&hidden, &self.self_attn_k)?.to_candle()?.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?.transpose(1, 2)?;
        let v = ops_fn::matmul(&hidden, &self.self_attn_v)?.to_candle()?.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?.transpose(1, 2)?;

        let scale = (self.head_dim as f32).powf(-0.5);
        let scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * (scale as f64))?;
        let device = scores.device();
        let mask = { let mut m = vec![0.0f32; seq_len * seq_len]; for i in 0..seq_len { for j in (i+1)..seq_len { m[i*seq_len+j] = f32::NEG_INFINITY; } } candle_core::Tensor::from_vec(m, &[1,1,seq_len,seq_len], device)? };
        let attn = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(&mask)?)?.matmul(&v.contiguous()?)?
            .transpose(1, 2)?.reshape(&[batch_size, seq_len, self.num_heads * self.head_dim])?;
        let hidden = ops_fn::add(&residual, &ops_fn::matmul(&Tensor::from_candle(attn), &self.self_attn_o)?)?;

        let residual = hidden.clone();
        let hidden = ops_fn::layer_norm(&hidden, &self.norm2, None, 1e-6)?;
        let hidden = ops_fn::gelu(&ops_fn::matmul(&hidden, &self.mlp_fc1)?)?;
        ops_fn::add(&residual, &ops_fn::matmul(&hidden, &self.mlp_fc2)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_florence_forward() {
        let config = FlorenceConfig { vocab_size: 100, hidden_size: 32, intermediate_size: 128, num_hidden_layers: 1, num_attention_heads: 2, ..Default::default() };
        let model = FlorenceModelV2::new(config).unwrap();
        let outputs = model.forward(&ModelInputs::text(ops_fn::zeros(&[1, 4], DataType::Int64, &Device::CPU).unwrap())).unwrap();
        match outputs { ModelOutputs::Logits { logits, .. } => assert_eq!(logits.shape(), &[1, 4, 100]), _ => panic!() }
    }
}
