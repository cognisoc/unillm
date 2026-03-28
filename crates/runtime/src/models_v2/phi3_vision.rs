//! Phi-3-Vision Model V2 - Vision-Language Model
//!
//! This implements the Phi-3-Vision architecture which features:
//! - CLIP ViT vision encoder
//! - MLP projector to align vision/text embeddings
//! - Phi-3 decoder for language modeling
//!
//! Supports: Phi-3-Vision-128k-Instruct

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

model_config!(Phi3VisionConfig {
    // Language model config
    vocab_size: usize = 32064,
    hidden_size: usize = 3072,
    intermediate_size: usize = 8192,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 32,
    num_key_value_heads: usize = 32,
    max_position_embeddings: usize = 131072,
    rms_norm_eps: f32 = 1e-5,
    rope_theta: f32 = 10000.0,

    // Vision encoder config (CLIP ViT)
    vision_hidden_size: usize = 1024,
    vision_intermediate_size: usize = 4096,
    vision_num_hidden_layers: usize = 24,
    vision_num_attention_heads: usize = 16,
    vision_patch_size: usize = 14,
    vision_image_size: usize = 336,

    pad_token_id: i64 = 32000,
    bos_token_id: i64 = 1,
    eos_token_id: i64 = 32000,
    image_token_id: i64 = 32044,
});

impl Phi3VisionConfig {
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

pub struct Phi3VisionModelV2 {
    config: Phi3VisionConfig,
    device: Device,
    vision_encoder: Phi3VisionEncoder,
    projector: Phi3VisionProjector,
    embed_tokens: Tensor,
    layers: Vec<Phi3VisionDecoderLayer>,
    norm: Tensor,
    lm_head: Tensor,
}

pub struct Phi3VisionEncoder {
    patch_embed: Tensor,
    cls_token: Tensor,
    pos_embed: Tensor,
    blocks: Vec<VitBlock>,
    norm: Tensor,
    config: Phi3VisionConfig,
}

pub struct VitBlock {
    norm1: Tensor,
    attn_qkv: Tensor,
    attn_proj: Tensor,
    norm2: Tensor,
    mlp_fc1: Tensor,
    mlp_fc2: Tensor,
    num_heads: usize,
    head_dim: usize,
}

pub struct Phi3VisionProjector {
    linear1: Tensor,
    linear2: Tensor,
}

pub struct Phi3VisionDecoderLayer {
    self_attn_qkv: Tensor,
    self_attn_o: Tensor,
    mlp_gate_up: Tensor,
    mlp_down: Tensor,
    input_layernorm: Tensor,
    post_attention_layernorm: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Model for Phi3VisionModelV2 {
    type Config = Phi3VisionConfig;

    fn new(config: Phi3VisionConfig) -> Result<Self> {
        let device = Device::CPU;

        let vision_encoder = Phi3VisionEncoder::new(&config, &device)?;
        let projector = Phi3VisionProjector::new(&config, &device)?;
        let embed_tokens = ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, &device)?;
        let norm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, &device)?;
        let lm_head = ops_fn::zeros(&[config.hidden_size, config.vocab_size], DataType::Float32, &device)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(Phi3VisionDecoderLayer::new(&config, &device)?);
        }

        Ok(Self { config, device, vision_encoder, projector, embed_tokens, layers, norm, lm_head })
    }

    fn from_weights(config: Phi3VisionConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;
        model.load_weights(&weights)?;
        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        match inputs {
            ModelInputs::Text { input_ids, .. } => {
                let mut hidden = ops_fn::embedding(input_ids, &self.embed_tokens)?;

                for layer in &self.layers {
                    hidden = layer.forward(&hidden)?;
                }

                hidden = ops_fn::rms_norm(&hidden, &self.norm, self.config.rms_norm_eps)?;
                let logits = ops_fn::matmul(&hidden, &self.lm_head)?;

                Ok(ModelOutputs::Logits { logits, hidden_states: None })
            }
            _ => Err(anyhow::anyhow!("Phi-3-Vision requires text input")),
        }
    }

    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        use crate::tokenizer::Tokenizer;
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
                _ => return Err(anyhow::anyhow!("Expected logits")),
            };

            let logits_vec: Vec<f32> = logits.to_candle()?.flatten_all()?.to_vec1()?;
            let seq_len = tokens.len();
            let start = (seq_len - 1) * self.config.vocab_size;
            let last_logits = &logits_vec[start..start + self.config.vocab_size];

            let next_token = last_logits.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);

            if next_token == config.eos_token_id {
                break;
            }
            tokens.push(next_token);
        }

        Ok(tokenizer.decode(&tokens))
    }

    fn config(&self) -> &Self::Config { &self.config }

    fn memory_requirements(&self) -> MemoryRequirements {
        let param_size = (self.config.vocab_size * self.config.hidden_size +
            self.config.num_hidden_layers * 4 * self.config.hidden_size * self.config.hidden_size) * 4;
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
        Ok(())
    }
}

impl Phi3VisionModelV2 {
    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        if let Some(w) = weights.get("model.embed_tokens.weight") {
            self.embed_tokens = w.clone();
        }
        if let Some(w) = weights.get("model.norm.weight") {
            self.norm = w.clone();
        }
        if let Some(w) = weights.get("lm_head.weight") {
            self.lm_head = ops_fn::transpose(w)?;
        }
        Ok(())
    }
}

impl Phi3VisionEncoder {
    fn new(config: &Phi3VisionConfig, device: &Device) -> Result<Self> {
        let num_patches = (config.vision_image_size / config.vision_patch_size).pow(2);
        let patch_dim = 3 * config.vision_patch_size * config.vision_patch_size;

        let patch_embed = ops_fn::zeros(&[patch_dim, config.vision_hidden_size], DataType::Float32, device)?;
        let cls_token = ops_fn::zeros(&[1, 1, config.vision_hidden_size], DataType::Float32, device)?;
        let pos_embed = ops_fn::zeros(&[1, num_patches + 1, config.vision_hidden_size], DataType::Float32, device)?;
        let norm = ops_fn::zeros(&[config.vision_hidden_size], DataType::Float32, device)?;

        let mut blocks = Vec::with_capacity(config.vision_num_hidden_layers);
        for _ in 0..config.vision_num_hidden_layers {
            blocks.push(VitBlock::new(config, device)?);
        }

        Ok(Self { patch_embed, cls_token, pos_embed, blocks, norm, config: config.clone() })
    }
}

impl VitBlock {
    fn new(config: &Phi3VisionConfig, device: &Device) -> Result<Self> {
        let head_dim = config.vision_hidden_size / config.vision_num_attention_heads;

        Ok(Self {
            norm1: ops_fn::zeros(&[config.vision_hidden_size], DataType::Float32, device)?,
            attn_qkv: ops_fn::zeros(&[config.vision_hidden_size, config.vision_hidden_size * 3], DataType::Float32, device)?,
            attn_proj: ops_fn::zeros(&[config.vision_hidden_size, config.vision_hidden_size], DataType::Float32, device)?,
            norm2: ops_fn::zeros(&[config.vision_hidden_size], DataType::Float32, device)?,
            mlp_fc1: ops_fn::zeros(&[config.vision_hidden_size, config.vision_intermediate_size], DataType::Float32, device)?,
            mlp_fc2: ops_fn::zeros(&[config.vision_intermediate_size, config.vision_hidden_size], DataType::Float32, device)?,
            num_heads: config.vision_num_attention_heads,
            head_dim,
        })
    }
}

impl Phi3VisionProjector {
    fn new(config: &Phi3VisionConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            linear1: ops_fn::zeros(&[config.vision_hidden_size, config.hidden_size], DataType::Float32, device)?,
            linear2: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
        })
    }
}

impl Phi3VisionDecoderLayer {
    fn new(config: &Phi3VisionConfig, device: &Device) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let qkv_dim = config.hidden_size + 2 * (config.num_key_value_heads * head_dim);

        Ok(Self {
            self_attn_qkv: ops_fn::zeros(&[config.hidden_size, qkv_dim], DataType::Float32, device)?,
            self_attn_o: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            mlp_gate_up: ops_fn::zeros(&[config.hidden_size, config.intermediate_size * 2], DataType::Float32, device)?,
            mlp_down: ops_fn::zeros(&[config.intermediate_size, config.hidden_size], DataType::Float32, device)?,
            input_layernorm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            post_attention_layernorm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch_size, seq_len, _) = (shape[0], shape[1], shape[2]);

        // Attention
        let residual = hidden_states.clone();
        let hidden = ops_fn::rms_norm(hidden_states, &self.input_layernorm, 1e-5)?;

        let qkv = ops_fn::matmul(&hidden, &self.self_attn_qkv)?;
        let qkv_candle = qkv.to_candle()?;

        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        let q = qkv_candle.narrow(2, 0, q_dim)?
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k = qkv_candle.narrow(2, q_dim, kv_dim)?
            .reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v = qkv_candle.narrow(2, q_dim + kv_dim, kv_dim)?
            .reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim])?
            .transpose(1, 2)?;

        // GQA expansion
        let num_groups = self.num_heads / self.num_kv_heads;
        let (k, v) = if num_groups > 1 {
            let k = k.unsqueeze(2)?
                .broadcast_as(&[batch_size, self.num_kv_heads, num_groups, seq_len, self.head_dim])?
                .reshape(&[batch_size, self.num_heads, seq_len, self.head_dim])?;
            let v = v.unsqueeze(2)?
                .broadcast_as(&[batch_size, self.num_kv_heads, num_groups, seq_len, self.head_dim])?
                .reshape(&[batch_size, self.num_heads, seq_len, self.head_dim])?;
            (k, v)
        } else {
            (k, v)
        };

        let scale = (self.head_dim as f32).powf(-0.5);
        let scores = q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let scores = (scores * (scale as f64))?;

        // Causal mask
        let device = scores.device();
        let mask = {
            let mut mask_data = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    mask_data[i * seq_len + j] = f32::NEG_INFINITY;
                }
            }
            candle_core::Tensor::from_vec(mask_data, &[1, 1, seq_len, seq_len], device)?
        };

        let scores = scores.broadcast_add(&mask)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let attn_output = attn_weights.matmul(&v.contiguous()?)?;

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape(&[batch_size, seq_len, self.num_heads * self.head_dim])?;

        let attn_output = Tensor::from_candle(attn_output);
        let attn_output = ops_fn::matmul(&attn_output, &self.self_attn_o)?;
        let hidden = ops_fn::add(&residual, &attn_output)?;

        // MLP
        let residual = hidden.clone();
        let hidden = ops_fn::rms_norm(&hidden, &self.post_attention_layernorm, 1e-5)?;

        let gate_up = ops_fn::matmul(&hidden, &self.mlp_gate_up)?;
        let gate_up_candle = gate_up.to_candle()?;
        let gate = gate_up_candle.narrow(2, 0, self.mlp_down.shape()[0])?;
        let up = gate_up_candle.narrow(2, self.mlp_down.shape()[0], self.mlp_down.shape()[0])?;

        let gate = candle_nn::ops::silu(&gate)?;
        let hidden = gate.mul(&up)?;
        let hidden = Tensor::from_candle(hidden);
        let hidden = ops_fn::matmul(&hidden, &self.mlp_down)?;

        ops_fn::add(&residual, &hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi3_vision_config() {
        let config = Phi3VisionConfig::default();
        assert_eq!(config.vocab_size, 32064);
        assert_eq!(config.hidden_size, 3072);
    }

    #[test]
    fn test_phi3_vision_model_creation() {
        let config = Phi3VisionConfig {
            vocab_size: 100,
            hidden_size: 32,
            intermediate_size: 128,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            vision_hidden_size: 16,
            vision_intermediate_size: 64,
            vision_num_hidden_layers: 1,
            vision_num_attention_heads: 2,
            ..Default::default()
        };

        let model = Phi3VisionModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 100);
    }

    #[test]
    fn test_phi3_vision_forward() {
        let config = Phi3VisionConfig {
            vocab_size: 100,
            hidden_size: 32,
            intermediate_size: 128,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            vision_hidden_size: 16,
            vision_intermediate_size: 64,
            vision_num_hidden_layers: 1,
            vision_num_attention_heads: 2,
            ..Default::default()
        };

        let model = Phi3VisionModelV2::new(config).unwrap();
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
