//! InternVL Model V2 - Vision-Language Model
//!
//! This implements the InternVL architecture: InternViT encoder + InternLM decoder

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

model_config!(InternVLConfig {
    vocab_size: usize = 92544,
    hidden_size: usize = 4096,
    intermediate_size: usize = 14336,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 32,
    num_key_value_heads: usize = 8,
    max_position_embeddings: usize = 8192,
    rms_norm_eps: f32 = 1e-5,
    rope_theta: f32 = 10000.0,
    vision_hidden_size: usize = 3200,
    vision_intermediate_size: usize = 12800,
    vision_num_hidden_layers: usize = 48,
    vision_num_attention_heads: usize = 25,
    vision_patch_size: usize = 14,
    vision_image_size: usize = 448,
    pad_token_id: i64 = 0,
    bos_token_id: i64 = 1,
    eos_token_id: i64 = 2,
});

impl InternVLConfig {
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

pub struct InternVLModelV2 {
    config: InternVLConfig,
    device: Device,
    vision_encoder: InternViTEncoder,
    mlp_projector: Tensor,
    embed_tokens: Tensor,
    layers: Vec<InternLMLayer>,
    norm: Tensor,
    lm_head: Tensor,
}

pub struct InternViTEncoder {
    patch_embed: Tensor,
    blocks: Vec<InternViTBlock>,
    norm: Tensor,
    hidden_size: usize,
}

pub struct InternViTBlock {
    norm1: Tensor,
    attn_qkv: Tensor,
    attn_proj: Tensor,
    norm2: Tensor,
    mlp_fc1: Tensor,
    mlp_fc2: Tensor,
    num_heads: usize,
}

pub struct InternLMLayer {
    self_attn: InternLMAttention,
    mlp: InternLMMLP,
    input_layernorm: Tensor,
    post_attention_layernorm: Tensor,
}

pub struct InternLMAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

pub struct InternLMMLP {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
}

impl Model for InternVLModelV2 {
    type Config = InternVLConfig;

    fn new(config: InternVLConfig) -> Result<Self> {
        let device = Device::CPU;

        let vision_encoder = InternViTEncoder::new(&config, &device)?;
        let mlp_projector = ops_fn::zeros(&[config.vision_hidden_size, config.hidden_size], DataType::Float32, &device)?;
        let embed_tokens = ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, &device)?;
        let norm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, &device)?;
        let lm_head = ops_fn::zeros(&[config.hidden_size, config.vocab_size], DataType::Float32, &device)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(InternLMLayer::new(&config, &device)?);
        }

        Ok(Self { config, device, vision_encoder, mlp_projector, embed_tokens, layers, norm, lm_head })
    }

    fn from_weights(config: InternVLConfig, weights: ModelWeights) -> Result<Self> {
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
            _ => Err(anyhow::anyhow!("InternVL requires text input")),
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
            let outputs = self.forward(&ModelInputs::text(input_ids))?;
            let logits = match outputs {
                ModelOutputs::Logits { logits, .. } => logits,
                _ => return Err(anyhow::anyhow!("Expected logits")),
            };

            let logits_vec: Vec<f32> = logits.to_candle()?.flatten_all()?.to_vec1()?;
            let start = (tokens.len() - 1) * self.config.vocab_size;
            let next_token = logits_vec[start..start + self.config.vocab_size].iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);

            if next_token == config.eos_token_id { break; }
            tokens.push(next_token);
        }

        Ok(tokenizer.decode(&tokens))
    }

    fn config(&self) -> &Self::Config { &self.config }

    fn memory_requirements(&self) -> MemoryRequirements {
        let param_size = (self.config.vocab_size * self.config.hidden_size) * 4;
        MemoryRequirements { gpu_memory: param_size, cpu_memory: param_size / 4, kv_cache_memory: param_size / 8, peak_memory: param_size * 2 }
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        self.embed_tokens = self.embed_tokens.to_device(device)?;
        self.norm = self.norm.to_device(device)?;
        self.lm_head = self.lm_head.to_device(device)?;
        Ok(())
    }
}

impl InternViTEncoder {
    fn new(config: &InternVLConfig, device: &Device) -> Result<Self> {
        let patch_dim = 3 * config.vision_patch_size * config.vision_patch_size;
        let patch_embed = ops_fn::zeros(&[patch_dim, config.vision_hidden_size], DataType::Float32, device)?;
        let norm = ops_fn::zeros(&[config.vision_hidden_size], DataType::Float32, device)?;

        let mut blocks = Vec::with_capacity(config.vision_num_hidden_layers);
        for _ in 0..config.vision_num_hidden_layers {
            blocks.push(InternViTBlock::new(config, device)?);
        }

        Ok(Self { patch_embed, blocks, norm, hidden_size: config.vision_hidden_size })
    }
}

impl InternViTBlock {
    fn new(config: &InternVLConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            norm1: ops_fn::zeros(&[config.vision_hidden_size], DataType::Float32, device)?,
            attn_qkv: ops_fn::zeros(&[config.vision_hidden_size, config.vision_hidden_size * 3], DataType::Float32, device)?,
            attn_proj: ops_fn::zeros(&[config.vision_hidden_size, config.vision_hidden_size], DataType::Float32, device)?,
            norm2: ops_fn::zeros(&[config.vision_hidden_size], DataType::Float32, device)?,
            mlp_fc1: ops_fn::zeros(&[config.vision_hidden_size, config.vision_intermediate_size], DataType::Float32, device)?,
            mlp_fc2: ops_fn::zeros(&[config.vision_intermediate_size, config.vision_hidden_size], DataType::Float32, device)?,
            num_heads: config.vision_num_attention_heads,
        })
    }
}

impl InternLMLayer {
    fn new(config: &InternVLConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            self_attn: InternLMAttention::new(config, device)?,
            mlp: InternLMMLP::new(config, device)?,
            input_layernorm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            post_attention_layernorm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let hidden = ops_fn::rms_norm(hidden_states, &self.input_layernorm, 1e-5)?;
        let hidden = self.self_attn.forward(&hidden)?;
        let hidden = ops_fn::add(&residual, &hidden)?;

        let residual = hidden.clone();
        let hidden = ops_fn::rms_norm(&hidden, &self.post_attention_layernorm, 1e-5)?;
        let hidden = self.mlp.forward(&hidden)?;
        ops_fn::add(&residual, &hidden)
    }
}

impl InternLMAttention {
    fn new(config: &InternVLConfig, device: &Device) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        Ok(Self {
            q_proj: ops_fn::zeros(&[config.hidden_size, config.num_attention_heads * head_dim], DataType::Float32, device)?,
            k_proj: ops_fn::zeros(&[config.hidden_size, config.num_key_value_heads * head_dim], DataType::Float32, device)?,
            v_proj: ops_fn::zeros(&[config.hidden_size, config.num_key_value_heads * head_dim], DataType::Float32, device)?,
            o_proj: ops_fn::zeros(&[config.num_attention_heads * head_dim, config.hidden_size], DataType::Float32, device)?,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch_size, seq_len, _) = (shape[0], shape[1], shape[2]);

        let q = ops_fn::matmul(hidden_states, &self.q_proj)?;
        let k = ops_fn::matmul(hidden_states, &self.k_proj)?;
        let v = ops_fn::matmul(hidden_states, &self.v_proj)?;

        let q = q.to_candle()?.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?.transpose(1, 2)?;
        let k = k.to_candle()?.reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim])?.transpose(1, 2)?;
        let v = v.to_candle()?.reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim])?.transpose(1, 2)?;

        let num_groups = self.num_heads / self.num_kv_heads;
        let (k, v) = if num_groups > 1 {
            (k.unsqueeze(2)?.broadcast_as(&[batch_size, self.num_kv_heads, num_groups, seq_len, self.head_dim])?.reshape(&[batch_size, self.num_heads, seq_len, self.head_dim])?,
             v.unsqueeze(2)?.broadcast_as(&[batch_size, self.num_kv_heads, num_groups, seq_len, self.head_dim])?.reshape(&[batch_size, self.num_heads, seq_len, self.head_dim])?)
        } else { (k, v) };

        let scale = (self.head_dim as f32).powf(-0.5);
        let scores = q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let scores = (scores * (scale as f64))?;

        let device = scores.device();
        let mask = {
            let mut m = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len { for j in (i + 1)..seq_len { m[i * seq_len + j] = f32::NEG_INFINITY; } }
            candle_core::Tensor::from_vec(m, &[1, 1, seq_len, seq_len], device)?
        };

        let scores = scores.broadcast_add(&mask)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let attn_output = attn_weights.matmul(&v.contiguous()?)?
            .transpose(1, 2)?
            .reshape(&[batch_size, seq_len, self.num_heads * self.head_dim])?;

        ops_fn::matmul(&Tensor::from_candle(attn_output), &self.o_proj)
    }
}

impl InternLMMLP {
    fn new(config: &InternVLConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            gate_proj: ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?,
            up_proj: ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?,
            down_proj: ops_fn::zeros(&[config.intermediate_size, config.hidden_size], DataType::Float32, device)?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let gate = ops_fn::silu(&ops_fn::matmul(hidden_states, &self.gate_proj)?)?;
        let up = ops_fn::matmul(hidden_states, &self.up_proj)?;
        ops_fn::matmul(&ops_fn::mul(&gate, &up)?, &self.down_proj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_internvl_config() {
        let config = InternVLConfig::default();
        assert_eq!(config.vocab_size, 92544);
    }

    #[test]
    fn test_internvl_forward() {
        let config = InternVLConfig {
            vocab_size: 100, hidden_size: 32, intermediate_size: 128,
            num_hidden_layers: 1, num_attention_heads: 2, num_key_value_heads: 2,
            vision_hidden_size: 16, vision_intermediate_size: 64,
            vision_num_hidden_layers: 1, vision_num_attention_heads: 2,
            ..Default::default()
        };

        let model = InternVLModelV2::new(config).unwrap();
        let input_ids = ops_fn::zeros(&[1, 4], DataType::Int64, &Device::CPU).unwrap();
        let outputs = model.forward(&ModelInputs::text(input_ids)).unwrap();

        match outputs {
            ModelOutputs::Logits { logits, .. } => assert_eq!(logits.shape(), &[1, 4, 100]),
            _ => panic!("Expected logits"),
        }
    }
}
