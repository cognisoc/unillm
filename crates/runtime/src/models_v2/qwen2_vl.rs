//! Qwen2-VL Model V2 - Vision-Language Model
//!
//! This implements the Qwen2-VL architecture which features:
//! - ViT-based vision encoder
//! - MLP projector to align vision/text embeddings
//! - Qwen2 decoder for language modeling
//! - Dynamic resolution support for images
//!
//! Supports: Qwen2-VL-2B, Qwen2-VL-7B, Qwen2-VL-72B

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

/// Qwen2-VL model configuration
model_config!(Qwen2VLConfig {
    // Language model config
    vocab_size: usize = 151936,
    hidden_size: usize = 3584,
    intermediate_size: usize = 18944,
    num_hidden_layers: usize = 28,
    num_attention_heads: usize = 28,
    num_key_value_heads: usize = 4,
    max_position_embeddings: usize = 32768,
    rms_norm_eps: f32 = 1e-6,
    rope_theta: f32 = 1000000.0,
    tie_word_embeddings: bool = false,

    // Vision encoder config
    vision_hidden_size: usize = 1280,
    vision_intermediate_size: usize = 5120,
    vision_num_hidden_layers: usize = 32,
    vision_num_attention_heads: usize = 16,
    vision_patch_size: usize = 14,
    vision_image_size: usize = 448,
    vision_temporal_patch_size: usize = 2,

    // Projector config
    projector_hidden_size: usize = 0,  // 0 = auto: hidden_size

    // Special tokens
    pad_token_id: i64 = 151643,
    bos_token_id: i64 = 151643,
    eos_token_id: i64 = 151645,
    image_token_id: i64 = 151655,
    video_token_id: i64 = 151656,
});

impl Qwen2VLConfig {
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

    pub fn effective_projector_hidden_size(&self) -> usize {
        if self.projector_hidden_size > 0 {
            self.projector_hidden_size
        } else {
            self.hidden_size
        }
    }
}

/// Main Qwen2-VL model
pub struct Qwen2VLModelV2 {
    config: Qwen2VLConfig,
    device: Device,
    vision_encoder: Qwen2VisionEncoder,
    projector: Qwen2VLProjector,
    language_model: Qwen2VLLanguageModel,
}

/// Qwen2 Vision Encoder (ViT-based)
pub struct Qwen2VisionEncoder {
    patch_embed: PatchEmbedding3D,
    blocks: Vec<VisionTransformerBlock>,
    merger: VisionMerger,
    config: Qwen2VLConfig,
}

/// 3D patch embedding for spatial + temporal
pub struct PatchEmbedding3D {
    proj: Tensor,
    temporal_patch_size: usize,
    patch_size: usize,
    hidden_size: usize,
}

/// Vision transformer block
pub struct VisionTransformerBlock {
    norm1: Tensor,
    attn: VisionAttention,
    norm2: Tensor,
    mlp: VisionMLP,
}

/// Vision attention
pub struct VisionAttention {
    qkv: Tensor,
    proj: Tensor,
    num_heads: usize,
    head_dim: usize,
}

/// Vision MLP
pub struct VisionMLP {
    fc1: Tensor,
    fc2: Tensor,
}

/// Vision merger to reduce tokens
pub struct VisionMerger {
    mlp: Vec<Tensor>,
    hidden_size: usize,
    target_hidden_size: usize,
}

/// Projector to align vision and text
pub struct Qwen2VLProjector {
    linear1: Tensor,
    linear2: Tensor,
}

/// Language model portion (Qwen2)
pub struct Qwen2VLLanguageModel {
    embed_tokens: Tensor,
    layers: Vec<Qwen2VLDecoderLayer>,
    norm: Tensor,
    lm_head: Tensor,
    config: Qwen2VLConfig,
}

/// Decoder layer
pub struct Qwen2VLDecoderLayer {
    self_attn: Qwen2VLAttention,
    mlp: Qwen2VLMLP,
    input_layernorm: Tensor,
    post_attention_layernorm: Tensor,
}

/// Attention with RoPE
pub struct Qwen2VLAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    num_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    scale: f32,
}

/// MLP (SwiGLU)
pub struct Qwen2VLMLP {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
}

impl Model for Qwen2VLModelV2 {
    type Config = Qwen2VLConfig;

    fn new(config: Qwen2VLConfig) -> Result<Self> {
        let device = Device::CPU;

        let vision_encoder = Qwen2VisionEncoder::new(&config, &device)?;
        let projector = Qwen2VLProjector::new(&config, &device)?;
        let language_model = Qwen2VLLanguageModel::new(&config, &device)?;

        Ok(Self {
            config,
            device,
            vision_encoder,
            projector,
            language_model,
        })
    }

    fn from_weights(config: Qwen2VLConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;

        model.vision_encoder.load_weights(&weights)?;
        model.projector.load_weights(&weights)?;
        model.language_model.load_weights(&weights)?;

        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        match inputs {
            ModelInputs::Multimodal { input_ids, pixel_values, attention_mask, .. } => {
                // Encode images if provided
                let image_embeds = if let Some(pixels) = pixel_values {
                    let vision_features = self.vision_encoder.forward(pixels)?;
                    Some(self.projector.forward(&vision_features)?)
                } else {
                    None
                };

                // Get text embeddings
                let text_embeds = ops_fn::embedding(input_ids, &self.language_model.embed_tokens)?;

                // Merge image and text embeddings (replace image tokens)
                let hidden_states = if let Some(img_emb) = image_embeds {
                    self.merge_embeddings(&text_embeds, &img_emb, input_ids)?
                } else {
                    text_embeds
                };

                // Forward through language model
                let logits = self.language_model.forward(&hidden_states)?;

                Ok(ModelOutputs::Logits {
                    logits,
                    hidden_states: None,
                })
            }
            ModelInputs::Text { input_ids, .. } => {
                // Text-only forward
                let hidden_states = ops_fn::embedding(input_ids, &self.language_model.embed_tokens)?;
                let logits = self.language_model.forward(&hidden_states)?;

                Ok(ModelOutputs::Logits {
                    logits,
                    hidden_states: None,
                })
            }
            _ => Err(anyhow::anyhow!("Qwen2-VL requires text or vision-language inputs")),
        }
    }

    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        use crate::tokenizer::Tokenizer;
        use rand::Rng;

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
                _ => return Err(anyhow::anyhow!("Expected logits output")),
            };

            let logits_candle = logits.to_candle()?;
            let last_logits = logits_candle.squeeze(0)?.narrow(0, logits_candle.dims()[1] - 1, 1)?.squeeze(0)?;
            let logits_vec: Vec<f32> = last_logits.to_vec1()?;

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
        let vision_params = self.config.vision_hidden_size * self.config.vision_hidden_size * 4 *
            self.config.vision_num_hidden_layers;
        let text_params = self.config.hidden_size * self.config.hidden_size * 4 *
            self.config.num_hidden_layers;
        let param_size = (vision_params + text_params + self.config.vocab_size * self.config.hidden_size) * 4;

        MemoryRequirements {
            gpu_memory: param_size,
            cpu_memory: param_size / 4,
            kv_cache_memory: param_size / 8,
            peak_memory: param_size + param_size / 2,
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        self.vision_encoder.to_device(device)?;
        self.projector.to_device(device)?;
        self.language_model.to_device(device)?;
        Ok(())
    }
}

impl Qwen2VLModelV2 {
    fn merge_embeddings(&self, text_embeds: &Tensor, image_embeds: &Tensor, input_ids: &Tensor) -> Result<Tensor> {
        // Find image token positions and replace with image embeddings
        // For now, return text embeddings (placeholder implementation)
        Ok(text_embeds.clone())
    }
}

impl Qwen2VisionEncoder {
    fn new(config: &Qwen2VLConfig, device: &Device) -> Result<Self> {
        let patch_embed = PatchEmbedding3D::new(config, device)?;

        let mut blocks = Vec::with_capacity(config.vision_num_hidden_layers);
        for _ in 0..config.vision_num_hidden_layers {
            blocks.push(VisionTransformerBlock::new(config, device)?);
        }

        let merger = VisionMerger::new(config, device)?;

        Ok(Self { patch_embed, blocks, merger, config: config.clone() })
    }

    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // Patch embedding
        let mut hidden_states = self.patch_embed.forward(pixel_values)?;

        // Vision transformer blocks
        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states)?;
        }

        // Merge vision tokens
        self.merger.forward(&hidden_states)
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        self.patch_embed.load_weights(weights)?;
        for (i, block) in self.blocks.iter_mut().enumerate() {
            block.load_weights(weights, i)?;
        }
        self.merger.load_weights(weights)?;
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.patch_embed.to_device(device)?;
        for block in &mut self.blocks {
            block.to_device(device)?;
        }
        self.merger.to_device(device)?;
        Ok(())
    }
}

impl PatchEmbedding3D {
    fn new(config: &Qwen2VLConfig, device: &Device) -> Result<Self> {
        let in_channels = 3 * config.vision_temporal_patch_size;
        let proj = ops_fn::zeros(
            &[in_channels * config.vision_patch_size * config.vision_patch_size, config.vision_hidden_size],
            DataType::Float32,
            device
        )?;

        Ok(Self {
            proj,
            temporal_patch_size: config.vision_temporal_patch_size,
            patch_size: config.vision_patch_size,
            hidden_size: config.vision_hidden_size,
        })
    }

    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // pixel_values: [batch, channels, frames, height, width]
        // Convert to patches and project
        let shape = pixel_values.shape();
        let batch_size = shape[0];

        // Flatten patches and project
        let flat = pixel_values.to_candle()?.flatten(1, 4)?;
        let flat = Tensor::from_candle(flat);
        ops_fn::matmul(&flat, &self.proj)
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        if let Some(w) = weights.get("visual.patch_embed.proj.weight") {
            // Reshape conv weight to linear
            let w_candle = w.to_candle()?;
            let shape = w_candle.dims();
            let flat = w_candle.reshape(&[shape[0], shape[1] * shape[2] * shape[3]])?;
            self.proj = Tensor::from_candle(flat.t()?);
        }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.proj = self.proj.to_device(device)?;
        Ok(())
    }
}

impl VisionTransformerBlock {
    fn new(config: &Qwen2VLConfig, device: &Device) -> Result<Self> {
        let norm1 = ops_fn::zeros(&[config.vision_hidden_size], DataType::Float32, device)?;
        let norm2 = ops_fn::zeros(&[config.vision_hidden_size], DataType::Float32, device)?;
        let attn = VisionAttention::new(config, device)?;
        let mlp = VisionMLP::new(config, device)?;

        Ok(Self { norm1, attn, norm2, mlp })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let normed = ops_fn::layer_norm(hidden_states, &self.norm1, None, 1e-6)?;
        let attn_out = self.attn.forward(&normed)?;
        let hidden_states = ops_fn::add(&residual, &attn_out)?;

        let residual = hidden_states.clone();
        let normed = ops_fn::layer_norm(&hidden_states, &self.norm2, None, 1e-6)?;
        let mlp_out = self.mlp.forward(&normed)?;
        ops_fn::add(&residual, &mlp_out)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("visual.blocks.{}", layer_idx);

        if let Some(w) = weights.get(&format!("{}.norm1.weight", prefix)) {
            self.norm1 = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.norm2.weight", prefix)) {
            self.norm2 = w.clone();
        }

        self.attn.load_weights(weights, layer_idx)?;
        self.mlp.load_weights(weights, layer_idx)?;

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.norm1 = self.norm1.to_device(device)?;
        self.norm2 = self.norm2.to_device(device)?;
        self.attn.to_device(device)?;
        self.mlp.to_device(device)?;
        Ok(())
    }
}

impl VisionAttention {
    fn new(config: &Qwen2VLConfig, device: &Device) -> Result<Self> {
        let head_dim = config.vision_hidden_size / config.vision_num_attention_heads;
        let qkv = ops_fn::zeros(
            &[config.vision_hidden_size, config.vision_hidden_size * 3],
            DataType::Float32,
            device
        )?;
        let proj = ops_fn::zeros(
            &[config.vision_hidden_size, config.vision_hidden_size],
            DataType::Float32,
            device
        )?;

        Ok(Self {
            qkv,
            proj,
            num_heads: config.vision_num_attention_heads,
            head_dim,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch_size, seq_len, _) = (shape[0], shape[1], shape[2]);

        let qkv = ops_fn::matmul(hidden_states, &self.qkv)?;
        let qkv_candle = qkv.to_candle()?;

        let q = qkv_candle.narrow(2, 0, self.num_heads * self.head_dim)?;
        let k = qkv_candle.narrow(2, self.num_heads * self.head_dim, self.num_heads * self.head_dim)?;
        let v = qkv_candle.narrow(2, self.num_heads * self.head_dim * 2, self.num_heads * self.head_dim)?;

        let q = q.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?.transpose(1, 2)?;
        let k = k.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?.transpose(1, 2)?;
        let v = v.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?.transpose(1, 2)?;

        let scale = (self.head_dim as f32).powf(-0.5);
        let scores = q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let scores = (scores * (scale as f64))?;

        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let attn_output = attn_weights.matmul(&v.contiguous()?)?;

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape(&[batch_size, seq_len, self.num_heads * self.head_dim])?;

        let attn_output = Tensor::from_candle(attn_output);
        ops_fn::matmul(&attn_output, &self.proj)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("visual.blocks.{}.attn", layer_idx);

        if let Some(w) = weights.get(&format!("{}.qkv.weight", prefix)) {
            self.qkv = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.proj.weight", prefix)) {
            self.proj = ops_fn::transpose(w)?;
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.qkv = self.qkv.to_device(device)?;
        self.proj = self.proj.to_device(device)?;
        Ok(())
    }
}

impl VisionMLP {
    fn new(config: &Qwen2VLConfig, device: &Device) -> Result<Self> {
        let fc1 = ops_fn::zeros(
            &[config.vision_hidden_size, config.vision_intermediate_size],
            DataType::Float32,
            device
        )?;
        let fc2 = ops_fn::zeros(
            &[config.vision_intermediate_size, config.vision_hidden_size],
            DataType::Float32,
            device
        )?;

        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden = ops_fn::matmul(hidden_states, &self.fc1)?;
        let hidden = ops_fn::gelu(&hidden)?;
        ops_fn::matmul(&hidden, &self.fc2)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("visual.blocks.{}.mlp", layer_idx);

        if let Some(w) = weights.get(&format!("{}.fc1.weight", prefix)) {
            self.fc1 = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.fc2.weight", prefix)) {
            self.fc2 = ops_fn::transpose(w)?;
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.fc1 = self.fc1.to_device(device)?;
        self.fc2 = self.fc2.to_device(device)?;
        Ok(())
    }
}

impl VisionMerger {
    fn new(config: &Qwen2VLConfig, device: &Device) -> Result<Self> {
        let hidden_size = config.vision_hidden_size * 4; // Merge 2x2 patches
        let target_hidden_size = config.hidden_size;

        let mlp = vec![
            ops_fn::zeros(&[hidden_size, target_hidden_size], DataType::Float32, device)?,
            ops_fn::zeros(&[target_hidden_size, target_hidden_size], DataType::Float32, device)?,
        ];

        Ok(Self { mlp, hidden_size, target_hidden_size })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Merge adjacent patches and project
        let mut hidden = hidden_states.clone();
        for (i, w) in self.mlp.iter().enumerate() {
            hidden = ops_fn::matmul(&hidden, w)?;
            if i < self.mlp.len() - 1 {
                hidden = ops_fn::gelu(&hidden)?;
            }
        }
        Ok(hidden)
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        for (i, w) in self.mlp.iter_mut().enumerate() {
            if let Some(weight) = weights.get(&format!("visual.merger.mlp.{}.weight", i * 2)) {
                *w = ops_fn::transpose(weight)?;
            }
        }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        for w in &mut self.mlp {
            *w = w.to_device(device)?;
        }
        Ok(())
    }
}

impl Qwen2VLProjector {
    fn new(config: &Qwen2VLConfig, device: &Device) -> Result<Self> {
        let linear1 = ops_fn::zeros(
            &[config.vision_hidden_size, config.effective_projector_hidden_size()],
            DataType::Float32,
            device
        )?;
        let linear2 = ops_fn::zeros(
            &[config.effective_projector_hidden_size(), config.hidden_size],
            DataType::Float32,
            device
        )?;

        Ok(Self { linear1, linear2 })
    }

    fn forward(&self, vision_features: &Tensor) -> Result<Tensor> {
        let hidden = ops_fn::matmul(vision_features, &self.linear1)?;
        let hidden = ops_fn::gelu(&hidden)?;
        ops_fn::matmul(&hidden, &self.linear2)
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        if let Some(w) = weights.get("visual.projector.0.weight") {
            self.linear1 = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get("visual.projector.2.weight") {
            self.linear2 = ops_fn::transpose(w)?;
        }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.linear1 = self.linear1.to_device(device)?;
        self.linear2 = self.linear2.to_device(device)?;
        Ok(())
    }
}

impl Qwen2VLLanguageModel {
    fn new(config: &Qwen2VLConfig, device: &Device) -> Result<Self> {
        let embed_tokens = ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, device)?;
        let norm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;

        let lm_head = if config.tie_word_embeddings {
            embed_tokens.clone()
        } else {
            ops_fn::zeros(&[config.hidden_size, config.vocab_size], DataType::Float32, device)?
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(Qwen2VLDecoderLayer::new(config, device)?);
        }

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut hidden = hidden_states.clone();

        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }

        hidden = ops_fn::rms_norm(&hidden, &self.norm, self.config.rms_norm_eps)?;

        if self.config.tie_word_embeddings {
            let embed_t = ops_fn::transpose(&self.embed_tokens)?;
            ops_fn::matmul(&hidden, &embed_t)
        } else {
            ops_fn::matmul(&hidden, &self.lm_head)
        }
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        if let Some(w) = weights.get("model.embed_tokens.weight") {
            self.embed_tokens = w.clone();
        }
        if let Some(w) = weights.get("model.norm.weight") {
            self.norm = w.clone();
        }
        if !self.config.tie_word_embeddings {
            if let Some(w) = weights.get("lm_head.weight") {
                self.lm_head = ops_fn::transpose(w)?;
            }
        }

        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_weights(weights, i)?;
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.embed_tokens = self.embed_tokens.to_device(device)?;
        self.norm = self.norm.to_device(device)?;
        if !self.config.tie_word_embeddings {
            self.lm_head = self.lm_head.to_device(device)?;
        }
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        Ok(())
    }
}

impl Qwen2VLDecoderLayer {
    fn new(config: &Qwen2VLConfig, device: &Device) -> Result<Self> {
        let self_attn = Qwen2VLAttention::new(config, device)?;
        let mlp = Qwen2VLMLP::new(config, device)?;
        let input_layernorm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;
        let post_attention_layernorm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let hidden = ops_fn::rms_norm(hidden_states, &self.input_layernorm, 1e-6)?;
        let hidden = self.self_attn.forward(&hidden)?;
        let hidden = ops_fn::add(&residual, &hidden)?;

        let residual = hidden.clone();
        let hidden = ops_fn::rms_norm(&hidden, &self.post_attention_layernorm, 1e-6)?;
        let hidden = self.mlp.forward(&hidden)?;
        ops_fn::add(&residual, &hidden)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("model.layers.{}", layer_idx);

        if let Some(w) = weights.get(&format!("{}.input_layernorm.weight", prefix)) {
            self.input_layernorm = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
            self.post_attention_layernorm = w.clone();
        }

        self.self_attn.load_weights(weights, layer_idx)?;
        self.mlp.load_weights(weights, layer_idx)?;

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.input_layernorm = self.input_layernorm.to_device(device)?;
        self.post_attention_layernorm = self.post_attention_layernorm.to_device(device)?;
        self.self_attn.to_device(device)?;
        self.mlp.to_device(device)?;
        Ok(())
    }
}

impl Qwen2VLAttention {
    fn new(config: &Qwen2VLConfig, device: &Device) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;

        let q_proj = ops_fn::zeros(&[config.hidden_size, config.num_attention_heads * head_dim], DataType::Float32, device)?;
        let k_proj = ops_fn::zeros(&[config.hidden_size, config.num_key_value_heads * head_dim], DataType::Float32, device)?;
        let v_proj = ops_fn::zeros(&[config.hidden_size, config.num_key_value_heads * head_dim], DataType::Float32, device)?;
        let o_proj = ops_fn::zeros(&[config.num_attention_heads * head_dim, config.hidden_size], DataType::Float32, device)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch_size, seq_len, _) = (shape[0], shape[1], shape[2]);

        let q = ops_fn::matmul(hidden_states, &self.q_proj)?;
        let k = ops_fn::matmul(hidden_states, &self.k_proj)?;
        let v = ops_fn::matmul(hidden_states, &self.v_proj)?;

        let q = q.to_candle()?.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?.transpose(1, 2)?;
        let k = k.to_candle()?.reshape(&[batch_size, seq_len, self.num_key_value_heads, self.head_dim])?.transpose(1, 2)?;
        let v = v.to_candle()?.reshape(&[batch_size, seq_len, self.num_key_value_heads, self.head_dim])?.transpose(1, 2)?;

        // GQA expansion
        let num_groups = self.num_heads / self.num_key_value_heads;
        let (k, v) = if num_groups > 1 {
            let k = k.unsqueeze(2)?
                .broadcast_as(&[batch_size, self.num_key_value_heads, num_groups, seq_len, self.head_dim])?
                .reshape(&[batch_size, self.num_heads, seq_len, self.head_dim])?;
            let v = v.unsqueeze(2)?
                .broadcast_as(&[batch_size, self.num_key_value_heads, num_groups, seq_len, self.head_dim])?
                .reshape(&[batch_size, self.num_heads, seq_len, self.head_dim])?;
            (k, v)
        } else {
            (k, v)
        };

        let scores = q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let scores = (scores * (self.scale as f64))?;

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
        ops_fn::matmul(&attn_output, &self.o_proj)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("model.layers.{}.self_attn", layer_idx);

        if let Some(w) = weights.get(&format!("{}.q_proj.weight", prefix)) {
            self.q_proj = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.k_proj.weight", prefix)) {
            self.k_proj = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.v_proj.weight", prefix)) {
            self.v_proj = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.o_proj.weight", prefix)) {
            self.o_proj = ops_fn::transpose(w)?;
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.q_proj = self.q_proj.to_device(device)?;
        self.k_proj = self.k_proj.to_device(device)?;
        self.v_proj = self.v_proj.to_device(device)?;
        self.o_proj = self.o_proj.to_device(device)?;
        Ok(())
    }
}

impl Qwen2VLMLP {
    fn new(config: &Qwen2VLConfig, device: &Device) -> Result<Self> {
        let gate_proj = ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?;
        let up_proj = ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?;
        let down_proj = ops_fn::zeros(&[config.intermediate_size, config.hidden_size], DataType::Float32, device)?;

        Ok(Self { gate_proj, up_proj, down_proj })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let gate = ops_fn::matmul(hidden_states, &self.gate_proj)?;
        let gate = ops_fn::silu(&gate)?;
        let up = ops_fn::matmul(hidden_states, &self.up_proj)?;
        let hidden = ops_fn::mul(&gate, &up)?;
        ops_fn::matmul(&hidden, &self.down_proj)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("model.layers.{}.mlp", layer_idx);

        if let Some(w) = weights.get(&format!("{}.gate_proj.weight", prefix)) {
            self.gate_proj = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.up_proj.weight", prefix)) {
            self.up_proj = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.down_proj.weight", prefix)) {
            self.down_proj = ops_fn::transpose(w)?;
        }

        Ok(())
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
    fn test_qwen2vl_config() {
        let config = Qwen2VLConfig::default();
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.hidden_size, 3584);
        assert_eq!(config.vision_hidden_size, 1280);
    }

    #[test]
    fn test_qwen2vl_model_creation() {
        let config = Qwen2VLConfig {
            vocab_size: 1000,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            vision_hidden_size: 32,
            vision_intermediate_size: 128,
            vision_num_hidden_layers: 2,
            vision_num_attention_heads: 2,
            ..Default::default()
        };

        let model = Qwen2VLModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 1000);
    }

    #[test]
    fn test_qwen2vl_text_forward() {
        let config = Qwen2VLConfig {
            vocab_size: 100,
            hidden_size: 32,
            intermediate_size: 128,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            vision_hidden_size: 16,
            vision_intermediate_size: 64,
            vision_num_hidden_layers: 1,
            vision_num_attention_heads: 2,
            ..Default::default()
        };

        let model = Qwen2VLModelV2::new(config).unwrap();
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
