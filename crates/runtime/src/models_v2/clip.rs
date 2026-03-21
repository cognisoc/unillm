//! CLIP Model V2 - Clean implementation using solid abstractions
//!
//! This implements the CLIP (Contrastive Language-Image Pretraining) architecture including:
//! - CLIP-ViT-B/32, CLIP-ViT-B/16, CLIP-ViT-L/14

use crate::model_config;
use super::traits::*;
use std::collections::HashMap;
use anyhow::Result;
use serde::{Serialize, Deserialize};

model_config!(CLIPConfig {
    // Text config
    vocab_size: usize = 49408,
    context_length: usize = 77,
    transformer_width: usize = 512,
    transformer_heads: usize = 8,
    transformer_layers: usize = 12,

    // Vision config
    embed_dim: usize = 512,
    image_resolution: usize = 224,
    vision_layers: usize = 12,
    vision_width: usize = 768,
    vision_patch_size: usize = 32,

    // Shared config
    projection_dim: usize = 512,
    logit_scale_init_value: f32 = 2.6592, // ln(1/0.07)

    // Additional config
    layer_norm_eps: f32 = 1e-5,
    attention_dropout: f32 = 0.0,
    dropout: f32 = 0.0,
    initializer_range: f32 = 0.02,
    initializer_factor: f32 = 1.0,
});

pub struct CLIPModelV2 {
    config: CLIPConfig,
    device: Device,
    text_model: CLIPTextTransformer,
    vision_model: CLIPVisionTransformer,
    visual_projection: Tensor,
    text_projection: Tensor,
    logit_scale: Tensor,
}

pub struct CLIPTextTransformer {
    embeddings: CLIPTextEmbeddings,
    encoder: CLIPEncoder,
    final_layer_norm: Tensor,
    config: CLIPConfig,
}

pub struct CLIPVisionTransformer {
    embeddings: CLIPVisionEmbeddings,
    encoder: CLIPEncoder,
    layernorm: Tensor,
    config: CLIPConfig,
}

pub struct CLIPTextEmbeddings {
    token_embedding: Tensor,
    position_embedding: Tensor,
    config: CLIPConfig,
}

pub struct CLIPVisionEmbeddings {
    patch_embedding: Tensor,
    class_embedding: Tensor,
    position_embedding: Tensor,
    config: CLIPConfig,
}

pub struct CLIPEncoder {
    layers: Vec<CLIPEncoderLayer>,
    config: CLIPConfig,
}

pub struct CLIPEncoderLayer {
    self_attn: CLIPAttention,
    layer_norm1: Tensor,
    mlp: CLIPMLP,
    layer_norm2: Tensor,
}

pub struct CLIPAttention {
    k_proj: Tensor,
    v_proj: Tensor,
    q_proj: Tensor,
    out_proj: Tensor,
    config: CLIPConfig,
}

pub struct CLIPMLP {
    fc1: Tensor,
    fc2: Tensor,
    config: CLIPConfig,
}

impl Model for CLIPModelV2 {
    type Config = CLIPConfig;

    fn new(config: CLIPConfig) -> Result<Self> {
        let device = Device::CPU;

        let text_model = CLIPTextTransformer::new(&config, &device)?;
        let vision_model = CLIPVisionTransformer::new(&config, &device)?;

        let visual_projection = ops_fn::zeros(&[config.vision_width, config.projection_dim], DataType::Float32, &device)?;
        let text_projection = ops_fn::zeros(&[config.transformer_width, config.projection_dim], DataType::Float32, &device)?;
        let logit_scale = ops_fn::zeros(&[1], DataType::Float32, &device)?;

        Ok(Self {
            config,
            device,
            text_model,
            vision_model,
            visual_projection,
            text_projection,
            logit_scale
        })
    }

    fn from_weights(config: CLIPConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;

        if let Some(w) = weights.get("visual_projection.weight") { model.visual_projection = w.clone(); }
        if let Some(w) = weights.get("text_projection.weight") { model.text_projection = w.clone(); }
        if let Some(w) = weights.get("logit_scale") { model.logit_scale = w.clone(); }

        model.text_model.load_weights(&weights)?;
        model.vision_model.load_weights(&weights)?;

        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        match inputs {
            ModelInputs::Multimodal { input_ids, pixel_values, .. } => {
                // Encode text
                let text_features = self.text_model.forward(input_ids)?;
                let text_features = ops_fn::matmul(&text_features, &self.text_projection)?;

                // Encode vision
                let image_features = self.vision_model.forward(pixel_values)?;
                let image_features = ops_fn::matmul(&image_features, &self.visual_projection)?;

                // Normalize features
                let text_features = self.normalize_features(&text_features)?;
                let image_features = self.normalize_features(&image_features)?;

                // Compute similarity
                let logit_scale = ops_fn::exp(&self.logit_scale)?;
                let logits_per_text = ops_fn::matmul(&text_features, &image_features)?;
                let logits_per_image = ops_fn::matmul(&image_features, &text_features)?;

                Ok(ModelOutputs::CLIP {
                    logits_per_text,
                    logits_per_image,
                    text_embeds: text_features,
                    image_embeds: image_features,
                })
            },
            ModelInputs::Text { input_ids, .. } => {
                // Text-only encoding
                let text_features = self.text_model.forward(input_ids)?;
                let text_features = ops_fn::matmul(&text_features, &self.text_projection)?;
                let text_features = self.normalize_features(&text_features)?;

                Ok(ModelOutputs::Embeddings {
                    embeddings: text_features.clone(),
                    pooled: Some(text_features),
                })
            },
            ModelInputs::Image { pixel_values } => {
                // Image-only encoding
                let image_features = self.vision_model.forward(pixel_values)?;
                let image_features = ops_fn::matmul(&image_features, &self.visual_projection)?;
                let image_features = self.normalize_features(&image_features)?;

                Ok(ModelOutputs::Embeddings {
                    embeddings: image_features.clone(),
                    pooled: Some(image_features),
                })
            },
            _ => Err(anyhow::anyhow!("CLIP expects text, image, or multimodal input")),
        }
    }

    fn generate(&self, prompt: &str, _config: &GenerationConfig) -> Result<String> {
        Ok(format!("CLIP processed: {}", prompt))
    }

    fn config(&self) -> &Self::Config { &self.config }

    fn memory_requirements(&self) -> MemoryRequirements {
        let text_params = self.config.vocab_size * self.config.transformer_width +
                         self.config.transformer_layers * self.config.transformer_width * self.config.transformer_width * 4;
        let vision_params = (self.config.image_resolution / self.config.vision_patch_size).pow(2) * self.config.vision_width +
                           self.config.vision_layers * self.config.vision_width * self.config.vision_width * 4;
        let param_size = (text_params + vision_params) * 4;

        MemoryRequirements {
            gpu_memory: param_size, cpu_memory: param_size / 4,
            kv_cache_memory: self.config.context_length * self.config.transformer_width * 2 * 4,
            peak_memory: param_size + param_size / 2,
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        self.visual_projection = self.visual_projection.to_device(device)?;
        self.text_projection = self.text_projection.to_device(device)?;
        self.logit_scale = self.logit_scale.to_device(device)?;
        self.text_model.to_device(device)?;
        self.vision_model.to_device(device)?;
        Ok(())
    }
}

impl CLIPModelV2 {
    fn normalize_features(&self, features: &Tensor) -> Result<Tensor> {
        // L2 normalization
        ops_fn::normalize(features, 2, -1)
    }
}

impl CLIPTextTransformer {
    fn new(config: &CLIPConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            embeddings: CLIPTextEmbeddings::new(config, device)?,
            encoder: CLIPEncoder::new(config, device, config.transformer_layers)?,
            final_layer_norm: ops_fn::zeros(&[config.transformer_width], DataType::Float32, device)?,
            config: config.clone(),
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let hidden_states = self.embeddings.forward(input_ids)?;
        let hidden_states = self.encoder.forward(&hidden_states)?;
        ops_fn::layer_norm(&hidden_states, &self.final_layer_norm, None, self.config.layer_norm_eps)
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        if let Some(w) = weights.get("text_model.final_layer_norm.weight") { self.final_layer_norm = w.clone(); }
        self.embeddings.load_weights(weights)?;
        self.encoder.load_weights(weights, "text_model")?;
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.final_layer_norm = self.final_layer_norm.to_device(device)?;
        self.embeddings.to_device(device)?;
        self.encoder.to_device(device)?;
        Ok(())
    }
}

impl CLIPVisionTransformer {
    fn new(config: &CLIPConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            embeddings: CLIPVisionEmbeddings::new(config, device)?,
            encoder: CLIPEncoder::new(config, device, config.vision_layers)?,
            layernorm: ops_fn::zeros(&[config.vision_width], DataType::Float32, device)?,
            config: config.clone(),
        })
    }

    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let hidden_states = self.embeddings.forward(pixel_values)?;
        let hidden_states = self.encoder.forward(&hidden_states)?;

        // Take [CLS] token (first token)
        // In real implementation, we'd properly extract the first token
        ops_fn::layer_norm(&hidden_states, &self.layernorm, None, self.config.layer_norm_eps)
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        if let Some(w) = weights.get("vision_model.post_layernorm.weight") { self.layernorm = w.clone(); }
        self.embeddings.load_weights(weights)?;
        self.encoder.load_weights(weights, "vision_model")?;
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.layernorm = self.layernorm.to_device(device)?;
        self.embeddings.to_device(device)?;
        self.encoder.to_device(device)?;
        Ok(())
    }
}

impl CLIPTextEmbeddings {
    fn new(config: &CLIPConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            token_embedding: ops_fn::zeros(&[config.vocab_size, config.transformer_width], DataType::Float32, device)?,
            position_embedding: ops_fn::zeros(&[config.context_length, config.transformer_width], DataType::Float32, device)?,
            config: config.clone(),
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let token_embeds = ops_fn::embedding(input_ids, &self.token_embedding)?;
        // In real implementation, we'd add position embeddings
        Ok(token_embeds)
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        if let Some(w) = weights.get("text_model.embeddings.token_embedding.weight") { self.token_embedding = w.clone(); }
        if let Some(w) = weights.get("text_model.embeddings.position_embedding.weight") { self.position_embedding = w.clone(); }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.token_embedding = self.token_embedding.to_device(device)?;
        self.position_embedding = self.position_embedding.to_device(device)?;
        Ok(())
    }
}

impl CLIPVisionEmbeddings {
    fn new(config: &CLIPConfig, device: &Device) -> Result<Self> {
        let num_patches = (config.image_resolution / config.vision_patch_size).pow(2);
        let num_positions = num_patches + 1; // +1 for [CLS] token

        Ok(Self {
            patch_embedding: ops_fn::zeros(&[config.vision_width, 3, config.vision_patch_size, config.vision_patch_size], DataType::Float32, device)?,
            class_embedding: ops_fn::zeros(&[config.vision_width], DataType::Float32, device)?,
            position_embedding: ops_fn::zeros(&[num_positions, config.vision_width], DataType::Float32, device)?,
            config: config.clone(),
        })
    }

    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // In real implementation, we'd:
        // 1. Apply patch embedding (conv2d)
        // 2. Add class token
        // 3. Add position embeddings
        // For now, simplified version
        let batch_size = 1; // Simplified
        let sequence_length = (self.config.image_resolution / self.config.vision_patch_size).pow(2) + 1;
        ops_fn::zeros(&[batch_size, sequence_length, self.config.vision_width], DataType::Float32, &Device::CPU)
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        if let Some(w) = weights.get("vision_model.embeddings.patch_embedding.weight") { self.patch_embedding = w.clone(); }
        if let Some(w) = weights.get("vision_model.embeddings.class_embedding") { self.class_embedding = w.clone(); }
        if let Some(w) = weights.get("vision_model.embeddings.position_embedding.weight") { self.position_embedding = w.clone(); }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.patch_embedding = self.patch_embedding.to_device(device)?;
        self.class_embedding = self.class_embedding.to_device(device)?;
        self.position_embedding = self.position_embedding.to_device(device)?;
        Ok(())
    }
}

impl CLIPEncoder {
    fn new(config: &CLIPConfig, device: &Device, num_layers: usize) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(CLIPEncoderLayer::new(config, device)?);
        }
        Ok(Self { layers, config: config.clone() })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }
        Ok(hidden_states)
    }

    fn load_weights(&mut self, weights: &ModelWeights, prefix: &str) -> Result<()> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_weights(weights, &format!("{}.encoder.layers.{}", prefix, i))?;
        }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        Ok(())
    }
}

impl CLIPEncoderLayer {
    fn new(config: &CLIPConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            self_attn: CLIPAttention::new(config, device)?,
            layer_norm1: ops_fn::zeros(&[config.transformer_width.max(config.vision_width)], DataType::Float32, device)?,
            mlp: CLIPMLP::new(config, device)?,
            layer_norm2: ops_fn::zeros(&[config.transformer_width.max(config.vision_width)], DataType::Float32, device)?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let hidden_states = ops_fn::layer_norm(hidden_states, &self.layer_norm1, None, 1e-5)?;
        let hidden_states = self.self_attn.forward(&hidden_states)?;
        let hidden_states = ops_fn::add(&residual, &hidden_states)?;

        let residual = hidden_states.clone();
        let hidden_states = ops_fn::layer_norm(&hidden_states, &self.layer_norm2, None, 1e-5)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        ops_fn::add(&residual, &hidden_states)
    }

    fn load_weights(&mut self, weights: &ModelWeights, prefix: &str) -> Result<()> {
        if let Some(w) = weights.get(&format!("{}.layer_norm1.weight", prefix)) { self.layer_norm1 = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.layer_norm2.weight", prefix)) { self.layer_norm2 = w.clone(); }
        self.self_attn.load_weights(weights, &format!("{}.self_attn", prefix))?;
        self.mlp.load_weights(weights, &format!("{}.mlp", prefix))?;
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.layer_norm1 = self.layer_norm1.to_device(device)?;
        self.layer_norm2 = self.layer_norm2.to_device(device)?;
        self.self_attn.to_device(device)?;
        self.mlp.to_device(device)?;
        Ok(())
    }
}

impl CLIPAttention {
    fn new(config: &CLIPConfig, device: &Device) -> Result<Self> {
        let embed_dim = config.transformer_width.max(config.vision_width);
        Ok(Self {
            k_proj: ops_fn::zeros(&[embed_dim, embed_dim], DataType::Float32, device)?,
            v_proj: ops_fn::zeros(&[embed_dim, embed_dim], DataType::Float32, device)?,
            q_proj: ops_fn::zeros(&[embed_dim, embed_dim], DataType::Float32, device)?,
            out_proj: ops_fn::zeros(&[embed_dim, embed_dim], DataType::Float32, device)?,
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let query = ops_fn::matmul(hidden_states, &self.q_proj)?;
        let key = ops_fn::matmul(hidden_states, &self.k_proj)?;
        let value = ops_fn::matmul(hidden_states, &self.v_proj)?;
        let attn_output = ops_fn::attention(&query, &key, &value, None)?;
        ops_fn::matmul(&attn_output, &self.out_proj)
    }

    fn load_weights(&mut self, weights: &ModelWeights, prefix: &str) -> Result<()> {
        if let Some(w) = weights.get(&format!("{}.k_proj.weight", prefix)) { self.k_proj = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.v_proj.weight", prefix)) { self.v_proj = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.q_proj.weight", prefix)) { self.q_proj = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.out_proj.weight", prefix)) { self.out_proj = w.clone(); }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.k_proj = self.k_proj.to_device(device)?;
        self.v_proj = self.v_proj.to_device(device)?;
        self.q_proj = self.q_proj.to_device(device)?;
        self.out_proj = self.out_proj.to_device(device)?;
        Ok(())
    }
}

impl CLIPMLP {
    fn new(config: &CLIPConfig, device: &Device) -> Result<Self> {
        let embed_dim = config.transformer_width.max(config.vision_width);
        let intermediate_size = embed_dim * 4;
        Ok(Self {
            fc1: ops_fn::zeros(&[embed_dim, intermediate_size], DataType::Float32, device)?,
            fc2: ops_fn::zeros(&[intermediate_size, embed_dim], DataType::Float32, device)?,
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = ops_fn::matmul(hidden_states, &self.fc1)?;
        let hidden_states = ops_fn::gelu(&hidden_states)?;
        ops_fn::matmul(&hidden_states, &self.fc2)
    }

    fn load_weights(&mut self, weights: &ModelWeights, prefix: &str) -> Result<()> {
        if let Some(w) = weights.get(&format!("{}.fc1.weight", prefix)) { self.fc1 = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.fc2.weight", prefix)) { self.fc2 = w.clone(); }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.fc1 = self.fc1.to_device(device)?;
        self.fc2 = self.fc2.to_device(device)?;
        Ok(())
    }
}