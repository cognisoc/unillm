//! CLIP Model V2 - Clean implementation using solid abstractions
//!
//! This implements the CLIP (Contrastive Language-Image Pretraining) architecture including:
//! - CLIP-ViT-B/32, CLIP-ViT-B/16, CLIP-ViT-L/14
//!
//! CLIP is a dual encoder model for vision-language tasks:
//! - Text encoder: standard transformer with causal masking
//! - Vision encoder: ViT-style with patch embeddings, CLS token, position embeddings
//! - Projects both to shared embedding space
//! - Outputs similarity scores between text and image embeddings
//! - Uses contrastive learning (logit_scale parameter)

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

// Note: CLIP uses different hidden sizes for text and vision, but model_config! macro
// requires hidden_size and num_hidden_layers. We use transformer_width as hidden_size
// and transformer_layers as num_hidden_layers for the text model.
model_config!(CLIPConfig {
    // Text config - these are used by ModelConfig trait
    vocab_size: usize = 49408,
    hidden_size: usize = 512,           // Same as transformer_width
    num_hidden_layers: usize = 12,      // Same as transformer_layers
    context_length: usize = 77,
    transformer_width: usize = 512,
    transformer_heads: usize = 8,
    transformer_layers: usize = 12,

    // Vision config
    image_resolution: usize = 224,
    vision_layers: usize = 12,
    vision_width: usize = 768,
    vision_heads: usize = 12,
    vision_patch_size: usize = 32,

    // Shared config
    embed_dim: usize = 512,             // Projection dimension
    projection_dim: usize = 512,

    // Additional config
    layer_norm_eps: f32 = 1e-5,
    attention_dropout: f32 = 0.0,
    initializer_range: f32 = 0.02,
});

impl CLIPConfig {
    /// Create CLIPConfig from GGUF model configuration
    /// Note: GGUF files for CLIP are rare, but this provides compatibility
    pub fn from_gguf_config(gguf: &crate::weight_loader_core::GGUFModelConfig) -> Self {
        Self {
            vocab_size: gguf.vocab_size,
            hidden_size: gguf.hidden_size,
            num_hidden_layers: gguf.num_hidden_layers,
            transformer_width: gguf.hidden_size,
            transformer_layers: gguf.num_hidden_layers,
            transformer_heads: gguf.num_attention_heads,
            vision_width: gguf.hidden_size,  // Approximate
            vision_layers: gguf.num_hidden_layers,
            vision_heads: gguf.num_attention_heads,
            ..Default::default()
        }
    }

    /// Get vision head dimension
    pub fn vision_head_dim(&self) -> usize {
        self.vision_width / self.vision_heads
    }

    /// Get text head dimension
    pub fn text_head_dim(&self) -> usize {
        self.transformer_width / self.transformer_heads
    }

    /// Get number of patches (excluding CLS token)
    pub fn num_patches(&self) -> usize {
        (self.image_resolution / self.vision_patch_size).pow(2)
    }

    /// Get number of positions (patches + CLS token)
    pub fn num_positions(&self) -> usize {
        self.num_patches() + 1
    }
}

/// Main CLIP model implementation
pub struct CLIPModelV2 {
    config: CLIPConfig,
    device: Device,
    text_model: CLIPTextTransformer,
    vision_model: CLIPVisionTransformer,
    text_projection: Tensor,
    visual_projection: Tensor,
    logit_scale: Tensor,
}

/// CLIP Text Transformer (causal self-attention)
pub struct CLIPTextTransformer {
    embeddings: CLIPTextEmbeddings,
    encoder: CLIPEncoder,
    final_layer_norm: Tensor,
    final_layer_norm_bias: Option<Tensor>,
    config: CLIPConfig,
    is_causal: bool,
}

/// CLIP Vision Transformer (bidirectional self-attention)
pub struct CLIPVisionTransformer {
    embeddings: CLIPVisionEmbeddings,
    pre_layernorm: Tensor,
    pre_layernorm_bias: Option<Tensor>,
    encoder: CLIPEncoder,
    post_layernorm: Tensor,
    post_layernorm_bias: Option<Tensor>,
    config: CLIPConfig,
}

/// Text embeddings: token + position
pub struct CLIPTextEmbeddings {
    token_embedding: Tensor,
    position_embedding: Tensor,
    config: CLIPConfig,
}

/// Vision embeddings: patch + class + position
pub struct CLIPVisionEmbeddings {
    patch_embedding: Tensor,      // Conv2D as linear projection
    class_embedding: Tensor,      // [CLS] token
    position_embedding: Tensor,
    config: CLIPConfig,
}

/// Shared encoder structure for both text and vision
pub struct CLIPEncoder {
    layers: Vec<CLIPEncoderLayer>,
    is_causal: bool,
}

/// Single encoder layer
pub struct CLIPEncoderLayer {
    self_attn: CLIPAttention,
    layer_norm1: Tensor,
    layer_norm1_bias: Option<Tensor>,
    mlp: CLIPMLP,
    layer_norm2: Tensor,
    layer_norm2_bias: Option<Tensor>,
    hidden_size: usize,
}

/// Multi-head attention
pub struct CLIPAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    out_proj: Tensor,
    q_bias: Option<Tensor>,
    k_bias: Option<Tensor>,
    v_bias: Option<Tensor>,
    out_bias: Option<Tensor>,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

/// MLP with QuickGELU activation
pub struct CLIPMLP {
    fc1: Tensor,
    fc1_bias: Option<Tensor>,
    fc2: Tensor,
    fc2_bias: Option<Tensor>,
    hidden_size: usize,
    intermediate_size: usize,
}

impl Model for CLIPModelV2 {
    type Config = CLIPConfig;

    fn new(config: CLIPConfig) -> Result<Self> {
        let device = Device::CPU;

        let text_model = CLIPTextTransformer::new(&config, &device)?;
        let vision_model = CLIPVisionTransformer::new(&config, &device)?;

        // Projection matrices map to shared embedding space
        // Note: These may need transposing when loading weights
        let text_projection = ops_fn::zeros(
            &[config.transformer_width, config.projection_dim],
            DataType::Float32,
            &device
        )?;
        let visual_projection = ops_fn::zeros(
            &[config.vision_width, config.projection_dim],
            DataType::Float32,
            &device
        )?;

        // Learned temperature parameter (logit_scale = log(1/0.07) initially)
        let logit_scale_data = vec![config.projection_dim as f32 * 0.01]; // Small init
        let logit_scale = Tensor::from_f32_slice(&[2.6592], &[1], &device)?; // ln(1/0.07)

        Ok(Self {
            config,
            device,
            text_model,
            vision_model,
            text_projection,
            visual_projection,
            logit_scale,
        })
    }

    fn from_weights(config: CLIPConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;

        // Load projection weights (transpose for matmul)
        if let Some(w) = weights.get("text_projection.weight") {
            model.text_projection = ops_fn::transpose(w)?;
        } else if let Some(w) = weights.get("text_projection") {
            // Some models store without .weight suffix
            model.text_projection = ops_fn::transpose(w)?;
        }

        if let Some(w) = weights.get("visual_projection.weight") {
            model.visual_projection = ops_fn::transpose(w)?;
        } else if let Some(w) = weights.get("visual_projection") {
            model.visual_projection = ops_fn::transpose(w)?;
        }

        if let Some(w) = weights.get("logit_scale") {
            model.logit_scale = w.clone();
        }

        model.text_model.load_weights(&weights)?;
        model.vision_model.load_weights(&weights)?;

        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        match inputs {
            ModelInputs::Multimodal { input_ids, pixel_values, .. } => {
                // Encode text
                let text_features = self.encode_text_internal(input_ids)?;

                // Encode vision
                let pixel_values = pixel_values.as_ref()
                    .ok_or_else(|| anyhow::anyhow!("CLIP multimodal forward requires pixel_values"))?;
                let image_features = self.encode_image_internal(pixel_values)?;

                // Compute similarity with learned temperature
                let logit_scale = ops_fn::exp(&self.logit_scale)?;
                let logit_scale_val = logit_scale.to_candle()?.to_vec1::<f32>()?[0];

                // logits_per_text = text_features @ image_features.T * logit_scale
                let image_features_t = ops_fn::transpose(&image_features)?;
                let logits_per_text = ops_fn::matmul(&text_features, &image_features_t)?;
                let logits_per_text = ops_fn::scale(&logits_per_text, logit_scale_val)?;

                // logits_per_image = logits_per_text.T
                let logits_per_image = ops_fn::transpose(&logits_per_text)?;

                Ok(ModelOutputs::CLIP {
                    logits_per_text,
                    logits_per_image,
                    text_embeds: text_features,
                    image_embeds: image_features,
                })
            },
            ModelInputs::Text { input_ids, .. } => {
                // Text-only encoding
                let text_features = self.encode_text_internal(input_ids)?;

                Ok(ModelOutputs::Embeddings {
                    embeddings: text_features.clone(),
                    pooled: Some(text_features),
                })
            },
            ModelInputs::Image { pixel_values, .. } => {
                // Image-only encoding
                let image_features = self.encode_image_internal(pixel_values)?;

                Ok(ModelOutputs::Embeddings {
                    embeddings: image_features.clone(),
                    pooled: Some(image_features),
                })
            },
            _ => Err(anyhow::anyhow!("CLIP expects text, image, or multimodal input")),
        }
    }

    fn generate(&self, _prompt: &str, _config: &GenerationConfig) -> Result<String> {
        // CLIP is not a generative model
        Err(anyhow::anyhow!(
            "CLIP is a contrastive model for computing image-text similarity. \
             It does not support text generation. Use encode_text() and encode_image() \
             to compute embeddings, then compare them for similarity."
        ))
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn memory_requirements(&self) -> MemoryRequirements {
        // Text model parameters
        let text_embedding_params = self.config.vocab_size * self.config.transformer_width +
                                   self.config.context_length * self.config.transformer_width;
        let text_layer_params = self.config.transformer_layers * (
            4 * self.config.transformer_width * self.config.transformer_width + // attention
            2 * self.config.transformer_width * (self.config.transformer_width * 4) // MLP
        );

        // Vision model parameters
        let num_patches = self.config.num_patches();
        let vision_embedding_params = 3 * self.config.vision_patch_size * self.config.vision_patch_size * self.config.vision_width +
                                     self.config.vision_width + // class embedding
                                     (num_patches + 1) * self.config.vision_width; // position embedding
        let vision_layer_params = self.config.vision_layers * (
            4 * self.config.vision_width * self.config.vision_width + // attention
            2 * self.config.vision_width * (self.config.vision_width * 4) // MLP
        );

        // Projections
        let projection_params = self.config.transformer_width * self.config.projection_dim +
                               self.config.vision_width * self.config.projection_dim;

        let total_params = text_embedding_params + text_layer_params +
                          vision_embedding_params + vision_layer_params +
                          projection_params;

        let param_bytes = total_params * 4; // float32

        MemoryRequirements {
            gpu_memory: param_bytes,
            cpu_memory: param_bytes / 4,
            kv_cache_memory: 0, // CLIP doesn't use KV cache
            peak_memory: param_bytes * 2, // Forward pass activations
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        self.text_projection = self.text_projection.to_device(device)?;
        self.visual_projection = self.visual_projection.to_device(device)?;
        self.logit_scale = self.logit_scale.to_device(device)?;
        self.text_model.to_device(device)?;
        self.vision_model.to_device(device)?;
        Ok(())
    }
}

impl CLIPModelV2 {
    /// Encode text into normalized embeddings
    pub fn encode_text(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.encode_text_internal(input_ids)
    }

    /// Encode image into normalized embeddings
    pub fn encode_image(&self, pixel_values: &Tensor) -> Result<Tensor> {
        self.encode_image_internal(pixel_values)
    }

    /// Compute similarity between text and image embeddings
    pub fn compute_similarity(&self, text_features: &Tensor, image_features: &Tensor) -> Result<Tensor> {
        let logit_scale = ops_fn::exp(&self.logit_scale)?;
        let logit_scale_val = logit_scale.to_candle()?.to_vec1::<f32>()?[0];

        let image_features_t = ops_fn::transpose(image_features)?;
        let similarity = ops_fn::matmul(text_features, &image_features_t)?;
        ops_fn::scale(&similarity, logit_scale_val)
    }

    /// Internal text encoding with projection and normalization
    fn encode_text_internal(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Get text encoder output
        let hidden_states = self.text_model.forward(input_ids)?;

        // Pool: take features from the EOS token position (highest index in each sequence)
        // For CLIP, this is typically the last non-padding token
        let pooled = self.pool_text_features(&hidden_states, input_ids)?;

        // Project to shared embedding space
        let text_features = ops_fn::matmul(&pooled, &self.text_projection)?;

        // L2 normalize
        self.normalize_features(&text_features)
    }

    /// Internal image encoding with projection and normalization
    fn encode_image_internal(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // Get vision encoder output (already pooled via CLS token)
        let image_features = self.vision_model.forward(pixel_values)?;

        // Project to shared embedding space
        let image_features = ops_fn::matmul(&image_features, &self.visual_projection)?;

        // L2 normalize
        self.normalize_features(&image_features)
    }

    /// Pool text features by taking the EOS token position
    fn pool_text_features(&self, hidden_states: &Tensor, input_ids: &Tensor) -> Result<Tensor> {
        // In CLIP, text features are pooled from the EOS token position
        // For simplicity, we take the last token in each sequence
        let candle_hidden = hidden_states.to_candle()?;
        let shape = candle_hidden.dims();

        if shape.len() == 3 {
            // [batch, seq, hidden] -> take last position -> [batch, hidden]
            let seq_len = shape[1];
            let pooled = candle_hidden.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
            Ok(Tensor::from_candle(pooled))
        } else if shape.len() == 2 {
            // Already [batch/seq, hidden] - just return as is
            Ok(hidden_states.clone())
        } else {
            Err(anyhow::anyhow!("Invalid hidden states shape: {:?}", shape))
        }
    }

    /// L2 normalize features
    fn normalize_features(&self, features: &Tensor) -> Result<Tensor> {
        ops_fn::normalize(features, 2, -1)
    }
}

// ============================================================================
// Text Transformer
// ============================================================================

impl CLIPTextTransformer {
    fn new(config: &CLIPConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            embeddings: CLIPTextEmbeddings::new(config, device)?,
            encoder: CLIPEncoder::new_text(config, device)?,
            final_layer_norm: ops_fn::zeros(&[config.transformer_width], DataType::Float32, device)?,
            final_layer_norm_bias: None,
            config: config.clone(),
            is_causal: true,
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Get embeddings (token + position)
        let hidden_states = self.embeddings.forward(input_ids)?;

        // Apply transformer layers with causal attention
        let hidden_states = self.encoder.forward(&hidden_states, self.is_causal)?;

        // Final layer norm
        ops_fn::layer_norm(&hidden_states, &self.final_layer_norm, self.final_layer_norm_bias.as_ref(), self.config.layer_norm_eps)
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        if let Some(w) = weights.get("text_model.final_layer_norm.weight") {
            self.final_layer_norm = w.clone();
        }
        if let Some(w) = weights.get("text_model.final_layer_norm.bias") {
            self.final_layer_norm_bias = Some(w.clone());
        }

        self.embeddings.load_weights(weights)?;
        self.encoder.load_weights(weights, "text_model")?;
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.final_layer_norm = self.final_layer_norm.to_device(device)?;
        if let Some(ref mut b) = self.final_layer_norm_bias {
            *b = b.to_device(device)?;
        }
        self.embeddings.to_device(device)?;
        self.encoder.to_device(device)?;
        Ok(())
    }
}

// ============================================================================
// Vision Transformer
// ============================================================================

impl CLIPVisionTransformer {
    fn new(config: &CLIPConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            embeddings: CLIPVisionEmbeddings::new(config, device)?,
            pre_layernorm: ops_fn::zeros(&[config.vision_width], DataType::Float32, device)?,
            pre_layernorm_bias: None,
            encoder: CLIPEncoder::new_vision(config, device)?,
            post_layernorm: ops_fn::zeros(&[config.vision_width], DataType::Float32, device)?,
            post_layernorm_bias: None,
            config: config.clone(),
        })
    }

    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // Get patch embeddings + CLS token + position embeddings
        let hidden_states = self.embeddings.forward(pixel_values)?;

        // Pre-layer norm (some CLIP variants)
        let hidden_states = ops_fn::layer_norm(
            &hidden_states,
            &self.pre_layernorm,
            self.pre_layernorm_bias.as_ref(),
            self.config.layer_norm_eps
        )?;

        // Apply transformer layers with bidirectional attention
        let hidden_states = self.encoder.forward(&hidden_states, false)?;

        // Extract CLS token (first position)
        let candle_hidden = hidden_states.to_candle()?;
        let cls_hidden = candle_hidden.narrow(1, 0, 1)?.squeeze(1)?;
        let cls_hidden = Tensor::from_candle(cls_hidden);

        // Post-layer norm on CLS token
        ops_fn::layer_norm(
            &cls_hidden,
            &self.post_layernorm,
            self.post_layernorm_bias.as_ref(),
            self.config.layer_norm_eps
        )
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        if let Some(w) = weights.get("vision_model.pre_layrnorm.weight") {
            self.pre_layernorm = w.clone();
        } else if let Some(w) = weights.get("vision_model.pre_layernorm.weight") {
            self.pre_layernorm = w.clone();
        }
        if let Some(w) = weights.get("vision_model.pre_layrnorm.bias") {
            self.pre_layernorm_bias = Some(w.clone());
        } else if let Some(w) = weights.get("vision_model.pre_layernorm.bias") {
            self.pre_layernorm_bias = Some(w.clone());
        }

        if let Some(w) = weights.get("vision_model.post_layernorm.weight") {
            self.post_layernorm = w.clone();
        }
        if let Some(w) = weights.get("vision_model.post_layernorm.bias") {
            self.post_layernorm_bias = Some(w.clone());
        }

        self.embeddings.load_weights(weights)?;
        self.encoder.load_weights(weights, "vision_model")?;
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.pre_layernorm = self.pre_layernorm.to_device(device)?;
        self.post_layernorm = self.post_layernorm.to_device(device)?;
        if let Some(ref mut b) = self.pre_layernorm_bias {
            *b = b.to_device(device)?;
        }
        if let Some(ref mut b) = self.post_layernorm_bias {
            *b = b.to_device(device)?;
        }
        self.embeddings.to_device(device)?;
        self.encoder.to_device(device)?;
        Ok(())
    }
}

// ============================================================================
// Embeddings
// ============================================================================

impl CLIPTextEmbeddings {
    fn new(config: &CLIPConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            token_embedding: ops_fn::zeros(
                &[config.vocab_size, config.transformer_width],
                DataType::Float32,
                device
            )?,
            position_embedding: ops_fn::zeros(
                &[config.context_length, config.transformer_width],
                DataType::Float32,
                device
            )?,
            config: config.clone(),
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Token embeddings
        let token_embeds = ops_fn::embedding(input_ids, &self.token_embedding)?;

        // Position embeddings (learned, not sinusoidal)
        let input_shape = input_ids.shape();
        let seq_len = if input_shape.len() >= 2 { input_shape[1] } else { input_shape[0] };

        // Slice position embeddings to match sequence length
        let pos_candle = self.position_embedding.to_candle()?;
        let pos_slice = pos_candle.narrow(0, 0, seq_len)?;
        let pos_embeds = Tensor::from_candle(pos_slice);

        // Add token and position embeddings
        ops_fn::add(&token_embeds, &pos_embeds)
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        if let Some(w) = weights.get("text_model.embeddings.token_embedding.weight") {
            self.token_embedding = w.clone();
        }
        if let Some(w) = weights.get("text_model.embeddings.position_embedding.weight") {
            self.position_embedding = w.clone();
        }
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
        let num_positions = config.num_positions();

        Ok(Self {
            // Patch embedding: linear projection of flattened patches
            // Input: [batch, 3, H, W] -> [batch, num_patches, vision_width]
            // Implemented as conv2d with kernel_size=patch_size, stride=patch_size
            patch_embedding: ops_fn::zeros(
                &[config.vision_width, 3, config.vision_patch_size, config.vision_patch_size],
                DataType::Float32,
                device
            )?,
            class_embedding: ops_fn::zeros(
                &[config.vision_width],
                DataType::Float32,
                device
            )?,
            position_embedding: ops_fn::zeros(
                &[num_positions, config.vision_width],
                DataType::Float32,
                device
            )?,
            config: config.clone(),
        })
    }

    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let pixel_candle = pixel_values.to_candle()?;
        let shape = pixel_candle.dims();

        // Expected input: [batch, channels, height, width]
        let batch_size = shape[0];

        // Apply patch embedding via conv2d
        // patch_embedding: [out_channels, in_channels, kH, kW] = [vision_width, 3, patch_size, patch_size]
        let patch_candle = self.patch_embedding.to_candle()?;

        // Conv2d with stride = kernel_size = patch_size
        let patch_embeds = pixel_candle.conv2d(
            &patch_candle,
            self.config.vision_patch_size,  // padding
            self.config.vision_patch_size,  // stride
            1,  // dilation
            1,  // groups
        )?;

        // Reshape: [batch, vision_width, H/patch, W/patch] -> [batch, num_patches, vision_width]
        let patch_shape = patch_embeds.dims();
        let num_patches = patch_shape[2] * patch_shape[3];
        let patch_embeds = patch_embeds
            .reshape(&[batch_size, self.config.vision_width, num_patches])?
            .transpose(1, 2)?; // [batch, num_patches, vision_width]

        // Prepend CLS token
        let class_candle = self.class_embedding.to_candle()?;
        let class_embeds = class_candle
            .unsqueeze(0)? // [1, vision_width]
            .unsqueeze(0)? // [1, 1, vision_width]
            .broadcast_as(&[batch_size, 1, self.config.vision_width])?;

        let embeddings = candle_core::Tensor::cat(&[&class_embeds, &patch_embeds], 1)?;

        // Add position embeddings
        let pos_candle = self.position_embedding.to_candle()?;
        let seq_len = embeddings.dims()[1];
        let pos_slice = pos_candle.narrow(0, 0, seq_len)?;
        let embeddings = embeddings.broadcast_add(&pos_slice)?;

        Ok(Tensor::from_candle(embeddings))
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        if let Some(w) = weights.get("vision_model.embeddings.patch_embedding.weight") {
            self.patch_embedding = w.clone();
        }
        if let Some(w) = weights.get("vision_model.embeddings.class_embedding") {
            self.class_embedding = w.clone();
        }
        if let Some(w) = weights.get("vision_model.embeddings.position_embedding.weight") {
            self.position_embedding = w.clone();
        }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.patch_embedding = self.patch_embedding.to_device(device)?;
        self.class_embedding = self.class_embedding.to_device(device)?;
        self.position_embedding = self.position_embedding.to_device(device)?;
        Ok(())
    }
}

// ============================================================================
// Encoder
// ============================================================================

impl CLIPEncoder {
    fn new_text(config: &CLIPConfig, device: &Device) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.transformer_layers {
            layers.push(CLIPEncoderLayer::new(
                config.transformer_width,
                config.transformer_heads,
                config.layer_norm_eps,
                device,
            )?);
        }
        Ok(Self { layers, is_causal: true })
    }

    fn new_vision(config: &CLIPConfig, device: &Device) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.vision_layers {
            layers.push(CLIPEncoderLayer::new(
                config.vision_width,
                config.vision_heads,
                config.layer_norm_eps,
                device,
            )?);
        }
        Ok(Self { layers, is_causal: false })
    }

    fn forward(&self, hidden_states: &Tensor, is_causal: bool) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, is_causal)?;
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
    fn new(hidden_size: usize, num_heads: usize, layer_norm_eps: f32, device: &Device) -> Result<Self> {
        let intermediate_size = hidden_size * 4;

        Ok(Self {
            self_attn: CLIPAttention::new(hidden_size, num_heads, device)?,
            layer_norm1: ops_fn::zeros(&[hidden_size], DataType::Float32, device)?,
            layer_norm1_bias: None,
            mlp: CLIPMLP::new(hidden_size, intermediate_size, device)?,
            layer_norm2: ops_fn::zeros(&[hidden_size], DataType::Float32, device)?,
            layer_norm2_bias: None,
            hidden_size,
        })
    }

    fn forward(&self, hidden_states: &Tensor, is_causal: bool) -> Result<Tensor> {
        // Pre-norm architecture
        let residual = hidden_states.clone();
        let hidden_states = ops_fn::layer_norm(hidden_states, &self.layer_norm1, self.layer_norm1_bias.as_ref(), 1e-5)?;
        let hidden_states = self.self_attn.forward(&hidden_states, is_causal)?;
        let hidden_states = ops_fn::add(&residual, &hidden_states)?;

        let residual = hidden_states.clone();
        let hidden_states = ops_fn::layer_norm(&hidden_states, &self.layer_norm2, self.layer_norm2_bias.as_ref(), 1e-5)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        ops_fn::add(&residual, &hidden_states)
    }

    fn load_weights(&mut self, weights: &ModelWeights, prefix: &str) -> Result<()> {
        if let Some(w) = weights.get(&format!("{}.layer_norm1.weight", prefix)) {
            self.layer_norm1 = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.layer_norm1.bias", prefix)) {
            self.layer_norm1_bias = Some(w.clone());
        }
        if let Some(w) = weights.get(&format!("{}.layer_norm2.weight", prefix)) {
            self.layer_norm2 = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.layer_norm2.bias", prefix)) {
            self.layer_norm2_bias = Some(w.clone());
        }
        self.self_attn.load_weights(weights, &format!("{}.self_attn", prefix))?;
        self.mlp.load_weights(weights, &format!("{}.mlp", prefix))?;
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.layer_norm1 = self.layer_norm1.to_device(device)?;
        self.layer_norm2 = self.layer_norm2.to_device(device)?;
        if let Some(ref mut b) = self.layer_norm1_bias {
            *b = b.to_device(device)?;
        }
        if let Some(ref mut b) = self.layer_norm2_bias {
            *b = b.to_device(device)?;
        }
        self.self_attn.to_device(device)?;
        self.mlp.to_device(device)?;
        Ok(())
    }
}

// ============================================================================
// Attention
// ============================================================================

impl CLIPAttention {
    fn new(hidden_size: usize, num_heads: usize, device: &Device) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        Ok(Self {
            q_proj: ops_fn::zeros(&[hidden_size, hidden_size], DataType::Float32, device)?,
            k_proj: ops_fn::zeros(&[hidden_size, hidden_size], DataType::Float32, device)?,
            v_proj: ops_fn::zeros(&[hidden_size, hidden_size], DataType::Float32, device)?,
            out_proj: ops_fn::zeros(&[hidden_size, hidden_size], DataType::Float32, device)?,
            q_bias: None,
            k_bias: None,
            v_bias: None,
            out_bias: None,
            num_heads,
            head_dim,
            scale,
        })
    }

    fn forward(&self, hidden_states: &Tensor, is_causal: bool) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch_size, seq_len, _) = if shape.len() == 3 {
            (shape[0], shape[1], shape[2])
        } else if shape.len() == 2 {
            (1, shape[0], shape[1])
        } else {
            return Err(anyhow::anyhow!("Invalid hidden_states shape: {:?}", shape));
        };

        // Project Q, K, V (with transpose for matmul)
        let query = ops_fn::matmul(hidden_states, &self.q_proj)?;
        let key = ops_fn::matmul(hidden_states, &self.k_proj)?;
        let value = ops_fn::matmul(hidden_states, &self.v_proj)?;

        // Add biases if present
        let query = if let Some(ref bias) = self.q_bias {
            ops_fn::add(&query, bias)?
        } else {
            query
        };
        let key = if let Some(ref bias) = self.k_bias {
            ops_fn::add(&key, bias)?
        } else {
            key
        };
        let value = if let Some(ref bias) = self.v_bias {
            ops_fn::add(&value, bias)?
        } else {
            value
        };

        // Reshape for multi-head attention
        // [batch, seq, hidden] -> [batch, heads, seq, head_dim]
        let q_candle = query.to_candle()?;
        let k_candle = key.to_candle()?;
        let v_candle = value.to_candle()?;

        let q_reshaped = q_candle
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k_reshaped = k_candle
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v_reshaped = v_candle
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;

        // Scaled dot-product attention
        let k_t = k_reshaped.transpose(2, 3)?;

        // Make tensors contiguous for matmul
        let q_contiguous = q_reshaped.contiguous()?;
        let k_contiguous = k_t.contiguous()?;

        let scores = q_contiguous.matmul(&k_contiguous)?;
        let scaled_scores = (scores * (self.scale as f64))?;

        // Apply causal mask if needed (for text encoder)
        let masked_scores = if is_causal {
            let device = scaled_scores.device();
            let mut mask_data = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    if j > i {
                        mask_data[i * seq_len + j] = f32::NEG_INFINITY;
                    }
                }
            }
            let causal_mask = candle_core::Tensor::from_vec(mask_data, &[1, 1, seq_len, seq_len], device)?;
            scaled_scores.broadcast_add(&causal_mask)?
        } else {
            scaled_scores
        };

        // Softmax
        let attention_weights = candle_nn::ops::softmax_last_dim(&masked_scores)?;

        // Apply attention to values
        let v_contiguous = v_reshaped.contiguous()?;
        let attn_output = attention_weights.matmul(&v_contiguous)?;

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape(&[batch_size, seq_len, self.num_heads * self.head_dim])?;

        let attn_output = Tensor::from_candle(attn_output);

        // Output projection
        let output = ops_fn::matmul(&attn_output, &self.out_proj)?;
        if let Some(ref bias) = self.out_bias {
            ops_fn::add(&output, bias)
        } else {
            Ok(output)
        }
    }

    fn load_weights(&mut self, weights: &ModelWeights, prefix: &str) -> Result<()> {
        // Load projection weights (transpose for matmul)
        if let Some(w) = weights.get(&format!("{}.q_proj.weight", prefix)) {
            self.q_proj = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.k_proj.weight", prefix)) {
            self.k_proj = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.v_proj.weight", prefix)) {
            self.v_proj = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.out_proj.weight", prefix)) {
            self.out_proj = ops_fn::transpose(w)?;
        }

        // Load biases
        if let Some(w) = weights.get(&format!("{}.q_proj.bias", prefix)) {
            self.q_bias = Some(w.clone());
        }
        if let Some(w) = weights.get(&format!("{}.k_proj.bias", prefix)) {
            self.k_bias = Some(w.clone());
        }
        if let Some(w) = weights.get(&format!("{}.v_proj.bias", prefix)) {
            self.v_bias = Some(w.clone());
        }
        if let Some(w) = weights.get(&format!("{}.out_proj.bias", prefix)) {
            self.out_bias = Some(w.clone());
        }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.q_proj = self.q_proj.to_device(device)?;
        self.k_proj = self.k_proj.to_device(device)?;
        self.v_proj = self.v_proj.to_device(device)?;
        self.out_proj = self.out_proj.to_device(device)?;
        if let Some(ref mut b) = self.q_bias { *b = b.to_device(device)?; }
        if let Some(ref mut b) = self.k_bias { *b = b.to_device(device)?; }
        if let Some(ref mut b) = self.v_bias { *b = b.to_device(device)?; }
        if let Some(ref mut b) = self.out_bias { *b = b.to_device(device)?; }
        Ok(())
    }
}

// ============================================================================
// MLP
// ============================================================================

impl CLIPMLP {
    fn new(hidden_size: usize, intermediate_size: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            fc1: ops_fn::zeros(&[hidden_size, intermediate_size], DataType::Float32, device)?,
            fc1_bias: None,
            fc2: ops_fn::zeros(&[intermediate_size, hidden_size], DataType::Float32, device)?,
            fc2_bias: None,
            hidden_size,
            intermediate_size,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // FC1
        let hidden_states = ops_fn::matmul(hidden_states, &self.fc1)?;
        let hidden_states = if let Some(ref bias) = self.fc1_bias {
            ops_fn::add(&hidden_states, bias)?
        } else {
            hidden_states
        };

        // QuickGELU activation: x * sigmoid(1.702 * x)
        let hidden_states = self.quick_gelu(&hidden_states)?;

        // FC2
        let hidden_states = ops_fn::matmul(&hidden_states, &self.fc2)?;
        if let Some(ref bias) = self.fc2_bias {
            ops_fn::add(&hidden_states, bias)
        } else {
            Ok(hidden_states)
        }
    }

    /// QuickGELU: x * sigmoid(1.702 * x)
    fn quick_gelu(&self, x: &Tensor) -> Result<Tensor> {
        let x_candle = x.to_candle()?;
        // x * sigmoid(1.702 * x)
        let scaled = (&x_candle * 1.702)?;
        let sigmoid = candle_nn::ops::sigmoid(&scaled)?;
        let result = (&x_candle * &sigmoid)?;
        Ok(Tensor::from_candle(result))
    }

    fn load_weights(&mut self, weights: &ModelWeights, prefix: &str) -> Result<()> {
        if let Some(w) = weights.get(&format!("{}.fc1.weight", prefix)) {
            self.fc1 = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.fc1.bias", prefix)) {
            self.fc1_bias = Some(w.clone());
        }
        if let Some(w) = weights.get(&format!("{}.fc2.weight", prefix)) {
            self.fc2 = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.fc2.bias", prefix)) {
            self.fc2_bias = Some(w.clone());
        }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.fc1 = self.fc1.to_device(device)?;
        self.fc2 = self.fc2.to_device(device)?;
        if let Some(ref mut b) = self.fc1_bias { *b = b.to_device(device)?; }
        if let Some(ref mut b) = self.fc2_bias { *b = b.to_device(device)?; }
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
    fn test_clip_config_creation() {
        let config = CLIPConfig::default();
        assert_eq!(config.vocab_size(), 49408);
        assert_eq!(config.hidden_size(), 512);
        assert_eq!(config.num_layers(), 12);
        assert_eq!(config.num_patches(), 49); // (224/32)^2
        assert_eq!(config.num_positions(), 50); // patches + CLS
    }

    #[test]
    fn test_clip_model_creation() {
        let config = CLIPConfig {
            vocab_size: 1000,
            hidden_size: 64,
            num_hidden_layers: 2,
            transformer_width: 64,
            transformer_layers: 2,
            transformer_heads: 4,
            vision_width: 64,
            vision_layers: 2,
            vision_heads: 4,
            vision_patch_size: 16,
            image_resolution: 64,
            projection_dim: 64,
            embed_dim: 64,
            ..Default::default()
        };

        let model = CLIPModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 1000);
    }

    #[test]
    fn test_clip_text_forward() {
        let config = CLIPConfig {
            vocab_size: 100,
            hidden_size: 32,
            num_hidden_layers: 1,
            transformer_width: 32,
            transformer_layers: 1,
            transformer_heads: 4,
            context_length: 16,
            vision_width: 32,
            vision_layers: 1,
            vision_heads: 4,
            vision_patch_size: 8,
            image_resolution: 32,
            projection_dim: 32,
            embed_dim: 32,
            ..Default::default()
        };

        let model = CLIPModelV2::new(config).unwrap();

        // Create text input
        let input_ids = ops_fn::zeros(&[2, 8], DataType::Int64, &Device::CPU).unwrap();
        let inputs = ModelInputs::text(input_ids);

        let outputs = model.forward(&inputs).unwrap();
        match outputs {
            ModelOutputs::Embeddings { embeddings, .. } => {
                assert_eq!(embeddings.shape()[0], 2); // batch
                assert_eq!(embeddings.shape()[1], 32); // projection_dim
            }
            _ => panic!("Expected embeddings output"),
        }
    }

    #[test]
    fn test_clip_generate_returns_error() {
        let config = CLIPConfig::default();
        let model = CLIPModelV2::new(config).unwrap();
        let gen_config = GenerationConfig::default();

        let result = model.generate("test", &gen_config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("contrastive model"));
    }
}
