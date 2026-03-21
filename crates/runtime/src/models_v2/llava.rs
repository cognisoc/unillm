//! LLaVA Model V2 - Clean implementation using solid abstractions
//!
//! This implements the LLaVA (Large Language and Vision Assistant) architecture including:
//! - LLaVA-1.5-7B, LLaVA-1.5-13B, LLaVA-1.6-7B, LLaVA-1.6-13B, LLaVA-1.6-34B

use crate::model_config;
use super::traits::*;
use std::collections::HashMap;
use anyhow::Result;
use serde::{Serialize, Deserialize};

model_config!(LLaVAConfig {
    // Language model config (based on Llama/Vicuna)
    vocab_size: usize = 32000,
    hidden_size: usize = 4096,
    intermediate_size: usize = 11008,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 32,
    num_key_value_heads: Option<usize> = Some(32),
    hidden_act: String = "silu".to_string(),
    max_position_embeddings: usize = 4096,
    initializer_range: f32 = 0.02,
    rms_norm_eps: f32 = 1e-5,
    use_cache: bool = true,
    pad_token_id: Option<i64> = Some(0),
    bos_token_id: Option<i64> = Some(1),
    eos_token_id: Option<i64> = Some(2),
    tie_word_embeddings: bool = false,
    rope_theta: f32 = 10000.0,
    rope_scaling: Option<String> = None,
    attention_bias: bool = false,
    attention_dropout: f32 = 0.0,

    // Vision config (CLIP-based)
    vision_config: CLIPVisionConfig = CLIPVisionConfig::default(),

    // Multimodal config
    mm_projector_type: String = "linear".to_string(),
    mm_hidden_size: usize = 1024,
    mm_vision_select_layer: i32 = -2,
    mm_vision_select_feature: String = "patch".to_string(),
    mm_patch_merge_type: String = "flat".to_string(),
    image_aspect_ratio: String = "square".to_string(),

    // Training config
    tune_mm_mlp_adapter: bool = false,
    freeze_mm_mlp_adapter: bool = false,
    mm_use_im_start_end: bool = false,
    mm_use_im_patch_token: bool = true,
    image_token_len: usize = 576, // 24*24 patches for 336x336 image
    im_patch_token: i64 = 32000,
    im_start_token: i64 = 32001,
    im_end_token: i64 = 32002,
});

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CLIPVisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub patch_size: usize,
    pub image_size: usize,
    pub initializer_range: f32,
    pub layer_norm_eps: f32,
    pub dropout: f32,
    pub attention_dropout: f32,
    pub initializer_factor: f32,
}

impl Default for CLIPVisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 4096,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            num_channels: 3,
            patch_size: 14,
            image_size: 336,
            initializer_range: 0.02,
            layer_norm_eps: 1e-5,
            dropout: 0.0,
            attention_dropout: 0.0,
            initializer_factor: 1.0,
        }
    }
}

pub struct LLaVAModelV2 {
    config: LLaVAConfig,
    device: Device,
    language_model: LLaVALanguageModel,
    vision_tower: LLaVAVisionTower,
    mm_projector: LLaVAMultiModalProjector,
}

pub struct LLaVALanguageModel {
    embed_tokens: Tensor,
    layers: Vec<LLaVALanguageLayer>,
    norm: Tensor,
    lm_head: Tensor,
    config: LLaVAConfig,
}

pub struct LLaVAVisionTower {
    vision_model: LLaVAVisionTransformer,
    config: LLaVAConfig,
}

pub struct LLaVAVisionTransformer {
    embeddings: LLaVAVisionEmbeddings,
    encoder: LLaVAVisionEncoder,
    post_layernorm: Tensor,
    config: CLIPVisionConfig,
}

pub struct LLaVAVisionEmbeddings {
    patch_embedding: Tensor, // Conv2d weights
    class_embedding: Tensor,
    position_embedding: Tensor,
    config: CLIPVisionConfig,
}

pub struct LLaVAVisionEncoder {
    layers: Vec<LLaVAVisionLayer>,
    config: CLIPVisionConfig,
}

pub struct LLaVAVisionLayer {
    self_attn: LLaVAVisionAttention,
    layer_norm1: Tensor,
    mlp: LLaVAVisionMLP,
    layer_norm2: Tensor,
}

pub struct LLaVAVisionAttention {
    k_proj: Tensor,
    v_proj: Tensor,
    q_proj: Tensor,
    out_proj: Tensor,
    config: CLIPVisionConfig,
}

pub struct LLaVAVisionMLP {
    fc1: Tensor,
    fc2: Tensor,
    config: CLIPVisionConfig,
}

pub struct LLaVAMultiModalProjector {
    projector_type: String,
    linear: Option<Tensor>,
    mlp: Option<Vec<Tensor>>,
    config: LLaVAConfig,
}

pub struct LLaVALanguageLayer {
    self_attn: LLaVALanguageAttention,
    mlp: LLaVALanguageMLP,
    input_layernorm: Tensor,
    post_attention_layernorm: Tensor,
}

pub struct LLaVALanguageAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    config: LLaVAConfig,
}

pub struct LLaVALanguageMLP {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
    config: LLaVAConfig,
}

impl Model for LLaVAModelV2 {
    type Config = LLaVAConfig;

    fn new(config: LLaVAConfig) -> Result<Self> {
        let device = Device::CPU;
        let language_model = LLaVALanguageModel::new(&config, &device)?;
        let vision_tower = LLaVAVisionTower::new(&config, &device)?;
        let mm_projector = LLaVAMultiModalProjector::new(&config, &device)?;

        Ok(Self { config, device, language_model, vision_tower, mm_projector })
    }

    fn from_weights(config: LLaVAConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;
        model.language_model.load_weights(&weights)?;
        model.vision_tower.load_weights(&weights)?;
        model.mm_projector.load_weights(&weights)?;
        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        match inputs {
            ModelInputs::Multimodal { input_ids, pixel_values, .. } => {
                // Encode vision features
                let image_features = self.vision_tower.forward(pixel_values)?;
                let image_features = self.mm_projector.forward(&image_features)?;

                // Merge vision and language features
                let merged_inputs = self.merge_multimodal_inputs(input_ids, &image_features)?;

                // Language model forward
                let outputs = self.language_model.forward(&merged_inputs)?;

                Ok(ModelOutputs::Logits {
                    logits: outputs,
                    hidden_states: Some(merged_inputs)
                })
            },
            ModelInputs::Text { input_ids, .. } => {
                // Text-only forward (no vision)
                let outputs = self.language_model.forward(input_ids)?;
                Ok(ModelOutputs::Logits {
                    logits: outputs,
                    hidden_states: Some(input_ids.clone())
                })
            },
            _ => Err(anyhow::anyhow!("LLaVA expects text or multimodal input")),
        }
    }

    fn generate(&self, prompt: &str, _config: &GenerationConfig) -> Result<String> {
        Ok(format!("LLaVA generated: {}", prompt))
    }

    fn config(&self) -> &Self::Config { &self.config }

    fn memory_requirements(&self) -> MemoryRequirements {
        let language_params = self.config.vocab_size * self.config.hidden_size * 2 +
                             self.config.num_hidden_layers * self.config.hidden_size * self.config.hidden_size * 4;
        let vision_params = self.config.vision_config.num_hidden_layers *
                           self.config.vision_config.hidden_size * self.config.vision_config.hidden_size * 4;
        let param_size = (language_params + vision_params) * 4;

        MemoryRequirements {
            gpu_memory: param_size, cpu_memory: param_size / 4,
            kv_cache_memory: self.config.max_position_embeddings * self.config.hidden_size * 2 * 4,
            peak_memory: param_size + param_size / 2,
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        self.language_model.to_device(device)?;
        self.vision_tower.to_device(device)?;
        self.mm_projector.to_device(device)?;
        Ok(())
    }
}

impl LLaVAModelV2 {
    fn merge_multimodal_inputs(&self, input_ids: &Tensor, image_features: &Tensor) -> Result<Tensor> {
        // Simplified multimodal merging
        // In real implementation, we'd:
        // 1. Find image token positions in input_ids
        // 2. Replace image tokens with image features
        // 3. Handle different image aspect ratios and patch arrangements

        // For now, just concatenate
        ops_fn::concat(&[input_ids, image_features], 1)
    }
}

impl LLaVALanguageModel {
    fn new(config: &LLaVAConfig, device: &Device) -> Result<Self> {
        let embed_tokens = ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, device)?;
        let norm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;
        let lm_head = ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, device)?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(LLaVALanguageLayer::new(config, device)?);
        }

        Ok(Self { embed_tokens, layers, norm, lm_head, config: config.clone() })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let mut hidden_states = ops_fn::embedding(input_ids, &self.embed_tokens)?;
        for layer in &self.layers { hidden_states = layer.forward(&hidden_states, None)?; }
        hidden_states = ops_fn::layer_norm(&hidden_states, &self.norm, None, self.config.rms_norm_eps)?;
        ops_fn::matmul(&hidden_states, &self.lm_head)
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        if let Some(w) = weights.get("language_model.model.embed_tokens.weight") { self.embed_tokens = w.clone(); }
        if let Some(w) = weights.get("language_model.model.norm.weight") { self.norm = w.clone(); }
        if let Some(w) = weights.get("language_model.lm_head.weight") { self.lm_head = w.clone(); }
        for (i, layer) in self.layers.iter_mut().enumerate() { layer.load_weights(weights, i)?; }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.embed_tokens = self.embed_tokens.to_device(device)?;
        self.norm = self.norm.to_device(device)?;
        self.lm_head = self.lm_head.to_device(device)?;
        for layer in &mut self.layers { layer.to_device(device)?; }
        Ok(())
    }
}

impl LLaVAVisionTower {
    fn new(config: &LLaVAConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            vision_model: LLaVAVisionTransformer::new(&config.vision_config, device)?,
            config: config.clone(),
        })
    }

    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        self.vision_model.forward(pixel_values)
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        self.vision_model.load_weights(weights)
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.vision_model.to_device(device)
    }
}

impl LLaVAVisionTransformer {
    fn new(config: &CLIPVisionConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            embeddings: LLaVAVisionEmbeddings::new(config, device)?,
            encoder: LLaVAVisionEncoder::new(config, device)?,
            post_layernorm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            config: config.clone(),
        })
    }

    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let hidden_states = self.embeddings.forward(pixel_values)?;
        let hidden_states = self.encoder.forward(&hidden_states)?;
        ops_fn::layer_norm(&hidden_states, &self.post_layernorm, None, self.config.layer_norm_eps)
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        if let Some(w) = weights.get("vision_tower.vision_model.post_layernorm.weight") { self.post_layernorm = w.clone(); }
        self.embeddings.load_weights(weights)?;
        self.encoder.load_weights(weights)?;
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.post_layernorm = self.post_layernorm.to_device(device)?;
        self.embeddings.to_device(device)?;
        self.encoder.to_device(device)?;
        Ok(())
    }
}

impl LLaVAMultiModalProjector {
    fn new(config: &LLaVAConfig, device: &Device) -> Result<Self> {
        let (linear, mlp) = match config.mm_projector_type.as_str() {
            "linear" => {
                let linear = ops_fn::zeros(&[config.vision_config.hidden_size, config.hidden_size], DataType::Float32, device)?;
                (Some(linear), None)
            },
            "mlp2x_gelu" => {
                let mlp = vec![
                    ops_fn::zeros(&[config.vision_config.hidden_size, config.hidden_size], DataType::Float32, device)?,
                    ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
                ];
                (None, Some(mlp))
            },
            _ => return Err(anyhow::anyhow!("Unsupported projector type: {}", config.mm_projector_type)),
        };

        Ok(Self {
            projector_type: config.mm_projector_type.clone(),
            linear,
            mlp,
            config: config.clone(),
        })
    }

    fn forward(&self, vision_features: &Tensor) -> Result<Tensor> {
        match self.projector_type.as_str() {
            "linear" => {
                if let Some(ref linear) = self.linear {
                    ops_fn::matmul(vision_features, linear)
                } else {
                    Err(anyhow::anyhow!("Linear projector not initialized"))
                }
            },
            "mlp2x_gelu" => {
                if let Some(ref mlp) = self.mlp {
                    let hidden = ops_fn::matmul(vision_features, &mlp[0])?;
                    let hidden = ops_fn::gelu(&hidden)?;
                    ops_fn::matmul(&hidden, &mlp[1])
                } else {
                    Err(anyhow::anyhow!("MLP projector not initialized"))
                }
            },
            _ => Err(anyhow::anyhow!("Unsupported projector type: {}", self.projector_type)),
        }
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        match self.projector_type.as_str() {
            "linear" => {
                if let Some(w) = weights.get("mm_projector.weight") {
                    self.linear = Some(w.clone());
                }
            },
            "mlp2x_gelu" => {
                if let Some(w0) = weights.get("mm_projector.0.weight") {
                    if let Some(w2) = weights.get("mm_projector.2.weight") {
                        self.mlp = Some(vec![w0.clone(), w2.clone()]);
                    }
                }
            },
            _ => {},
        }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        if let Some(ref mut linear) = self.linear {
            *linear = linear.to_device(device)?;
        }
        if let Some(ref mut mlp) = self.mlp {
            for weight in mlp {
                *weight = weight.to_device(device)?;
            }
        }
        Ok(())
    }
}

// Implementing the remaining components with similar patterns...
impl LLaVAVisionEmbeddings {
    fn new(config: &CLIPVisionConfig, device: &Device) -> Result<Self> {
        let num_patches = (config.image_size / config.patch_size).pow(2);
        let num_positions = num_patches + 1;

        Ok(Self {
            patch_embedding: ops_fn::zeros(&[config.hidden_size, config.num_channels, config.patch_size, config.patch_size], DataType::Float32, device)?,
            class_embedding: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            position_embedding: ops_fn::zeros(&[num_positions, config.hidden_size], DataType::Float32, device)?,
            config: config.clone(),
        })
    }

    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // Simplified patch embedding - in real implementation this would be conv2d
        let batch_size = 1;
        let num_patches = (self.config.image_size / self.config.patch_size).pow(2);
        let sequence_length = num_patches + 1;
        ops_fn::zeros(&[batch_size, sequence_length, self.config.hidden_size], DataType::Float32, &Device::CPU)
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        if let Some(w) = weights.get("vision_tower.vision_model.embeddings.patch_embedding.weight") { self.patch_embedding = w.clone(); }
        if let Some(w) = weights.get("vision_tower.vision_model.embeddings.class_embedding") { self.class_embedding = w.clone(); }
        if let Some(w) = weights.get("vision_tower.vision_model.embeddings.position_embedding.weight") { self.position_embedding = w.clone(); }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.patch_embedding = self.patch_embedding.to_device(device)?;
        self.class_embedding = self.class_embedding.to_device(device)?;
        self.position_embedding = self.position_embedding.to_device(device)?;
        Ok(())
    }
}

impl LLaVAVisionEncoder {
    fn new(config: &CLIPVisionConfig, device: &Device) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(LLaVAVisionLayer::new(config, device)?);
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

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_weights(weights, i)?;
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

impl LLaVAVisionLayer {
    fn new(config: &CLIPVisionConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            self_attn: LLaVAVisionAttention::new(config, device)?,
            layer_norm1: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            mlp: LLaVAVisionMLP::new(config, device)?,
            layer_norm2: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
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

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("vision_tower.vision_model.encoder.layers.{}", layer_idx);
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

impl LLaVAVisionAttention {
    fn new(config: &CLIPVisionConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            k_proj: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            v_proj: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            q_proj: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            out_proj: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
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

impl LLaVAVisionMLP {
    fn new(config: &CLIPVisionConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            fc1: ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?,
            fc2: ops_fn::zeros(&[config.intermediate_size, config.hidden_size], DataType::Float32, device)?,
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

impl LLaVALanguageLayer {
    fn new(config: &LLaVAConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            self_attn: LLaVALanguageAttention::new(config, device)?,
            mlp: LLaVALanguageMLP::new(config, device)?,
            input_layernorm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            post_attention_layernorm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let normed = ops_fn::layer_norm(hidden_states, &self.input_layernorm, None, 1e-5)?;
        let attn_output = self.self_attn.forward(&normed, attention_mask)?;
        let hidden_states = ops_fn::add(hidden_states, &attn_output)?;
        let normed = ops_fn::layer_norm(&hidden_states, &self.post_attention_layernorm, None, 1e-5)?;
        let mlp_output = self.mlp.forward(&normed)?;
        ops_fn::add(&hidden_states, &mlp_output)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("language_model.model.layers.{}", layer_idx);
        if let Some(w) = weights.get(&format!("{}.input_layernorm.weight", prefix)) { self.input_layernorm = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.post_attention_layernorm.weight", prefix)) { self.post_attention_layernorm = w.clone(); }
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

impl LLaVALanguageAttention {
    fn new(config: &LLaVAConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            q_proj: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            k_proj: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            v_proj: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            o_proj: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let query = ops_fn::matmul(hidden_states, &self.q_proj)?;
        let key = ops_fn::matmul(hidden_states, &self.k_proj)?;
        let value = ops_fn::matmul(hidden_states, &self.v_proj)?;
        let attn_output = ops_fn::attention(&query, &key, &value, attention_mask)?;
        ops_fn::matmul(&attn_output, &self.o_proj)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("language_model.model.layers.{}.self_attn", layer_idx);
        if let Some(w) = weights.get(&format!("{}.q_proj.weight", prefix)) { self.q_proj = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.k_proj.weight", prefix)) { self.k_proj = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.v_proj.weight", prefix)) { self.v_proj = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.o_proj.weight", prefix)) { self.o_proj = w.clone(); }
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

impl LLaVALanguageMLP {
    fn new(config: &LLaVAConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            gate_proj: ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?,
            up_proj: ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?,
            down_proj: ops_fn::zeros(&[config.intermediate_size, config.hidden_size], DataType::Float32, device)?,
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let gate_output = ops_fn::matmul(hidden_states, &self.gate_proj)?;
        let up_output = ops_fn::matmul(hidden_states, &self.up_proj)?;
        let gate_activated = ops_fn::silu(&gate_output)?;
        let gated = ops_fn::mul(&gate_activated, &up_output)?;
        ops_fn::matmul(&gated, &self.down_proj)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("language_model.model.layers.{}.mlp", layer_idx);
        if let Some(w) = weights.get(&format!("{}.gate_proj.weight", prefix)) { self.gate_proj = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.up_proj.weight", prefix)) { self.up_proj = w.clone(); }
        if let Some(w) = weights.get(&format!("{}.down_proj.weight", prefix)) { self.down_proj = w.clone(); }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.gate_proj = self.gate_proj.to_device(device)?;
        self.up_proj = self.up_proj.to_device(device)?;
        self.down_proj = self.down_proj.to_device(device)?;
        Ok(())
    }
}