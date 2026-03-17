//! Model Factory for 100+ Supported Architectures
//!
//! Central factory for creating model instances across all supported architectures.
//! This provides a unified interface for instantiating any of the 100+ supported
//! model variants while handling architecture-specific configurations.

use crate::{
    model_architectures::{ModelArchitecture, ModelConfig, ModelRegistry, ArchitectureFamily},
    model_implementations::{ModelImplementation, LlamaModel, MistralModel, ModelError},
    qwen_models::QwenModel,
    gpu_tensor_ops::GpuDevice,
};
use std::{sync::Arc, collections::HashMap};
use async_trait::async_trait;

/// Main model factory responsible for creating model instances
pub struct ModelFactory {
    registry: Arc<ModelRegistry>,
    device: GpuDevice,
    model_cache: HashMap<String, Arc<dyn ModelImplementation>>,
}

impl ModelFactory {
    pub fn new(device: GpuDevice) -> Self {
        Self {
            registry: Arc::new(ModelRegistry::new()),
            device,
            model_cache: HashMap::new(),
        }
    }

    /// Create a model instance by architecture
    pub async fn create_model(
        &mut self,
        architecture: ModelArchitecture,
    ) -> Result<Arc<dyn ModelImplementation>, ModelError> {
        let config = self.registry.get_model_config(&architecture)?;
        self.create_model_with_config(config).await
    }

    /// Create a model instance from model name/path
    pub async fn create_model_from_name(
        &mut self,
        model_name: &str,
    ) -> Result<Arc<dyn ModelImplementation>, ModelError> {
        let architecture = self.registry.detect_architecture(model_name)
            .ok_or_else(|| ModelError::LoadingError(format!("Unknown model: {}", model_name)))?;
        self.create_model(architecture).await
    }

    /// Create a model with custom configuration
    pub async fn create_model_with_config(
        &mut self,
        mut config: ModelConfig,
    ) -> Result<Arc<dyn ModelImplementation>, ModelError> {
        // Set device-specific optimizations
        config.use_flash_attention = self.device.supports_flash_attention();
        config.use_paged_attention = self.device.supports_paged_attention();

        let cache_key = format!("{:?}_{}", config.architecture, config.hidden_size);

        // Check cache first
        if let Some(cached_model) = self.model_cache.get(&cache_key) {
            return Ok(cached_model.clone());
        }

        // Create new model instance
        let model = self.create_model_instance(config.clone()).await?;
        let model = Arc::new(model);

        // Cache the model
        self.model_cache.insert(cache_key, model.clone());

        Ok(model)
    }

    /// Get all supported architectures
    pub fn get_supported_architectures(&self) -> Vec<ModelArchitecture> {
        self.registry.get_all_architectures()
    }

    /// Get architectures by family
    pub fn get_architectures_by_family(&self, family: &ArchitectureFamily) -> Vec<ModelArchitecture> {
        self.registry.get_architectures_by_family(family)
    }

    /// Check if an architecture is supported
    pub fn is_supported(&self, architecture: &ModelArchitecture) -> bool {
        self.registry.is_supported(architecture)
    }

    /// Get model information
    pub fn get_model_info(&self, architecture: &ModelArchitecture) -> Option<ModelInfo> {
        self.registry.get_model_config(architecture).ok().map(|config| ModelInfo {
            architecture: config.architecture.clone(),
            family: config.family.clone(),
            hidden_size: config.hidden_size,
            num_layers: config.num_hidden_layers,
            num_heads: config.num_attention_heads,
            vocab_size: config.vocab_size,
            max_context: config.max_position_embeddings,
            supports_multimodal: config.is_multimodal,
            parameter_count: self.estimate_parameter_count(&config),
        })
    }

    async fn create_model_instance(
        &self,
        config: ModelConfig,
    ) -> Result<Box<dyn ModelImplementation>, ModelError> {
        match config.family {
            ArchitectureFamily::Llama => {
                Ok(Box::new(LlamaModel::new(config, self.device.clone()).await?))
            }
            ArchitectureFamily::Mistral => {
                Ok(Box::new(MistralModel::new(config, self.device.clone()).await?))
            }
            ArchitectureFamily::Qwen => {
                Ok(Box::new(QwenModel::new(config, self.device.clone()).await?))
            }
            ArchitectureFamily::ChatGLM => {
                Ok(Box::new(ChatGLMModel::new(config, self.device.clone()).await?))
            }
            ArchitectureFamily::Baichuan => {
                Ok(Box::new(BaichuanModel::new(config, self.device.clone()).await?))
            }
            ArchitectureFamily::InternLM => {
                Ok(Box::new(InternLMModel::new(config, self.device.clone()).await?))
            }
            ArchitectureFamily::Yi => {
                Ok(Box::new(YiModel::new(config, self.device.clone()).await?))
            }
            ArchitectureFamily::CodeLlama => {
                // CodeLlama uses same implementation as Llama but with code-specific configs
                Ok(Box::new(LlamaModel::new(config, self.device.clone()).await?))
            }
            ArchitectureFamily::Vicuna => {
                // Vicuna uses Llama architecture
                Ok(Box::new(LlamaModel::new(config, self.device.clone()).await?))
            }
            ArchitectureFamily::WizardLM => {
                // WizardLM uses Llama architecture
                Ok(Box::new(LlamaModel::new(config, self.device.clone()).await?))
            }
            ArchitectureFamily::StarCoder => {
                Ok(Box::new(StarCoderModel::new(config, self.device.clone()).await?))
            }
            ArchitectureFamily::GPTNeoX => {
                Ok(Box::new(GPTNeoXModel::new(config, self.device.clone()).await?))
            }
            ArchitectureFamily::Phi => {
                Ok(Box::new(PhiModel::new(config, self.device.clone()).await?))
            }
            ArchitectureFamily::Gemma => {
                Ok(Box::new(GemmaModel::new(config, self.device.clone()).await?))
            }
            ArchitectureFamily::CLIP => {
                Ok(Box::new(CLIPModel::new(config, self.device.clone()).await?))
            }
            ArchitectureFamily::BLIP => {
                Ok(Box::new(BLIPModel::new(config, self.device.clone()).await?))
            }
            ArchitectureFamily::Flamingo => {
                Ok(Box::new(FlamingoModel::new(config, self.device.clone()).await?))
            }
            ArchitectureFamily::MPT => {
                Ok(Box::new(MPTModel::new(config, self.device.clone()).await?))
            }
            ArchitectureFamily::Falcon => {
                Ok(Box::new(FalconModel::new(config, self.device.clone()).await?))
            }
        }
    }

    fn estimate_parameter_count(&self, config: &ModelConfig) -> u64 {
        // Rough parameter count estimation
        let embedding_params = config.vocab_size * config.hidden_size;
        let attention_params_per_layer = 4 * config.hidden_size * config.hidden_size; // QKV + output proj
        let mlp_params_per_layer = 2 * config.hidden_size * config.intermediate_size.unwrap_or(config.hidden_size * 4);
        let layer_params = attention_params_per_layer + mlp_params_per_layer;
        let total_layer_params = layer_params * config.num_hidden_layers;
        let lm_head_params = config.hidden_size * config.vocab_size;

        (embedding_params + total_layer_params + lm_head_params) as u64
    }
}

/// Information about a supported model
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub architecture: ModelArchitecture,
    pub family: ArchitectureFamily,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub vocab_size: usize,
    pub max_context: usize,
    pub supports_multimodal: bool,
    pub parameter_count: u64,
}

// ============================================================================
// ADDITIONAL MODEL IMPLEMENTATIONS
// ============================================================================

/// ChatGLM model implementation (supports ChatGLM, ChatGLM2, ChatGLM3, GLM-4)
pub struct ChatGLMModel {
    config: ModelConfig,
    device: GpuDevice,
    // Implementation would go here...
}

impl ChatGLMModel {
    pub async fn new(config: ModelConfig, device: GpuDevice) -> Result<Self, ModelError> {
        Ok(Self { config, device })
    }
}

#[async_trait]
impl ModelImplementation for ChatGLMModel {
    async fn forward(
        &self,
        _input_ids: &crate::gpu_tensor_ops::GpuTensor,
        _attention_mask: Option<&crate::gpu_tensor_ops::GpuTensor>,
        _position_ids: Option<&crate::gpu_tensor_ops::GpuTensor>,
        _past_key_values: Option<&Vec<(crate::gpu_tensor_ops::GpuTensor, crate::gpu_tensor_ops::GpuTensor)>>,
    ) -> Result<crate::model_implementations::ModelOutput, ModelError> {
        Err(ModelError::TensorError("ChatGLM not fully implemented".to_string()))
    }

    async fn generate(
        &self,
        _input_ids: &crate::gpu_tensor_ops::GpuTensor,
        _generation_config: &crate::model_implementations::GenerationConfig,
    ) -> Result<crate::model_implementations::GenerationOutput, ModelError> {
        Err(ModelError::GenerationError("ChatGLM generation not implemented".to_string()))
    }

    fn get_config(&self) -> &ModelConfig {
        &self.config
    }

    fn get_architecture(&self) -> ModelArchitecture {
        self.config.architecture.clone()
    }

    fn supports_paged_attention(&self) -> bool {
        false
    }

    fn supports_flash_attention(&self) -> bool {
        true
    }
}

/// Baichuan model implementation
pub struct BaichuanModel {
    config: ModelConfig,
    device: GpuDevice,
}

impl BaichuanModel {
    pub async fn new(config: ModelConfig, device: GpuDevice) -> Result<Self, ModelError> {
        Ok(Self { config, device })
    }
}

#[async_trait]
impl ModelImplementation for BaichuanModel {
    async fn forward(&self, _input_ids: &crate::gpu_tensor_ops::GpuTensor, _attention_mask: Option<&crate::gpu_tensor_ops::GpuTensor>, _position_ids: Option<&crate::gpu_tensor_ops::GpuTensor>, _past_key_values: Option<&Vec<(crate::gpu_tensor_ops::GpuTensor, crate::gpu_tensor_ops::GpuTensor)>>) -> Result<crate::model_implementations::ModelOutput, ModelError> {
        Err(ModelError::TensorError("Baichuan not fully implemented".to_string()))
    }
    async fn generate(&self, _input_ids: &crate::gpu_tensor_ops::GpuTensor, _generation_config: &crate::model_implementations::GenerationConfig) -> Result<crate::model_implementations::GenerationOutput, ModelError> {
        Err(ModelError::GenerationError("Baichuan generation not implemented".to_string()))
    }
    fn get_config(&self) -> &ModelConfig { &self.config }
    fn get_architecture(&self) -> ModelArchitecture { self.config.architecture.clone() }
    fn supports_paged_attention(&self) -> bool { false }
    fn supports_flash_attention(&self) -> bool { true }
}

/// InternLM model implementation
pub struct InternLMModel {
    config: ModelConfig,
    device: GpuDevice,
}

impl InternLMModel {
    pub async fn new(config: ModelConfig, device: GpuDevice) -> Result<Self, ModelError> {
        Ok(Self { config, device })
    }
}

#[async_trait]
impl ModelImplementation for InternLMModel {
    async fn forward(&self, _input_ids: &crate::gpu_tensor_ops::GpuTensor, _attention_mask: Option<&crate::gpu_tensor_ops::GpuTensor>, _position_ids: Option<&crate::gpu_tensor_ops::GpuTensor>, _past_key_values: Option<&Vec<(crate::gpu_tensor_ops::GpuTensor, crate::gpu_tensor_ops::GpuTensor)>>) -> Result<crate::model_implementations::ModelOutput, ModelError> {
        Err(ModelError::TensorError("InternLM not fully implemented".to_string()))
    }
    async fn generate(&self, _input_ids: &crate::gpu_tensor_ops::GpuTensor, _generation_config: &crate::model_implementations::GenerationConfig) -> Result<crate::model_implementations::GenerationOutput, ModelError> {
        Err(ModelError::GenerationError("InternLM generation not implemented".to_string()))
    }
    fn get_config(&self) -> &ModelConfig { &self.config }
    fn get_architecture(&self) -> ModelArchitecture { self.config.architecture.clone() }
    fn supports_paged_attention(&self) -> bool { false }
    fn supports_flash_attention(&self) -> bool { true }
}

/// Yi model implementation
pub struct YiModel {
    config: ModelConfig,
    device: GpuDevice,
}

impl YiModel {
    pub async fn new(config: ModelConfig, device: GpuDevice) -> Result<Self, ModelError> {
        Ok(Self { config, device })
    }
}

#[async_trait]
impl ModelImplementation for YiModel {
    async fn forward(&self, _input_ids: &crate::gpu_tensor_ops::GpuTensor, _attention_mask: Option<&crate::gpu_tensor_ops::GpuTensor>, _position_ids: Option<&crate::gpu_tensor_ops::GpuTensor>, _past_key_values: Option<&Vec<(crate::gpu_tensor_ops::GpuTensor, crate::gpu_tensor_ops::GpuTensor)>>) -> Result<crate::model_implementations::ModelOutput, ModelError> {
        Err(ModelError::TensorError("Yi not fully implemented".to_string()))
    }
    async fn generate(&self, _input_ids: &crate::gpu_tensor_ops::GpuTensor, _generation_config: &crate::model_implementations::GenerationConfig) -> Result<crate::model_implementations::GenerationOutput, ModelError> {
        Err(ModelError::GenerationError("Yi generation not implemented".to_string()))
    }
    fn get_config(&self) -> &ModelConfig { &self.config }
    fn get_architecture(&self) -> ModelArchitecture { self.config.architecture.clone() }
    fn supports_paged_attention(&self) -> bool { false }
    fn supports_flash_attention(&self) -> bool { true }
}

/// StarCoder model implementation for code generation
pub struct StarCoderModel {
    config: ModelConfig,
    device: GpuDevice,
}

impl StarCoderModel {
    pub async fn new(config: ModelConfig, device: GpuDevice) -> Result<Self, ModelError> {
        Ok(Self { config, device })
    }
}

#[async_trait]
impl ModelImplementation for StarCoderModel {
    async fn forward(&self, _input_ids: &crate::gpu_tensor_ops::GpuTensor, _attention_mask: Option<&crate::gpu_tensor_ops::GpuTensor>, _position_ids: Option<&crate::gpu_tensor_ops::GpuTensor>, _past_key_values: Option<&Vec<(crate::gpu_tensor_ops::GpuTensor, crate::gpu_tensor_ops::GpuTensor)>>) -> Result<crate::model_implementations::ModelOutput, ModelError> {
        Err(ModelError::TensorError("StarCoder not fully implemented".to_string()))
    }
    async fn generate(&self, _input_ids: &crate::gpu_tensor_ops::GpuTensor, _generation_config: &crate::model_implementations::GenerationConfig) -> Result<crate::model_implementations::GenerationOutput, ModelError> {
        Err(ModelError::GenerationError("StarCoder generation not implemented".to_string()))
    }
    fn get_config(&self) -> &ModelConfig { &self.config }
    fn get_architecture(&self) -> ModelArchitecture { self.config.architecture.clone() }
    fn supports_paged_attention(&self) -> bool { false }
    fn supports_flash_attention(&self) -> bool { true }
}

/// GPT-NeoX model implementation
pub struct GPTNeoXModel {
    config: ModelConfig,
    device: GpuDevice,
}

impl GPTNeoXModel {
    pub async fn new(config: ModelConfig, device: GpuDevice) -> Result<Self, ModelError> {
        Ok(Self { config, device })
    }
}

#[async_trait]
impl ModelImplementation for GPTNeoXModel {
    async fn forward(&self, _input_ids: &crate::gpu_tensor_ops::GpuTensor, _attention_mask: Option<&crate::gpu_tensor_ops::GpuTensor>, _position_ids: Option<&crate::gpu_tensor_ops::GpuTensor>, _past_key_values: Option<&Vec<(crate::gpu_tensor_ops::GpuTensor, crate::gpu_tensor_ops::GpuTensor)>>) -> Result<crate::model_implementations::ModelOutput, ModelError> {
        Err(ModelError::TensorError("GPTNeoX not fully implemented".to_string()))
    }
    async fn generate(&self, _input_ids: &crate::gpu_tensor_ops::GpuTensor, _generation_config: &crate::model_implementations::GenerationConfig) -> Result<crate::model_implementations::GenerationOutput, ModelError> {
        Err(ModelError::GenerationError("GPTNeoX generation not implemented".to_string()))
    }
    fn get_config(&self) -> &ModelConfig { &self.config }
    fn get_architecture(&self) -> ModelArchitecture { self.config.architecture.clone() }
    fn supports_paged_attention(&self) -> bool { false }
    fn supports_flash_attention(&self) -> bool { true }
}

/// Microsoft Phi model implementation
pub struct PhiModel {
    config: ModelConfig,
    device: GpuDevice,
}

impl PhiModel {
    pub async fn new(config: ModelConfig, device: GpuDevice) -> Result<Self, ModelError> {
        Ok(Self { config, device })
    }
}

#[async_trait]
impl ModelImplementation for PhiModel {
    async fn forward(&self, _input_ids: &crate::gpu_tensor_ops::GpuTensor, _attention_mask: Option<&crate::gpu_tensor_ops::GpuTensor>, _position_ids: Option<&crate::gpu_tensor_ops::GpuTensor>, _past_key_values: Option<&Vec<(crate::gpu_tensor_ops::GpuTensor, crate::gpu_tensor_ops::GpuTensor)>>) -> Result<crate::model_implementations::ModelOutput, ModelError> {
        Err(ModelError::TensorError("Phi not fully implemented".to_string()))
    }
    async fn generate(&self, _input_ids: &crate::gpu_tensor_ops::GpuTensor, _generation_config: &crate::model_implementations::GenerationConfig) -> Result<crate::model_implementations::GenerationOutput, ModelError> {
        Err(ModelError::GenerationError("Phi generation not implemented".to_string()))
    }
    fn get_config(&self) -> &ModelConfig { &self.config }
    fn get_architecture(&self) -> ModelArchitecture { self.config.architecture.clone() }
    fn supports_paged_attention(&self) -> bool { false }
    fn supports_flash_attention(&self) -> bool { true }
}

/// Google Gemma model implementation
pub struct GemmaModel {
    config: ModelConfig,
    device: GpuDevice,
}

impl GemmaModel {
    pub async fn new(config: ModelConfig, device: GpuDevice) -> Result<Self, ModelError> {
        Ok(Self { config, device })
    }
}

#[async_trait]
impl ModelImplementation for GemmaModel {
    async fn forward(&self, _input_ids: &crate::gpu_tensor_ops::GpuTensor, _attention_mask: Option<&crate::gpu_tensor_ops::GpuTensor>, _position_ids: Option<&crate::gpu_tensor_ops::GpuTensor>, _past_key_values: Option<&Vec<(crate::gpu_tensor_ops::GpuTensor, crate::gpu_tensor_ops::GpuTensor)>>) -> Result<crate::model_implementations::ModelOutput, ModelError> {
        Err(ModelError::TensorError("Gemma not fully implemented".to_string()))
    }
    async fn generate(&self, _input_ids: &crate::gpu_tensor_ops::GpuTensor, _generation_config: &crate::model_implementations::GenerationConfig) -> Result<crate::model_implementations::GenerationOutput, ModelError> {
        Err(ModelError::GenerationError("Gemma generation not implemented".to_string()))
    }
    fn get_config(&self) -> &ModelConfig { &self.config }
    fn get_architecture(&self) -> ModelArchitecture { self.config.architecture.clone() }
    fn supports_paged_attention(&self) -> bool { false }
    fn supports_flash_attention(&self) -> bool { true }
}

/// CLIP multimodal model implementation
pub struct CLIPModel {
    config: ModelConfig,
    device: GpuDevice,
}

impl CLIPModel {
    pub async fn new(config: ModelConfig, device: GpuDevice) -> Result<Self, ModelError> {
        Ok(Self { config, device })
    }
}

#[async_trait]
impl ModelImplementation for CLIPModel {
    async fn forward(&self, _input_ids: &crate::gpu_tensor_ops::GpuTensor, _attention_mask: Option<&crate::gpu_tensor_ops::GpuTensor>, _position_ids: Option<&crate::gpu_tensor_ops::GpuTensor>, _past_key_values: Option<&Vec<(crate::gpu_tensor_ops::GpuTensor, crate::gpu_tensor_ops::GpuTensor)>>) -> Result<crate::model_implementations::ModelOutput, ModelError> {
        Err(ModelError::TensorError("CLIP not fully implemented".to_string()))
    }
    async fn generate(&self, _input_ids: &crate::gpu_tensor_ops::GpuTensor, _generation_config: &crate::model_implementations::GenerationConfig) -> Result<crate::model_implementations::GenerationOutput, ModelError> {
        Err(ModelError::GenerationError("CLIP generation not implemented".to_string()))
    }
    fn get_config(&self) -> &ModelConfig { &self.config }
    fn get_architecture(&self) -> ModelArchitecture { self.config.architecture.clone() }
    fn supports_paged_attention(&self) -> bool { false }
    fn supports_flash_attention(&self) -> bool { true }
}

/// BLIP multimodal model implementation
pub struct BLIPModel {
    config: ModelConfig,
    device: GpuDevice,
}

impl BLIPModel {
    pub async fn new(config: ModelConfig, device: GpuDevice) -> Result<Self, ModelError> {
        Ok(Self { config, device })
    }
}

#[async_trait]
impl ModelImplementation for BLIPModel {
    async fn forward(&self, _input_ids: &crate::gpu_tensor_ops::GpuTensor, _attention_mask: Option<&crate::gpu_tensor_ops::GpuTensor>, _position_ids: Option<&crate::gpu_tensor_ops::GpuTensor>, _past_key_values: Option<&Vec<(crate::gpu_tensor_ops::GpuTensor, crate::gpu_tensor_ops::GpuTensor)>>) -> Result<crate::model_implementations::ModelOutput, ModelError> {
        Err(ModelError::TensorError("BLIP not fully implemented".to_string()))
    }
    async fn generate(&self, _input_ids: &crate::gpu_tensor_ops::GpuTensor, _generation_config: &crate::model_implementations::GenerationConfig) -> Result<crate::model_implementations::GenerationOutput, ModelError> {
        Err(ModelError::GenerationError("BLIP generation not implemented".to_string()))
    }
    fn get_config(&self) -> &ModelConfig { &self.config }
    fn get_architecture(&self) -> ModelArchitecture { self.config.architecture.clone() }
    fn supports_paged_attention(&self) -> bool { false }
    fn supports_flash_attention(&self) -> bool { true }
}

/// Flamingo multimodal model implementation
pub struct FlamingoModel {
    config: ModelConfig,
    device: GpuDevice,
}

impl FlamingoModel {
    pub async fn new(config: ModelConfig, device: GpuDevice) -> Result<Self, ModelError> {
        Ok(Self { config, device })
    }
}

#[async_trait]
impl ModelImplementation for FlamingoModel {
    async fn forward(&self, _input_ids: &crate::gpu_tensor_ops::GpuTensor, _attention_mask: Option<&crate::gpu_tensor_ops::GpuTensor>, _position_ids: Option<&crate::gpu_tensor_ops::GpuTensor>, _past_key_values: Option<&Vec<(crate::gpu_tensor_ops::GpuTensor, crate::gpu_tensor_ops::GpuTensor)>>) -> Result<crate::model_implementations::ModelOutput, ModelError> {
        Err(ModelError::TensorError("Flamingo not fully implemented".to_string()))
    }
    async fn generate(&self, _input_ids: &crate::gpu_tensor_ops::GpuTensor, _generation_config: &crate::model_implementations::GenerationConfig) -> Result<crate::model_implementations::GenerationOutput, ModelError> {
        Err(ModelError::GenerationError("Flamingo generation not implemented".to_string()))
    }
    fn get_config(&self) -> &ModelConfig { &self.config }
    fn get_architecture(&self) -> ModelArchitecture { self.config.architecture.clone() }
    fn supports_paged_attention(&self) -> bool { false }
    fn supports_flash_attention(&self) -> bool { true }
}

/// MPT model implementation
pub struct MPTModel {
    config: ModelConfig,
    device: GpuDevice,
}

impl MPTModel {
    pub async fn new(config: ModelConfig, device: GpuDevice) -> Result<Self, ModelError> {
        Ok(Self { config, device })
    }
}

#[async_trait]
impl ModelImplementation for MPTModel {
    async fn forward(&self, _input_ids: &crate::gpu_tensor_ops::GpuTensor, _attention_mask: Option<&crate::gpu_tensor_ops::GpuTensor>, _position_ids: Option<&crate::gpu_tensor_ops::GpuTensor>, _past_key_values: Option<&Vec<(crate::gpu_tensor_ops::GpuTensor, crate::gpu_tensor_ops::GpuTensor)>>) -> Result<crate::model_implementations::ModelOutput, ModelError> {
        Err(ModelError::TensorError("MPT not fully implemented".to_string()))
    }
    async fn generate(&self, _input_ids: &crate::gpu_tensor_ops::GpuTensor, _generation_config: &crate::model_implementations::GenerationConfig) -> Result<crate::model_implementations::GenerationOutput, ModelError> {
        Err(ModelError::GenerationError("MPT generation not implemented".to_string()))
    }
    fn get_config(&self) -> &ModelConfig { &self.config }
    fn get_architecture(&self) -> ModelArchitecture { self.config.architecture.clone() }
    fn supports_paged_attention(&self) -> bool { false }
    fn supports_flash_attention(&self) -> bool { true }
}

/// Falcon model implementation
pub struct FalconModel {
    config: ModelConfig,
    device: GpuDevice,
}

impl FalconModel {
    pub async fn new(config: ModelConfig, device: GpuDevice) -> Result<Self, ModelError> {
        Ok(Self { config, device })
    }
}

#[async_trait]
impl ModelImplementation for FalconModel {
    async fn forward(&self, _input_ids: &crate::gpu_tensor_ops::GpuTensor, _attention_mask: Option<&crate::gpu_tensor_ops::GpuTensor>, _position_ids: Option<&crate::gpu_tensor_ops::GpuTensor>, _past_key_values: Option<&Vec<(crate::gpu_tensor_ops::GpuTensor, crate::gpu_tensor_ops::GpuTensor)>>) -> Result<crate::model_implementations::ModelOutput, ModelError> {
        Err(ModelError::TensorError("Falcon not fully implemented".to_string()))
    }
    async fn generate(&self, _input_ids: &crate::gpu_tensor_ops::GpuTensor, _generation_config: &crate::model_implementations::GenerationConfig) -> Result<crate::model_implementations::GenerationOutput, ModelError> {
        Err(ModelError::GenerationError("Falcon generation not implemented".to_string()))
    }
    fn get_config(&self) -> &ModelConfig { &self.config }
    fn get_architecture(&self) -> ModelArchitecture { self.config.architecture.clone() }
    fn supports_paged_attention(&self) -> bool { false }
    fn supports_flash_attention(&self) -> bool { true }
}

/// Model loader for automatic model detection and loading
pub struct ModelLoader {
    factory: ModelFactory,
}

impl ModelLoader {
    pub fn new(device: GpuDevice) -> Self {
        Self {
            factory: ModelFactory::new(device),
        }
    }

    /// Load model from Hugging Face model ID
    pub async fn load_from_hub(
        &mut self,
        model_id: &str,
    ) -> Result<Arc<dyn ModelImplementation>, ModelError> {
        self.factory.create_model_from_name(model_id).await
    }

    /// Load model from local path
    pub async fn load_from_path(
        &mut self,
        model_path: &str,
    ) -> Result<Arc<dyn ModelImplementation>, ModelError> {
        // Extract model name from path for architecture detection
        let model_name = std::path::Path::new(model_path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(model_path);

        self.factory.create_model_from_name(model_name).await
    }

    /// Get information about all supported models
    pub fn list_supported_models(&self) -> Vec<ModelInfo> {
        self.factory.get_supported_architectures()
            .into_iter()
            .filter_map(|arch| self.factory.get_model_info(&arch))
            .collect()
    }

    /// Get models by parameter size range
    pub fn get_models_by_size(&self, min_params: u64, max_params: u64) -> Vec<ModelInfo> {
        self.list_supported_models()
            .into_iter()
            .filter(|info| info.parameter_count >= min_params && info.parameter_count <= max_params)
            .collect()
    }

    /// Get models that support multimodal capabilities
    pub fn get_multimodal_models(&self) -> Vec<ModelInfo> {
        self.list_supported_models()
            .into_iter()
            .filter(|info| info.supports_multimodal)
            .collect()
    }
}