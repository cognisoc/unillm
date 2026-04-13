//! Core Model Abstraction
//!
//! This module provides the foundational model abstraction that all
//! model implementations build upon. It defines clean interfaces that
//! hide implementation complexity.

use crate::tensor_core::{DataType, Device, Tensor};
use anyhow::Result;
use candle_core::quantized::QMatMul;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

/// Core model trait - the single interface all models implement
pub trait Model: Send + Sync {
    /// Model configuration type
    type Config: ModelConfig;

    /// Create new model instance
    fn new(config: Self::Config) -> Result<Self>
    where
        Self: Sized;

    /// Load model with weights
    fn from_weights(config: Self::Config, weights: ModelWeights) -> Result<Self>
    where
        Self: Sized;

    /// Forward pass - core inference method
    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs>;

    /// Generate text (high-level interface)
    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String>;

    /// Get model configuration
    fn config(&self) -> &Self::Config;

    /// Get model memory requirements
    fn memory_requirements(&self) -> MemoryRequirements;

    /// Move model to device
    fn to_device(&mut self, device: &Device) -> Result<()>;
}

/// Model configuration trait
pub trait ModelConfig: Send + Sync + std::fmt::Debug {
    /// Get model architecture name
    fn architecture(&self) -> &str;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;

    /// Get hidden dimension
    fn hidden_size(&self) -> usize;

    /// Get number of layers
    fn num_layers(&self) -> usize;

    /// Validate configuration
    fn validate(&self) -> Result<()>;
}

/// Unified model inputs - supports all model types
#[derive(Debug, Clone)]
pub enum ModelInputs {
    /// Text-only inputs (most language models)
    Text {
        input_ids: Tensor,
        attention_mask: Option<Tensor>,
        position_ids: Option<Tensor>,
    },
    /// Image-only inputs (vision models)
    Image {
        pixel_values: Tensor,
        image_mask: Option<Tensor>,
    },
    /// Multimodal inputs (vision-language models)
    Multimodal {
        input_ids: Tensor,
        pixel_values: Option<Tensor>,
        attention_mask: Option<Tensor>,
        image_mask: Option<Tensor>,
    },
    /// Audio inputs (speech models)
    Audio {
        input_features: Tensor,
        attention_mask: Option<Tensor>,
    },
}

/// Unified model outputs - supports all model types
#[derive(Debug, Clone)]
pub enum ModelOutputs {
    /// Language model logits
    Logits {
        logits: Tensor,
        hidden_states: Option<Tensor>,
    },
    /// Embeddings/representations
    Embeddings {
        embeddings: Tensor,
        pooled: Option<Tensor>,
    },
    /// Multimodal outputs
    Multimodal {
        text_logits: Option<Tensor>,
        image_logits: Option<Tensor>,
        text_embeddings: Option<Tensor>,
        image_embeddings: Option<Tensor>,
    },
    /// Sequence-to-sequence outputs
    Sequence {
        logits: Tensor,
        encoder_hidden_states: Option<Tensor>,
        decoder_hidden_states: Option<Tensor>,
    },
    /// CLIP contrastive outputs
    CLIP {
        logits_per_text: Tensor,
        logits_per_image: Tensor,
        text_embeds: Tensor,
        image_embeds: Tensor,
    },
}

/// Model weights container
#[derive(Clone)]
pub struct ModelWeights {
    /// Tensor weights by name (F32 dequantized, for compatibility)
    pub tensors: HashMap<String, Tensor>,
    /// Metadata
    pub metadata: WeightMetadata,
    /// GGUF-specific config (populated when loading from GGUF)
    pub gguf_config: Option<crate::weight_loader_core::GGUFModelConfig>,
    /// GGUF tokenizer data (populated when loading from GGUF)
    pub gguf_tokenizer: Option<crate::weight_loader_core::GGUFTokenizer>,
    /// Quantized weights (QMatMul) for efficient inference
    pub quantized_tensors: HashMap<String, Arc<QMatMul>>,
    /// Native SIMD quantized weights for maximum performance
    #[cfg(feature = "simd")]
    pub simd_quantized: HashMap<String, Arc<crate::simd::quant::QuantizedTensor>>,
}

impl std::fmt::Debug for ModelWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("ModelWeights");
        debug
            .field("tensors", &format!("{} tensors", self.tensors.len()))
            .field("metadata", &self.metadata)
            .field("gguf_config", &self.gguf_config)
            .field(
                "gguf_tokenizer",
                &self.gguf_tokenizer.as_ref().map(|_| "..."),
            )
            .field(
                "quantized_tensors",
                &format!("{} quantized", self.quantized_tensors.len()),
            );

        #[cfg(feature = "simd")]
        debug.field(
            "simd_quantized",
            &format!("{} simd", self.simd_quantized.len()),
        );

        debug.finish()
    }
}

/// Weight metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightMetadata {
    /// Model architecture
    pub architecture: String,
    /// Total parameters
    pub total_params: usize,
    /// Weight format
    pub format: WeightFormat,
    /// Precision
    pub dtype: String,
}

/// Supported weight formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WeightFormat {
    SafeTensors,
    PyTorch,
    GGUF,
    HuggingFace,
}

/// Generation configuration
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub min_p: f32,
    pub do_sample: bool,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub stop_sequences: Vec<String>,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 100,
            temperature: 1.0,
            top_p: 1.0,
            top_k: None,
            min_p: 0.0,
            do_sample: true,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_sequences: vec![],
            eos_token_id: 2,
            pad_token_id: 0,
        }
    }
}

/// Memory requirements for a model
#[derive(Debug, Clone)]
pub struct MemoryRequirements {
    /// GPU memory in bytes
    pub gpu_memory: usize,
    /// CPU memory in bytes
    pub cpu_memory: usize,
    /// KV cache memory in bytes
    pub kv_cache_memory: usize,
    /// Peak memory during inference
    pub peak_memory: usize,
}

// ============================================================================
// ModelRunner - Unified generation interface for CLI/server
// ============================================================================

/// Unified model runner trait for streaming generation across architectures.
/// This is the interface used by the CLI binary and HTTP server.
pub trait ModelRunner: Send + Sync {
    /// Generate tokens with streaming callback.
    /// Returns generated token ids (excluding prompt).
    /// Callback receives (token_id, decoded_text) and returns true to continue.
    fn generate_streaming(
        &self,
        tokens: &[u32],
        config: &GenerationConfig,
        callback: &mut dyn FnMut(u32, &str) -> bool,
    ) -> Result<Vec<u32>>;

    /// Get the model's EOS token ID
    fn eos_token_id(&self) -> u32;

    /// Get the architecture name
    fn architecture(&self) -> &str;

    /// Get the model's context size
    fn context_size(&self) -> usize;
}

/// Performance metrics for a generation run
#[derive(Debug, Clone)]
pub struct GenerationMetrics {
    /// Time to load model in milliseconds
    pub model_load_ms: f64,
    /// Number of prompt tokens processed
    pub prompt_tokens: usize,
    /// Time to process prompt (prefill) in milliseconds
    pub prompt_eval_ms: f64,
    /// Prompt evaluation speed
    pub prompt_tokens_per_sec: f64,
    /// Number of tokens generated
    pub generated_tokens: usize,
    /// Time to generate tokens (decode) in milliseconds
    pub generation_ms: f64,
    /// Generation speed
    pub generation_tokens_per_sec: f64,
    /// Total time in milliseconds
    pub total_ms: f64,
}

impl GenerationMetrics {
    pub fn new() -> Self {
        Self {
            model_load_ms: 0.0,
            prompt_tokens: 0,
            prompt_eval_ms: 0.0,
            prompt_tokens_per_sec: 0.0,
            generated_tokens: 0,
            generation_ms: 0.0,
            generation_tokens_per_sec: 0.0,
            total_ms: 0.0,
        }
    }

    /// Compute derived fields from raw timing data
    pub fn finalize(&mut self) {
        if self.prompt_eval_ms > 0.0 {
            self.prompt_tokens_per_sec =
                (self.prompt_tokens as f64) / (self.prompt_eval_ms / 1000.0);
        }
        if self.generation_ms > 0.0 {
            self.generation_tokens_per_sec =
                (self.generated_tokens as f64) / (self.generation_ms / 1000.0);
        }
        self.total_ms = self.model_load_ms + self.prompt_eval_ms + self.generation_ms;
    }

    /// Print metrics summary (llama.cpp style)
    pub fn print_summary(&self) {
        eprintln!();
        eprintln!("Model load time:     {:>8.2} ms", self.model_load_ms);
        eprintln!(
            "Prompt eval:         {:>8.2} ms / {:>5} tokens ({:>8.2} tokens/s)",
            self.prompt_eval_ms, self.prompt_tokens, self.prompt_tokens_per_sec
        );
        eprintln!(
            "Generation:          {:>8.2} ms / {:>5} tokens ({:>8.2} tokens/s)",
            self.generation_ms, self.generated_tokens, self.generation_tokens_per_sec
        );
        eprintln!("Total time:          {:>8.2} ms", self.total_ms);
    }
}

impl Default for GenerationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Model factory trait for creating models by name
pub trait ModelFactory: Send + Sync {
    /// Create model from configuration
    fn create_model(
        &self,
        config_json: &str,
    ) -> Result<Box<dyn Model<Config = Box<dyn ModelConfig>>>>;

    /// Get supported architecture names
    fn supported_architectures(&self) -> Vec<&str>;

    /// Check if architecture is supported
    fn supports(&self, architecture: &str) -> bool;
}

/// Model registry for managing all supported architectures
pub struct ModelRegistry {
    factories: HashMap<String, Box<dyn ModelFactory>>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Register a model factory
    pub fn register<F>(&mut self, architecture: &str, factory: F)
    where
        F: ModelFactory + 'static,
    {
        self.factories
            .insert(architecture.to_string(), Box::new(factory));
    }

    /// Create model by architecture name
    pub fn create_model(
        &self,
        architecture: &str,
        config_json: &str,
    ) -> Result<Box<dyn Model<Config = Box<dyn ModelConfig>>>> {
        let factory = self
            .factories
            .get(architecture)
            .ok_or_else(|| anyhow::anyhow!("Unsupported architecture: {}", architecture))?;

        factory.create_model(config_json)
    }

    /// Get all supported architectures
    pub fn supported_architectures(&self) -> Vec<String> {
        self.factories.keys().cloned().collect()
    }

    /// Check if architecture is supported
    pub fn supports(&self, architecture: &str) -> bool {
        self.factories.contains_key(architecture)
    }
}

/// Global model registry
static MODEL_REGISTRY: std::sync::OnceLock<std::sync::Mutex<ModelRegistry>> =
    std::sync::OnceLock::new();

/// Get global model registry
pub fn registry() -> &'static std::sync::Mutex<ModelRegistry> {
    MODEL_REGISTRY.get_or_init(|| std::sync::Mutex::new(ModelRegistry::new()))
}

/// Utility functions for model inputs/outputs
impl ModelInputs {
    /// Create text inputs
    pub fn text(input_ids: Tensor) -> Self {
        Self::Text {
            input_ids,
            attention_mask: None,
            position_ids: None,
        }
    }

    /// Create text inputs with attention mask
    pub fn text_with_mask(input_ids: Tensor, attention_mask: Tensor) -> Self {
        Self::Text {
            input_ids,
            attention_mask: Some(attention_mask),
            position_ids: None,
        }
    }

    /// Create image inputs
    pub fn image(pixel_values: Tensor) -> Self {
        Self::Image {
            pixel_values,
            image_mask: None,
        }
    }

    /// Create multimodal inputs
    pub fn multimodal(input_ids: Tensor, pixel_values: Option<Tensor>) -> Self {
        Self::Multimodal {
            input_ids,
            pixel_values,
            attention_mask: None,
            image_mask: None,
        }
    }

    /// Get batch size
    pub fn batch_size(&self) -> usize {
        match self {
            Self::Text { input_ids, .. } => input_ids.shape()[0],
            Self::Image { pixel_values, .. } => pixel_values.shape()[0],
            Self::Multimodal { input_ids, .. } => input_ids.shape()[0],
            Self::Audio { input_features, .. } => input_features.shape()[0],
        }
    }

    /// Get sequence length (for text inputs)
    pub fn sequence_length(&self) -> Option<usize> {
        match self {
            Self::Text { input_ids, .. } => Some(input_ids.shape()[1]),
            Self::Multimodal { input_ids, .. } => Some(input_ids.shape()[1]),
            _ => None,
        }
    }
}

impl ModelOutputs {
    /// Create logits output
    pub fn logits(logits: Tensor) -> Self {
        Self::Logits {
            logits,
            hidden_states: None,
        }
    }

    /// Create embeddings output
    pub fn embeddings(embeddings: Tensor) -> Self {
        Self::Embeddings {
            embeddings,
            pooled: None,
        }
    }

    /// Get main tensor output
    pub fn main_tensor(&self) -> &Tensor {
        match self {
            Self::Logits { logits, .. } => logits,
            Self::Embeddings { embeddings, .. } => embeddings,
            Self::Multimodal {
                text_logits: Some(logits),
                ..
            } => logits,
            Self::Multimodal {
                image_logits: Some(logits),
                ..
            } => logits,
            Self::Sequence { logits, .. } => logits,
            _ => panic!("No main tensor available"),
        }
    }
}

impl ModelWeights {
    /// Create new model weights
    pub fn new(tensors: HashMap<String, Tensor>, metadata: WeightMetadata) -> Self {
        Self {
            tensors,
            metadata,
            gguf_config: None,
            gguf_tokenizer: None,
            quantized_tensors: HashMap::new(),
            #[cfg(feature = "simd")]
            simd_quantized: HashMap::new(),
        }
    }

    /// Create new model weights with GGUF config
    pub fn with_gguf_config(
        tensors: HashMap<String, Tensor>,
        metadata: WeightMetadata,
        gguf_config: crate::weight_loader_core::GGUFModelConfig,
    ) -> Self {
        Self {
            tensors,
            metadata,
            gguf_config: Some(gguf_config),
            gguf_tokenizer: None,
            quantized_tensors: HashMap::new(),
            #[cfg(feature = "simd")]
            simd_quantized: HashMap::new(),
        }
    }

    /// Create new model weights with GGUF config and tokenizer
    pub fn with_gguf_config_and_tokenizer(
        tensors: HashMap<String, Tensor>,
        metadata: WeightMetadata,
        gguf_config: crate::weight_loader_core::GGUFModelConfig,
        gguf_tokenizer: Option<crate::weight_loader_core::GGUFTokenizer>,
    ) -> Self {
        Self {
            tensors,
            metadata,
            gguf_config: Some(gguf_config),
            gguf_tokenizer,
            quantized_tensors: HashMap::new(),
            #[cfg(feature = "simd")]
            simd_quantized: HashMap::new(),
        }
    }

    /// Create new model weights with GGUF config, tokenizer, and quantized tensors
    pub fn with_quantized(
        tensors: HashMap<String, Tensor>,
        metadata: WeightMetadata,
        gguf_config: crate::weight_loader_core::GGUFModelConfig,
        gguf_tokenizer: Option<crate::weight_loader_core::GGUFTokenizer>,
        quantized_tensors: HashMap<String, Arc<QMatMul>>,
    ) -> Self {
        Self {
            tensors,
            metadata,
            gguf_config: Some(gguf_config),
            gguf_tokenizer,
            quantized_tensors,
            #[cfg(feature = "simd")]
            simd_quantized: HashMap::new(),
        }
    }

    /// Create new model weights with GGUF config, tokenizer, quantized tensors, and SIMD quantized
    #[cfg(feature = "simd")]
    pub fn with_simd_quantized(
        tensors: HashMap<String, Tensor>,
        metadata: WeightMetadata,
        gguf_config: crate::weight_loader_core::GGUFModelConfig,
        gguf_tokenizer: Option<crate::weight_loader_core::GGUFTokenizer>,
        quantized_tensors: HashMap<String, Arc<QMatMul>>,
        simd_quantized: HashMap<String, Arc<crate::simd::quant::QuantizedTensor>>,
    ) -> Self {
        Self {
            tensors,
            metadata,
            gguf_config: Some(gguf_config),
            gguf_tokenizer,
            quantized_tensors,
            simd_quantized,
        }
    }

    /// Get tensor by name
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Get tensor by name (required)
    pub fn require(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Required tensor '{}' not found", name))
    }

    /// Get quantized tensor (QMatMul) by name
    pub fn get_quantized(&self, name: &str) -> Option<Arc<QMatMul>> {
        self.quantized_tensors.get(name).cloned()
    }

    /// Check if a tensor has a quantized version
    pub fn has_quantized(&self, name: &str) -> bool {
        self.quantized_tensors.contains_key(name)
    }

    /// Get SIMD quantized tensor by name
    #[cfg(feature = "simd")]
    pub fn get_simd_quantized(
        &self,
        name: &str,
    ) -> Option<Arc<crate::simd::quant::QuantizedTensor>> {
        self.simd_quantized.get(name).cloned()
    }

    /// Check if a tensor has a SIMD quantized version
    #[cfg(feature = "simd")]
    pub fn has_simd_quantized(&self, name: &str) -> bool {
        self.simd_quantized.contains_key(name)
    }

    /// Get all tensor names
    pub fn tensor_names(&self) -> Vec<&String> {
        self.tensors.keys().collect()
    }

    /// Move all tensors to device
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        for tensor in self.tensors.values_mut() {
            *tensor = tensor.to_device(device)?;
        }
        Ok(())
    }
}

// ============================================================================
// Mixture of Experts (MoE) Support
// ============================================================================

/// MLP trait for expert networks in MoE layers
pub trait MLPLayer: Send + Sync {
    /// Forward pass through the MLP
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor>;
}

/// Configuration for MoE layer
#[derive(Debug, Clone)]
pub struct MoEConfig {
    /// Number of experts
    pub num_experts: usize,
    /// Number of experts activated per token
    pub num_experts_per_tok: usize,
    /// Whether to use auxiliary load balancing loss
    pub aux_loss: bool,
    /// Router type (softmax, top-k, etc.)
    pub router_type: RouterType,
}

impl Default for MoEConfig {
    fn default() -> Self {
        Self {
            num_experts: 8,
            num_experts_per_tok: 2,
            aux_loss: true,
            router_type: RouterType::TopK,
        }
    }
}

/// Router types for MoE
#[derive(Debug, Clone)]
pub enum RouterType {
    /// Standard top-k routing
    TopK,
    /// Expert choice routing (each expert chooses tokens)
    ExpertChoice,
    /// Soft routing with weighted combination
    Soft,
}

/// Mixture of Experts layer
/// Routes tokens to top-k experts and combines their outputs
#[derive(Debug)]
pub struct MoELayer {
    /// Router weights: [hidden_size, num_experts]
    pub router_weights: Tensor,
    /// Number of experts
    pub num_experts: usize,
    /// Number of experts per token
    pub num_experts_per_tok: usize,
    /// Device
    pub device: Device,
}

impl MoELayer {
    /// Create new MoE layer
    pub fn new(
        router_weights: Tensor,
        num_experts: usize,
        num_experts_per_tok: usize,
        device: Device,
    ) -> Self {
        Self {
            router_weights,
            num_experts,
            num_experts_per_tok,
            device,
        }
    }

    /// Compute routing weights and indices for each token
    /// Returns (routing_weights, expert_indices) where:
    /// - routing_weights: [batch * seq_len, num_experts_per_tok] - normalized weights
    /// - expert_indices: [batch * seq_len, num_experts_per_tok] - expert indices
    pub fn route(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor)> {
        use crate::tensor_core::ops_fn;

        // hidden_states: [batch, seq_len, hidden_size]
        let shape = hidden_states.shape();
        let (batch, seq_len, _hidden_size) = (shape[0], shape[1], shape[2]);
        let num_tokens = batch * seq_len;

        // Reshape to [batch * seq_len, hidden_size]
        let flat_hidden = hidden_states.reshape(&[num_tokens, shape[2]])?;

        // Compute router logits: [batch * seq_len, num_experts]
        let router_logits = ops_fn::matmul(&flat_hidden, &self.router_weights)?;

        // Get top-k experts
        let (topk_weights, topk_indices) =
            ops_fn::topk(&router_logits, self.num_experts_per_tok, -1)?;

        // Softmax over the selected experts to get normalized weights
        let routing_weights = ops_fn::softmax(&topk_weights, -1)?;

        Ok((routing_weights, topk_indices))
    }

    /// Forward pass through MoE layer
    /// expert_fn: function that takes (hidden_states, expert_idx) and returns expert output
    pub fn forward_with_experts<F>(&self, hidden_states: &Tensor, expert_fn: F) -> Result<Tensor>
    where
        F: Fn(&Tensor, usize) -> Result<Tensor>,
    {
        use crate::tensor_core::ops_fn;

        let shape = hidden_states.shape();
        let (batch, seq_len, hidden_size) = (shape[0], shape[1], shape[2]);
        let num_tokens = batch * seq_len;

        // Get routing weights and indices
        let (routing_weights, expert_indices) = self.route(hidden_states)?;

        // Flatten hidden states for processing
        let flat_hidden = hidden_states.reshape(&[num_tokens, hidden_size])?;

        // Initialize output tensor
        let mut output = ops_fn::zeros(
            &[num_tokens, hidden_size],
            hidden_states.dtype(),
            &self.device,
        )?;

        // Process each expert
        // This is a basic implementation - production would batch tokens by expert
        for expert_idx in 0..self.num_experts {
            // Find tokens routed to this expert
            // For each position in top-k, check if it matches this expert
            let expert_indices_candle = expert_indices.to_candle()?;
            let routing_weights_candle = routing_weights.to_candle()?;

            for tok_idx in 0..num_tokens {
                for k in 0..self.num_experts_per_tok {
                    let idx_val: Vec<i64> = expert_indices_candle.get(tok_idx)?.to_vec1()?;
                    if idx_val[k] as usize == expert_idx {
                        // Get this token's hidden state
                        let token_hidden = flat_hidden.to_candle()?.get(tok_idx)?;
                        let token_tensor = Tensor::from_candle(token_hidden.unsqueeze(0)?);

                        // Get expert output
                        let expert_output = expert_fn(&token_tensor, expert_idx)?;

                        // Get routing weight for this expert
                        let weight_val: Vec<f32> =
                            routing_weights_candle.get(tok_idx)?.to_vec1()?;
                        let weight = weight_val[k];

                        // Scale by routing weight and add to output
                        let scaled_output = ops_fn::scale(&expert_output, weight)?;

                        // Add to output at this position
                        let output_candle = output.to_candle()?;
                        let current = output_candle.get(tok_idx)?;
                        let new_val = (current + scaled_output.to_candle()?.squeeze(0)?)?;

                        // Update output tensor at this position
                        // This is inefficient but works for the basic implementation
                        let mut output_data: Vec<f32> =
                            output.to_candle()?.flatten_all()?.to_vec1()?;
                        let new_data: Vec<f32> = new_val.to_vec1()?;
                        for (i, v) in new_data.iter().enumerate() {
                            output_data[tok_idx * hidden_size + i] = *v;
                        }
                        output = Tensor::from_f32_slice(
                            &output_data,
                            &[num_tokens, hidden_size],
                            &self.device,
                        )?;
                    }
                }
            }
        }

        // Reshape back to [batch, seq_len, hidden_size]
        output.reshape(&[batch, seq_len, hidden_size])
    }
}

/// Helper struct for a standard MoE expert (simple MLP)
#[derive(Debug)]
pub struct MoEExpert {
    /// Gate projection: hidden_size -> intermediate_size
    pub gate_proj: Tensor,
    /// Up projection: hidden_size -> intermediate_size
    pub up_proj: Tensor,
    /// Down projection: intermediate_size -> hidden_size
    pub down_proj: Tensor,
}

impl MoEExpert {
    pub fn new(gate_proj: Tensor, up_proj: Tensor, down_proj: Tensor) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    /// Forward pass: SwiGLU activation
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        use crate::tensor_core::ops_fn;

        // SwiGLU: down(silu(gate(x)) * up(x))
        let gate = ops_fn::matmul(hidden_states, &self.gate_proj)?;
        let gate_activated = ops_fn::silu(&gate)?;
        let up = ops_fn::matmul(hidden_states, &self.up_proj)?;
        let gated = ops_fn::mul(&gate_activated, &up)?;
        ops_fn::matmul(&gated, &self.down_proj)
    }
}

// ============================================================================
// Model Configuration Macro
// ============================================================================

/// Helper macro for creating model configurations
#[macro_export]
macro_rules! model_config {
    ($name:ident {
        $($field:ident: $type:ty = $default:expr),* $(,)?
    }) => {
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct $name {
            $(pub $field: $type,)*
        }

        impl Default for $name {
            fn default() -> Self {
                Self {
                    $($field: $default,)*
                }
            }
        }

        impl ModelConfig for $name {
            fn architecture(&self) -> &str {
                stringify!($name)
            }

            fn vocab_size(&self) -> usize {
                self.vocab_size
            }

            fn hidden_size(&self) -> usize {
                self.hidden_size
            }

            fn num_layers(&self) -> usize {
                self.num_hidden_layers
            }

            fn validate(&self) -> Result<()> {
                if self.vocab_size() == 0 {
                    return Err(anyhow::anyhow!("vocab_size must be > 0"));
                }
                if self.hidden_size() == 0 {
                    return Err(anyhow::anyhow!("hidden_size must be > 0"));
                }
                if self.num_layers() == 0 {
                    return Err(anyhow::anyhow!("num_layers must be > 0"));
                }
                Ok(())
            }
        }
    };
}

// Example usage would be:
// model_config!(LlamaConfig {
//     vocab_size: usize = 32000,
//     hidden_size: usize = 4096,
//     // ... other fields
// });

// ============================================================================
// Error types and result aliases (unified from legacy types module)
// ============================================================================

/// Error types for model operations
#[derive(Debug)]
pub enum ModelError {
    InitializationFailed(String),
    ComputationFailed(String),
    InvalidInput(String),
    DeviceError(String),
    MemoryError(String),
    UnsupportedOperation(String),
    GenerationFailed(String),
    ValidationFailed(String),
    ServerError(String),
    LoadingError(String),
    ConfigurationError(String),
    NetworkError(String),
    FileSystemError(String),
    CommunicationError(String),
    RuntimeError(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::InitializationFailed(msg) => write!(f, "Initialization failed: {}", msg),
            ModelError::ComputationFailed(msg) => write!(f, "Computation failed: {}", msg),
            ModelError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            ModelError::DeviceError(msg) => write!(f, "Device error: {}", msg),
            ModelError::MemoryError(msg) => write!(f, "Memory error: {}", msg),
            ModelError::UnsupportedOperation(msg) => write!(f, "Unsupported operation: {}", msg),
            ModelError::GenerationFailed(msg) => write!(f, "Generation failed: {}", msg),
            ModelError::ValidationFailed(msg) => write!(f, "Validation failed: {}", msg),
            ModelError::ServerError(msg) => write!(f, "Server error: {}", msg),
            ModelError::LoadingError(msg) => write!(f, "Loading error: {}", msg),
            ModelError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            ModelError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            ModelError::FileSystemError(msg) => write!(f, "File system error: {}", msg),
            ModelError::CommunicationError(msg) => write!(f, "Communication error: {}", msg),
            ModelError::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
        }
    }
}

impl std::error::Error for ModelError {}

/// Result type for model operations
pub type ModelResult<T> = Result<T, ModelError>;

/// Model features that can be supported
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFeature {
    FlashAttention,
    GroupedQueryAttention,
    RotaryEmbedding,
    RMSNorm,
    LayerNorm,
    SwiGLU,
    GELU,
    PrefixCaching,
    ChunkedPrefill,
    DynamicBatching,
    ContinuousBatching,
    LongContext,
    SlidingWindow,
    Quantization,
    LoRA,
}

/// Normalization types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationType {
    LayerNorm,
    RMSNorm,
    GroupNorm,
}

/// Position embedding types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionEmbeddingType {
    Absolute,
    Relative,
    Rotary,
    ALiBi,
}

/// Model precision types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelPrecision {
    Float32,
    Float16,
    BFloat16,
    Int8,
    Int4,
}

impl fmt::Display for ModelPrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelPrecision::Float32 => write!(f, "fp32"),
            ModelPrecision::Float16 => write!(f, "fp16"),
            ModelPrecision::BFloat16 => write!(f, "bf16"),
            ModelPrecision::Int8 => write!(f, "int8"),
            ModelPrecision::Int4 => write!(f, "int4"),
        }
    }
}

/// Generation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub time_to_first_token_ms: f64,
    pub tokens_per_second: f64,
    pub total_time_ms: f64,
    pub cache_hit_rate: f64,
    pub memory_usage_mb: f64,
}

// ============================================================================
// KV Cache Provider Trait - Unified interface for KV caches
// ============================================================================

/// Unified trait for KV cache providers.
///
/// This trait defines the interface that all KV cache implementations must satisfy,
/// enabling models to use any KV cache backend (Candle-based, paged, hybrid) without
/// knowing the implementation details.
pub trait KvCacheProvider: Send + Sync {
    /// Get the number of layers in this cache
    fn num_layers(&self) -> usize;

    /// Get current cached sequence length
    fn seq_len(&self) -> usize;

    /// Clear all cached values (may keep buffers allocated for reuse)
    fn clear(&mut self);

    /// Check if cache is empty
    fn is_empty(&self) -> bool {
        self.seq_len() == 0
    }
}

// ============================================================================
// Model Executor - Bridge between scheduler and model execution
// ============================================================================

/// Result of a single execution step
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Generated token IDs (excluding prompt)
    pub token_ids: Vec<u32>,
    /// Decoded text output
    pub text: String,
    /// Number of prompt tokens processed
    pub prompt_tokens: usize,
    /// Number of completion tokens generated
    pub completion_tokens: usize,
    /// Time to first token in milliseconds
    pub time_to_first_token_ms: f64,
    /// Total generation time in milliseconds
    pub total_time_ms: f64,
    /// Tokens per second
    pub tokens_per_second: f64,
}

/// Execution context provided by the scheduler
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// GPU device assignment (if applicable)
    pub gpu_id: usize,
    /// Memory pressure level (0.0-1.0)
    pub memory_pressure: f32,
    /// Estimated latency budget in milliseconds
    pub latency_budget_ms: Option<f64>,
    /// Whether to use KV cache
    pub use_cache: bool,
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self {
            gpu_id: 0,
            memory_pressure: 0.0,
            latency_budget_ms: None,
            use_cache: true,
        }
    }
}

/// Model executor trait that bridges the scheduler/batch processor
/// to actual model execution. This decouples the scheduling layer from
/// the model implementation, allowing different execution strategies.
pub trait ModelExecutor: Send + Sync {
    /// Execute generation for a single prompt
    fn execute_generation(
        &self,
        prompt_tokens: &[u32],
        config: &GenerationConfig,
        context: &ExecutionContext,
    ) -> Result<ExecutionResult>;

    /// Get the model's EOS token ID
    fn eos_token_id(&self) -> u32;

    /// Get the model's context size
    fn context_size(&self) -> usize;

    /// Get the architecture name
    fn architecture(&self) -> &str;
}

/// Blanket implementation of ModelExecutor for any ModelRunner
impl<T: ModelRunner> ModelExecutor for T {
    fn execute_generation(
        &self,
        prompt_tokens: &[u32],
        config: &GenerationConfig,
        _context: &ExecutionContext,
    ) -> Result<ExecutionResult> {
        use std::time::Instant;

        let start = Instant::now();
        let mut ttft: Option<f64> = None;
        let mut text = String::new();
        let mut token_ids = Vec::new();

        let result = self.generate_streaming(prompt_tokens, config, &mut |token_id, token_text| {
            if ttft.is_none() {
                ttft = Some(start.elapsed().as_secs_f64() * 1000.0);
            }
            token_ids.push(token_id);
            text.push_str(token_text);
            true
        });

        result.map(|ids| {
            let total_time = start.elapsed().as_secs_f64() * 1000.0;
            let completion_tokens = ids.len();
            let ttft_val = ttft.unwrap_or(0.0);
            let tps = if total_time > 0.0 {
                completion_tokens as f64 / (total_time / 1000.0)
            } else {
                0.0
            };

            ExecutionResult {
                token_ids: ids,
                text,
                prompt_tokens: prompt_tokens.len(),
                completion_tokens,
                time_to_first_token_ms: ttft_val,
                total_time_ms: total_time,
                tokens_per_second: tps,
            }
        })
    }

    fn eos_token_id(&self) -> u32 {
        ModelRunner::eos_token_id(self)
    }

    fn context_size(&self) -> usize {
        ModelRunner::context_size(self)
    }

    fn architecture(&self) -> &str {
        ModelRunner::architecture(self)
    }
}
