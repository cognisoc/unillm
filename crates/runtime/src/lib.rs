//! UniLLM Runtime
//!
//! High-performance inference runtime for large language models.
//!
//! This crate provides a clean, solid abstraction system for LLM inference
//! with support for multiple model architectures and deployment targets.

// === CORE ABSTRACTION LAYERS ===

/// Unified tensor operations and device management
pub mod tensor_core;

/// Model trait and configuration system
pub mod model_core;

/// Weight loading from various formats
pub mod weight_loader_core;

// === MODEL IMPLEMENTATIONS ===

/// Clean model implementations using solid abstractions
pub mod models_v2;

// === INFERENCE PIPELINE ===

/// Tokenization utilities
pub mod tokenizer;

/// Basic inference implementation
pub mod inference;

/// Sampling and decoding
pub mod sampler;

// === UTILITIES ===

/// Type definitions
pub mod types;

/// Simple observability
pub mod simple_observability;

/// Ollama registry client
pub mod ollama;

// === RE-EXPORTS ===

pub use tensor_core::{Tensor, Device, DataType};
pub use model_core::{Model, ModelInputs, ModelOutputs, GenerationConfig, MemoryRequirements, ModelWeights};
pub use weight_loader_core::{WeightLoader};

/// Main runtime instance
pub struct Runtime {
    _placeholder: (),
}

impl Runtime {
    /// Create a new runtime instance
    pub fn new() -> Self {
        Self {
            _placeholder: (),
        }
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}