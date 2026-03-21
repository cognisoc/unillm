//! Models V2 - Clean implementations using solid abstractions
//!
//! This module contains model implementations that use our new solid abstractions:
//! - TensorCore for unified tensor operations
//! - ModelCore for clean model interfaces
//! - WeightLoaderCore for format-agnostic weight loading
//!
//! All models in this module implement the Model trait and use consistent patterns.

// Core model families (completed migration) - WORKING MODELS
pub mod llama;
// pub mod qwen;
// pub mod gemma;
// pub mod phi;
// pub mod deepseek;

// Additional model families - TEMPORARILY DISABLED while fixing compilation
// pub mod yi;
// pub mod baichuan;
// pub mod internlm;
// pub mod chatglm;
// pub mod falcon;
// pub mod bert;

// Specialized architectures - TEMPORARILY DISABLED while fixing compilation
// pub mod t5;
// pub mod whisper;
// pub mod clip;
// pub mod llava;
// pub mod mamba;
// pub mod minicpm;

pub mod traits {
    //! Common traits and utilities for V2 model implementations

    pub use crate::model_core::*;
    pub use crate::tensor_core::*;
    pub use crate::weight_loader_core::*;
}