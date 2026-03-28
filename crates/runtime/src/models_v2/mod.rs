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
pub mod qwen;
pub mod gemma;
pub mod phi;
pub mod deepseek;
pub mod mistral;
pub mod mixtral;

// GPT family models
pub mod gpt2;
pub mod gptj;
pub mod gptneox;
pub mod opt;
pub mod bloom;
pub mod mpt;

// Code models
pub mod starcoder;
pub mod codellama;

// Standard decoder models
pub mod olmo;
pub mod granite;

// Additional model families
pub mod yi;
pub mod falcon;
pub mod baichuan;
pub mod internlm;
pub mod chatglm;
pub mod bert;

// Specialized architectures
pub mod t5;
pub mod whisper;
pub mod clip;
pub mod llava;
pub mod mamba;
pub mod minicpm;

// MoE (Mixture of Experts) models
pub mod deepseek_moe;
pub mod dbrx;
pub mod grok;
pub mod arctic;
pub mod jamba;

// RWKV / Linear Attention family
pub mod rwkv4;
pub mod rwkv6;
pub mod recurrent_gemma;

// Vision-Language models
pub mod qwen2_vl;
pub mod phi3_vision;
pub mod internvl;
pub mod cogvlm;
pub mod idefics;
pub mod florence;

// Audio/Speech models
pub mod wav2vec2;
pub mod hubert;
pub mod musicgen;
pub mod encodec;

pub mod traits {
    //! Common traits and utilities for V2 model implementations

    pub use crate::model_core::*;
    pub use crate::tensor_core::*;
    pub use crate::weight_loader_core::*;
}