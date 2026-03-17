//! Model architectures for UniLLM
//!
//! Fully functional model implementations with working inference capabilities:
//! - Llama: Complete CPU implementation with real computation
//! - LLaVA: Vision-language model for multimodal understanding
//! - More architectures coming soon

pub mod llama;
pub mod llava;
pub mod traits;

// Re-export the main types
pub use llama::LlamaModel;
pub use llava::{LLaVAModel, LLaVAConfig, VisionEncoder};
pub use traits::*;

// TODO: Add these models as they are implemented
// pub mod mistral;
// pub mod qwen;
// pub mod gemma;
// pub mod phi;
// pub mod chatglm;
// pub mod baichuan;
// pub mod deepseek;
// pub mod cohere;
// pub mod mamba;
// pub mod mixtral;
// pub mod gptneox;
// pub mod bloom;
// pub mod falcon;
// pub mod arctic;
// pub mod jamba;
// pub mod internlm;
// pub mod exaone;
// pub mod granite;
// pub mod nemotron;
// pub mod olmo;
// pub mod solar;
// pub mod starcoder;
// pub mod xverse;
// pub mod minicpm;