//! Benchmark comparison module for UniLLM vs llama.cpp
//!
//! This module provides infrastructure for comparing inference performance
//! between UniLLM and llama.cpp using the same GGUF model files.

mod metrics;
mod runner;
mod unillm_backend;

#[cfg(feature = "benchmark")]
mod llama_cpp_backend;

pub use metrics::*;
pub use runner::*;
pub use unillm_backend::*;

#[cfg(feature = "benchmark")]
pub use llama_cpp_backend::*;
