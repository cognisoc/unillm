//! llama.cpp inference backend for benchmarking
//!
//! This module is only compiled when the "benchmark" feature is enabled.
//! Requires llama-cpp-2 crate and system dependencies (clang, cmake).

use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;

use super::{GenerationResult, InferenceBackend};

/// llama.cpp inference backend
pub struct LlamaCppBackend {
    backend: Option<LlamaBackend>,
    model: Option<LlamaModel>,
    n_ctx: u32,
}

impl LlamaCppBackend {
    /// Create new llama.cpp backend
    pub fn new() -> Self {
        Self {
            backend: None,
            model: None,
            n_ctx: 2048,
        }
    }

    /// Create with custom context length
    pub fn with_context_length(mut self, n_ctx: u32) -> Self {
        self.n_ctx = n_ctx;
        self
    }
}

impl Default for LlamaCppBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for LlamaCppBackend {
    fn name(&self) -> &str {
        "llama.cpp"
    }

    fn load_model(&mut self, path: &Path) -> Result<f64> {
        let start = Instant::now();

        // Initialize backend
        let backend = LlamaBackend::init()?;

        // Load model
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, path, &model_params)?;

        self.backend = Some(backend);
        self.model = Some(model);

        Ok(start.elapsed().as_secs_f64() * 1000.0)
    }

    fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<GenerationResult> {
        let backend = self
            .backend
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Backend not initialized"))?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model not loaded"))?;

        // Create context
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(self.n_ctx))
            .with_n_batch(512);
        let mut ctx = model.new_context(backend, ctx_params)?;

        // Tokenize prompt
        let tokens = model.str_to_token(prompt, llama_cpp_2::model::AddBos::Always)?;
        let prompt_tokens = tokens.len();

        // Create batch for prompt processing
        let mut batch = LlamaBatch::new(512, 1);

        // Add prompt tokens to batch
        let last_idx = tokens.len() - 1;
        for (i, token) in tokens.iter().enumerate() {
            batch.add(*token, i as i32, &[0], i == last_idx)?;
        }

        // Timing
        let start = Instant::now();
        let mut first_token_time: Option<std::time::Duration> = None;

        // Process prompt (prefill)
        ctx.decode(&mut batch)?;

        // Record TTFT after prefill
        first_token_time = Some(start.elapsed());

        // Generation loop
        let mut generated_tokens = Vec::new();
        let mut n_cur = tokens.len();

        for _ in 0..max_tokens {
            // Sample next token
            let candidates = LlamaTokenDataArray::from_iter(
                ctx.candidates_ith(batch.n_tokens() - 1),
                false,
            );

            // Greedy sampling - get top token
            let new_token_id = candidates
                .data
                .iter()
                .max_by(|a, b| a.logit().partial_cmp(&b.logit()).unwrap())
                .map(|t| t.id())
                .ok_or_else(|| anyhow::anyhow!("No candidates"))?;

            // Check for EOS
            if model.is_eog_token(new_token_id) {
                break;
            }

            generated_tokens.push(new_token_id);

            // Prepare next batch
            batch.clear();
            batch.add(new_token_id, n_cur as i32, &[0], true)?;
            n_cur += 1;

            // Decode
            ctx.decode(&mut batch)?;
        }

        let total_time = start.elapsed();

        // Decode output text
        let output_text = generated_tokens
            .iter()
            .map(|&t| model.token_to_str(t, llama_cpp_2::model::Special::Tokenize))
            .collect::<Result<Vec<_>, _>>()?
            .join("");

        Ok(GenerationResult {
            output_text,
            tokens_generated: generated_tokens.len(),
            prompt_tokens,
            time_to_first_token_ms: first_token_time
                .map(|t| t.as_secs_f64() * 1000.0)
                .unwrap_or(0.0),
            total_time_ms: total_time.as_secs_f64() * 1000.0,
        })
    }

    fn memory_usage(&self) -> u64 {
        super::runner::get_process_memory()
    }

    fn unload(&mut self) {
        self.model = None;
        self.backend = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_creation() {
        let backend = LlamaCppBackend::new();
        assert_eq!(backend.name(), "llama.cpp");
    }
}
