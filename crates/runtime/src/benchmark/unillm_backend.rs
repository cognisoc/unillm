//! UniLLM inference backend for benchmarking

use std::path::Path;
use std::time::Instant;

use anyhow::Result;

use crate::kv_cache::KVCache;
use crate::model_core::{Model, ModelInputs, ModelOutputs};
use crate::models_v2::llama::{LlamaConfig, LlamaModelV2};
use crate::tensor_core::{Device, Tensor};
use crate::tokenizer::Tokenizer;
use crate::weight_loader_core::UnifiedWeightLoader;

use super::{GenerationResult, InferenceBackend};

/// UniLLM inference backend
pub struct UniLLMBackend {
    model: Option<LlamaModelV2>,
    tokenizer: Option<Tokenizer>,
    device: Device,
}

impl UniLLMBackend {
    /// Create new UniLLM backend
    pub fn new() -> Self {
        Self {
            model: None,
            tokenizer: None,
            device: Device::CPU,
        }
    }
}

impl Default for UniLLMBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for UniLLMBackend {
    fn name(&self) -> &str {
        "UniLLM"
    }

    fn load_model(&mut self, path: &Path) -> Result<f64> {
        let start = Instant::now();

        // Load weights
        let loader = UnifiedWeightLoader::new();
        let weights = loader.load_weights(path)?;

        // Create config from GGUF metadata
        let config = if let Some(ref gguf_config) = weights.gguf_config {
            LlamaConfig::from_gguf_config(gguf_config)
        } else {
            LlamaConfig::default()
        };

        // Create tokenizer
        self.tokenizer = Some(Tokenizer::from_model_weights(&weights)?);

        // Create model
        self.model = Some(LlamaModelV2::from_weights(config, weights)?);

        Ok(start.elapsed().as_secs_f64() * 1000.0)
    }

    fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<GenerationResult> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model not loaded"))?;
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Tokenizer not loaded"))?;

        // Encode prompt
        let prompt_tokens_vec: Vec<u32> = tokenizer.encode_with_special_tokens(prompt, true, false);
        let prompt_tokens = prompt_tokens_vec.len();
        let mut tokens = prompt_tokens_vec.clone();

        // Initialize KV cache
        let num_layers = model.config().num_hidden_layers;
        let mut cache = KVCache::new(num_layers);

        // Track timing
        let start = Instant::now();
        let mut first_token_time: Option<std::time::Duration> = None;

        // === PREFILL PHASE ===
        // Process entire prompt at once
        let prompt_i64: Vec<i64> = prompt_tokens_vec.iter().map(|&t| t as i64).collect();
        let prompt_tensor = Tensor::from_i64_slice(&prompt_i64, &[1, prompt_tokens], &self.device)?;
        let inputs = ModelInputs::Text {
            input_ids: prompt_tensor,
            attention_mask: None,
            position_ids: None,
        };

        let outputs = model.forward_with_cache(&inputs, Some(&mut cache))?;

        // Record time to first token (after prefill)
        first_token_time = Some(start.elapsed());

        // Get logits and sample first new token
        let logits = match outputs {
            ModelOutputs::Logits { logits, .. } => logits,
            _ => return Err(anyhow::anyhow!("Expected logits output")),
        };

        let logits_candle = logits.to_candle()?;
        let shape = logits_candle.dims();
        let seq_len = if shape.len() == 3 { shape[1] } else { shape[0] };
        let last_logits = if shape.len() == 3 {
            logits_candle.narrow(1, seq_len - 1, 1)?.squeeze(1)?.squeeze(0)?
        } else {
            logits_candle.narrow(0, seq_len - 1, 1)?.squeeze(0)?
        };

        // Greedy sampling
        let logits_vec: Vec<f32> = last_logits.to_vec1()?;
        let mut max_idx = 0;
        let mut max_val = logits_vec[0];
        for (idx, &val) in logits_vec.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = idx;
            }
        }
        let mut next_token = max_idx as u32;

        // Check EOS
        if next_token == tokenizer.eos_token_id() {
            let total_time = start.elapsed();
            return Ok(GenerationResult {
                output_text: String::new(),
                tokens_generated: 0,
                prompt_tokens,
                time_to_first_token_ms: first_token_time.map(|t| t.as_secs_f64() * 1000.0).unwrap_or(0.0),
                total_time_ms: total_time.as_secs_f64() * 1000.0,
            });
        }

        tokens.push(next_token);

        // === DECODE PHASE ===
        // Generate tokens one at a time using cache
        for _ in 1..max_tokens {
            // Create input tensor for SINGLE new token
            let input_tensor = Tensor::from_i64_slice(
                &[next_token as i64],
                &[1, 1],
                &self.device
            )?;

            let inputs = ModelInputs::Text {
                input_ids: input_tensor,
                attention_mask: None,
                position_ids: None,
            };

            // Forward with cache - only processes the new token!
            let outputs = model.forward_with_cache(&inputs, Some(&mut cache))?;

            // Get logits
            let logits = match outputs {
                ModelOutputs::Logits { logits, .. } => logits,
                _ => return Err(anyhow::anyhow!("Expected logits output")),
            };

            let logits_candle = logits.to_candle()?;
            let last_logits = logits_candle.squeeze(0)?.squeeze(0)?;

            // Greedy sampling
            let logits_vec: Vec<f32> = last_logits.to_vec1()?;
            let mut max_idx = 0;
            let mut max_val = logits_vec[0];
            for (idx, &val) in logits_vec.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_idx = idx;
                }
            }
            next_token = max_idx as u32;

            // Check EOS
            if next_token == tokenizer.eos_token_id() {
                break;
            }

            tokens.push(next_token);
        }

        let total_time = start.elapsed();
        let tokens_generated = tokens.len() - prompt_tokens;

        // Decode output
        let output_text = tokenizer.decode(&tokens[prompt_tokens..]);

        Ok(GenerationResult {
            output_text,
            tokens_generated,
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
        self.tokenizer = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_creation() {
        let backend = UniLLMBackend::new();
        assert_eq!(backend.name(), "UniLLM");
    }
}
