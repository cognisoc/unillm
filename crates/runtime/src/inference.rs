//! Complete inference pipeline and advanced sampling for text generation
//!
//! This module ties together tokenization, model inference, and text generation
//! into a complete pipeline that can process text input and generate responses.
//!
//! The Sampler implements a proper sampling pipeline:
//! 1. Logit preprocessing (repetition/frequency/presence penalty)
//! 2. Temperature scaling
//! 3. Top-k filtering
//! 4. Top-p (nucleus) filtering
//! 5. Min-p filtering
//! 6. Sampling from the final distribution

use crate::model_core::{
    GenerationConfig, Model, ModelConfig, ModelError, ModelInputs, ModelResult,
};
use crate::models_v2::llama::{LlamaConfig, LlamaModelV2};
use crate::tensor_core::{Device, Tensor};
use crate::tokenizer::Tokenizer;
use std::collections::HashMap;

/// Sampling strategy for text generation
pub struct Sampler {
    _tensor_ops: crate::tensor_core::CpuTensorOpsImpl,
}

impl Sampler {
    pub fn new() -> Self {
        Self {
            _tensor_ops: crate::tensor_core::CpuTensorOpsImpl::new(),
        }
    }

    /// Greedy sampling - always pick the highest probability token
    pub fn sample_greedy(&self, logits: &[f32]) -> ModelResult<u32> {
        if logits.is_empty() {
            return Err(ModelError::ComputationFailed("Empty logits".to_string()));
        }

        let mut max_idx = 0;
        let mut max_val = logits[0];

        for (idx, &val) in logits.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = idx;
            }
        }

        Ok(max_idx as u32)
    }

    /// Temperature scaling + softmax + random sampling
    pub fn sample_temperature(&self, logits: &[f32], temperature: f32) -> ModelResult<u32> {
        if logits.is_empty() {
            return Err(ModelError::ComputationFailed("Empty logits".to_string()));
        }

        let scaled_logits: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
        let probabilities = softmax(&scaled_logits)?;
        sample_from_probabilities(&probabilities)
    }

    /// Top-k sampling: keep only top k tokens, then sample
    pub fn sample_top_k(&self, logits: &[f32], top_k: usize, temperature: f32) -> ModelResult<u32> {
        if logits.is_empty() {
            return Err(ModelError::ComputationFailed("Empty logits".to_string()));
        }

        let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

        let mut indexed: Vec<(usize, f32)> =
            scaled.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(top_k);

        let max_val = indexed[0].1;
        let exp_sum: f32 = indexed.iter().map(|(_, v)| (v - max_val).exp()).sum();
        let probs: Vec<(usize, f32)> = indexed
            .iter()
            .map(|&(i, v)| (i, (v - max_val).exp() / exp_sum))
            .collect();

        use rand::Rng;
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen();
        let mut cum = 0.0;
        for (idx, prob) in &probs {
            cum += prob;
            if r <= cum {
                return Ok(*idx as u32);
            }
        }
        Ok(probs.last().unwrap().0 as u32)
    }

    /// Top-p (nucleus) sampling: keep smallest set of tokens with cumulative prob >= top_p
    pub fn sample_top_p(&self, logits: &[f32], top_p: f32, temperature: f32) -> ModelResult<u32> {
        if logits.is_empty() {
            return Err(ModelError::ComputationFailed("Empty logits".to_string()));
        }

        let scaled_logits: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
        let probabilities = softmax(&scaled_logits)?;

        let mut indexed_probs: Vec<(usize, f32)> = probabilities
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut cumulative_prob = 0.0;
        let mut cutoff_idx = indexed_probs.len();

        for (i, (_, prob)) in indexed_probs.iter().enumerate() {
            cumulative_prob += prob;
            if cumulative_prob >= top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        let mut filtered_probs = vec![0.0f32; logits.len()];
        let mut total_prob = 0.0;

        for (idx, prob) in indexed_probs.iter().take(cutoff_idx) {
            filtered_probs[*idx] = *prob;
            total_prob += prob;
        }

        if total_prob > 0.0 {
            for prob in &mut filtered_probs {
                *prob /= total_prob;
            }
        }

        sample_from_probabilities(&filtered_probs)
    }

    /// Combined sampling pipeline with proper ordering:
    /// 1. Apply repetition/frequency/presence penalties
    /// 2. Apply temperature scaling
    /// 3. Apply top-k filtering (if set)
    /// 4. Apply top-p filtering (if set)
    /// 5. Apply min-p filtering (if set)
    /// 6. Sample from final distribution
    ///
    /// This follows the HuggingFace transformers ordering where
    /// top-k is applied before top-p, enabling combined filtering.
    pub fn sample_with_config(
        &self,
        logits: &[f32],
        config: &crate::model_core::GenerationConfig,
        generated_tokens: &[u32],
    ) -> ModelResult<u32> {
        let mut logits = logits.to_vec();

        // Step 1: Apply repetition penalty (scales logits of previously seen tokens)
        Self::apply_repetition_penalty(&mut logits, generated_tokens, config.repetition_penalty);

        // Step 2: Apply frequency and presence penalties
        Self::apply_frequency_presence_penalty(
            &mut logits,
            generated_tokens,
            config.frequency_penalty,
            config.presence_penalty,
        );

        // Step 3: Greedy if not sampling
        if !config.do_sample || config.temperature <= 0.0 {
            return self.sample_greedy(&logits);
        }

        // Step 4: Temperature scaling
        let temperature = config.temperature;
        let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

        // Step 5: Find the max logit for computing probabilities
        let max_logit = scaled.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Convert to exponentials (avoid overflow)
        let mut exp_logits: Vec<f32> = scaled.iter().map(|&x| (x - max_logit).exp()).collect();

        // Step 6: Top-k filtering (keeps top_k logits, zeros out the rest)
        if let Some(top_k) = config.top_k {
            if top_k > 0 && top_k < exp_logits.len() {
                // Find the k-th largest value using partial sort
                let mut indexed: Vec<(usize, f32)> = exp_logits
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (i, v))
                    .collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let kth_value = indexed[top_k - 1].1;
                for val in &mut exp_logits {
                    if *val < kth_value {
                        *val = 0.0;
                    }
                }
            }
        }

        // Step 7: Top-p (nucleus) filtering
        if config.top_p < 1.0 {
            let total_sum: f32 = exp_logits.iter().sum();
            if total_sum > 0.0 {
                let probs: Vec<f32> = exp_logits.iter().map(|&x| x / total_sum).collect();
                let mut indexed_probs: Vec<(usize, f32)> =
                    probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
                indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                let mut cum_sum = 0.0;
                let mut mask = vec![true; exp_logits.len()];

                for (i, (idx, prob)) in indexed_probs.iter().enumerate() {
                    cum_sum += prob;
                    mask[*idx] = true;
                    if cum_sum >= config.top_p {
                        // Zero out everything after this
                        for (_, (remaining_idx, _)) in indexed_probs[i + 1..].iter().enumerate() {
                            mask[*remaining_idx] = false;
                        }
                        break;
                    }
                }

                for (i, should_keep) in mask.iter().enumerate() {
                    if !should_keep {
                        exp_logits[i] = 0.0;
                    }
                }
            }
        }

        // Step 8: Min-p filtering (keeps tokens with prob >= min_p * max_prob)
        if config.min_p > 0.0 && config.min_p < 1.0 {
            let total_sum: f32 = exp_logits.iter().sum();
            if total_sum > 0.0 {
                let max_prob = exp_logits
                    .iter()
                    .map(|&x| x / total_sum)
                    .fold(0.0f32, |a, b| a.max(b));
                let threshold = max_prob * config.min_p;
                let probs: Vec<f32> = exp_logits.iter().map(|&x| x / total_sum).collect();
                for (i, prob) in probs.iter().enumerate() {
                    if *prob < threshold {
                        exp_logits[i] = 0.0;
                    }
                }
            }
        }

        // Step 9: Renormalize and sample
        let sum: f32 = exp_logits.iter().sum();
        if sum <= 0.0 {
            // Fallback to greedy if all tokens were filtered
            return self.sample_greedy(logits.as_slice());
        }

        let probabilities: Vec<f32> = exp_logits.iter().map(|&x| x / sum).collect();
        sample_from_probabilities(&probabilities)
    }

    /// Apply repetition penalty to logits in-place.
    /// Divides positive logits and multiplies negative logits by the penalty.
    pub fn apply_repetition_penalty(logits: &mut [f32], generated_tokens: &[u32], penalty: f32) {
        if penalty == 1.0 {
            return;
        }
        for &token_id in generated_tokens {
            let idx = token_id as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= penalty;
                } else {
                    logits[idx] *= penalty;
                }
            }
        }
    }

    /// Apply frequency and presence penalties.
    /// Frequency penalty scales by token frequency count.
    /// Presence penalty adds a constant if token appeared at all.
    pub fn apply_frequency_presence_penalty(
        logits: &mut [f32],
        generated_tokens: &[u32],
        frequency_penalty: f32,
        presence_penalty: f32,
    ) {
        if frequency_penalty == 0.0 && presence_penalty == 0.0 {
            return;
        }

        // Count token frequencies
        let mut token_counts: HashMap<u32, usize> = HashMap::new();
        for &token_id in generated_tokens {
            *token_counts.entry(token_id).or_insert(0) += 1;
        }

        for (&token_id, &count) in &token_counts {
            let idx = token_id as usize;
            if idx < logits.len() {
                // Frequency penalty: proportional to count
                logits[idx] -= frequency_penalty * count as f32;
                // Presence penalty: constant if present
                logits[idx] -= presence_penalty;
            }
        }
    }

    /// Check if any stop sequence matches the end of the decoded text.
    /// Returns the length of the matching stop sequence, or 0 if none match.
    pub fn check_stop_sequences(text: &str, stop_sequences: &[String]) -> usize {
        for stop in stop_sequences {
            if text.ends_with(stop) {
                return stop.len();
            }
        }
        0
    }

    /// Trim stop sequences from the end of generated text.
    pub fn trim_stop_sequences(text: &str, stop_sequences: &[String]) -> String {
        let mut result = text.to_string();
        for stop in stop_sequences {
            if let Some(pos) = result.rfind(stop) {
                if pos + stop.len() == result.len() {
                    result.truncate(pos);
                }
            }
        }
        result
    }
}

/// Numerically stable softmax
fn softmax(logits: &[f32]) -> ModelResult<Vec<f32>> {
    if logits.is_empty() {
        return Ok(Vec::new());
    }
    let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut exp_vals = Vec::with_capacity(logits.len());
    let mut sum = 0.0f32;
    for &logit in logits {
        let exp_val = (logit - max_val).exp();
        exp_vals.push(exp_val);
        sum += exp_val;
    }
    for exp_val in &mut exp_vals {
        *exp_val /= sum;
    }
    Ok(exp_vals)
}

/// Sample a token index from a probability distribution using CDF
fn sample_from_probabilities(probabilities: &[f32]) -> ModelResult<u32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let random_val: f32 = rng.gen();
    let mut cumulative = 0.0;

    for (idx, &prob) in probabilities.iter().enumerate() {
        cumulative += prob;
        if random_val <= cumulative {
            return Ok(idx as u32);
        }
    }

    // Fallback to last non-zero probability token
    for i in (0..probabilities.len()).rev() {
        if probabilities[i] > 0.0 {
            return Ok(i as u32);
        }
    }

    Ok((probabilities.len() - 1) as u32)
}

/// Complete inference pipeline
pub struct InferencePipeline {
    model: LlamaModelV2,
    tokenizer: Tokenizer,
    sampler: Sampler,
}

impl InferencePipeline {
    /// Create new inference pipeline
    pub fn new(model: LlamaModelV2, tokenizer: Tokenizer) -> Self {
        Self {
            model,
            tokenizer,
            sampler: Sampler::new(),
        }
    }

    /// Generate text from a prompt
    pub fn generate(&self, prompt: &str, config: &GenerationConfig) -> ModelResult<String> {
        let input_tokens = self.tokenizer.encode(prompt);
        let generated_tokens = self.generate_tokens(&input_tokens, config)?;
        let output_text = self.tokenizer.decode(&generated_tokens);
        Ok(output_text)
    }

    /// Generate token sequence using the full sampling pipeline
    fn generate_tokens(
        &self,
        input_tokens: &[u32],
        config: &GenerationConfig,
    ) -> ModelResult<Vec<u32>> {
        let mut current_tokens = input_tokens.to_vec();
        let mut generated: Vec<u32> = Vec::new();

        while generated.len() < config.max_new_tokens {
            let input_tensor = self.create_input_tensor(&current_tokens)?;
            let inputs = ModelInputs::Text {
                input_ids: input_tensor,
                attention_mask: None,
                position_ids: None,
            };

            let outputs = self.model.forward(&inputs).map_err(|e| {
                ModelError::ComputationFailed(format!("Forward pass failed: {}", e))
            })?;

            let logits = match outputs {
                crate::model_core::ModelOutputs::Logits { logits, .. } => logits,
                _ => {
                    return Err(ModelError::ComputationFailed(
                        "Expected logits output".to_string(),
                    ))
                }
            };

            let last_token_logits = self.extract_last_token_logits(&logits)?;

            // Use the full sampling pipeline
            let next_token =
                self.sampler
                    .sample_with_config(&last_token_logits, config, &generated)?;

            // Check EOS
            if next_token == config.eos_token_id {
                break;
            }

            // Check stop sequences
            if !config.stop_sequences.is_empty() {
                let mut all_tokens = current_tokens.clone();
                all_tokens.push(next_token);
                let decoded = self.tokenizer.decode(&all_tokens);
                if Sampler::check_stop_sequences(&decoded, &config.stop_sequences) > 0 {
                    break;
                }
            }

            current_tokens.push(next_token);
            generated.push(next_token);
        }

        Ok(current_tokens)
    }

    fn create_input_tensor(&self, tokens: &[u32]) -> ModelResult<Tensor> {
        let tokens_i64: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
        let shape = &[1, tokens.len()];
        Tensor::from_i64_slice(&tokens_i64, shape, &Device::CPU).map_err(|e| {
            ModelError::ComputationFailed(format!("Failed to create input tensor: {}", e))
        })
    }

    fn extract_last_token_logits(&self, logits: &Tensor) -> ModelResult<Vec<f32>> {
        let shape = logits.shape();
        if shape.len() < 2 {
            return Err(ModelError::ComputationFailed(format!(
                "Logits tensor has unexpected shape: {:?}",
                shape
            )));
        }

        let candle_logits = logits.to_candle().map_err(|e| {
            ModelError::ComputationFailed(format!("Failed to convert logits: {}", e))
        })?;

        let vocab_size = if shape.len() == 3 { shape[2] } else { shape[1] };

        let last_logits = if shape.len() == 3 {
            let seq_len = shape[1];
            candle_logits
                .narrow(1, seq_len - 1, 1)
                .map_err(|e| ModelError::ComputationFailed(format!("narrow failed: {}", e)))?
                .squeeze(1)
                .map_err(|e| ModelError::ComputationFailed(format!("squeeze failed: {}", e)))?
                .squeeze(0)
                .map_err(|e| ModelError::ComputationFailed(format!("squeeze failed: {}", e)))?
        } else {
            let seq_len = shape[0];
            candle_logits
                .narrow(0, seq_len - 1, 1)
                .map_err(|e| ModelError::ComputationFailed(format!("narrow failed: {}", e)))?
                .squeeze(0)
                .map_err(|e| ModelError::ComputationFailed(format!("squeeze failed: {}", e)))?
        };

        let logits_vec: Vec<f32> = last_logits.to_vec1().map_err(|e| {
            ModelError::ComputationFailed(format!("Failed to convert to vec: {}", e))
        })?;

        if logits_vec.len() != vocab_size {
            return Err(ModelError::ComputationFailed(format!(
                "Logits size mismatch: got {}, expected {}",
                logits_vec.len(),
                vocab_size
            )));
        }

        Ok(logits_vec)
    }

    pub fn model_config(&self) -> &LlamaConfig {
        self.model.config()
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

/// Builder for creating inference pipelines
pub struct InferencePipelineBuilder {
    model_config: Option<LlamaConfig>,
    tokenizer: Option<Tokenizer>,
}

impl InferencePipelineBuilder {
    pub fn new() -> Self {
        Self {
            model_config: None,
            tokenizer: None,
        }
    }

    pub fn with_model_config(mut self, config: LlamaConfig) -> Self {
        self.model_config = Some(config);
        self
    }

    pub fn with_tokenizer(mut self, tokenizer: Tokenizer) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    pub fn build(self) -> ModelResult<InferencePipeline> {
        let config = self.model_config.ok_or_else(|| {
            ModelError::InitializationFailed("Model config is required".to_string())
        })?;
        let tokenizer = self
            .tokenizer
            .ok_or_else(|| ModelError::InitializationFailed("Tokenizer is required".to_string()))?;
        let model = LlamaModelV2::new(config)
            .map_err(|e| ModelError::InitializationFailed(e.to_string()))?;

        Ok(InferencePipeline::new(model, tokenizer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampler_greedy() {
        let sampler = Sampler::new();
        let logits = vec![0.1, 0.5, 0.9, 0.3];
        let token = sampler.sample_greedy(&logits).unwrap();
        assert_eq!(token, 2);
    }

    #[test]
    fn test_sampler_temperature() {
        let sampler = Sampler::new();
        let logits = vec![0.1, 0.2, 10.0, 0.3];
        // With very low temperature, should pick the highest logit
        let config = GenerationConfig {
            temperature: 0.01,
            do_sample: true,
            ..Default::default()
        };
        let token = sampler.sample_with_config(&logits, &config, &[]).unwrap();
        assert_eq!(token, 2);
    }

    #[test]
    fn test_sampler_top_k() {
        let sampler = Sampler::new();
        let logits = vec![0.1, 10.0, 0.3, 0.2];
        let config = GenerationConfig {
            temperature: 0.01,
            top_k: Some(2),
            do_sample: true,
            ..Default::default()
        };
        // With very low temp and top_k=2, should pick index 1 (highest)
        let token = sampler.sample_with_config(&logits, &config, &[]).unwrap();
        assert_eq!(token, 1);
    }

    #[test]
    fn test_repetition_penalty() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        let generated = vec![2u32]; // Token 2 was already generated
        Sampler::apply_repetition_penalty(&mut logits, &generated, 2.0);
        // Token 2 has positive logit (3.0), so it should be divided by 2.0
        assert!(logits[2] < 3.0);
        // Other tokens should be unchanged
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], 2.0);
        assert_eq!(logits[3], 4.0);
    }

    #[test]
    fn test_frequency_presence_penalty() {
        let mut logits = vec![1.0, 1.0, 1.0, 1.0];
        let generated = vec![0u32, 0u32, 1u32]; // Token 0 appeared 2x, token 1 appeared 1x
        Sampler::apply_frequency_presence_penalty(&mut logits, &generated, 0.5, 0.1);
        // Token 0: 1.0 - 0.5*2 - 0.1 = -0.1
        assert!((logits[0] - (-0.1)).abs() < 0.001);
        // Token 1: 1.0 - 0.5*1 - 0.1 = 0.4
        assert!((logits[1] - 0.4).abs() < 0.001);
        // Token 2 & 3: unchanged
        assert_eq!(logits[2], 1.0);
        assert_eq!(logits[3], 1.0);
    }

    #[test]
    fn test_combined_top_k_and_top_p() {
        let sampler = Sampler::new();
        // When both top_k and top_p are set, top_k should filter first
        let logits = vec![0.0; 100];
        let config = GenerationConfig {
            temperature: 1.0,
            top_k: Some(10),
            top_p: 0.9,
            do_sample: true,
            ..Default::default()
        };
        // This should not panic - the pipeline should handle combined top-k + top-p
        let result = sampler.sample_with_config(&logits, &config, &[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_min_p_filtering() {
        let sampler = Sampler::new();
        // With a dominant token and min-p=0.5, only the dominant token
        // and near-dominant tokens should survive
        let logits = vec![10.0, 1.0, 0.5, 0.01, 0.01];
        let config = GenerationConfig {
            temperature: 1.0,
            min_p: 0.5,
            do_sample: true,
            ..Default::default()
        };
        // Should prefer the dominant token (index 0)
        let result = sampler.sample_with_config(&logits, &config, &[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_stop_sequences() {
        let text = "Hello world\n";
        assert_eq!(Sampler::check_stop_sequences(text, &["\n".to_string()]), 1);
        assert_eq!(
            Sampler::check_stop_sequences(text, &["STOP".to_string()]),
            0
        );

        let text2 = "Hello world</s>More text</s>";
        assert_eq!(
            Sampler::check_stop_sequences(text2, &["</s>".to_string()]),
            4
        );

        let trimmed = Sampler::trim_stop_sequences("Hello world\n", &["\n".to_string()]);
        assert_eq!(trimmed, "Hello world");
    }
}
