//! InferenceService — thin service layer bridging the HTTP server to model execution.
//!
//! This replaces direct `ModelRunner` usage in the server, providing:
//! - Unified generation API with proper stop sequence handling
//! - Streaming generation with per-token stop sequence checking
//! - Generation metrics collection
//! - Concurrency limiting via semaphore

use crate::chat_template::ChatTemplateFormat;
use crate::inference::Sampler;
use crate::model_core::{GenerationConfig, GenerationMetrics, ModelRunner};
use crate::tokenizer::Tokenizer;
use std::time::Instant;

/// Result of a non-streaming generation call.
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: FinishReason,
    pub metrics: GenerationMetrics,
}

/// Why generation stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// Natural end-of-sequence token
    Stop,
    /// Hit max_new_tokens limit
    Length,
    /// Matched a user-specified stop sequence
    StopSequence(String),
}

/// Shared service state used by the HTTP server.
pub struct InferenceService {
    model: Box<dyn ModelRunner>,
    tokenizer: Tokenizer,
    chat_template: ChatTemplateFormat,
    model_name: String,
    context_size: usize,
}

impl InferenceService {
    pub fn new(
        model: Box<dyn ModelRunner>,
        tokenizer: Tokenizer,
        chat_template: ChatTemplateFormat,
        model_name: String,
        context_size: usize,
        max_concurrency: usize,
    ) -> Self {
        let _ = max_concurrency; // Concurrency handled by server semaphore
        Self {
            model,
            tokenizer,
            chat_template,
            model_name,
            context_size,
        }
    }

    /// Encode a prompt string into token IDs with special tokens.
    pub fn encode_prompt(&self, prompt: &str, add_bos: bool, add_eos: bool) -> Vec<u32> {
        self.tokenizer
            .encode_with_special_tokens(prompt, add_bos, add_eos)
    }

    /// Render chat messages into a prompt string.
    pub fn render_chat(&self, messages: &[(String, String)]) -> String {
        let chat_messages: Vec<crate::chat_template::ChatMessage> = messages
            .iter()
            .map(|(role, content)| crate::chat_template::ChatMessage {
                role: role.clone(),
                content: content.clone(),
            })
            .collect();
        self.chat_template.render(&chat_messages)
    }

    /// Build a GenerationConfig from server request parameters, including stop sequences.
    pub fn make_gen_config(
        &self,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
        min_p: Option<f32>,
        repetition_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        stop_sequences: Vec<String>,
        do_sample: bool,
    ) -> GenerationConfig {
        GenerationConfig {
            max_new_tokens: max_tokens,
            temperature,
            top_p,
            top_k,
            min_p: min_p.unwrap_or(0.0),
            do_sample,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            stop_sequences,
            eos_token_id: self.model.eos_token_id(),
            pad_token_id: 0,
        }
    }

    /// Non-streaming generation: returns the complete output.
    /// Properly checks stop sequences during generation.
    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        config: &GenerationConfig,
    ) -> anyhow::Result<GenerationOutput> {
        let start = Instant::now();
        let mut ttft: Option<f64> = None;
        let mut output_text = String::new();
        let mut token_ids = Vec::new();

        let all_ids =
            self.model
                .generate_streaming(prompt_tokens, config, &mut |token_id, text| {
                    if ttft.is_none() {
                        ttft = Some(start.elapsed().as_secs_f64() * 1000.0);
                    }
                    token_ids.push(token_id);
                    output_text.push_str(text);

                    // Check stop sequences during generation
                    if !config.stop_sequences.is_empty() {
                        if Sampler::check_stop_sequences(&output_text, &config.stop_sequences) > 0 {
                            return false;
                        }
                    }

                    true
                })?;

        // Trim stop sequences from output
        let finish_reason = if !config.stop_sequences.is_empty() {
            if let Some(matched) = find_matching_stop_sequence(&output_text, &config.stop_sequences)
            {
                output_text = Sampler::trim_stop_sequences(&output_text, &config.stop_sequences);
                FinishReason::StopSequence(matched)
            } else if token_ids.len() >= config.max_new_tokens {
                FinishReason::Length
            } else {
                FinishReason::Stop
            }
        } else if token_ids.len() >= config.max_new_tokens {
            FinishReason::Length
        } else {
            FinishReason::Stop
        };

        let total_ms = start.elapsed().as_secs_f64() * 1000.0;
        let gen_ms = ttft.map(|t| total_ms - t).unwrap_or(total_ms);
        let prompt_tokens = prompt_tokens.len();

        let mut metrics = GenerationMetrics::new();
        metrics.model_load_ms = 0.0;
        metrics.prompt_tokens = prompt_tokens;
        metrics.prompt_eval_ms = ttft.unwrap_or(total_ms);
        metrics.prompt_tokens_per_sec = if metrics.prompt_eval_ms > 0.0 {
            prompt_tokens as f64 / (metrics.prompt_eval_ms / 1000.0)
        } else {
            0.0
        };
        metrics.generated_tokens = token_ids.len();
        metrics.generation_ms = gen_ms;
        metrics.generation_tokens_per_sec = if gen_ms > 0.0 {
            token_ids.len() as f64 / (gen_ms / 1000.0)
        } else {
            0.0
        };
        metrics.total_ms = total_ms;

        Ok(GenerationOutput {
            text: output_text,
            token_ids: all_ids,
            prompt_tokens,
            completion_tokens: token_ids.len(),
            finish_reason,
            metrics,
        })
    }

    /// Streaming generation: yields tokens one-by-one via a callback.
    /// Checks stop sequences on each token and stops when matched.
    /// Returns (final_text, token_ids, prompt_tokens, completion_tokens, finish_reason).
    pub fn generate_streaming(
        &self,
        prompt_tokens: &[u32],
        config: &GenerationConfig,
        on_token: &mut dyn FnMut(u32, &str) -> bool,
    ) -> anyhow::Result<StreamingOutput> {
        let start = Instant::now();
        let mut ttft: Option<f64> = None;
        let mut completed_text = String::new();
        let mut token_ids = Vec::new();
        let mut stop_reason = FinishReason::Stop;

        let all_ids =
            self.model
                .generate_streaming(prompt_tokens, config, &mut |token_id, text| {
                    if ttft.is_none() {
                        ttft = Some(start.elapsed().as_secs_f64() * 1000.0);
                    }
                    token_ids.push(token_id);
                    completed_text.push_str(text);

                    // Check stop sequences on every token during streaming
                    if !config.stop_sequences.is_empty() {
                        if Sampler::check_stop_sequences(&completed_text, &config.stop_sequences)
                            > 0
                        {
                            stop_reason = FinishReason::StopSequence(
                                find_matching_stop_sequence(
                                    &completed_text,
                                    &config.stop_sequences,
                                )
                                .unwrap_or_default(),
                            );
                            // Still call the callback with this text so the server can
                            // send it, but we'll trim the stop sequence from the final output.
                            let _should_continue = on_token(token_id, text);
                            return false; // Signal the model to stop generating
                        }
                    }

                    // Check if we're about to hit max_new_tokens
                    if token_ids.len() >= config.max_new_tokens {
                        stop_reason = FinishReason::Length;
                        let _should_continue = on_token(token_id, text);
                        return false;
                    }

                    on_token(token_id, text)
                })?;

        // Trim stop sequences from final text
        if matches!(stop_reason, FinishReason::StopSequence(_)) {
            completed_text = Sampler::trim_stop_sequences(&completed_text, &config.stop_sequences);
        }

        let total_ms = start.elapsed().as_secs_f64() * 1000.0;
        let gen_ms = ttft.map(|t| total_ms - t).unwrap_or(total_ms);

        Ok(StreamingOutput {
            text: completed_text,
            token_ids: all_ids,
            prompt_tokens: prompt_tokens.len(),
            completion_tokens: token_ids.len(),
            finish_reason: stop_reason,
            time_to_first_token_ms: ttft.unwrap_or(0.0),
            total_time_ms: total_ms,
            generation_ms: gen_ms,
        })
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn context_size(&self) -> usize {
        self.context_size
    }

    pub fn eos_token_id(&self) -> u32 {
        self.model.eos_token_id()
    }

    pub fn architecture(&self) -> &str {
        self.model.architecture()
    }
}

/// Output of a streaming generation call.
pub struct StreamingOutput {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: FinishReason,
    pub time_to_first_token_ms: f64,
    pub total_time_ms: f64,
    pub generation_ms: f64,
}

/// Find the first matching stop sequence in the text, returning the sequence itself.
fn find_matching_stop_sequence(text: &str, stop_sequences: &[String]) -> Option<String> {
    for stop in stop_sequences {
        if text.ends_with(stop) {
            return Some(stop.clone());
        }
    }
    None
}
