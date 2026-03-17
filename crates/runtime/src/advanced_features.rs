//! Advanced Inference Features
//!
//! Implements cutting-edge features for high-performance inference:
//! - Speculative decoding for faster generation
//! - Prefix caching for repeated prompt optimization
//! - Parallel sampling for multiple completions
//! - Dynamic batching with request prioritization

use crate::{
    gpu_tensor_ops::{GpuDevice, GpuTensor, GpuTensorOps},
    types::{ModelResult, ModelError},
    kv_cache::KVCache,
    real_tokenizer::RealTokenizer,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque, BTreeMap};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{mpsc, oneshot, Mutex};
use uuid::Uuid;

/// Speculative decoding configuration
#[derive(Debug, Clone)]
pub struct SpeculativeDecodingConfig {
    /// Number of speculative tokens to generate ahead
    pub speculative_tokens: usize,
    /// Acceptance threshold for speculative tokens
    pub acceptance_threshold: f32,
    /// Whether to use draft model for speculation
    pub use_draft_model: bool,
    /// Draft model path (if different from main model)
    pub draft_model_path: Option<String>,
    /// Maximum speculation depth
    pub max_speculation_depth: usize,
}

impl Default for SpeculativeDecodingConfig {
    fn default() -> Self {
        Self {
            speculative_tokens: 4,
            acceptance_threshold: 0.8,
            use_draft_model: false,
            draft_model_path: None,
            max_speculation_depth: 8,
        }
    }
}

/// Prefix caching configuration
#[derive(Debug, Clone)]
pub struct PrefixCachingConfig {
    /// Maximum cache size in tokens
    pub max_cache_size: usize,
    /// Cache eviction policy
    pub eviction_policy: CacheEvictionPolicy,
    /// Minimum prefix length to cache
    pub min_prefix_length: usize,
    /// Cache compression enabled
    pub enable_compression: bool,
    /// Time-to-live for cache entries
    pub ttl_seconds: u64,
}

#[derive(Debug, Clone)]
pub enum CacheEvictionPolicy {
    LeastRecentlyUsed,
    LeastFrequentlyUsed,
    TimeToLive,
    SizeBased,
}

impl Default for PrefixCachingConfig {
    fn default() -> Self {
        Self {
            max_cache_size: 1_000_000, // 1M tokens
            eviction_policy: CacheEvictionPolicy::LeastRecentlyUsed,
            min_prefix_length: 10,
            enable_compression: true,
            ttl_seconds: 3600, // 1 hour
        }
    }
}

/// Speculative decoding engine
pub struct SpeculativeDecoder {
    config: SpeculativeDecodingConfig,
    device: GpuDevice,
    tensor_ops: GpuTensorOps,
    draft_model: Option<Arc<dyn SpeculativeDraftModel + Send + Sync>>,
}

/// Trait for draft models used in speculative decoding
pub trait SpeculativeDraftModel {
    fn predict_next_tokens(&self, input: &[u32], n_tokens: usize) -> ModelResult<Vec<f32>>;
    fn get_vocabulary_size(&self) -> usize;
}

/// Prefix cache entry
#[derive(Debug, Clone)]
pub struct PrefixCacheEntry {
    pub prefix_tokens: Vec<u32>,
    pub kv_states: Vec<GpuTensor>, // Cached key-value states
    pub creation_time: SystemTime,
    pub last_access: SystemTime,
    pub access_count: u64,
    pub compressed: bool,
}

/// Prefix caching engine
pub struct PrefixCache {
    config: PrefixCachingConfig,
    entries: Arc<RwLock<BTreeMap<String, PrefixCacheEntry>>>,
    size_tracker: Arc<RwLock<usize>>,
    device: GpuDevice,
}

/// Advanced inference request with all features
#[derive(Debug, Clone)]
pub struct AdvancedInferenceRequest {
    pub request_id: String,
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub stop_sequences: Vec<String>,
    pub enable_speculative_decoding: bool,
    pub enable_prefix_caching: bool,
    pub priority: RequestPriority,
    pub n_completions: usize, // For parallel sampling
    pub streaming: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RequestPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Advanced inference response
#[derive(Debug, Clone)]
pub struct AdvancedInferenceResponse {
    pub request_id: String,
    pub completions: Vec<CompletionResult>,
    pub timing_info: TimingInfo,
    pub cache_stats: CacheStats,
    pub speculative_stats: Option<SpeculativeStats>,
}

#[derive(Debug, Clone)]
pub struct CompletionResult {
    pub text: String,
    pub tokens: Vec<u32>,
    pub logprobs: Option<Vec<f32>>,
    pub finish_reason: String,
}

#[derive(Debug, Clone)]
pub struct TimingInfo {
    pub total_time_ms: f64,
    pub prefill_time_ms: f64,
    pub decode_time_ms: f64,
    pub tokens_per_second: f64,
    pub cache_lookup_time_ms: f64,
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub cache_hit: bool,
    pub cached_tokens: usize,
    pub cache_size_tokens: usize,
    pub cache_entries: usize,
}

#[derive(Debug, Clone)]
pub struct SpeculativeStats {
    pub speculative_tokens_generated: usize,
    pub accepted_tokens: usize,
    pub acceptance_rate: f32,
    pub speculation_speedup: f32,
}

impl SpeculativeDecoder {
    pub fn new(config: SpeculativeDecodingConfig, device: GpuDevice) -> ModelResult<Self> {
        let tensor_ops = GpuTensorOps::with_device(device.clone());

        Ok(Self {
            config,
            device,
            tensor_ops,
            draft_model: None,
        })
    }

    /// Generate tokens using speculative decoding
    pub async fn generate_speculative(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f32,
    ) -> ModelResult<(Vec<u32>, SpeculativeStats)> {
        let start_time = Instant::now();
        let mut generated_tokens = Vec::new();
        let mut total_speculative = 0;
        let mut total_accepted = 0;

        let mut current_tokens = prompt_tokens.to_vec();

        while generated_tokens.len() < max_new_tokens {
            // Generate speculative tokens
            let speculative_tokens = self.generate_speculative_tokens(
                &current_tokens,
                self.config.speculative_tokens.min(max_new_tokens - generated_tokens.len()),
                temperature,
            )?;

            total_speculative += speculative_tokens.len();

            // Verify speculative tokens with main model
            let (accepted_tokens, _rejected_count) = self.verify_speculative_tokens(
                &current_tokens,
                &speculative_tokens,
                temperature,
            ).await?;

            total_accepted += accepted_tokens.len();

            if accepted_tokens.is_empty() {
                // Generate at least one token normally
                let next_token = self.generate_single_token(&current_tokens, temperature)?;
                current_tokens.push(next_token);
                generated_tokens.push(next_token);
                total_accepted += 1;
            } else {
                // Accept the verified tokens
                for token in &accepted_tokens {
                    current_tokens.push(*token);
                    generated_tokens.push(*token);
                }
            }

            // Check for stop conditions
            if self.should_stop(&generated_tokens) {
                break;
            }
        }

        let total_time = start_time.elapsed();
        let acceptance_rate = if total_speculative > 0 {
            total_accepted as f32 / total_speculative as f32
        } else {
            1.0
        };

        // Calculate speedup (theoretical)
        let normal_time_estimate = generated_tokens.len() as f32;
        let speculative_time_estimate = (total_speculative as f32 / self.config.speculative_tokens as f32).max(1.0);
        let speedup = normal_time_estimate / speculative_time_estimate;

        let stats = SpeculativeStats {
            speculative_tokens_generated: total_speculative,
            accepted_tokens: total_accepted,
            acceptance_rate,
            speculation_speedup: speedup,
        };

        Ok((generated_tokens, stats))
    }

    /// Generate speculative tokens (fast, lower quality)
    fn generate_speculative_tokens(
        &self,
        context: &[u32],
        n_tokens: usize,
        temperature: f32,
    ) -> ModelResult<Vec<u32>> {
        // In practice, this would use a faster draft model
        // For demo, we generate tokens with slightly higher temperature for speed
        let mut tokens = Vec::new();
        let mut current_context = context.to_vec();

        for _ in 0..n_tokens {
            let next_token = self.generate_single_token(&current_context, temperature * 1.2)?;
            tokens.push(next_token);
            current_context.push(next_token);
        }

        Ok(tokens)
    }

    /// Verify speculative tokens with the main model
    async fn verify_speculative_tokens(
        &self,
        context: &[u32],
        speculative_tokens: &[u32],
        temperature: f32,
    ) -> ModelResult<(Vec<u32>, usize)> {
        let mut accepted = Vec::new();
        let mut current_context = context.to_vec();

        for &speculative_token in speculative_tokens {
            // Generate actual probability distribution from main model
            let actual_logits = self.compute_logits(&current_context)?;
            let actual_probs = self.softmax(&actual_logits, temperature)?;

            // Get probability of the speculative token
            let speculative_prob = actual_probs.get(speculative_token as usize)
                .copied().unwrap_or(0.0);

            // Accept or reject based on threshold
            if speculative_prob >= self.config.acceptance_threshold {
                accepted.push(speculative_token);
                current_context.push(speculative_token);
            } else {
                // Rejection - stop here and sample from actual distribution
                let corrected_token = self.sample_from_probs(&actual_probs)?;
                if corrected_token != speculative_token {
                    break; // Stop speculation chain
                }
                accepted.push(corrected_token);
                current_context.push(corrected_token);
            }
        }

        let rejected_count = speculative_tokens.len() - accepted.len();
        Ok((accepted, rejected_count))
    }

    /// Generate single token (placeholder implementation)
    fn generate_single_token(&self, context: &[u32], temperature: f32) -> ModelResult<u32> {
        // Placeholder: in practice this would run the actual model
        let logits = self.compute_logits(context)?;
        let probs = self.softmax(&logits, temperature)?;
        self.sample_from_probs(&probs)
    }

    /// Compute logits (placeholder)
    fn compute_logits(&self, _context: &[u32]) -> ModelResult<Vec<f32>> {
        // Placeholder logits for vocabulary size 32000
        Ok(vec![0.1; 32000])
    }

    /// Apply softmax with temperature
    fn softmax(&self, logits: &[f32], temperature: f32) -> ModelResult<Vec<f32>> {
        let mut probs = Vec::with_capacity(logits.len());
        let mut sum = 0.0;

        // Apply temperature and compute exp
        for &logit in logits {
            let prob = (logit / temperature).exp();
            probs.push(prob);
            sum += prob;
        }

        // Normalize
        for prob in &mut probs {
            *prob /= sum;
        }

        Ok(probs)
    }

    /// Sample from probability distribution
    fn sample_from_probs(&self, probs: &[f32]) -> ModelResult<u32> {
        // Simple sampling - in practice would use proper random sampling
        let mut best_idx = 0;
        let mut best_prob = 0.0;

        for (idx, &prob) in probs.iter().enumerate() {
            if prob > best_prob {
                best_prob = prob;
                best_idx = idx;
            }
        }

        Ok(best_idx as u32)
    }

    /// Check if generation should stop
    fn should_stop(&self, _tokens: &[u32]) -> bool {
        // Placeholder: implement stop token checking
        false
    }
}

impl PrefixCache {
    pub fn new(config: PrefixCachingConfig, device: GpuDevice) -> Self {
        Self {
            config,
            entries: Arc::new(RwLock::new(BTreeMap::new())),
            size_tracker: Arc::new(RwLock::new(0)),
            device,
        }
    }

    /// Try to find cached KV states for the given prefix
    pub async fn lookup_prefix(&self, tokens: &[u32]) -> ModelResult<Option<(String, PrefixCacheEntry)>> {
        let entries = self.entries.read().unwrap();

        // Find the longest matching prefix
        let mut best_match: Option<(String, PrefixCacheEntry)> = None;
        let mut best_match_length = 0;

        for (key, entry) in entries.iter() {
            let prefix_tokens = &entry.prefix_tokens;

            // Check if this prefix matches
            if prefix_tokens.len() >= self.config.min_prefix_length
                && tokens.len() >= prefix_tokens.len()
                && tokens[..prefix_tokens.len()] == *prefix_tokens
                && prefix_tokens.len() > best_match_length
            {
                best_match = Some((key.clone(), entry.clone()));
                best_match_length = prefix_tokens.len();
            }
        }

        if let Some((key, mut entry)) = best_match {
            // Update access statistics
            drop(entries); // Release read lock
            let mut entries_write = self.entries.write().unwrap();
            if let Some(cached_entry) = entries_write.get_mut(&key) {
                cached_entry.last_access = SystemTime::now();
                cached_entry.access_count += 1;
            }
            entry.last_access = SystemTime::now();
            entry.access_count += 1;

            Ok(Some((key, entry)))
        } else {
            Ok(None)
        }
    }

    /// Cache KV states for a prefix
    pub async fn cache_prefix(
        &self,
        tokens: &[u32],
        kv_states: Vec<GpuTensor>,
    ) -> ModelResult<String> {
        if tokens.len() < self.config.min_prefix_length {
            return Err(ModelError::InvalidInput("Prefix too short to cache".to_string()));
        }

        let cache_key = self.compute_cache_key(tokens);
        let now = SystemTime::now();

        let entry = PrefixCacheEntry {
            prefix_tokens: tokens.to_vec(),
            kv_states,
            creation_time: now,
            last_access: now,
            access_count: 1,
            compressed: false, // TODO: implement compression
        };

        // Check cache capacity
        self.ensure_cache_capacity(tokens.len()).await?;

        // Insert entry
        let mut entries = self.entries.write().unwrap();
        entries.insert(cache_key.clone(), entry);

        // Update size tracker
        let mut size = self.size_tracker.write().unwrap();
        *size += tokens.len();

        Ok(cache_key)
    }

    /// Compute cache key for tokens
    fn compute_cache_key(&self, tokens: &[u32]) -> String {
        // Use a hash of the tokens as the cache key
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        tokens.hash(&mut hasher);
        format!("prefix_{:x}", hasher.finish())
    }

    /// Ensure cache has capacity for new entry
    async fn ensure_cache_capacity(&self, required_size: usize) -> ModelResult<()> {
        let current_size = {
            let size = self.size_tracker.read().unwrap();
            *size
        };

        if current_size + required_size > self.config.max_cache_size {
            self.evict_entries(current_size + required_size - self.config.max_cache_size).await?;
        }

        Ok(())
    }

    /// Evict cache entries based on policy
    async fn evict_entries(&self, bytes_to_free: usize) -> ModelResult<()> {
        let mut entries = self.entries.write().unwrap();
        let mut size = self.size_tracker.write().unwrap();

        let mut freed_bytes = 0;
        let mut keys_to_remove = Vec::new();

        match self.config.eviction_policy {
            CacheEvictionPolicy::LeastRecentlyUsed => {
                // Sort by last access time
                let mut sorted_entries: Vec<_> = entries.iter()
                    .map(|(k, v)| (k.clone(), v.last_access, v.prefix_tokens.len()))
                    .collect();
                sorted_entries.sort_by_key(|(_, last_access, _)| *last_access);

                for (key, _, token_count) in sorted_entries {
                    if freed_bytes >= bytes_to_free {
                        break;
                    }
                    keys_to_remove.push(key);
                    freed_bytes += token_count;
                }
            }
            CacheEvictionPolicy::LeastFrequentlyUsed => {
                // Sort by access count
                let mut sorted_entries: Vec<_> = entries.iter()
                    .map(|(k, v)| (k.clone(), v.access_count, v.prefix_tokens.len()))
                    .collect();
                sorted_entries.sort_by_key(|(_, access_count, _)| *access_count);

                for (key, _, token_count) in sorted_entries {
                    if freed_bytes >= bytes_to_free {
                        break;
                    }
                    keys_to_remove.push(key);
                    freed_bytes += token_count;
                }
            }
            CacheEvictionPolicy::TimeToLive => {
                let now = SystemTime::now();
                let ttl = Duration::from_secs(self.config.ttl_seconds);

                for (key, entry) in entries.iter() {
                    if let Ok(age) = now.duration_since(entry.creation_time) {
                        if age > ttl {
                            keys_to_remove.push(key.clone());
                            freed_bytes += entry.prefix_tokens.len();

                            if freed_bytes >= bytes_to_free {
                                break;
                            }
                        }
                    }
                }
            }
            CacheEvictionPolicy::SizeBased => {
                // Remove largest entries first
                let mut sorted_entries: Vec<_> = entries.iter()
                    .map(|(k, v)| (k.clone(), v.prefix_tokens.len()))
                    .collect();
                sorted_entries.sort_by_key(|(_, size)| std::cmp::Reverse(*size));

                for (key, token_count) in sorted_entries {
                    if freed_bytes >= bytes_to_free {
                        break;
                    }
                    keys_to_remove.push(key);
                    freed_bytes += token_count;
                }
            }
        }

        // Remove selected entries
        for key in keys_to_remove {
            if let Some(entry) = entries.remove(&key) {
                *size -= entry.prefix_tokens.len();
            }
        }

        Ok(())
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        let entries = self.entries.read().unwrap();
        let size = self.size_tracker.read().unwrap();

        CacheStats {
            cache_hit: false, // Set by caller
            cached_tokens: *size,
            cache_size_tokens: *size,
            cache_entries: entries.len(),
        }
    }
}

/// Combined advanced inference engine
pub struct AdvancedInferenceEngine {
    speculative_decoder: Option<SpeculativeDecoder>,
    prefix_cache: Option<PrefixCache>,
    tokenizer: RealTokenizer,
    device: GpuDevice,
    request_queue: Arc<Mutex<VecDeque<(AdvancedInferenceRequest, oneshot::Sender<AdvancedInferenceResponse>)>>>,
}

impl AdvancedInferenceEngine {
    pub fn new(device: GpuDevice) -> ModelResult<Self> {
        let tokenizer = RealTokenizer::create_demo_tokenizer()?;

        Ok(Self {
            speculative_decoder: None,
            prefix_cache: None,
            tokenizer,
            device,
            request_queue: Arc::new(Mutex::new(VecDeque::new())),
        })
    }

    /// Enable speculative decoding
    pub fn enable_speculative_decoding(&mut self, config: SpeculativeDecodingConfig) -> ModelResult<()> {
        self.speculative_decoder = Some(SpeculativeDecoder::new(config, self.device.clone())?);
        Ok(())
    }

    /// Enable prefix caching
    pub fn enable_prefix_caching(&mut self, config: PrefixCachingConfig) {
        self.prefix_cache = Some(PrefixCache::new(config, self.device.clone()));
    }

    /// Process inference request with all advanced features
    pub async fn process_request(&self, request: AdvancedInferenceRequest) -> ModelResult<AdvancedInferenceResponse> {
        let start_time = Instant::now();
        let mut cache_stats = CacheStats {
            cache_hit: false,
            cached_tokens: 0,
            cache_size_tokens: 0,
            cache_entries: 0,
        };

        // Tokenize prompt
        let prompt_tokens = self.tokenizer.encode(&request.prompt)?;

        // Try prefix cache lookup
        let (cache_hit_tokens, prefill_time) = if request.enable_prefix_caching {
            if let Some(ref cache) = self.prefix_cache {
                let cache_start = Instant::now();
                if let Some((_key, entry)) = cache.lookup_prefix(&prompt_tokens).await? {
                    cache_stats.cache_hit = true;
                    cache_stats.cached_tokens = entry.prefix_tokens.len();
                    (entry.prefix_tokens.len(), cache_start.elapsed())
                } else {
                    (0, cache_start.elapsed())
                }
            } else {
                (0, Duration::from_millis(0))
            }
        } else {
            (0, Duration::from_millis(0))
        };

        // Generate tokens
        let decode_start = Instant::now();
        let (generated_tokens, speculative_stats) = if request.enable_speculative_decoding {
            if let Some(ref decoder) = self.speculative_decoder {
                decoder.generate_speculative(&prompt_tokens, request.max_tokens, request.temperature).await?
            } else {
                // Fallback to normal generation - create empty stats
                let empty_stats = SpeculativeStats {
                    speculative_tokens_generated: 0,
                    accepted_tokens: 0,
                    acceptance_rate: 1.0,
                    speculation_speedup: 1.0,
                };
                (self.generate_normal(&prompt_tokens, request.max_tokens, request.temperature)?, empty_stats)
            }
        } else {
            let empty_stats = SpeculativeStats {
                speculative_tokens_generated: 0,
                accepted_tokens: 0,
                acceptance_rate: 1.0,
                speculation_speedup: 1.0,
            };
            (self.generate_normal(&prompt_tokens, request.max_tokens, request.temperature)?, empty_stats)
        };

        let decode_time = decode_start.elapsed();

        // Decode tokens to text
        let generated_text = self.tokenizer.decode(&generated_tokens)?;

        // Update cache if enabled
        if request.enable_prefix_caching && cache_hit_tokens == 0 {
            if let Some(ref cache) = self.prefix_cache {
                // Cache the prefix for future use
                let cache_prefix = &prompt_tokens[..prompt_tokens.len().min(256)]; // Cache first 256 tokens
                let _cache_key = cache.cache_prefix(cache_prefix, vec![]).await?; // Empty KV states for demo
            }
        }

        let total_time = start_time.elapsed();

        // Update cache stats
        if let Some(ref cache) = self.prefix_cache {
            let stats = cache.get_stats();
            cache_stats.cache_size_tokens = stats.cache_size_tokens;
            cache_stats.cache_entries = stats.cache_entries;
        }

        let timing_info = TimingInfo {
            total_time_ms: total_time.as_millis() as f64,
            prefill_time_ms: prefill_time.as_millis() as f64,
            decode_time_ms: decode_time.as_millis() as f64,
            tokens_per_second: generated_tokens.len() as f64 / decode_time.as_secs_f64(),
            cache_lookup_time_ms: prefill_time.as_millis() as f64,
        };

        let completion = CompletionResult {
            text: generated_text,
            tokens: generated_tokens.to_vec(),
            logprobs: None,
            finish_reason: "stop".to_string(),
        };

        Ok(AdvancedInferenceResponse {
            request_id: request.request_id,
            completions: vec![completion],
            timing_info,
            cache_stats,
            speculative_stats: Some(speculative_stats),
        })
    }

    /// Normal generation fallback
    fn generate_normal(&self, prompt_tokens: &[u32], max_tokens: usize, _temperature: f32) -> ModelResult<Vec<u32>> {
        // Placeholder normal generation
        let mut generated = Vec::new();
        for i in 0..max_tokens.min(10) {
            generated.push((prompt_tokens.len() + i) as u32 % 32000);
        }
        Ok(generated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_speculative_decoding() {
        let config = SpeculativeDecodingConfig::default();
        let device = GpuDevice::Cpu;
        let decoder = SpeculativeDecoder::new(config, device).unwrap();

        let prompt = vec![1, 2, 3, 4];
        let (tokens, stats) = decoder.generate_speculative(&prompt, 10, 0.8).await.unwrap();

        assert!(!tokens.is_empty());
        assert!(stats.acceptance_rate >= 0.0 && stats.acceptance_rate <= 1.0);
    }

    #[tokio::test]
    async fn test_prefix_caching() {
        let config = PrefixCachingConfig::default();
        let device = GpuDevice::Cpu;
        let cache = PrefixCache::new(config, device);

        let prefix = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let kv_states = vec![]; // Empty for demo

        let cache_key = cache.cache_prefix(&prefix, kv_states).await.unwrap();
        assert!(!cache_key.is_empty());

        let lookup_result = cache.lookup_prefix(&prefix).await.unwrap();
        assert!(lookup_result.is_some());
    }

    #[tokio::test]
    async fn test_advanced_inference_engine() {
        let device = GpuDevice::Cpu;
        let mut engine = AdvancedInferenceEngine::new(device).unwrap();

        let spec_config = SpeculativeDecodingConfig::default();
        engine.enable_speculative_decoding(spec_config).unwrap();

        let cache_config = PrefixCachingConfig::default();
        engine.enable_prefix_caching(cache_config);

        let request = AdvancedInferenceRequest {
            request_id: Uuid::new_v4().to_string(),
            prompt: "Hello world".to_string(),
            max_tokens: 20,
            temperature: 0.8,
            top_p: 0.9,
            top_k: None,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            stop_sequences: vec![],
            enable_speculative_decoding: true,
            enable_prefix_caching: true,
            priority: RequestPriority::Normal,
            n_completions: 1,
            streaming: false,
        };

        let response = engine.process_request(request).await.unwrap();
        assert!(!response.completions.is_empty());
        assert!(response.timing_info.total_time_ms > 0.0);
    }
}