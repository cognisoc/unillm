//! Dynamic request batching for high-throughput inference
//!
//! This module implements continuous batching that allows processing multiple
//! requests of varying sequence lengths simultaneously, maximizing GPU utilization.

use crate::types::*;
use crate::gpu_tensor_ops::{GpuTensor, GpuTensorOps, GpuDevice};
use crate::kv_cache::GenerationContext;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Request identifier
pub type RequestId = String;

/// Generation request with metadata
#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub request_id: RequestId,
    pub input_tokens: Vec<u32>,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub stop_tokens: Vec<u32>,
    pub created_at: Instant,
    pub priority: RequestPriority,
}

/// Request priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RequestPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Active generation state for a request
#[derive(Debug)]
pub struct GenerationState {
    pub request: GenerationRequest,
    pub generated_tokens: Vec<u32>,
    pub current_position: usize,
    pub is_finished: bool,
    pub context: GenerationContext,
    pub last_update: Instant,
}

impl GenerationState {
    pub fn new(
        request: GenerationRequest,
        num_heads: usize,
        head_dim: usize,
        num_layers: usize,
        device: GpuDevice,
    ) -> ModelResult<Self> {
        let max_context_length = request.input_tokens.len() + request.max_new_tokens;
        let context = GenerationContext::new(
            1, // batch_size = 1 for individual request
            num_heads,
            head_dim,
            max_context_length,
            num_layers,
            device,
        )?;

        Ok(Self {
            request,
            generated_tokens: Vec::new(),
            current_position: 0,
            is_finished: false,
            context,
            last_update: Instant::now(),
        })
    }

    pub fn total_tokens(&self) -> usize {
        self.request.input_tokens.len() + self.generated_tokens.len()
    }

    pub fn should_stop(&self, new_token: u32) -> bool {
        // Check stop conditions
        self.generated_tokens.len() >= self.request.max_new_tokens ||
        self.request.stop_tokens.contains(&new_token) ||
        !self.context.can_continue()
    }

    pub fn add_token(&mut self, token: u32) -> ModelResult<()> {
        if self.is_finished {
            return Err(ModelError::ComputationFailed("Request already finished".to_string()));
        }

        self.generated_tokens.push(token);
        self.context.step()?;
        self.last_update = Instant::now();

        if self.should_stop(token) {
            self.is_finished = true;
        }

        Ok(())
    }
}

/// Batch of requests being processed together
#[derive(Debug)]
pub struct RequestBatch {
    pub requests: Vec<GenerationState>,
    pub batch_size: usize,
    pub max_sequence_length: usize,
    pub created_at: Instant,
}

impl Default for RequestBatch {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestBatch {
    pub fn new() -> Self {
        Self {
            requests: Vec::new(),
            batch_size: 0,
            max_sequence_length: 0,
            created_at: Instant::now(),
        }
    }

    pub fn add_request(&mut self, request: GenerationState) {
        let seq_len = request.total_tokens();
        if seq_len > self.max_sequence_length {
            self.max_sequence_length = seq_len;
        }
        self.requests.push(request);
        self.batch_size = self.requests.len();
    }

    pub fn remove_finished(&mut self) -> Vec<GenerationState> {
        let mut finished = Vec::new();
        let mut active = Vec::new();

        for request in self.requests.drain(..) {
            if request.is_finished {
                finished.push(request);
            } else {
                active.push(request);
            }
        }

        self.requests = active;
        self.batch_size = self.requests.len();

        // Recalculate max sequence length
        self.max_sequence_length = self.requests
            .iter()
            .map(|r| r.total_tokens())
            .max()
            .unwrap_or(0);

        finished
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    pub fn average_priority(&self) -> f32 {
        if self.requests.is_empty() {
            return 0.0;
        }

        let sum: u32 = self.requests
            .iter()
            .map(|r| r.request.priority as u32)
            .sum();
        sum as f32 / self.requests.len() as f32
    }
}

/// Dynamic request batching configuration
#[derive(Debug, Clone)]
pub struct BatchingConfig {
    pub max_batch_size: usize,
    pub max_sequence_length: usize,
    pub batch_timeout_ms: u64,
    pub max_wait_time_ms: u64,
    pub enable_padding: bool,
    pub enable_priority_scheduling: bool,
}

impl Default for BatchingConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            max_sequence_length: 2048,
            batch_timeout_ms: 100,
            max_wait_time_ms: 1000,
            enable_padding: true,
            enable_priority_scheduling: true,
        }
    }
}

/// Continuous batching scheduler
pub struct ContinuousBatcher {
    config: BatchingConfig,
    pending_requests: VecDeque<GenerationRequest>,
    active_batches: Vec<RequestBatch>,
    finished_requests: HashMap<RequestId, GenerationResult>,
    tensor_ops: GpuTensorOps,
    device: GpuDevice,
    num_heads: usize,
    head_dim: usize,
    num_layers: usize,
    stats: BatchingStats,
}

/// Generation result for a completed request
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub request_id: RequestId,
    pub input_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub generation_time: Duration,
    pub total_time: Duration,
    pub tokens_per_second: f64,
}

/// Batching performance statistics
#[derive(Debug, Clone)]
pub struct BatchingStats {
    pub requests_processed: usize,
    pub requests_pending: usize,
    pub active_batches: usize,
    pub average_batch_size: f64,
    pub average_throughput_tps: f64,
    pub average_latency_ms: f64,
    pub gpu_utilization: f64,
    pub memory_usage_mb: f64,
}

impl ContinuousBatcher {
    pub fn new(
        config: BatchingConfig,
        num_heads: usize,
        head_dim: usize,
        num_layers: usize,
        device: GpuDevice,
    ) -> Self {
        let tensor_ops = GpuTensorOps::with_device(device.clone());

        Self {
            config,
            pending_requests: VecDeque::new(),
            active_batches: Vec::new(),
            finished_requests: HashMap::new(),
            tensor_ops,
            device,
            num_heads,
            head_dim,
            num_layers,
            stats: BatchingStats {
                requests_processed: 0,
                requests_pending: 0,
                active_batches: 0,
                average_batch_size: 0.0,
                average_throughput_tps: 0.0,
                average_latency_ms: 0.0,
                gpu_utilization: 0.0,
                memory_usage_mb: 0.0,
            },
        }
    }

    /// Add a new generation request
    pub fn add_request(&mut self, request: GenerationRequest) {
        if self.config.enable_priority_scheduling {
            // Insert based on priority (higher priority first)
            let insert_pos = self.pending_requests
                .iter()
                .position(|r| r.priority < request.priority)
                .unwrap_or(self.pending_requests.len());
            self.pending_requests.insert(insert_pos, request);
        } else {
            self.pending_requests.push_back(request);
        }
        self.stats.requests_pending = self.pending_requests.len();
    }

    /// Process one step of continuous batching
    pub fn step(&mut self) -> ModelResult<Vec<GenerationResult>> {
        let mut completed_requests = Vec::new();

        // 1. Form new batches from pending requests
        self.form_new_batches()?;

        // 2. Process active batches
        let mut batch_idx = 0;
        while batch_idx < self.active_batches.len() {
            let batch_results = self.process_batch(batch_idx)?;
            completed_requests.extend(batch_results);

            // Remove empty batches
            if self.active_batches[batch_idx].is_empty() {
                self.active_batches.remove(batch_idx);
            } else {
                batch_idx += 1;
            }
        }

        // 3. Update statistics
        self.update_statistics();

        // 4. Store completed requests
        for result in &completed_requests {
            self.finished_requests.insert(result.request_id.clone(), result.clone());
            self.stats.requests_processed += 1;
        }

        Ok(completed_requests)
    }

    /// Form new batches from pending requests
    fn form_new_batches(&mut self) -> ModelResult<()> {
        while !self.pending_requests.is_empty() {
            let mut new_batch = RequestBatch::new();
            let mut batch_tokens = 0;

            // Fill batch with compatible requests
            while let Some(request) = self.pending_requests.pop_front() {
                let request_tokens = request.input_tokens.len() + request.max_new_tokens;

                // Check batch constraints
                if new_batch.batch_size >= self.config.max_batch_size {
                    // Batch is full, put request back
                    self.pending_requests.push_front(request);
                    break;
                }

                if request_tokens > self.config.max_sequence_length {
                    // Request too long, skip (in production, might handle differently)
                    continue;
                }

                let max_batch_tokens = if self.config.enable_padding {
                    std::cmp::max(batch_tokens, request_tokens)
                } else {
                    batch_tokens + request_tokens
                };

                // Convert request to generation state
                let gen_state = GenerationState::new(
                    request,
                    self.num_heads,
                    self.head_dim,
                    self.num_layers,
                    self.device.clone(),
                )?;

                new_batch.add_request(gen_state);
                batch_tokens = max_batch_tokens;

                // If not padding, we can continue adding more requests
                if !self.config.enable_padding {
                    continue;
                }

                // With padding, check if we should add more requests of similar length
                let current_max_len = new_batch.max_sequence_length;
                let remaining = self.pending_requests.iter().take(5).find(|r| {
                    let req_len = r.input_tokens.len() + r.max_new_tokens;
                    (req_len as i32 - current_max_len as i32).abs() < 100
                });

                if remaining.is_none() {
                    break; // No similar requests, start batch now
                }
            }

            if !new_batch.is_empty() {
                self.active_batches.push(new_batch);
            }
        }

        self.stats.requests_pending = self.pending_requests.len();
        self.stats.active_batches = self.active_batches.len();

        Ok(())
    }

    /// Process a single batch for one generation step
    fn process_batch(&mut self, batch_idx: usize) -> ModelResult<Vec<GenerationResult>> {
        if batch_idx >= self.active_batches.len() {
            return Ok(Vec::new());
        }

        if self.active_batches[batch_idx].is_empty() {
            return Ok(Vec::new());
        }

        // Create batched tensors for all requests
        let batched_input = {
            let batch = &self.active_batches[batch_idx];
            self.create_batched_input(batch)?
        };

        // Run inference (placeholder - would call actual model)
        let outputs = self.run_batched_inference(&batched_input)?;

        // Process outputs and update request states
        // Extract the batch temporarily to avoid double borrow
        let mut current_batch = std::mem::take(&mut self.active_batches[batch_idx]);
        self.process_batch_outputs(&mut current_batch, &outputs)?;
        self.active_batches[batch_idx] = current_batch;

        // Remove completed requests from batch
        let completed = self.active_batches[batch_idx].remove_finished();

        // Convert to results
        let mut results = Vec::new();
        for finished_state in completed {
            let total_time = finished_state.last_update.duration_since(finished_state.request.created_at);
            let generation_time = total_time; // Simplified

            let tokens_per_second = if generation_time.as_secs_f64() > 0.0 {
                finished_state.generated_tokens.len() as f64 / generation_time.as_secs_f64()
            } else {
                0.0
            };

            results.push(GenerationResult {
                request_id: finished_state.request.request_id,
                input_tokens: finished_state.request.input_tokens,
                generated_tokens: finished_state.generated_tokens,
                generation_time,
                total_time,
                tokens_per_second,
            });
        }

        Ok(results)
    }

    /// Create batched input tensors from multiple requests
    fn create_batched_input(&self, batch: &RequestBatch) -> ModelResult<BatchedInput> {
        let batch_size = batch.batch_size;
        let seq_len = batch.max_sequence_length;

        // In a full implementation, we would:
        // 1. Pad all sequences to max_sequence_length
        // 2. Create attention masks
        // 3. Stack into batched tensors

        // For now, create placeholder tensors
        let input_ids = GpuTensor::zeros(vec![batch_size, seq_len], self.device.clone())?;
        let attention_mask = GpuTensor::ones(vec![batch_size, seq_len], self.device.clone())?;

        Ok(BatchedInput {
            input_ids,
            attention_mask,
            batch_size,
            sequence_length: seq_len,
        })
    }

    /// Run inference on batched input
    fn run_batched_inference(&self, input: &BatchedInput) -> ModelResult<BatchedOutput> {
        // Placeholder inference - in reality would call the model
        let logits = GpuTensor::randn(
            vec![input.batch_size, input.sequence_length, 32000], // vocab_size
            self.device.clone()
        )?;

        Ok(BatchedOutput {
            logits,
            batch_size: input.batch_size,
        })
    }

    /// Process batch outputs and update request states
    fn process_batch_outputs(
        &mut self,
        batch: &mut RequestBatch,
        outputs: &BatchedOutput,
    ) -> ModelResult<()> {
        // For each request in the batch, sample next token and update state
        for (req_idx, request_state) in batch.requests.iter_mut().enumerate() {
            if request_state.is_finished {
                continue;
            }

            // Extract logits for this request (simplified)
            let next_token = self.sample_token(req_idx, outputs)?;

            // Update request state
            request_state.add_token(next_token)?;
        }

        Ok(())
    }

    /// Sample next token for a request
    fn sample_token(&self, _request_idx: usize, _outputs: &BatchedOutput) -> ModelResult<u32> {
        // Simplified sampling - just return a random token
        // In reality, would apply temperature, top-p, etc.
        Ok(42) // Placeholder token
    }

    /// Update performance statistics
    fn update_statistics(&mut self) {
        let total_requests = self.active_batches.iter()
            .map(|b| b.batch_size)
            .sum::<usize>();

        if !self.active_batches.is_empty() {
            self.stats.average_batch_size = total_requests as f64 / self.active_batches.len() as f64;
        }

        // Update other statistics (simplified)
        self.stats.gpu_utilization = if total_requests > 0 { 85.0 } else { 0.0 };
        self.stats.memory_usage_mb = (total_requests * 100) as f64; // Rough estimate
    }

    /// Get current statistics
    pub fn statistics(&self) -> BatchingStats {
        self.stats.clone()
    }

    /// Get completed request result
    pub fn get_result(&mut self, request_id: &RequestId) -> Option<GenerationResult> {
        self.finished_requests.remove(request_id)
    }

    /// Check if request is complete
    pub fn is_complete(&self, request_id: &RequestId) -> bool {
        self.finished_requests.contains_key(request_id)
    }
}

/// Batched input tensors
#[derive(Debug)]
pub struct BatchedInput {
    pub input_ids: GpuTensor,
    pub attention_mask: GpuTensor,
    pub batch_size: usize,
    pub sequence_length: usize,
}

/// Batched output tensors
#[derive(Debug)]
pub struct BatchedOutput {
    pub logits: GpuTensor,
    pub batch_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_request(id: &str, input_len: usize, max_tokens: usize) -> GenerationRequest {
        GenerationRequest {
            request_id: id.to_string(),
            input_tokens: vec![1; input_len],
            max_new_tokens: max_tokens,
            temperature: 1.0,
            top_p: 0.9,
            stop_tokens: vec![2],
            created_at: Instant::now(),
            priority: RequestPriority::Normal,
        }
    }

    #[test]
    fn test_generation_request_creation() {
        let request = create_test_request("test", 10, 50);
        assert_eq!(request.request_id, "test");
        assert_eq!(request.input_tokens.len(), 10);
        assert_eq!(request.max_new_tokens, 50);
    }

    #[test]
    fn test_generation_state_creation() {
        let device = GpuDevice::auto_detect();
        let request = create_test_request("test", 5, 10);
        let state = GenerationState::new(request, 8, 64, 6, device);

        assert!(state.is_ok());
        let state = state.unwrap();
        assert_eq!(state.total_tokens(), 5);
        assert!(!state.is_finished);
        assert!(state.context.can_continue());
    }

    #[test]
    fn test_generation_state_token_addition() {
        let device = GpuDevice::auto_detect();
        let request = create_test_request("test", 5, 3);
        let mut state = GenerationState::new(request, 8, 64, 6, device).unwrap();

        // Add tokens until finished
        assert!(state.add_token(10).is_ok());
        assert_eq!(state.generated_tokens.len(), 1);
        assert!(!state.is_finished);

        assert!(state.add_token(11).is_ok());
        assert_eq!(state.generated_tokens.len(), 2);
        assert!(!state.is_finished);

        assert!(state.add_token(12).is_ok());
        assert_eq!(state.generated_tokens.len(), 3);
        assert!(state.is_finished); // Reached max_new_tokens
    }

    #[test]
    fn test_generation_state_stop_token() {
        let device = GpuDevice::auto_detect();
        let request = create_test_request("test", 5, 10);
        let mut state = GenerationState::new(request, 8, 64, 6, device).unwrap();

        // Add stop token
        assert!(state.add_token(2).is_ok()); // 2 is in stop_tokens
        assert!(state.is_finished);
    }

    #[test]
    fn test_request_batch_operations() {
        let device = GpuDevice::auto_detect();
        let mut batch = RequestBatch::new();

        let request1 = create_test_request("req1", 10, 5);
        let state1 = GenerationState::new(request1, 8, 64, 6, device.clone()).unwrap();

        let request2 = create_test_request("req2", 15, 5);
        let mut state2 = GenerationState::new(request2, 8, 64, 6, device).unwrap();

        batch.add_request(state1);
        batch.add_request(state2);

        assert_eq!(batch.batch_size, 2);
        assert_eq!(batch.max_sequence_length, 15); // max(10, 15) since no tokens generated yet

        // Mark one as finished
        batch.requests[1].is_finished = true;
        let finished = batch.remove_finished();

        assert_eq!(finished.len(), 1);
        assert_eq!(batch.batch_size, 1);
        assert_eq!(batch.max_sequence_length, 10); // Recalculated after removing req2 (15)
    }

    #[test]
    fn test_continuous_batcher_creation() {
        let device = GpuDevice::auto_detect();
        let config = BatchingConfig::default();
        let batcher = ContinuousBatcher::new(config, 8, 64, 6, device);

        assert_eq!(batcher.stats.requests_pending, 0);
        assert_eq!(batcher.stats.active_batches, 0);
    }

    #[test]
    fn test_continuous_batcher_add_request() {
        let device = GpuDevice::auto_detect();
        let config = BatchingConfig::default();
        let mut batcher = ContinuousBatcher::new(config, 8, 64, 6, device);

        let request = create_test_request("test", 10, 20);
        batcher.add_request(request);

        assert_eq!(batcher.stats.requests_pending, 1);
        assert!(batcher.pending_requests.len() == 1);
    }

    #[test]
    fn test_priority_scheduling() {
        let device = GpuDevice::auto_detect();
        let mut config = BatchingConfig::default();
        config.enable_priority_scheduling = true;
        let mut batcher = ContinuousBatcher::new(config, 8, 64, 6, device);

        // Add requests with different priorities
        let mut low_req = create_test_request("low", 10, 5);
        low_req.priority = RequestPriority::Low;

        let mut high_req = create_test_request("high", 10, 5);
        high_req.priority = RequestPriority::High;

        let mut normal_req = create_test_request("normal", 10, 5);
        normal_req.priority = RequestPriority::Normal;

        // Add in random order
        batcher.add_request(low_req);
        batcher.add_request(high_req);
        batcher.add_request(normal_req);

        // High priority should be first
        assert_eq!(batcher.pending_requests[0].request_id, "high");
        assert_eq!(batcher.pending_requests[1].request_id, "normal");
        assert_eq!(batcher.pending_requests[2].request_id, "low");
    }

    #[test]
    fn test_batching_config() {
        let config = BatchingConfig::default();
        assert_eq!(config.max_batch_size, 64);
        assert_eq!(config.max_sequence_length, 2048);
        assert!(config.enable_padding);
        assert!(config.enable_priority_scheduling);
    }

    #[test]
    fn test_batch_statistics() {
        let device = GpuDevice::auto_detect();
        let config = BatchingConfig::default();
        let batcher = ContinuousBatcher::new(config, 8, 64, 6, device);

        let stats = batcher.statistics();
        assert_eq!(stats.requests_processed, 0);
        assert_eq!(stats.requests_pending, 0);
        assert_eq!(stats.active_batches, 0);
        assert_eq!(stats.average_batch_size, 0.0);
    }
}