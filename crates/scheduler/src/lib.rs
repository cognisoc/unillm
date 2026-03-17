//! Intelligent scheduler implementation for UniLLM
//!
//! This crate implements both continuous batching and intelligent scheduling
//! that integrates directly with the GPU memory management system.

mod chunked_prefill;
mod stream_overlap;
mod intelligent_scheduler;
mod cache_analyzer;
mod gpu_memory_tracker;
mod adaptive_policy;
mod types;

use std::collections::{HashMap, VecDeque, BinaryHeap};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::cmp::Ordering;

// Existing components
pub use chunked_prefill::{ChunkedPrefillManager, ChunkedPrefillConfig, PrefillChunk, PrefillStats};
pub use stream_overlap::{
    StreamOverlapManager, StreamType, StreamPriority, StreamOperation, OperationData,
    StreamOverlapStats, create_h2d_operation, create_compute_operation,
    create_prefill_operation, create_decode_operation
};

// Intelligent scheduler components
pub use intelligent_scheduler::{IntelligentScheduler, SchedulingPolicy, InferenceRequest, ScheduleInfo};
pub use cache_analyzer::CacheAwareAnalyzer;
pub use gpu_memory_tracker::{GpuMemoryTracker, GpuMemoryStats};
pub use adaptive_policy::{AdaptivePolicyEngine, PolicySummary, WorkloadAnalysis};
pub use types::*;

/// Request priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RequestPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Request state in the scheduler
#[derive(Debug, Clone)]
pub enum RequestState {
    /// Request is waiting to be admitted
    Waiting,
    /// Request is being processed (prefill phase)
    Prefilling,
    /// Request is in decode phase
    Decoding,
    /// Request is completed
    Completed,
    /// Request failed or was cancelled
    Failed,
}

/// A request in the continuous batching system
#[derive(Debug, Clone)]
pub struct Request {
    /// Unique request ID
    pub id: u32,
    /// Input prompt tokens
    pub prompt_tokens: Vec<u32>,
    /// Maximum output length
    pub max_output_length: usize,
    /// Current output length
    pub current_output_length: usize,
    /// Priority level
    pub priority: RequestPriority,
    /// Request state
    pub state: RequestState,
    /// Creation timestamp
    pub created_at: Instant,
    /// When request started processing
    pub started_at: Option<Instant>,
    /// When request completed
    pub completed_at: Option<Instant>,
    /// KV sequence ID
    pub kv_seq_id: Option<u32>,
    /// Number of tokens processed in prefill
    pub prefill_tokens_processed: usize,
    /// Whether this is a prefill request
    pub is_prefill: bool,
}

impl Request {
    /// Create a new request
    pub fn new(
        id: u32,
        prompt_tokens: Vec<u32>,
        max_output_length: usize,
        priority: RequestPriority,
    ) -> Self {
        Self {
            id,
            prompt_tokens: prompt_tokens.clone(),
            max_output_length,
            current_output_length: 0,
            priority,
            state: RequestState::Waiting,
            created_at: Instant::now(),
            started_at: None,
            completed_at: None,
            kv_seq_id: None,
            prefill_tokens_processed: 0,
            is_prefill: true,
        }
    }
    
    /// Get the total sequence length (prompt + output)
    pub fn total_length(&self) -> usize {
        self.prompt_tokens.len() + self.current_output_length
    }
    
    /// Get the remaining output length
    pub fn remaining_output_length(&self) -> usize {
        self.max_output_length - self.current_output_length
    }
    
    /// Check if request is completed
    pub fn is_completed(&self) -> bool {
        matches!(self.state, RequestState::Completed | RequestState::Failed)
    }
    
    /// Get request age
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
    
    /// Get processing time
    pub fn processing_time(&self) -> Option<Duration> {
        if let Some(started) = self.started_at {
            if let Some(completed) = self.completed_at {
                Some(completed.duration_since(started))
            } else {
                Some(started.elapsed())
            }
        } else {
            None
        }
    }
}

/// Batch of requests being processed together
#[derive(Debug, Clone)]
pub struct RequestBatch {
    /// Requests in this batch
    pub requests: Vec<Request>,
    /// Batch creation timestamp
    pub created_at: Instant,
    /// Batch ID
    pub batch_id: u32,
    /// Whether this is a prefill batch
    pub is_prefill_batch: bool,
    /// Maximum sequence length in this batch
    pub max_sequence_length: usize,
    /// Number of tokens in this batch
    pub total_tokens: usize,
}

impl RequestBatch {
    /// Create a new request batch
    pub fn new(batch_id: u32, requests: Vec<Request>) -> Self {
        let max_sequence_length = requests.iter().map(|r| r.total_length()).max().unwrap_or(0);
        let total_tokens = requests.iter().map(|r| r.total_length()).sum();
        let is_prefill_batch = requests.iter().any(|r| r.is_prefill);
        
        Self {
            requests,
            created_at: Instant::now(),
            batch_id,
            is_prefill_batch,
            max_sequence_length,
            total_tokens,
        }
    }
    
    /// Get the number of requests in this batch
    pub fn size(&self) -> usize {
        self.requests.len()
    }
    
    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }
    
    /// Get batch age
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

/// Continuous batching scheduler
pub struct ContinuousBatchingScheduler {
    /// Pending requests waiting to be admitted
    pending_requests: VecDeque<Request>,
    /// Requests currently being processed
    active_requests: HashMap<u32, Request>,
    /// Completed requests
    completed_requests: VecDeque<Request>,
    /// Current batch being processed
    current_batch: Option<RequestBatch>,
    /// Next request ID
    next_request_id: u32,
    /// Next batch ID
    next_batch_id: u32,
    /// Maximum batch size
    max_batch_size: usize,
    /// Maximum sequence length
    max_sequence_length: usize,
    /// Batching window duration
    batching_window: Duration,
    /// Last batch creation time
    last_batch_time: Instant,
    /// Statistics
    stats: SchedulerStats,
}

impl ContinuousBatchingScheduler {
    /// Create a new continuous batching scheduler
    /// 
    /// # Arguments
    /// * `max_batch_size` - Maximum number of requests per batch
    /// * `max_sequence_length` - Maximum sequence length
    /// * `batching_window_ms` - Batching window in milliseconds
    pub fn new(
        max_batch_size: usize,
        max_sequence_length: usize,
        batching_window_ms: u64,
    ) -> Self {
        Self {
            pending_requests: VecDeque::new(),
            active_requests: HashMap::new(),
            completed_requests: VecDeque::new(),
            current_batch: None,
            next_request_id: 0,
            next_batch_id: 0,
            max_batch_size,
            max_sequence_length,
            batching_window: Duration::from_millis(batching_window_ms),
            last_batch_time: Instant::now(),
            stats: SchedulerStats::new(),
        }
    }
    
    /// Add a new request to the scheduler
    pub fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_output_length: usize,
        priority: RequestPriority,
    ) -> u32 {
        let request_id = self.next_request_id;
        self.next_request_id += 1;
        
        let request = Request::new(request_id, prompt_tokens.clone(), max_output_length, priority);
        let prompt_len = prompt_tokens.len(); // Store before moving
        self.pending_requests.push_back(request);

        self.stats.total_requests += 1;

        println!("Added request {} with {} prompt tokens, max output: {}",
                 request_id, prompt_len, max_output_length);
        
        request_id
    }
    
    /// Create a new batch from pending requests
    pub fn create_batch(&mut self) -> Option<RequestBatch> {
        if self.pending_requests.is_empty() {
            return None;
        }
        
        let mut batch_requests = Vec::new();
        let mut current_tokens = 0;
        
        // Sort pending requests by priority (highest first)
        let mut sorted_requests: Vec<_> = self.pending_requests.drain(..).collect();
        sorted_requests.sort_by(|a, b| b.priority.cmp(&a.priority));
        
        for request in sorted_requests {
            let request_tokens = request.total_length();
            
            // Check if we can fit this request in the current batch
            if batch_requests.len() < self.max_batch_size 
                && current_tokens + request_tokens <= self.max_sequence_length {
                
                batch_requests.push(request);
                current_tokens += request_tokens;
            } else {
                // Put the request back in pending
                self.pending_requests.push_back(request);
                break;
            }
        }
        
        if batch_requests.is_empty() {
            return None;
        }
        
        let batch_id = self.next_batch_id;
        self.next_batch_id += 1;
        
        let batch = RequestBatch::new(batch_id, batch_requests);
        
        // Move requests to active
        for request in &batch.requests {
            self.active_requests.insert(request.id, request.clone());
        }
        
        self.current_batch = Some(batch.clone());
        self.last_batch_time = Instant::now();
        
        self.stats.batches_created += 1;
        self.stats.active_requests += batch.size();
        
        println!("Created batch {} with {} requests, {} total tokens", 
                 batch_id, batch.size(), batch.total_tokens);
        
        Some(batch)
    }
    
    /// Check if it's time to create a new batch
    pub fn should_create_batch(&self) -> bool {
        !self.pending_requests.is_empty() 
            && (self.current_batch.is_none() 
                || self.last_batch_time.elapsed() >= self.batching_window)
    }
    
    /// Complete a request
    pub fn complete_request(&mut self, request_id: u32) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(mut request) = self.active_requests.remove(&request_id) {
            request.state = RequestState::Completed;
            request.completed_at = Some(Instant::now());
            self.completed_requests.push_back(request);
            
            self.stats.active_requests -= 1;
            self.stats.completed_requests += 1;
            
            println!("Completed request {}", request_id);
        }
        
        Ok(())
    }
    
    /// Fail a request
    pub fn fail_request(&mut self, request_id: u32, reason: &str) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(mut request) = self.active_requests.remove(&request_id) {
            request.state = RequestState::Failed;
            request.completed_at = Some(Instant::now());
            self.completed_requests.push_back(request);
            
            self.stats.active_requests -= 1;
            self.stats.failed_requests += 1;
            
            println!("Failed request {}: {}", request_id, reason);
        }
        
        Ok(())
    }
    
    /// Get the current batch
    pub fn get_current_batch(&self) -> Option<&RequestBatch> {
        self.current_batch.as_ref()
    }
    
    /// Clear the current batch
    pub fn clear_current_batch(&mut self) {
        self.current_batch = None;
    }
    
    /// Get pending requests count
    pub fn pending_count(&self) -> usize {
        self.pending_requests.len()
    }
    
    /// Get active requests count
    pub fn active_count(&self) -> usize {
        self.active_requests.len()
    }
    
    /// Get scheduler statistics
    pub fn get_stats(&self) -> &SchedulerStats {
        &self.stats
    }
    
    /// Get all active requests
    pub fn get_active_requests(&self) -> Vec<&Request> {
        self.active_requests.values().collect()
    }
    
    /// Get completed requests
    pub fn get_completed_requests(&self) -> Vec<&Request> {
        self.completed_requests.iter().collect()
    }
    
    /// Update request state
    pub fn update_request_state(&mut self, request_id: u32, new_state: RequestState) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(request) = self.active_requests.get_mut(&request_id) {
            let state_check = new_state.clone(); // Clone for the matches! check
            request.state = new_state;
            if matches!(state_check, RequestState::Prefilling | RequestState::Decoding) && request.started_at.is_none() {
                request.started_at = Some(Instant::now());
            }
        }

        Ok(())
    }
    
    /// Get request by ID
    pub fn get_request(&self, request_id: u32) -> Option<&Request> {
        self.active_requests.get(&request_id)
    }
}

/// Scheduler statistics
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub total_requests: usize,
    pub active_requests: usize,
    pub completed_requests: usize,
    pub failed_requests: usize,
    pub batches_created: usize,
    pub average_batch_size: f64,
    pub average_request_latency: Duration,
}

impl SchedulerStats {
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            active_requests: 0,
            completed_requests: 0,
            failed_requests: 0,
            batches_created: 0,
            average_batch_size: 0.0,
            average_request_latency: Duration::from_millis(0),
        }
    }
}

impl std::fmt::Display for SchedulerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Scheduler Stats: {} total, {} active, {} completed, {} failed, {} batches",
               self.total_requests, self.active_requests, self.completed_requests, self.failed_requests, self.batches_created)
    }
}