//! Request admission control and resource management

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Resource limits for the system
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Maximum number of concurrent requests
    pub max_concurrent_requests: usize,
    /// Maximum total sequence length across all requests
    pub max_total_sequence_length: usize,
    /// Maximum memory usage (in bytes)
    pub max_memory_usage: usize,
    /// Maximum GPU utilization percentage
    pub max_gpu_utilization: f64,
    /// Maximum queue wait time
    pub max_queue_wait_time: Duration,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_concurrent_requests: 32,
            max_total_sequence_length: 65536, // 64K tokens
            max_memory_usage: 8 * 1024 * 1024 * 1024, // 8GB
            max_gpu_utilization: 90.0, // 90%
            max_queue_wait_time: Duration::from_secs(30),
        }
    }
}

/// Current resource usage
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Number of active requests
    pub active_requests: usize,
    /// Total sequence length across all requests
    pub total_sequence_length: usize,
    /// Current memory usage (in bytes)
    pub memory_usage: usize,
    /// Current GPU utilization percentage
    pub gpu_utilization: f64,
    /// Number of requests in queue
    pub queued_requests: usize,
    /// Average queue wait time
    pub average_queue_wait_time: Duration,
}

impl ResourceUsage {
    pub fn new() -> Self {
        Self {
            active_requests: 0,
            total_sequence_length: 0,
            memory_usage: 0,
            gpu_utilization: 0.0,
            queued_requests: 0,
            average_queue_wait_time: Duration::from_millis(0),
        }
    }
}

/// Admission decision result
#[derive(Debug, Clone)]
pub enum AdmissionDecision {
    /// Request is admitted immediately
    Admitted,
    /// Request is queued for later processing
    Queued { estimated_wait_time: Duration },
    /// Request is rejected due to resource constraints
    Rejected { reason: String },
}

/// Request admission controller
pub struct AdmissionController {
    /// Resource limits
    limits: ResourceLimits,
    /// Current resource usage
    usage: ResourceUsage,
    /// Request queue
    request_queue: VecDeque<QueuedRequest>,
    /// Request statistics
    stats: AdmissionStats,
}

/// A queued request waiting for admission
#[derive(Debug, Clone)]
pub struct QueuedRequest {
    /// Request ID
    pub request_id: u32,
    /// Prompt tokens
    pub prompt_tokens: Vec<u32>,
    /// Maximum output length
    pub max_output_length: usize,
    /// Priority
    pub priority: crate::RequestPriority,
    /// Queue timestamp
    pub queued_at: Instant,
    /// Estimated resource requirements
    pub resource_requirements: ResourceRequirements,
}

/// Resource requirements for a request
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    /// Estimated memory usage (in bytes)
    pub memory_usage: usize,
    /// Estimated sequence length
    pub sequence_length: usize,
    /// Estimated GPU utilization
    pub gpu_utilization: f64,
    /// Estimated processing time
    pub processing_time: Duration,
}

impl ResourceRequirements {
    /// Calculate resource requirements for a request
    pub fn calculate(
        prompt_tokens: &[u32],
        max_output_length: usize,
        model_config: &crate::runtime::ModelConfig,
    ) -> Self {
        let prompt_length = prompt_tokens.len();
        let total_length = prompt_length + max_output_length;
        
        // Estimate memory usage (rough calculation)
        let kv_cache_size = total_length * model_config.hidden_size * 2 * 4; // 2 * hidden_size * 4 bytes
        let attention_memory = prompt_length * prompt_length * 4; // Attention matrix
        let total_memory = kv_cache_size + attention_memory;
        
        // Estimate GPU utilization based on sequence length
        let gpu_utilization = if total_length < 1024 {
            20.0 // Short sequences
        } else if total_length < 4096 {
            50.0 // Medium sequences
        } else {
            80.0 // Long sequences
        };
        
        // Estimate processing time
        let processing_time = if prompt_length < 512 {
            Duration::from_millis(100) // Fast
        } else if prompt_length < 2048 {
            Duration::from_millis(500) // Medium
        } else {
            Duration::from_millis(2000) // Slow
        };
        
        Self {
            memory_usage: total_memory,
            sequence_length: total_length,
            gpu_utilization,
            processing_time,
        }
    }
}

impl AdmissionController {
    /// Create a new admission controller
    pub fn new(limits: ResourceLimits) -> Self {
        Self {
            limits,
            usage: ResourceUsage::new(),
            request_queue: VecDeque::new(),
            stats: AdmissionStats::new(),
        }
    }
    
    /// Try to admit a request
    pub fn admit_request(
        &mut self,
        request_id: u32,
        prompt_tokens: Vec<u32>,
        max_output_length: usize,
        priority: crate::RequestPriority,
        model_config: &crate::runtime::ModelConfig,
    ) -> AdmissionDecision {
        let resource_requirements = ResourceRequirements::calculate(
            &prompt_tokens,
            max_output_length,
            model_config,
        );
        
        // Check if we can admit the request immediately
        if self.can_admit_immediately(&resource_requirements) {
            self.admit_immediately(request_id, resource_requirements);
            self.stats.requests_admitted += 1;
            return AdmissionDecision::Admitted;
        }
        
        // Check if we can queue the request
        if self.can_queue_request(&resource_requirements) {
            let estimated_wait_time = self.estimate_queue_wait_time(&resource_requirements);
            self.queue_request(request_id, prompt_tokens, max_output_length, priority, resource_requirements);
            self.stats.requests_queued += 1;
            return AdmissionDecision::Queued { estimated_wait_time };
        }
        
        // Reject the request
        let reason = self.get_rejection_reason(&resource_requirements);
        self.stats.requests_rejected += 1;
        AdmissionDecision::Rejected { reason }
    }
    
    /// Check if a request can be admitted immediately
    fn can_admit_immediately(&self, requirements: &ResourceRequirements) -> bool {
        // Check concurrent request limit
        if self.usage.active_requests >= self.limits.max_concurrent_requests {
            return false;
        }
        
        // Check total sequence length limit
        if self.usage.total_sequence_length + requirements.sequence_length > self.limits.max_total_sequence_length {
            return false;
        }
        
        // Check memory limit
        if self.usage.memory_usage + requirements.memory_usage > self.limits.max_memory_usage {
            return false;
        }
        
        // Check GPU utilization limit
        if self.usage.gpu_utilization + requirements.gpu_utilization > self.limits.max_gpu_utilization {
            return false;
        }
        
        true
    }
    
    /// Check if a request can be queued
    fn can_queue_request(&self, requirements: &ResourceRequirements) -> bool {
        // Don't queue if we're already at capacity
        if self.request_queue.len() >= self.limits.max_concurrent_requests * 2 {
            return false;
        }
        
        // Don't queue requests that are too resource-intensive
        if requirements.memory_usage > self.limits.max_memory_usage / 2 {
            return false;
        }
        
        true
    }
    
    /// Admit a request immediately
    fn admit_immediately(&mut self, request_id: u32, requirements: ResourceRequirements) {
        self.usage.active_requests += 1;
        self.usage.total_sequence_length += requirements.sequence_length;
        self.usage.memory_usage += requirements.memory_usage;
        self.usage.gpu_utilization += requirements.gpu_utilization;
        
        println!("Admitted request {} immediately", request_id);
    }
    
    /// Queue a request for later processing
    fn queue_request(
        &mut self,
        request_id: u32,
        prompt_tokens: Vec<u32>,
        max_output_length: usize,
        priority: crate::RequestPriority,
        resource_requirements: ResourceRequirements,
    ) {
        let queued_request = QueuedRequest {
            request_id,
            prompt_tokens,
            max_output_length,
            priority,
            queued_at: Instant::now(),
            resource_requirements,
        };
        
        // Insert request in priority order
        let mut inserted = false;
        for (i, existing_request) in self.request_queue.iter().enumerate() {
            if priority > existing_request.priority {
                self.request_queue.insert(i, queued_request);
                inserted = true;
                break;
            }
        }
        
        if !inserted {
            self.request_queue.push_back(queued_request);
        }
        
        self.usage.queued_requests = self.request_queue.len();
        
        println!("Queued request {} with priority {:?}", request_id, priority);
    }
    
    /// Estimate queue wait time
    fn estimate_queue_wait_time(&self, requirements: &ResourceRequirements) -> Duration {
        if self.request_queue.is_empty() {
            return Duration::from_millis(0);
        }
        
        // Calculate average processing time for similar requests
        let similar_requests = self.request_queue
            .iter()
            .filter(|req| req.resource_requirements.sequence_length <= requirements.sequence_length * 2)
            .count();
        
        let average_processing_time = if similar_requests > 0 {
            self.request_queue
                .iter()
                .take(similar_requests)
                .map(|req| req.resource_requirements.processing_time)
                .sum::<Duration>() / similar_requests as u32
        } else {
            Duration::from_millis(500) // Default estimate
        };
        
        // Estimate based on queue position and processing time
        let queue_position = self.request_queue.len();
        average_processing_time * queue_position as u32
    }
    
    /// Get rejection reason
    fn get_rejection_reason(&self, requirements: &ResourceRequirements) -> String {
        if self.usage.active_requests >= self.limits.max_concurrent_requests {
            "Maximum concurrent requests exceeded".to_string()
        } else if self.usage.total_sequence_length + requirements.sequence_length > self.limits.max_total_sequence_length {
            "Total sequence length limit exceeded".to_string()
        } else if self.usage.memory_usage + requirements.memory_usage > self.limits.max_memory_usage {
            "Memory limit exceeded".to_string()
        } else if self.usage.gpu_utilization + requirements.gpu_utilization > self.limits.max_gpu_utilization {
            "GPU utilization limit exceeded".to_string()
        } else {
            "Resource constraints not met".to_string()
        }
    }
    
    /// Process completed request and update resource usage
    pub fn complete_request(&mut self, request_id: u32, requirements: ResourceRequirements) {
        self.usage.active_requests = self.usage.active_requests.saturating_sub(1);
        self.usage.total_sequence_length = self.usage.total_sequence_length.saturating_sub(requirements.sequence_length);
        self.usage.memory_usage = self.usage.memory_usage.saturating_sub(requirements.memory_usage);
        self.usage.gpu_utilization = (self.usage.gpu_utilization - requirements.gpu_utilization).max(0.0);
        
        // Try to admit queued requests
        self.process_queue();
        
        println!("Completed request {}, updated resource usage", request_id);
    }
    
    /// Process the request queue
    fn process_queue(&mut self) {
        let mut to_admit = Vec::new();
        
        // Find requests that can now be admitted
        for (i, queued_request) in self.request_queue.iter().enumerate() {
            if self.can_admit_immediately(&queued_request.resource_requirements) {
                to_admit.push(i);
            }
        }
        
        // Admit requests in reverse order to maintain indices
        for &i in to_admit.iter().rev() {
            if let Some(queued_request) = self.request_queue.remove(i) {
                self.admit_immediately(queued_request.request_id, queued_request.resource_requirements);
                self.stats.requests_admitted += 1;
                self.stats.requests_queued -= 1;
            }
        }
        
        self.usage.queued_requests = self.request_queue.len();
    }
    
    /// Get current resource usage
    pub fn get_resource_usage(&self) -> &ResourceUsage {
        &self.usage
    }
    
    /// Get resource limits
    pub fn get_resource_limits(&self) -> &ResourceLimits {
        &self.limits
    }
    
    /// Get admission statistics
    pub fn get_stats(&self) -> &AdmissionStats {
        &self.stats
    }
    
    /// Get queued requests
    pub fn get_queued_requests(&self) -> Vec<&QueuedRequest> {
        self.request_queue.iter().collect()
    }
    
    /// Update resource usage (for external monitoring)
    pub fn update_resource_usage(&mut self, usage: ResourceUsage) {
        self.usage = usage;
    }
}

/// Admission control statistics
#[derive(Debug, Clone)]
pub struct AdmissionStats {
    pub requests_admitted: usize,
    pub requests_queued: usize,
    pub requests_rejected: usize,
    pub average_queue_wait_time: Duration,
    pub admission_rate: f64,
}

impl AdmissionStats {
    pub fn new() -> Self {
        Self {
            requests_admitted: 0,
            requests_queued: 0,
            requests_rejected: 0,
            average_queue_wait_time: Duration::from_millis(0),
            admission_rate: 0.0,
        }
    }
    
    /// Update admission rate
    pub fn update_admission_rate(&mut self) {
        let total_requests = self.requests_admitted + self.requests_queued + self.requests_rejected;
        if total_requests > 0 {
            self.admission_rate = self.requests_admitted as f64 / total_requests as f64;
        }
    }
}

impl std::fmt::Display for AdmissionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Admission Stats: {} admitted, {} queued, {} rejected ({:.1}% admission rate)",
               self.requests_admitted, self.requests_queued, self.requests_rejected, self.admission_rate * 100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::ModelConfig;
    
    #[test]
    fn test_admission_controller() {
        let limits = ResourceLimits::default();
        let mut controller = AdmissionController::new(limits);
        let model_config = ModelConfig::new(32000, 4096, 32, 32);
        
        // Test immediate admission
        let decision = controller.admit_request(
            1,
            vec![1, 2, 3, 4, 5],
            100,
            crate::RequestPriority::Normal,
            &model_config,
        );
        
        match decision {
            AdmissionDecision::Admitted => {
                assert_eq!(controller.get_resource_usage().active_requests, 1);
            },
            _ => panic!("Expected immediate admission"),
        }
    }
    
    #[test]
    fn test_resource_requirements() {
        let model_config = ModelConfig::new(32000, 4096, 32, 32);
        let requirements = ResourceRequirements::calculate(
            &vec![1; 1000], // 1000 token prompt
            2000, // 2000 token max output
            &model_config,
        );
        
        assert!(requirements.sequence_length > 0);
        assert!(requirements.memory_usage > 0);
        assert!(requirements.gpu_utilization > 0.0);
    }
}