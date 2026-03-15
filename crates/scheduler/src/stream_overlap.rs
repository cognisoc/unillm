//! Stream overlap implementation for H2D/compute overlap

use std::collections::{VecDeque, HashMap};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Stream types for different operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamType {
    /// Host-to-device transfer stream
    H2D,
    /// Device-to-host transfer stream
    D2H,
    /// Compute stream for inference
    Compute,
    /// Prefill stream for prompt processing
    Prefill,
    /// Decode stream for token generation
    Decode,
}

/// Stream priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StreamPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// A stream operation
#[derive(Debug, Clone)]
pub struct StreamOperation {
    /// Operation ID
    pub op_id: u32,
    /// Stream type
    pub stream_type: StreamType,
    /// Priority
    pub priority: StreamPriority,
    /// Operation data
    pub data: OperationData,
    /// Creation timestamp
    pub created_at: Instant,
    /// Estimated duration
    pub estimated_duration: Duration,
    /// Dependencies (operation IDs that must complete first)
    pub dependencies: Vec<u32>,
    /// Completion timestamp
    pub completed_at: Option<Instant>,
}

/// Operation data for different stream types
#[derive(Debug, Clone)]
pub enum OperationData {
    /// Host-to-device transfer
    H2DTransfer {
        src_ptr: u64,
        dst_ptr: u64,
        size: usize,
    },
    /// Device-to-host transfer
    D2HTransfer {
        src_ptr: u64,
        dst_ptr: u64,
        size: usize,
    },
    /// Compute operation
    Compute {
        kernel_id: u32,
        grid_dim: [u32; 3],
        block_dim: [u32; 3],
        shared_mem: u32,
        args: Vec<u64>,
    },
    /// Prefill operation
    Prefill {
        request_id: u32,
        chunk_id: u32,
        tokens: Vec<u32>,
    },
    /// Decode operation
    Decode {
        request_id: u32,
        batch_id: u32,
        num_tokens: usize,
    },
}

/// Stream overlap manager
pub struct StreamOverlapManager {
    /// Stream pools for different operation types
    stream_pools: HashMap<StreamType, VecDeque<u32>>,
    /// Pending operations waiting for streams
    pending_operations: VecDeque<StreamOperation>,
    /// Active operations currently running
    active_operations: HashMap<u32, StreamOperation>,
    /// Completed operations
    completed_operations: VecDeque<StreamOperation>,
    /// Next operation ID
    next_op_id: u32,
    /// Statistics
    stats: StreamOverlapStats,
}

impl StreamOverlapManager {
    /// Create a new stream overlap manager
    pub fn new() -> Self {
        let mut stream_pools = HashMap::new();
        
        // Initialize stream pools for each stream type
        for stream_type in [StreamType::H2D, StreamType::D2H, StreamType::Compute, StreamType::Prefill, StreamType::Decode] {
            stream_pools.insert(stream_type, VecDeque::new());
        }
        
        Self {
            stream_pools,
            pending_operations: VecDeque::new(),
            active_operations: HashMap::new(),
            completed_operations: VecDeque::new(),
            next_op_id: 0,
            stats: StreamOverlapStats::new(),
        }
    }
    
    /// Add a stream to the pool
    pub fn add_stream(&mut self, stream_type: StreamType, stream_id: u32) {
        self.stream_pools
            .entry(stream_type)
            .or_insert_with(VecDeque::new)
            .push_back(stream_id);
        
        println!("Added stream {} to {:?} pool", stream_id, stream_type);
    }
    
    /// Submit an operation for execution
    pub fn submit_operation(
        &mut self,
        stream_type: StreamType,
        priority: StreamPriority,
        data: OperationData,
        estimated_duration: Duration,
        dependencies: Vec<u32>,
    ) -> u32 {
        let op_id = self.next_op_id;
        self.next_op_id += 1;
        
        let operation = StreamOperation {
            op_id,
            stream_type,
            priority,
            data,
            created_at: Instant::now(),
            estimated_duration,
            dependencies,
            completed_at: None,
        };
        
        self.pending_operations.push_back(operation);
        self.stats.operations_submitted += 1;
        
        println!("Submitted operation {} for {:?} stream", op_id, stream_type);
        
        op_id
    }
    
    /// Schedule operations to available streams
    pub fn schedule_operations(&mut self) -> Vec<(u32, u32)> {
        let mut scheduled = Vec::new();
        
        // Sort pending operations by priority
        let mut sorted_ops: Vec<_> = self.pending_operations.drain(..).collect();
        sorted_ops.sort_by(|a, b| b.priority.cmp(&a.priority));
        
        for operation in sorted_ops {
            // Check if dependencies are satisfied
            if self.are_dependencies_satisfied(&operation) {
                // Try to find an available stream
                if let Some(stream_id) = self.get_available_stream(operation.stream_type) {
                    // Schedule the operation
                    self.active_operations.insert(operation.op_id, operation.clone());
                    scheduled.push((operation.op_id, stream_id));
                    
                    println!("Scheduled operation {} on stream {}", operation.op_id, stream_id);
                } else {
                    // No stream available, put back in pending
                    self.pending_operations.push_back(operation);
                }
            } else {
                // Dependencies not satisfied, put back in pending
                self.pending_operations.push_back(operation);
            }
        }
        
        scheduled
    }
    
    /// Complete an operation
    pub fn complete_operation(&mut self, op_id: u32) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(mut operation) = self.active_operations.remove(&op_id) {
            operation.completed_at = Some(Instant::now());
            self.completed_operations.push_back(operation);
            
            self.stats.operations_completed += 1;
            
            println!("Completed operation {}", op_id);
        }
        
        Ok(())
    }
    
    /// Check if dependencies are satisfied
    fn are_dependencies_satisfied(&self, operation: &StreamOperation) -> bool {
        for dep_id in &operation.dependencies {
            // Check if dependency is completed
            let is_completed = self.completed_operations
                .iter()
                .any(|op| op.op_id == *dep_id);
            
            if !is_completed {
                return false;
            }
        }
        
        true
    }
    
    /// Get an available stream of the specified type
    fn get_available_stream(&mut self, stream_type: StreamType) -> Option<u32> {
        self.stream_pools
            .get_mut(&stream_type)
            .and_then(|pool| pool.pop_front())
    }
    
    /// Return a stream to the pool
    pub fn return_stream(&mut self, stream_type: StreamType, stream_id: u32) {
        self.stream_pools
            .entry(stream_type)
            .or_insert_with(VecDeque::new)
            .push_back(stream_id);
        
        println!("Returned stream {} to {:?} pool", stream_id, stream_type);
    }
    
    /// Get pending operations count
    pub fn pending_count(&self) -> usize {
        self.pending_operations.len()
    }
    
    /// Get active operations count
    pub fn active_count(&self) -> usize {
        self.active_operations.len()
    }
    
    /// Get stream overlap statistics
    pub fn get_stats(&self) -> &StreamOverlapStats {
        &self.stats
    }
    
    /// Get active operations
    pub fn get_active_operations(&self) -> Vec<&StreamOperation> {
        self.active_operations.values().collect()
    }
    
    /// Estimate overlap efficiency
    pub fn estimate_overlap_efficiency(&self) -> f64 {
        if self.stats.operations_completed == 0 {
            return 0.0;
        }
        
        // Calculate theoretical vs actual execution time
        let total_estimated_time: Duration = self.completed_operations
            .iter()
            .map(|op| op.estimated_duration)
            .sum();
        
        let total_actual_time: Duration = self.completed_operations
            .iter()
            .filter_map(|op| {
                op.completed_at.map(|completed| completed.duration_since(op.created_at))
            })
            .sum();
        
        if total_actual_time.as_nanos() > 0 {
            total_estimated_time.as_nanos() as f64 / total_actual_time.as_nanos() as f64
        } else {
            0.0
        }
    }
}

/// Stream overlap statistics
#[derive(Debug, Clone)]
pub struct StreamOverlapStats {
    pub operations_submitted: usize,
    pub operations_completed: usize,
    pub operations_failed: usize,
    pub total_overlap_time: Duration,
    pub average_operation_duration: Duration,
    pub overlap_efficiency: f64,
}

impl StreamOverlapStats {
    pub fn new() -> Self {
        Self {
            operations_submitted: 0,
            operations_completed: 0,
            operations_failed: 0,
            total_overlap_time: Duration::from_millis(0),
            average_operation_duration: Duration::from_millis(0),
            overlap_efficiency: 0.0,
        }
    }
}

impl std::fmt::Display for StreamOverlapStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Stream Overlap Stats: {} submitted, {} completed, {:.2}% efficiency",
               self.operations_submitted, self.operations_completed, self.overlap_efficiency * 100.0)
    }
}

/// Helper function to create H2D transfer operation
pub fn create_h2d_operation(
    src_ptr: u64,
    dst_ptr: u64,
    size: usize,
    priority: StreamPriority,
) -> OperationData {
    OperationData::H2DTransfer {
        src_ptr,
        dst_ptr,
        size,
    }
}

/// Helper function to create compute operation
pub fn create_compute_operation(
    kernel_id: u32,
    grid_dim: [u32; 3],
    block_dim: [u32; 3],
    shared_mem: u32,
    args: Vec<u64>,
) -> OperationData {
    OperationData::Compute {
        kernel_id,
        grid_dim,
        block_dim,
        shared_mem,
        args,
    }
}

/// Helper function to create prefill operation
pub fn create_prefill_operation(
    request_id: u32,
    chunk_id: u32,
    tokens: Vec<u32>,
) -> OperationData {
    OperationData::Prefill {
        request_id,
        chunk_id,
        tokens,
    }
}

/// Helper function to create decode operation
pub fn create_decode_operation(
    request_id: u32,
    batch_id: u32,
    num_tokens: usize,
) -> OperationData {
    OperationData::Decode {
        request_id,
        batch_id,
        num_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_stream_overlap_manager() {
        let mut manager = StreamOverlapManager::new();
        
        // Add some streams
        manager.add_stream(StreamType::H2D, 1);
        manager.add_stream(StreamType::Compute, 2);
        
        // Submit operations
        let op1 = manager.submit_operation(
            StreamType::H2D,
            StreamPriority::High,
            create_h2d_operation(0x1000, 0x2000, 1024, StreamPriority::High),
            Duration::from_millis(10),
            vec![],
        );
        
        let op2 = manager.submit_operation(
            StreamType::Compute,
            StreamPriority::Normal,
            create_compute_operation(1, [1, 1, 1], [256, 1, 1], 0, vec![]),
            Duration::from_millis(50),
            vec![op1], // Depends on op1
        );
        
        // Schedule operations
        let scheduled = manager.schedule_operations();
        assert_eq!(scheduled.len(), 1); // Only op1 should be scheduled (op2 depends on it)
        
        // Complete op1
        manager.complete_operation(op1).unwrap();
        
        // Schedule again
        let scheduled = manager.schedule_operations();
        assert_eq!(scheduled.len(), 1); // Now op2 should be scheduled
    }
}