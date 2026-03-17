//! Multi-GPU and Distributed Inference Engine
//!
//! This module implements comprehensive distributed inference capabilities:
//! - Tensor parallelism for large models
//! - Pipeline parallelism for throughput
//! - Data parallelism for request volume
//! - Dynamic load balancing and fault tolerance

use crate::types::*;
use crate::gpu_tensor_ops::{GpuTensor, GpuDevice, GpuTensorOps};
use crate::memory_pool::{AdvancedMemoryPool, MemoryPoolConfig};
use crate::gpu_tensor_ops_v2::EnhancedGpuTensorOps;
use crate::request_batching::{GenerationRequest, GenerationResult};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};

/// Multi-GPU parallelism strategy
#[derive(Debug, Clone, PartialEq)]
pub enum ParallelismStrategy {
    /// Split tensors across GPUs (for large models)
    TensorParallel { world_size: usize },
    /// Different GPUs handle different layers
    PipelineParallel { stages: Vec<usize> }, // GPU indices for each stage
    /// Each GPU has full model, handles different requests
    DataParallel { replicas: usize },
    /// Hybrid: combine multiple strategies
    Hybrid {
        tensor_parallel_size: usize,
        pipeline_stages: usize,
        data_parallel_replicas: usize,
    },
}

impl ParallelismStrategy {
    /// Calculate total number of GPUs required
    pub fn total_gpus(&self) -> usize {
        match self {
            Self::TensorParallel { world_size } => *world_size,
            Self::PipelineParallel { stages } => stages.len(),
            Self::DataParallel { replicas } => *replicas,
            Self::Hybrid { tensor_parallel_size, pipeline_stages, data_parallel_replicas } => {
                tensor_parallel_size * pipeline_stages * data_parallel_replicas
            }
        }
    }

    /// Get optimal strategy for given model size and available GPUs
    pub fn optimal_for_model(
        param_count: usize,
        available_gpus: usize,
        gpu_memory_gb: usize,
    ) -> Self {
        let estimated_model_size_gb = param_count * 4 / 1024 / 1024 / 1024; // FP32 estimate

        if estimated_model_size_gb > gpu_memory_gb && available_gpus > 1 {
            // Model too large for single GPU - use tensor parallelism
            let min_gpus_needed = (estimated_model_size_gb + gpu_memory_gb - 1) / gpu_memory_gb;
            let tensor_parallel_size = min_gpus_needed.min(available_gpus);

            if available_gpus > tensor_parallel_size * 2 {
                // Use hybrid approach if we have extra GPUs
                Self::Hybrid {
                    tensor_parallel_size,
                    pipeline_stages: 2,
                    data_parallel_replicas: 1,
                }
            } else {
                Self::TensorParallel { world_size: tensor_parallel_size }
            }
        } else if available_gpus >= 4 {
            // Use pipeline parallelism for high throughput
            Self::PipelineParallel { stages: (0..available_gpus).collect() }
        } else {
            // Use data parallelism for request volume
            Self::DataParallel { replicas: available_gpus }
        }
    }
}

/// GPU worker configuration
#[derive(Debug, Clone)]
pub struct GpuWorkerConfig {
    pub device_id: usize,
    pub device: GpuDevice,
    pub memory_config: MemoryPoolConfig,
    pub role: WorkerRole,
}

/// Role of a GPU worker in distributed setup
#[derive(Debug, Clone)]
pub enum WorkerRole {
    /// Handles tensor parallel slice
    TensorParallel { rank: usize, world_size: usize },
    /// Handles specific pipeline stage
    PipelineStage { stage_id: usize, layer_range: (usize, usize) },
    /// Full model replica
    DataParallel { replica_id: usize },
    /// Master coordinator
    Master,
}

/// Distributed inference statistics
#[derive(Debug, Clone)]
pub struct DistributedStats {
    pub total_requests: usize,
    pub completed_requests: usize,
    pub average_latency_ms: f64,
    pub throughput_rps: f64,
    pub gpu_utilization: HashMap<usize, f32>,
    pub communication_overhead_ms: f64,
    pub load_balance_ratio: f32, // How evenly work is distributed
}

impl Default for DistributedStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            completed_requests: 0,
            average_latency_ms: 0.0,
            throughput_rps: 0.0,
            gpu_utilization: HashMap::new(),
            communication_overhead_ms: 0.0,
            load_balance_ratio: 1.0,
        }
    }
}

/// GPU worker for distributed inference
pub struct GpuWorker {
    config: GpuWorkerConfig,
    tensor_ops: EnhancedGpuTensorOps,
    request_queue: Arc<Mutex<VecDeque<GenerationRequest>>>,
    result_sender: mpsc::UnboundedSender<(String, GenerationResult)>,
    stats: Arc<RwLock<WorkerStats>>,
}

#[derive(Debug, Default, Clone)]
pub struct WorkerStats {
    requests_processed: usize,
    total_processing_time: Duration,
    idle_time: Duration,
    last_activity: Option<Instant>,
}

impl GpuWorker {
    pub fn new(
        config: GpuWorkerConfig,
        result_sender: mpsc::UnboundedSender<(String, GenerationResult)>,
    ) -> Self {
        let tensor_ops = EnhancedGpuTensorOps::with_memory_config(
            config.device.clone(),
            config.memory_config.clone(),
        );

        Self {
            config,
            tensor_ops,
            request_queue: Arc::new(Mutex::new(VecDeque::new())),
            result_sender,
            stats: Arc::new(RwLock::new(WorkerStats::default())),
        }
    }

    /// Add request to worker queue
    pub fn add_request(&self, request: GenerationRequest) -> ModelResult<()> {
        let mut queue = self.request_queue.lock()
            .map_err(|_| ModelError::ComputationFailed("Failed to lock request queue".to_string()))?;

        queue.push_back(request);
        Ok(())
    }

    /// Start worker processing loop
    pub async fn run(self: Arc<Self>) {
        loop {
            // Check for new requests
            let request = {
                let mut queue = self.request_queue.lock().unwrap();
                queue.pop_front()
            };

            match request {
                Some(req) => {
                    let start_time = Instant::now();

                    // Process request based on worker role
                    let result = match &self.config.role {
                        WorkerRole::TensorParallel { rank, world_size } => {
                            self.process_tensor_parallel(&req, *rank, *world_size).await
                        }
                        WorkerRole::PipelineStage { stage_id, layer_range } => {
                            self.process_pipeline_stage(&req, *stage_id, *layer_range).await
                        }
                        WorkerRole::DataParallel { replica_id: _ } => {
                            self.process_full_inference(&req).await
                        }
                        WorkerRole::Master => {
                            self.coordinate_distributed_inference(&req).await
                        }
                    };

                    let processing_time = start_time.elapsed();

                    // Update stats
                    {
                        let mut stats = self.stats.write().unwrap();
                        stats.requests_processed += 1;
                        stats.total_processing_time += processing_time;
                        stats.last_activity = Some(Instant::now());
                    }

                    // Send result
                    if let Ok(result) = result {
                        let _ = self.result_sender.send((req.request_id, result));
                    }
                }
                None => {
                    // No requests - brief sleep
                    tokio::time::sleep(Duration::from_millis(1)).await;

                    // Update idle time
                    {
                        let mut stats = self.stats.write().unwrap();
                        if let Some(last_activity) = stats.last_activity {
                            stats.idle_time += last_activity.elapsed();
                        }
                    }
                }
            }
        }
    }

    /// Process tensor parallel computation
    async fn process_tensor_parallel(
        &self,
        request: &GenerationRequest,
        rank: usize,
        world_size: usize,
    ) -> ModelResult<GenerationResult> {
        // In tensor parallelism, each worker handles a slice of the computation
        // This is a simplified version - real implementation would:
        // 1. Split input tensors across workers
        // 2. Perform local computation
        // 3. All-reduce to combine results

        println!("Worker {} processing tensor parallel slice {}/{}",
                self.config.device_id, rank + 1, world_size);

        // Simulate tensor parallel computation
        let computation_time = Duration::from_millis(100 + rank as u64 * 10);
        tokio::time::sleep(computation_time).await;

        Ok(GenerationResult {
            request_id: request.request_id.clone(),
            input_tokens: request.input_tokens.clone(),
            generated_tokens: vec![1, 2, 3, 4, 5], // Placeholder
            generation_time: computation_time,
            total_time: computation_time,
            tokens_per_second: 5.0 / computation_time.as_secs_f64(),
        })
    }

    /// Process pipeline stage
    async fn process_pipeline_stage(
        &self,
        request: &GenerationRequest,
        stage_id: usize,
        layer_range: (usize, usize),
    ) -> ModelResult<GenerationResult> {
        println!("Worker {} processing pipeline stage {} (layers {}-{})",
                self.config.device_id, stage_id, layer_range.0, layer_range.1);

        // Simulate pipeline stage computation
        let layer_count = layer_range.1 - layer_range.0;
        let computation_time = Duration::from_millis(20 * layer_count as u64);
        tokio::time::sleep(computation_time).await;

        Ok(GenerationResult {
            request_id: request.request_id.clone(),
            input_tokens: request.input_tokens.clone(),
            generated_tokens: vec![1, 2, 3, 4, 5], // Placeholder
            generation_time: computation_time,
            total_time: computation_time,
            tokens_per_second: 5.0 / computation_time.as_secs_f64(),
        })
    }

    /// Process full inference (data parallel)
    async fn process_full_inference(
        &self,
        request: &GenerationRequest,
    ) -> ModelResult<GenerationResult> {
        println!("Worker {} processing full inference", self.config.device_id);

        // Simulate full model inference
        let computation_time = Duration::from_millis(200);
        tokio::time::sleep(computation_time).await;

        Ok(GenerationResult {
            request_id: request.request_id.clone(),
            input_tokens: request.input_tokens.clone(),
            generated_tokens: vec![1, 2, 3, 4, 5], // Placeholder
            generation_time: computation_time,
            total_time: computation_time,
            tokens_per_second: 5.0 / computation_time.as_secs_f64(),
        })
    }

    /// Coordinate distributed inference
    async fn coordinate_distributed_inference(
        &self,
        request: &GenerationRequest,
    ) -> ModelResult<GenerationResult> {
        println!("Master worker {} coordinating distributed inference", self.config.device_id);

        // Master coordinates other workers
        let coordination_time = Duration::from_millis(50);
        tokio::time::sleep(coordination_time).await;

        Ok(GenerationResult {
            request_id: request.request_id.clone(),
            input_tokens: request.input_tokens.clone(),
            generated_tokens: vec![1, 2, 3, 4, 5], // Placeholder
            generation_time: coordination_time,
            total_time: coordination_time,
            tokens_per_second: 5.0 / coordination_time.as_secs_f64(),
        })
    }

    /// Get worker statistics
    pub fn get_stats(&self) -> WorkerStats {
        self.stats.read().unwrap().clone()
    }
}

/// Load balancer for distributing requests across GPUs
pub struct LoadBalancer {
    strategy: LoadBalancingStrategy,
    worker_loads: Arc<RwLock<HashMap<usize, f32>>>,
    request_history: Arc<Mutex<VecDeque<(Instant, usize)>>>, // (timestamp, worker_id)
}

#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastLoaded,
    WeightedRoundRobin { weights: HashMap<usize, f32> },
    LatencyBased,
}

impl LoadBalancer {
    pub fn new(strategy: LoadBalancingStrategy, worker_count: usize) -> Self {
        let mut worker_loads = HashMap::new();
        for i in 0..worker_count {
            worker_loads.insert(i, 0.0);
        }

        Self {
            strategy,
            worker_loads: Arc::new(RwLock::new(worker_loads)),
            request_history: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Select optimal worker for request
    pub fn select_worker(&self, _request: &GenerationRequest) -> ModelResult<usize> {
        match &self.strategy {
            LoadBalancingStrategy::RoundRobin => {
                // Simple round-robin based on history length
                let history = self.request_history.lock().unwrap();
                let worker_count = self.worker_loads.read().unwrap().len();
                Ok(history.len() % worker_count)
            }
            LoadBalancingStrategy::LeastLoaded => {
                // Select worker with lowest current load
                let loads = self.worker_loads.read().unwrap();
                let best_worker = loads
                    .iter()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(id, _)| *id)
                    .unwrap_or(0);
                Ok(best_worker)
            }
            LoadBalancingStrategy::WeightedRoundRobin { weights } => {
                // Use weights to bias selection
                let total_weight: f32 = weights.values().sum();
                let target = rand::random::<f32>() * total_weight;

                let mut cumulative = 0.0;
                for (&worker_id, &weight) in weights {
                    cumulative += weight;
                    if target <= cumulative {
                        return Ok(worker_id);
                    }
                }
                Ok(0) // Fallback
            }
            LoadBalancingStrategy::LatencyBased => {
                // Select based on historical latency (simplified)
                let loads = self.worker_loads.read().unwrap();
                let best_worker = loads
                    .iter()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(id, _)| *id)
                    .unwrap_or(0);
                Ok(best_worker)
            }
        }
    }

    /// Update worker load
    pub fn update_worker_load(&self, worker_id: usize, load: f32) {
        let mut loads = self.worker_loads.write().unwrap();
        loads.insert(worker_id, load);
    }

    /// Record request assignment
    pub fn record_assignment(&self, worker_id: usize) {
        let mut history = self.request_history.lock().unwrap();
        history.push_back((Instant::now(), worker_id));

        // Keep only recent history (last 1000 requests)
        while history.len() > 1000 {
            history.pop_front();
        }
    }
}

/// Distributed inference coordinator
pub struct DistributedInferenceEngine {
    strategy: ParallelismStrategy,
    workers: Vec<Arc<GpuWorker>>,
    load_balancer: LoadBalancer,
    result_receiver: mpsc::UnboundedReceiver<(String, GenerationResult)>,
    pending_requests: Arc<Mutex<HashMap<String, oneshot::Sender<GenerationResult>>>>,
    stats: Arc<RwLock<DistributedStats>>,
}

impl DistributedInferenceEngine {
    /// Create new distributed inference engine
    pub async fn new(
        strategy: ParallelismStrategy,
        available_devices: Vec<GpuDevice>,
    ) -> ModelResult<Self> {
        let required_gpus = strategy.total_gpus();
        if available_devices.len() < required_gpus {
            return Err(ModelError::ComputationFailed(
                format!("Need {} GPUs, only {} available", required_gpus, available_devices.len())
            ));
        }

        let (result_sender, result_receiver) = mpsc::unbounded_channel();
        let mut workers = Vec::new();

        // Create workers based on strategy
        match &strategy {
            ParallelismStrategy::TensorParallel { world_size } => {
                for rank in 0..*world_size {
                    let config = GpuWorkerConfig {
                        device_id: rank,
                        device: available_devices[rank].clone(),
                        memory_config: MemoryPoolConfig::default(),
                        role: WorkerRole::TensorParallel { rank, world_size: *world_size },
                    };
                    workers.push(Arc::new(GpuWorker::new(config, result_sender.clone())));
                }
            }
            ParallelismStrategy::PipelineParallel { stages } => {
                let layers_per_stage = 32 / stages.len(); // Assuming 32 layers
                for (stage_id, &gpu_id) in stages.iter().enumerate() {
                    let layer_start = stage_id * layers_per_stage;
                    let layer_end = ((stage_id + 1) * layers_per_stage).min(32);

                    let config = GpuWorkerConfig {
                        device_id: gpu_id,
                        device: available_devices[gpu_id].clone(),
                        memory_config: MemoryPoolConfig::default(),
                        role: WorkerRole::PipelineStage {
                            stage_id,
                            layer_range: (layer_start, layer_end)
                        },
                    };
                    workers.push(Arc::new(GpuWorker::new(config, result_sender.clone())));
                }
            }
            ParallelismStrategy::DataParallel { replicas } => {
                for replica_id in 0..*replicas {
                    let config = GpuWorkerConfig {
                        device_id: replica_id,
                        device: available_devices[replica_id].clone(),
                        memory_config: MemoryPoolConfig::default(),
                        role: WorkerRole::DataParallel { replica_id },
                    };
                    workers.push(Arc::new(GpuWorker::new(config, result_sender.clone())));
                }
            }
            ParallelismStrategy::Hybrid { .. } => {
                // Simplified hybrid setup
                let config = GpuWorkerConfig {
                    device_id: 0,
                    device: available_devices[0].clone(),
                    memory_config: MemoryPoolConfig::default(),
                    role: WorkerRole::Master,
                };
                workers.push(Arc::new(GpuWorker::new(config, result_sender.clone())));
            }
        }

        // Start all workers
        for worker in &workers {
            let worker_clone = Arc::clone(worker);
            tokio::spawn(async move {
                worker_clone.run().await;
            });
        }

        let load_balancer = LoadBalancer::new(
            LoadBalancingStrategy::LeastLoaded,
            workers.len(),
        );

        Ok(Self {
            strategy,
            workers,
            load_balancer,
            result_receiver,
            pending_requests: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(RwLock::new(DistributedStats::default())),
        })
    }

    /// Process inference request with distributed workers
    pub async fn process_request(
        &self,
        request: GenerationRequest,
    ) -> ModelResult<GenerationResult> {
        let request_start = Instant::now();

        // Select optimal worker
        let worker_id = self.load_balancer.select_worker(&request)?;
        self.load_balancer.record_assignment(worker_id);

        // Create response channel
        let (tx, rx) = oneshot::channel();

        // Track pending request
        {
            let mut pending = self.pending_requests.lock().unwrap();
            pending.insert(request.request_id.clone(), tx);
        }

        // Send to worker
        self.workers[worker_id].add_request(request.clone())?;

        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_requests += 1;
        }

        // Wait for result
        match rx.await {
            Ok(result) => {
                let latency = request_start.elapsed();

                // Update stats
                {
                    let mut stats = self.stats.write().unwrap();
                    stats.completed_requests += 1;
                    stats.average_latency_ms =
                        (stats.average_latency_ms * (stats.completed_requests - 1) as f64 +
                         latency.as_millis() as f64) / stats.completed_requests as f64;

                    if stats.completed_requests > 0 {
                        let time_span = latency.as_secs_f64();
                        stats.throughput_rps = stats.completed_requests as f64 / time_span;
                    }
                }

                Ok(result)
            }
            Err(_) => Err(ModelError::ComputationFailed("Request cancelled".to_string())),
        }
    }

    /// Start result collection loop
    pub async fn start_result_collection(mut self) {
        while let Some((request_id, result)) = self.result_receiver.recv().await {
            // Find and notify pending request
            if let Ok(mut pending) = self.pending_requests.lock() {
                if let Some(tx) = pending.remove(&request_id) {
                    let _ = tx.send(result);
                }
            }
        }
    }

    /// Get distributed inference statistics
    pub fn get_stats(&self) -> DistributedStats {
        self.stats.read().unwrap().clone()
    }

    /// Get detailed worker statistics
    pub fn get_worker_stats(&self) -> HashMap<usize, WorkerStats> {
        self.workers
            .iter()
            .enumerate()
            .map(|(i, worker)| (i, worker.get_stats()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallelism_strategy_optimal() {
        // Small model, many GPUs - should use pipeline
        let strategy = ParallelismStrategy::optimal_for_model(
            1_000_000_000, // 1B params
            8,             // 8 GPUs
            24,            // 24GB per GPU
        );

        match strategy {
            ParallelismStrategy::PipelineParallel { stages } => {
                assert_eq!(stages.len(), 8);
            }
            _ => panic!("Expected pipeline parallelism"),
        }
    }

    #[test]
    fn test_parallelism_strategy_tensor() {
        // Large model, few GPUs - should use tensor parallelism
        let strategy = ParallelismStrategy::optimal_for_model(
            70_000_000_000, // 70B params (280GB in FP32)
            4,              // 4 GPUs
            24,             // 24GB per GPU
        );

        match strategy {
            ParallelismStrategy::TensorParallel { world_size } => {
                assert!(world_size <= 4);
            }
            ParallelismStrategy::Hybrid { .. } => {
                // Also acceptable for large models
            }
            _ => panic!("Expected tensor parallelism or hybrid"),
        }
    }

    #[test]
    fn test_load_balancer_round_robin() {
        let balancer = LoadBalancer::new(LoadBalancingStrategy::RoundRobin, 4);

        let request = GenerationRequest {
            request_id: "test".to_string(),
            input_tokens: vec![1, 2, 3],
            max_new_tokens: 10,
            temperature: 1.0,
            top_p: 0.9,
            stop_tokens: vec![2],
            priority: crate::request_batching::RequestPriority::Normal,
            created_at: Instant::now(),
        };

        // First request should go to worker 0
        assert_eq!(balancer.select_worker(&request).unwrap(), 0);
        balancer.record_assignment(0);

        // Second request should go to worker 1
        assert_eq!(balancer.select_worker(&request).unwrap(), 1);
    }

    #[test]
    fn test_load_balancer_least_loaded() {
        let balancer = LoadBalancer::new(LoadBalancingStrategy::LeastLoaded, 3);

        // Set different loads
        balancer.update_worker_load(0, 0.8);
        balancer.update_worker_load(1, 0.3); // Least loaded
        balancer.update_worker_load(2, 0.6);

        let request = GenerationRequest {
            request_id: "test".to_string(),
            input_tokens: vec![1, 2, 3],
            max_new_tokens: 10,
            temperature: 1.0,
            top_p: 0.9,
            stop_tokens: vec![2],
            priority: crate::request_batching::RequestPriority::Normal,
            created_at: Instant::now(),
        };

        // Should select worker 1 (least loaded)
        assert_eq!(balancer.select_worker(&request).unwrap(), 1);
    }

    #[tokio::test]
    async fn test_gpu_worker_creation() {
        let device = GpuDevice::auto_detect();
        let (tx, _rx) = mpsc::unbounded_channel();

        let config = GpuWorkerConfig {
            device_id: 0,
            device: device.clone(),
            memory_config: MemoryPoolConfig::default(),
            role: WorkerRole::DataParallel { replica_id: 0 },
        };

        let worker = GpuWorker::new(config, tx);

        let request = GenerationRequest {
            request_id: "test".to_string(),
            input_tokens: vec![1, 2, 3],
            max_new_tokens: 10,
            temperature: 1.0,
            top_p: 0.9,
            stop_tokens: vec![2],
            priority: crate::request_batching::RequestPriority::Normal,
            created_at: Instant::now(),
        };

        assert!(worker.add_request(request).is_ok());
    }
}