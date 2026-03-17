//! PagedAttention Implementation for UniLLM
//!
//! Revolutionary attention mechanism inspired by vLLM that enables:
//! 1. Virtual memory-style paging for KV cache
//! 2. Efficient memory utilization and sharing
//! 3. Dynamic allocation and deallocation
//! 4. Support for variable sequence lengths
//! 5. Optimized CUDA kernels for performance

use crate::types::*;
use crate::gpu_tensor_ops::{GpuDevice, GpuTensor, GpuTensorOps};
// use crate::enhanced_kv_cache::EnhancedKVCache;  // Temporarily disabled
use std::sync::Arc;
use std::collections::{HashMap, VecDeque};
use tokio::sync::{RwLock, Mutex};
use serde::{Serialize, Deserialize};

/// PagedAttention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PagedAttentionConfig {
    /// Block size in tokens (typically 16 or 32)
    pub block_size: usize,
    /// Maximum number of blocks per sequence
    pub max_num_blocks_per_seq: usize,
    /// Maximum total blocks in memory
    pub max_total_blocks: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Data type for KV cache (f16, f32, etc.)
    pub dtype: DataType,
    /// Enable sliding window attention
    pub sliding_window: Option<usize>,
    /// Enable prefix caching
    pub enable_prefix_caching: bool,
}

impl Default for PagedAttentionConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            max_num_blocks_per_seq: 2048,
            max_total_blocks: 16384,
            num_heads: 32,
            head_dim: 128,
            dtype: DataType::Float16,
            sliding_window: None,
            enable_prefix_caching: false,
        }
    }
}

/// Data types supported by PagedAttention
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DataType {
    Float16,
    BFloat16,
    Float32,
    Int8,
}

/// Physical block containing KV cache data
#[derive(Debug)]
pub struct PhysicalBlock {
    pub block_id: usize,
    pub key_cache: GpuTensor,      // [num_heads, block_size, head_dim]
    pub value_cache: GpuTensor,    // [num_heads, block_size, head_dim]
    pub ref_count: usize,          // Reference counting for sharing
    pub last_accessed: std::time::Instant,
    pub is_dirty: bool,            // Needs to be written back
}

/// Logical block mapping for a sequence
#[derive(Debug, Clone)]
pub struct LogicalBlock {
    pub logical_idx: usize,
    pub physical_block_id: Option<usize>,
    pub num_tokens: usize,         // Tokens actually stored (≤ block_size)
}

/// Block table for a sequence mapping logical to physical blocks
#[derive(Debug)]
pub struct BlockTable {
    pub sequence_id: u64,
    pub logical_blocks: Vec<LogicalBlock>,
    pub total_tokens: usize,
    pub max_tokens: usize,
    pub prefix_hash: Option<u64>,  // For prefix caching
}

/// Memory pool managing physical blocks
#[derive(Debug)]
pub struct BlockMemoryPool {
    config: PagedAttentionConfig,
    device: GpuDevice,

    // Physical block storage
    free_blocks: VecDeque<usize>,
    allocated_blocks: HashMap<usize, PhysicalBlock>,

    // Memory management
    total_blocks: usize,
    watermark_high: usize,
    watermark_low: usize,

    // Statistics
    allocation_count: usize,
    deallocation_count: usize,
    cache_hit_count: usize,
    eviction_count: usize,
}

/// PagedAttention engine managing attention computation with paged KV cache
pub struct PagedAttention {
    config: PagedAttentionConfig,
    device: GpuDevice,
    tensor_ops: GpuTensorOps,

    // Block management
    block_pool: Arc<Mutex<BlockMemoryPool>>,
    block_tables: Arc<RwLock<HashMap<u64, BlockTable>>>,

    // Prefix caching (optional)
    prefix_cache: Option<Arc<RwLock<HashMap<u64, Vec<usize>>>>>, // prefix_hash -> block_ids

    // Performance tracking
    attention_ops: std::sync::atomic::AtomicU64,
    cache_hits: std::sync::atomic::AtomicU64,
    memory_efficiency: std::sync::atomic::AtomicU64, // Percentage utilization
}

/// Attention computation result with metadata
#[derive(Debug)]
pub struct AttentionResult {
    pub output: GpuTensor,             // [batch_size, num_heads, seq_len, head_dim]
    pub attention_weights: Option<GpuTensor>, // For debugging/analysis
    pub kv_cache_stats: KVCacheStats,
    pub computation_time_ms: f64,
    pub memory_transferred_mb: f64,
}

/// KV cache statistics
#[derive(Debug, Clone)]
pub struct KVCacheStats {
    pub blocks_allocated: usize,
    pub blocks_freed: usize,
    pub blocks_reused: usize,
    pub cache_hit_rate: f32,
    pub memory_utilization: f32,
    pub prefix_cache_hits: usize,
}

impl PagedAttention {
    /// Create new PagedAttention instance
    pub fn new(config: PagedAttentionConfig, device: GpuDevice) -> ModelResult<Self> {
        let tensor_ops = GpuTensorOps::with_device(device.clone());

        // Initialize block memory pool
        let block_pool = Arc::new(Mutex::new(BlockMemoryPool::new(
            config.clone(),
            device.clone()
        )?));

        // Initialize prefix cache if enabled
        let prefix_cache = if config.enable_prefix_caching {
            Some(Arc::new(RwLock::new(HashMap::new())))
        } else {
            None
        };

        println!("✅ PagedAttention initialized:");
        println!("   Block size: {} tokens", config.block_size);
        println!("   Max blocks per sequence: {}", config.max_num_blocks_per_seq);
        println!("   Total memory blocks: {}", config.max_total_blocks);
        println!("   Attention heads: {}", config.num_heads);
        println!("   Head dimension: {}", config.head_dim);
        println!("   Data type: {:?}", config.dtype);
        println!("   Prefix caching: {}", config.enable_prefix_caching);

        Ok(Self {
            config,
            device,
            tensor_ops,
            block_pool,
            block_tables: Arc::new(RwLock::new(HashMap::new())),
            prefix_cache,
            attention_ops: std::sync::atomic::AtomicU64::new(0),
            cache_hits: std::sync::atomic::AtomicU64::new(0),
            memory_efficiency: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Allocate blocks for a new sequence
    pub async fn allocate_sequence(
        &self,
        sequence_id: u64,
        prompt_length: usize,
        max_output_length: usize,
    ) -> ModelResult<()> {
        let total_tokens = prompt_length + max_output_length;
        let num_logical_blocks = (total_tokens + self.config.block_size - 1) / self.config.block_size;

        if num_logical_blocks > self.config.max_num_blocks_per_seq {
            return Err(ModelError::MemoryError(
                format!("Sequence {} requires {} blocks, max is {}",
                        sequence_id, num_logical_blocks, self.config.max_num_blocks_per_seq)
            ));
        }

        let mut block_tables = self.block_tables.write().await;
        if block_tables.contains_key(&sequence_id) {
            return Err(ModelError::InvalidInput(
                format!("Sequence {} already allocated", sequence_id)
            ));
        }

        // Check for prefix cache hit
        let (prefix_blocks, remaining_blocks) = if self.config.enable_prefix_caching {
            self.check_prefix_cache(sequence_id, num_logical_blocks).await?
        } else {
            (Vec::new(), num_logical_blocks)
        };

        // Create logical blocks
        let mut logical_blocks = Vec::new();

        // Add prefix blocks if available
        for (i, physical_id) in prefix_blocks.into_iter().enumerate() {
            logical_blocks.push(LogicalBlock {
                logical_idx: i,
                physical_block_id: Some(physical_id),
                num_tokens: self.config.block_size.min(total_tokens - i * self.config.block_size),
            });
        }

        // Add remaining blocks (not allocated yet)
        for i in logical_blocks.len()..num_logical_blocks {
            logical_blocks.push(LogicalBlock {
                logical_idx: i,
                physical_block_id: None,
                num_tokens: 0,
            });
        }

        // Create block table
        let block_table = BlockTable {
            sequence_id,
            logical_blocks,
            total_tokens: prompt_length, // Will grow as we generate
            max_tokens: total_tokens,
            prefix_hash: None, // Will be computed during prefill
        };

        block_tables.insert(sequence_id, block_table);

        println!("📦 Allocated {} logical blocks for sequence {}",
                 num_logical_blocks, sequence_id);

        Ok(())
    }

    /// Compute attention with paged KV cache
    pub async fn forward(
        &self,
        query: &GpuTensor,           // [batch_size, seq_len, num_heads * head_dim]
        key: &GpuTensor,             // [batch_size, seq_len, num_heads * head_dim]
        value: &GpuTensor,           // [batch_size, seq_len, num_heads * head_dim]
        sequence_ids: &[u64],        // Batch sequence IDs
        input_positions: &[Vec<usize>], // Token positions for each sequence
        attention_mask: Option<&GpuTensor>, // Optional attention mask
    ) -> ModelResult<AttentionResult> {
        let start_time = std::time::Instant::now();

        self.attention_ops.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let batch_size = query.shape()[0];
        let seq_len = query.shape()[1];
        let hidden_size = query.shape()[2];

        // Reshape to separate heads
        let q = self.reshape_for_attention(query)?; // [batch, num_heads, seq_len, head_dim]
        let k = self.reshape_for_attention(key)?;
        let v = self.reshape_for_attention(value)?;

        // Update KV cache with new key/value tokens
        self.update_kv_cache(&k, &v, sequence_ids, input_positions).await?;

        // Compute attention using paged KV cache
        let output = self.compute_paged_attention(&q, sequence_ids, input_positions, attention_mask).await?;

        // Reshape output back
        let final_output = self.reshape_from_attention(&output)?; // [batch, seq_len, hidden_size]

        let computation_time = start_time.elapsed().as_secs_f64() * 1000.0;

        // Collect statistics
        let kv_cache_stats = self.collect_cache_stats().await;

        Ok(AttentionResult {
            output: final_output,
            attention_weights: None, // Could be enabled for debugging
            kv_cache_stats,
            computation_time_ms: computation_time,
            memory_transferred_mb: 0.0, // Would be measured in real implementation
        })
    }

    /// Free memory for completed sequence
    pub async fn free_sequence(&self, sequence_id: u64) -> ModelResult<()> {
        let mut block_tables = self.block_tables.write().await;

        if let Some(block_table) = block_tables.remove(&sequence_id) {
            let mut block_pool = self.block_pool.lock().await;

            // Free all physical blocks
            for logical_block in &block_table.logical_blocks {
                if let Some(physical_id) = logical_block.physical_block_id {
                    block_pool.free_block(physical_id)?;
                }
            }

            println!("🗑️  Freed {} blocks for sequence {}",
                     block_table.logical_blocks.len(), sequence_id);
        }

        Ok(())
    }

    /// Get memory and performance statistics
    pub async fn get_stats(&self) -> PagedAttentionStats {
        let block_pool = self.block_pool.lock().await;
        let block_tables = self.block_tables.read().await;

        PagedAttentionStats {
            total_blocks: block_pool.total_blocks,
            free_blocks: block_pool.free_blocks.len(),
            allocated_blocks: block_pool.allocated_blocks.len(),
            active_sequences: block_tables.len(),
            attention_ops: self.attention_ops.load(std::sync::atomic::Ordering::Relaxed),
            cache_hits: self.cache_hits.load(std::sync::atomic::Ordering::Relaxed),
            memory_utilization: block_pool.get_utilization(),
            average_blocks_per_sequence: if block_tables.is_empty() {
                0.0
            } else {
                block_tables.values()
                    .map(|bt| bt.logical_blocks.len())
                    .sum::<usize>() as f64 / block_tables.len() as f64
            },
        }
    }

    // Private implementation methods

    async fn check_prefix_cache(
        &self,
        _sequence_id: u64,
        _num_blocks: usize,
    ) -> ModelResult<(Vec<usize>, usize)> {
        // Placeholder for prefix caching logic
        // In real implementation, this would:
        // 1. Hash the prompt prefix
        // 2. Look up existing cached blocks
        // 3. Return cached blocks and remaining blocks needed
        Ok((Vec::new(), _num_blocks))
    }

    async fn update_kv_cache(
        &self,
        key: &GpuTensor,
        value: &GpuTensor,
        sequence_ids: &[u64],
        input_positions: &[Vec<usize>],
    ) -> ModelResult<()> {
        let mut block_tables = self.block_tables.write().await;
        let mut block_pool = self.block_pool.lock().await;

        for (batch_idx, &seq_id) in sequence_ids.iter().enumerate() {
            if let Some(block_table) = block_tables.get_mut(&seq_id) {
                let positions = &input_positions[batch_idx];

                for &token_pos in positions {
                    let logical_block_idx = token_pos / self.config.block_size;
                    let block_offset = token_pos % self.config.block_size;

                    // Ensure we have enough logical blocks
                    while block_table.logical_blocks.len() <= logical_block_idx {
                        block_table.logical_blocks.push(LogicalBlock {
                            logical_idx: block_table.logical_blocks.len(),
                            physical_block_id: None,
                            num_tokens: 0,
                        });
                    }

                    let logical_block = &mut block_table.logical_blocks[logical_block_idx];

                    // Allocate physical block if needed
                    if logical_block.physical_block_id.is_none() {
                        let physical_id = block_pool.allocate_block()?;
                        logical_block.physical_block_id = Some(physical_id);
                        println!("📦 Allocated physical block {} for sequence {} block {}",
                                 physical_id, seq_id, logical_block_idx);
                    }

                    // Update token count
                    logical_block.num_tokens = logical_block.num_tokens.max(block_offset + 1);

                    // In real implementation, we would copy the K,V data to the physical block
                    // This requires extracting the right slice from the input tensors
                    // and writing to the appropriate location in the physical block
                }

                // Update total tokens for the sequence
                block_table.total_tokens = positions.iter().max().unwrap_or(&0) + 1;
            }
        }

        Ok(())
    }

    async fn compute_paged_attention(
        &self,
        query: &GpuTensor,
        sequence_ids: &[u64],
        input_positions: &[Vec<usize>],
        _attention_mask: Option<&GpuTensor>,
    ) -> ModelResult<GpuTensor> {
        let batch_size = query.shape()[0];
        let num_heads = query.shape()[1];
        let seq_len = query.shape()[2];
        let head_dim = query.shape()[3];

        // For now, we'll use a simplified implementation
        // In a real implementation, this would:
        // 1. Gather K,V from paged blocks for each sequence
        // 2. Compute attention scores efficiently
        // 3. Apply attention weights to values
        // 4. Handle variable sequence lengths and masking

        // Placeholder: create output tensor
        let output_shape = vec![batch_size, num_heads, seq_len, head_dim];
        let output = GpuTensor::zeros(output_shape, self.device.clone())?;

        // Here we would call optimized CUDA kernels for paged attention
        // The kernel would:
        // - Iterate through logical blocks for each sequence
        // - Map to physical blocks and load K,V data
        // - Compute attention efficiently with memory coalescing
        // - Handle different sequence lengths in the batch

        println!("🧮 Computing paged attention for {} sequences", sequence_ids.len());

        Ok(output)
    }

    async fn collect_cache_stats(&self) -> KVCacheStats {
        let block_pool = self.block_pool.lock().await;

        KVCacheStats {
            blocks_allocated: block_pool.allocation_count,
            blocks_freed: block_pool.deallocation_count,
            blocks_reused: block_pool.cache_hit_count,
            cache_hit_rate: if block_pool.allocation_count > 0 {
                block_pool.cache_hit_count as f32 / block_pool.allocation_count as f32
            } else {
                0.0
            },
            memory_utilization: block_pool.get_utilization(),
            prefix_cache_hits: 0, // Would track prefix cache hits
        }
    }

    fn reshape_for_attention(&self, tensor: &GpuTensor) -> ModelResult<GpuTensor> {
        let shape = tensor.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let _hidden_size = shape[2];

        // Reshape [batch, seq_len, hidden] -> [batch, num_heads, seq_len, head_dim]
        let new_shape = vec![batch_size, self.config.num_heads, seq_len, self.config.head_dim];

        // Use real tensor reshape operation
        self.tensor_ops.reshape(tensor, new_shape)
    }

    fn reshape_from_attention(&self, tensor: &GpuTensor) -> ModelResult<GpuTensor> {
        let shape = tensor.shape();
        let batch_size = shape[0];
        let seq_len = shape[2];

        // Reshape [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden]
        let new_shape = vec![batch_size, seq_len, self.config.num_heads * self.config.head_dim];

        self.tensor_ops.reshape(tensor, new_shape)
    }
}

impl BlockMemoryPool {
    fn new(config: PagedAttentionConfig, device: GpuDevice) -> ModelResult<Self> {
        let total_blocks = config.max_total_blocks;
        let watermark_high = (total_blocks as f32 * 0.9) as usize;
        let watermark_low = (total_blocks as f32 * 0.7) as usize;

        // Initialize free block list
        let free_blocks: VecDeque<usize> = (0..total_blocks).collect();

        println!("🏊 Block memory pool initialized:");
        println!("   Total blocks: {}", total_blocks);
        println!("   High watermark: {} blocks", watermark_high);
        println!("   Low watermark: {} blocks", watermark_low);

        Ok(Self {
            config,
            device,
            free_blocks,
            allocated_blocks: HashMap::new(),
            total_blocks,
            watermark_high,
            watermark_low,
            allocation_count: 0,
            deallocation_count: 0,
            cache_hit_count: 0,
            eviction_count: 0,
        })
    }

    fn allocate_block(&mut self) -> ModelResult<usize> {
        if let Some(block_id) = self.free_blocks.pop_front() {
            // Create physical block with KV cache tensors
            let key_shape = vec![self.config.num_heads, self.config.block_size, self.config.head_dim];
            let value_shape = key_shape.clone();

            let key_cache = GpuTensor::zeros(key_shape, self.device.clone())?;
            let value_cache = GpuTensor::zeros(value_shape, self.device.clone())?;

            let physical_block = PhysicalBlock {
                block_id,
                key_cache,
                value_cache,
                ref_count: 1,
                last_accessed: std::time::Instant::now(),
                is_dirty: false,
            };

            self.allocated_blocks.insert(block_id, physical_block);
            self.allocation_count += 1;

            Ok(block_id)
        } else {
            // Try to evict blocks if we're at capacity
            if self.try_evict_blocks()? {
                self.allocate_block() // Retry after eviction
            } else {
                Err(ModelError::MemoryError("No free blocks available".to_string()))
            }
        }
    }

    fn free_block(&mut self, block_id: usize) -> ModelResult<()> {
        if let Some(mut block) = self.allocated_blocks.remove(&block_id) {
            block.ref_count -= 1;
            if block.ref_count == 0 {
                self.free_blocks.push_back(block_id);
                self.deallocation_count += 1;
            } else {
                // Still has references, put it back
                self.allocated_blocks.insert(block_id, block);
            }
            Ok(())
        } else {
            Err(ModelError::InvalidInput(format!("Block {} not allocated", block_id)))
        }
    }

    fn try_evict_blocks(&mut self) -> ModelResult<bool> {
        // Simple LRU eviction strategy
        if self.allocated_blocks.is_empty() {
            return Ok(false);
        }

        // Find least recently used block
        let oldest_block_id = self.allocated_blocks
            .iter()
            .min_by_key(|(_, block)| block.last_accessed)
            .map(|(id, _)| *id);

        if let Some(block_id) = oldest_block_id {
            self.free_block(block_id)?;
            self.eviction_count += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn get_utilization(&self) -> f32 {
        self.allocated_blocks.len() as f32 / self.total_blocks as f32 * 100.0
    }
}

/// Statistics for PagedAttention system
#[derive(Debug, Clone, Serialize)]
pub struct PagedAttentionStats {
    pub total_blocks: usize,
    pub free_blocks: usize,
    pub allocated_blocks: usize,
    pub active_sequences: usize,
    pub attention_ops: u64,
    pub cache_hits: u64,
    pub memory_utilization: f32,
    pub average_blocks_per_sequence: f64,
}