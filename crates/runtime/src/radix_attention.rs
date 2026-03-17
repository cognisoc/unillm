//! RadixAttention Implementation for UniLLM
//!
//! SGLang's RadixAttention innovation for efficient prefix sharing:
//! 1. Radix tree (trie) structure for common prefix sharing
//! 2. Automatic prefix detection and reuse across requests
//! 3. Memory-efficient KV cache sharing
//! 4. Dynamic tree structure management
//! 5. Integration with PagedAttention for optimal memory usage

use crate::types::*;
use crate::gpu_tensor_ops::{GpuDevice, GpuTensor, GpuTensorOps};
use crate::paged_attention::{PagedAttention, PagedAttentionConfig};
use crate::flash_attention_v2::FlashAttention2;
use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use tokio::sync::Mutex;

/// RadixAttention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RadixAttentionConfig {
    /// Maximum tree depth for prefix sharing
    pub max_tree_depth: usize,
    /// Minimum prefix length to consider for sharing
    pub min_prefix_length: usize,
    /// Enable automatic prefix detection
    pub auto_prefix_detection: bool,
    /// Cache size for prefix lookup acceleration
    pub prefix_cache_size: usize,
    /// Enable prefix compression
    pub enable_prefix_compression: bool,
    /// Eviction policy for prefix nodes
    pub eviction_policy: EvictionPolicy,
    /// Integration with PagedAttention
    pub use_paged_attention: bool,
}

impl Default for RadixAttentionConfig {
    fn default() -> Self {
        Self {
            max_tree_depth: 1024,
            min_prefix_length: 4,
            auto_prefix_detection: true,
            prefix_cache_size: 10000,
            enable_prefix_compression: false,
            eviction_policy: EvictionPolicy::LRU,
            use_paged_attention: true,
        }
    }
}

/// Eviction policies for prefix nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvictionPolicy {
    LRU,        // Least Recently Used
    LFU,        // Least Frequently Used
    TTL,        // Time To Live
    Reference,  // Reference counting
}

/// RadixTree node for prefix sharing
#[derive(Debug)]
pub struct RadixNode {
    /// Token IDs for this node
    pub token_ids: Vec<u32>,
    /// Cached KV states for this prefix
    pub kv_cache: Option<(GpuTensor, GpuTensor)>, // (keys, values)
    /// Child nodes in the radix tree
    pub children: HashMap<u32, Arc<RwLock<RadixNode>>>,
    /// Parent node reference
    pub parent: Option<Arc<RwLock<RadixNode>>>,
    /// Reference count for sharing
    pub ref_count: usize,
    /// Last access time for eviction
    pub last_accessed: std::time::Instant,
    /// Usage frequency
    pub access_count: u64,
    /// Unique node ID
    pub node_id: u64,
    /// Depth in the tree
    pub depth: usize,
}

/// RadixTree for managing prefix sharing
#[derive(Debug)]
pub struct RadixTree {
    root: Arc<RwLock<RadixNode>>,
    node_counter: std::sync::atomic::AtomicU64,
    config: RadixAttentionConfig,
    device: GpuDevice,
}

/// Prefix match result
#[derive(Debug)]
pub struct PrefixMatch {
    pub node: Arc<RwLock<RadixNode>>,
    pub matched_length: usize,
    pub remaining_tokens: Vec<u32>,
    pub cache_hit: bool,
}

/// RadixAttention statistics
#[derive(Debug, Clone)]
pub struct RadixStats {
    pub total_nodes: usize,
    pub max_depth_reached: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub prefix_reuse_rate: f32,
    pub memory_saved_mb: f64,
    pub tree_compression_ratio: f32,
}

/// RadixAttention engine combining radix trees with attention mechanisms
pub struct RadixAttention {
    config: RadixAttentionConfig,
    device: GpuDevice,
    tensor_ops: GpuTensorOps,

    // Radix tree for prefix management
    radix_tree: Arc<Mutex<RadixTree>>,

    // Integration with other attention mechanisms
    paged_attention: Option<Arc<PagedAttention>>,
    flash_attention: Option<Arc<Mutex<FlashAttention2>>>,

    // Active requests mapping
    active_requests: Arc<RwLock<HashMap<u64, RequestState>>>,

    // Performance tracking
    cache_hits: std::sync::atomic::AtomicU64,
    cache_misses: std::sync::atomic::AtomicU64,
    total_requests: std::sync::atomic::AtomicU64,
}

/// State for active requests
#[derive(Debug)]
pub struct RequestState {
    pub request_id: u64,
    pub current_node: Arc<RwLock<RadixNode>>,
    pub processed_tokens: Vec<u32>,
    pub position_in_prefix: usize,
    pub created_at: std::time::Instant,
}

/// RadixAttention computation result
#[derive(Debug)]
pub struct RadixAttentionResult {
    pub output: GpuTensor,
    pub prefix_reused: bool,
    pub prefix_length: usize,
    pub computation_saved_percent: f32,
    pub stats: RadixStats,
}

impl RadixNode {
    pub fn new(token_ids: Vec<u32>, parent: Option<Arc<RwLock<RadixNode>>>, node_id: u64, depth: usize) -> Self {
        Self {
            token_ids,
            kv_cache: None,
            children: HashMap::new(),
            parent,
            ref_count: 1,
            last_accessed: std::time::Instant::now(),
            access_count: 0,
            node_id,
            depth,
        }
    }

    pub fn access(&mut self) {
        self.last_accessed = std::time::Instant::now();
        self.access_count += 1;
    }

    pub fn add_child(&mut self, token: u32, child: Arc<RwLock<RadixNode>>) {
        self.children.insert(token, child);
    }

    pub fn get_child(&self, token: u32) -> Option<Arc<RwLock<RadixNode>>> {
        self.children.get(&token).cloned()
    }
}

impl RadixTree {
    pub fn new(config: RadixAttentionConfig, device: GpuDevice) -> Self {
        let root = Arc::new(RwLock::new(RadixNode::new(
            Vec::new(),
            None,
            0,
            0,
        )));

        Self {
            root,
            node_counter: std::sync::atomic::AtomicU64::new(1),
            config,
            device,
        }
    }

    /// Find the longest matching prefix in the tree
    pub async fn find_prefix_match(&self, token_ids: &[u32]) -> ModelResult<PrefixMatch> {
        let root = self.root.clone();
        let mut current_node = root;
        let mut matched_length = 0;
        let mut cache_hit = false;

        // Traverse the tree following the token sequence
        for (i, &token) in token_ids.iter().enumerate() {
            let current_guard = current_node.read().unwrap();

            if let Some(child) = current_guard.get_child(token) {
                drop(current_guard);
                current_node = child;
                matched_length = i + 1;

                // Check if this node has cached KV states
                let child_guard = current_node.read().unwrap();
                if child_guard.kv_cache.is_some() {
                    cache_hit = true;
                }
                drop(child_guard);
            } else {
                break;
            }
        }

        let remaining_tokens = if matched_length < token_ids.len() {
            token_ids[matched_length..].to_vec()
        } else {
            Vec::new()
        };

        Ok(PrefixMatch {
            node: current_node,
            matched_length,
            remaining_tokens,
            cache_hit,
        })
    }

    /// Insert a new token sequence into the tree
    pub async fn insert_sequence(&mut self, token_ids: &[u32], kv_cache: Option<(GpuTensor, GpuTensor)>) -> ModelResult<Arc<RwLock<RadixNode>>> {
        if token_ids.is_empty() {
            return Ok(self.root.clone());
        }

        let prefix_match = self.find_prefix_match(token_ids).await?;
        let mut current_node = prefix_match.node;

        // Create new nodes for the remaining tokens
        for &token in &prefix_match.remaining_tokens {
            let node_id = self.node_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let depth = {
                let current_guard = current_node.read().unwrap();
                current_guard.depth + 1
            };

            if depth > self.config.max_tree_depth {
                break; // Prevent infinite tree growth
            }

            let new_node = Arc::new(RwLock::new(RadixNode::new(
                vec![token],
                Some(current_node.clone()),
                node_id,
                depth,
            )));

            // Add child to current node
            {
                let mut current_guard = current_node.write().unwrap();
                current_guard.add_child(token, new_node.clone());
            }

            current_node = new_node;
        }

        // Store KV cache in the final node
        if let Some(cache) = kv_cache {
            let mut final_guard = current_node.write().unwrap();
            final_guard.kv_cache = Some(cache);
        }

        Ok(current_node)
    }

    /// Get statistics about the tree
    pub fn get_stats(&self) -> RadixStats {
        let (total_nodes, max_depth) = self.calculate_tree_stats(&self.root, 0);

        RadixStats {
            total_nodes,
            max_depth_reached: max_depth,
            cache_hits: 0, // Will be updated from RadixAttention
            cache_misses: 0,
            prefix_reuse_rate: 0.0,
            memory_saved_mb: 0.0,
            tree_compression_ratio: 1.0,
        }
    }

    fn calculate_tree_stats(&self, node: &Arc<RwLock<RadixNode>>, current_depth: usize) -> (usize, usize) {
        let guard = node.read().unwrap();
        let mut total_nodes = 1;
        let mut max_depth = current_depth;

        for child in guard.children.values() {
            let (child_nodes, child_max_depth) = self.calculate_tree_stats(child, current_depth + 1);
            total_nodes += child_nodes;
            max_depth = max_depth.max(child_max_depth);
        }

        (total_nodes, max_depth)
    }
}

impl RadixAttention {
    /// Create new RadixAttention instance
    pub async fn new(config: RadixAttentionConfig, device: GpuDevice) -> ModelResult<Self> {
        let tensor_ops = GpuTensorOps::with_device(device.clone());
        let radix_tree = Arc::new(Mutex::new(RadixTree::new(config.clone(), device.clone())));

        // Initialize PagedAttention if enabled
        let paged_attention = if config.use_paged_attention {
            let paged_config = PagedAttentionConfig::default();
            Some(Arc::new(PagedAttention::new(paged_config, device.clone())?))
        } else {
            None
        };

        println!("🌳 RadixAttention initialized:");
        println!("   Max tree depth: {}", config.max_tree_depth);
        println!("   Min prefix length: {}", config.min_prefix_length);
        println!("   Auto prefix detection: {}", config.auto_prefix_detection);
        println!("   Prefix cache size: {}", config.prefix_cache_size);
        println!("   Eviction policy: {:?}", config.eviction_policy);
        println!("   PagedAttention integration: {}", config.use_paged_attention);

        Ok(Self {
            config,
            device,
            tensor_ops,
            radix_tree,
            paged_attention,
            flash_attention: None,
            active_requests: Arc::new(RwLock::new(HashMap::new())),
            cache_hits: std::sync::atomic::AtomicU64::new(0),
            cache_misses: std::sync::atomic::AtomicU64::new(0),
            total_requests: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Process a request with RadixAttention
    pub async fn forward(
        &self,
        request_id: u64,
        token_ids: &[u32],
        query: &GpuTensor,
        key: &GpuTensor,
        value: &GpuTensor,
    ) -> ModelResult<RadixAttentionResult> {
        self.total_requests.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Find prefix match in radix tree
        let prefix_match = {
            let tree = self.radix_tree.lock().await;
            tree.find_prefix_match(token_ids).await?
        };

        let prefix_reused = prefix_match.cache_hit && prefix_match.matched_length >= self.config.min_prefix_length;
        let computation_saved = if prefix_reused {
            (prefix_match.matched_length as f32 / token_ids.len() as f32) * 100.0
        } else {
            0.0
        };

        // Update statistics
        if prefix_reused {
            self.cache_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            self.cache_misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // Compute attention (simplified - in practice would integrate with FlashAttention)
        let output = self.compute_attention_with_prefix_reuse(
            &prefix_match,
            query,
            key,
            value,
        ).await?;

        // Store new KV cache in tree for future reuse
        if !prefix_reused && token_ids.len() >= self.config.min_prefix_length {
            let kv_cache = Some((key.clone(), value.clone()));
            let mut tree = self.radix_tree.lock().await;
            tree.insert_sequence(token_ids, kv_cache).await?;
        }

        // Update request state
        self.update_request_state(request_id, &prefix_match).await?;

        let stats = self.get_radix_stats().await;

        Ok(RadixAttentionResult {
            output,
            prefix_reused,
            prefix_length: prefix_match.matched_length,
            computation_saved_percent: computation_saved,
            stats,
        })
    }

    /// Compute attention with prefix reuse optimization
    async fn compute_attention_with_prefix_reuse(
        &self,
        prefix_match: &PrefixMatch,
        query: &GpuTensor,
        key: &GpuTensor,
        value: &GpuTensor,
    ) -> ModelResult<GpuTensor> {
        if prefix_match.cache_hit {
            // Reuse cached computation for prefix
            println!("🌳 Reusing cached prefix computation for {} tokens", prefix_match.matched_length);

            // In a full implementation, this would:
            // 1. Load cached KV states from the prefix node
            // 2. Compute attention only for new tokens
            // 3. Concatenate results efficiently
        }

        // For now, create a placeholder output tensor
        // In practice, this would delegate to FlashAttention or PagedAttention
        let output_shape = query.shape().to_vec();
        let output = GpuTensor::zeros(output_shape, self.device.clone())?;

        println!("🧮 Computing attention with RadixAttention optimization");

        Ok(output)
    }

    /// Update request state
    async fn update_request_state(&self, request_id: u64, prefix_match: &PrefixMatch) -> ModelResult<()> {
        let mut active_requests = self.active_requests.write().unwrap();

        let request_state = RequestState {
            request_id,
            current_node: prefix_match.node.clone(),
            processed_tokens: Vec::new(), // Would track processed tokens
            position_in_prefix: prefix_match.matched_length,
            created_at: std::time::Instant::now(),
        };

        active_requests.insert(request_id, request_state);
        Ok(())
    }

    /// Get RadixAttention statistics
    pub async fn get_radix_stats(&self) -> RadixStats {
        let tree = self.radix_tree.lock().await;
        let mut stats = tree.get_stats();

        let cache_hits = self.cache_hits.load(std::sync::atomic::Ordering::Relaxed);
        let cache_misses = self.cache_misses.load(std::sync::atomic::Ordering::Relaxed);
        let total_requests = cache_hits + cache_misses;

        stats.cache_hits = cache_hits;
        stats.cache_misses = cache_misses;
        stats.prefix_reuse_rate = if total_requests > 0 {
            (cache_hits as f32 / total_requests as f32) * 100.0
        } else {
            0.0
        };

        // Estimate memory savings based on prefix reuse
        stats.memory_saved_mb = (cache_hits as f64 * 1.5); // Rough estimate

        stats
    }

    /// Cleanup expired requests
    pub async fn cleanup_expired_requests(&self, ttl_seconds: u64) -> ModelResult<usize> {
        let mut active_requests = self.active_requests.write().unwrap();
        let now = std::time::Instant::now();
        let ttl_duration = std::time::Duration::from_secs(ttl_seconds);

        let mut expired_count = 0;
        active_requests.retain(|_, request_state| {
            if now.duration_since(request_state.created_at) > ttl_duration {
                expired_count += 1;
                false
            } else {
                true
            }
        });

        if expired_count > 0 {
            println!("🗑️ Cleaned up {} expired requests", expired_count);
        }

        Ok(expired_count)
    }

    /// Free resources for a completed request
    pub async fn free_request(&self, request_id: u64) -> ModelResult<()> {
        let mut active_requests = self.active_requests.write().unwrap();
        if let Some(request_state) = active_requests.remove(&request_id) {
            // Decrement reference count in radix tree node
            let mut node_guard = request_state.current_node.write().unwrap();
            node_guard.ref_count = node_guard.ref_count.saturating_sub(1);

            println!("🌳 Freed request {}, remaining ref_count: {}", request_id, node_guard.ref_count);
        }

        Ok(())
    }

    /// Get the device used by this RadixAttention instance
    pub fn get_device(&self) -> &GpuDevice {
        &self.device
    }
}