//! Hybrid KV Cache combining RadixAttention and PagedAttention
//!
//! This module implements UniLLM's innovative hybrid caching system that combines:
//! - SGLang's RadixAttention for token-level prefix sharing (L1)
//! - vLLM's PagedAttention for block-level efficiency (L2)
//! - Compressed storage for cold data (L3)

use std::collections::{HashMap, BTreeMap, VecDeque};
use std::sync::{Arc, Mutex, atomic::{AtomicU32, AtomicU64, Ordering}};
use std::time::{Duration, Instant};

use crate::{PagedKvAllocator, KvAllocatorStats};

/// Token ID type
pub type TokenId = u32;

/// Node ID for radix tree
pub type NodeId = u64;

/// Sequence ID
pub type SequenceId = u32;

/// Block ID for paged cache
pub type BlockId = u32;

/// Handle to a cache entry
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheHandle {
    pub tier: CacheTier,
    pub id: u64,
}

/// Cache tier enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheTier {
    L1Radix,     // Hot: Token-level sharing
    L2Paged,     // Warm: Block-level efficiency
    L3Compressed,// Cold: Compressed storage
}

/// KV tensor pair representation
#[derive(Debug, Clone)]
pub struct KVTensorPair {
    pub key_ptr: u64,      // Device pointer to key tensor
    pub value_ptr: u64,    // Device pointer to value tensor
    pub size_bytes: usize, // Total size in bytes
    pub token_count: usize,// Number of tokens
    pub head_dim: usize,   // Dimension per attention head
    pub num_heads: usize,  // Number of attention heads
}

impl KVTensorPair {
    pub fn new(key_ptr: u64, value_ptr: u64, token_count: usize, head_dim: usize, num_heads: usize) -> Self {
        let size_bytes = token_count * head_dim * num_heads * 2 * 2; // 2 for K+V, 2 bytes for f16
        Self {
            key_ptr,
            value_ptr,
            size_bytes,
            token_count,
            head_dim,
            num_heads,
        }
    }
}

/// Radix tree node for token-level prefix sharing
#[derive(Debug)]
pub struct RadixNode {
    /// Token sequence this node represents
    pub tokens: Vec<TokenId>,

    /// Child nodes (next tokens)
    pub children: HashMap<TokenId, Box<RadixNode>>,

    /// KV data for this node (if any)
    pub kv_data: Option<KVTensorPair>,

    /// Reference count for active usage
    pub ref_count: AtomicU32,

    /// Last access timestamp (for LRU)
    pub last_access: AtomicU64,

    /// Node ID
    pub node_id: NodeId,

    /// Parent node (for navigation)
    pub parent: Option<NodeId>,
}

impl RadixNode {
    pub fn new(node_id: NodeId, tokens: Vec<TokenId>) -> Self {
        Self {
            tokens,
            children: HashMap::new(),
            kv_data: None,
            ref_count: AtomicU32::new(0),
            last_access: AtomicU64::new(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            ),
            node_id,
            parent: None,
        }
    }

    pub fn add_ref(&self) {
        self.ref_count.fetch_add(1, Ordering::Relaxed);
        self.update_access_time();
    }

    pub fn remove_ref(&self) {
        self.ref_count.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn get_ref_count(&self) -> u32 {
        self.ref_count.load(Ordering::Relaxed)
    }

    pub fn update_access_time(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.last_access.store(now, Ordering::Relaxed);
    }

    pub fn get_last_access(&self) -> u64 {
        self.last_access.load(Ordering::Relaxed)
    }
}

/// RadixCache implementation for L1 token-level sharing
#[derive(Debug)]
pub struct RadixCache {
    /// Root node of the radix tree
    root: RadixNode,

    /// All nodes indexed by node ID
    nodes: HashMap<NodeId, Box<RadixNode>>,

    /// Next available node ID
    next_node_id: NodeId,

    /// LRU eviction tracking
    lru_order: VecDeque<NodeId>,

    /// Maximum number of nodes
    max_nodes: usize,

    /// Cache statistics
    stats: RadixCacheStats,
}

#[derive(Debug, Default)]
pub struct RadixCacheStats {
    pub total_nodes: usize,
    pub active_refs: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub evictions: u64,
    pub memory_usage_bytes: usize,
}

impl RadixCache {
    pub fn new(max_nodes: usize) -> Self {
        let _root = RadixNode::new(0, Vec::new());
        let mut nodes = HashMap::new();
        nodes.insert(0, Box::new(RadixNode::new(0, Vec::new())));

        Self {
            root: RadixNode::new(0, Vec::new()),
            nodes,
            next_node_id: 1,
            lru_order: VecDeque::new(),
            max_nodes,
            stats: RadixCacheStats::default(),
        }
    }

    /// Find the longest prefix match for a token sequence
    pub fn find_longest_prefix(&mut self, tokens: &[TokenId]) -> (Option<NodeId>, usize) {
        if tokens.is_empty() {
            return (Some(0), 0); // Root node matches empty sequence
        }

        let mut current_node_id = 0;
        let mut matched_length = 0;
        let mut best_match = Some(0);
        let mut best_length = 0;

        while matched_length < tokens.len() {
            if let Some(node) = self.nodes.get(&current_node_id) {
                // Update access time
                node.update_access_time();

                // Try to find child for next token
                let next_token = tokens[matched_length];
                if let Some(child_node) = node.children.get(&next_token) {
                    current_node_id = child_node.node_id;
                    matched_length += 1;
                    
                    // Check if the child node has KV data (valid endpoint)
                    if let Some(child_full_node) = self.nodes.get(&current_node_id) {
                        if child_full_node.kv_data.is_some() {
                            best_match = Some(current_node_id);
                            best_length = matched_length;
                        }
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        // Update statistics
        if best_length > 0 {
            self.stats.cache_hits += 1;
        } else {
            self.stats.cache_misses += 1;
        }

        (best_match, best_length)
    }

    /// Insert a new sequence into the radix tree
    pub fn insert_sequence(&mut self, tokens: &[TokenId], kv_data: KVTensorPair) -> Result<NodeId, Box<dyn std::error::Error>> {
        if tokens.is_empty() {
            return Err("Cannot insert empty token sequence".into());
        }

        // Find insertion point
        let (prefix_node_id, prefix_length) = self.find_longest_prefix(tokens);
        let mut current_node_id = prefix_node_id.unwrap_or(0);

        // Create new nodes for remaining tokens
        for i in prefix_length..tokens.len() {
            let token = tokens[i];
            let new_node_id = self.next_node_id;
            self.next_node_id += 1;

            // Create new node
            let mut new_node = RadixNode::new(new_node_id, tokens[..=i].to_vec());
            new_node.parent = Some(current_node_id);

            // Insert into nodes map first
            self.nodes.insert(new_node_id, Box::new(new_node));

            // Insert into parent's children
            if let Some(parent_node) = self.nodes.get_mut(&current_node_id) {
                parent_node.children.insert(token, Box::new(RadixNode::new(new_node_id, vec![token])));
            }

            current_node_id = new_node_id;
        }

        // Set KV data on the final node
        if let Some(node) = self.nodes.get_mut(&current_node_id) {
            node.kv_data = Some(kv_data);
            node.add_ref(); // Add reference for the insertion
        }

        // Update LRU order
        self.lru_order.push_back(current_node_id);

        // Check if we need to evict
        if self.nodes.len() > self.max_nodes {
            self.evict_lru_nodes(self.nodes.len() - self.max_nodes)?;
        }

        self.stats.total_nodes = self.nodes.len();
        Ok(current_node_id)
    }

    /// Evict LRU nodes
    fn evict_lru_nodes(&mut self, count: usize) -> Result<(), Box<dyn std::error::Error>> {
        let mut evicted = 0;
        let mut nodes_to_remove = Vec::new();

        while evicted < count && !self.lru_order.is_empty() {
            if let Some(node_id) = self.lru_order.pop_front() {
                if let Some(node) = self.nodes.get(&node_id) {
                    // Don't evict nodes with active references
                    if node.get_ref_count() == 0 {
                        // Collect info needed for removal
                        let parent_id = node.parent;
                        let last_token = node.tokens.last().copied();

                        nodes_to_remove.push((node_id, parent_id, last_token));
                        evicted += 1;
                        self.stats.evictions += 1;
                    } else {
                        // Put back at end if still referenced
                        self.lru_order.push_back(node_id);
                    }
                }
            }
        }

        // Actually remove the nodes
        for (node_id, parent_id, last_token) in nodes_to_remove {
            // Remove from parent's children
            if let Some(parent_id) = parent_id {
                if let Some(parent) = self.nodes.get_mut(&parent_id) {
                    if let Some(token) = last_token {
                        parent.children.remove(&token);
                    }
                }
            }

            // Remove the node
            self.nodes.remove(&node_id);
        }

        Ok(())
    }

    pub fn get_stats(&self) -> &RadixCacheStats {
        &self.stats
    }
}

/// Adaptive cache policy for managing tier allocation
#[derive(Debug)]
pub struct AdaptiveCachePolicy {
    /// Current policy mode
    current_policy: CachePolicy,

    /// Workload analysis metrics
    prefix_sharing_ratio: f64,
    access_pattern_entropy: f64,
    memory_pressure: f64,

    /// Policy optimization parameters
    l1_promotion_threshold: f64,
    l2_demotion_threshold: f64,
    l3_compression_threshold: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum CachePolicy {
    RadixPreferred,      // Favor L1 radix cache
    PagedPreferred,      // Favor L2 paged cache
    Balanced,            // Mixed approach
    WorkloadAdaptive,    // ML-optimized policy
}

impl AdaptiveCachePolicy {
    pub fn new() -> Self {
        Self {
            current_policy: CachePolicy::Balanced,
            prefix_sharing_ratio: 0.0,
            access_pattern_entropy: 0.0,
            memory_pressure: 0.0,
            l1_promotion_threshold: 0.7,
            l2_demotion_threshold: 0.3,
            l3_compression_threshold: 0.1,
        }
    }

    /// Analyze workload characteristics and adjust policy
    pub fn analyze_workload(&mut self, access_patterns: &[AccessPattern]) {
        // Calculate prefix sharing ratio
        let total_accesses = access_patterns.len() as f64;
        let shared_prefixes = access_patterns.iter()
            .filter(|p| p.prefix_length > 0)
            .count() as f64;

        self.prefix_sharing_ratio = shared_prefixes / total_accesses;

        // Update policy based on analysis
        self.current_policy = match self.prefix_sharing_ratio {
            ratio if ratio > 0.6 => CachePolicy::RadixPreferred,
            ratio if ratio < 0.2 => CachePolicy::PagedPreferred,
            _ => CachePolicy::Balanced,
        };
    }

    /// Determine which tier should handle a new allocation
    pub fn select_tier(&self, sequence_info: &SequenceInfo) -> CacheTier {
        match self.current_policy {
            CachePolicy::RadixPreferred => {
                if sequence_info.has_common_prefix {
                    CacheTier::L1Radix
                } else {
                    CacheTier::L2Paged
                }
            },
            CachePolicy::PagedPreferred => CacheTier::L2Paged,
            CachePolicy::Balanced => {
                if sequence_info.has_common_prefix && sequence_info.access_frequency > self.l1_promotion_threshold {
                    CacheTier::L1Radix
                } else {
                    CacheTier::L2Paged
                }
            },
            CachePolicy::WorkloadAdaptive => {
                // TODO: Implement ML-based policy
                CacheTier::L2Paged
            }
        }
    }

    /// Determine if data should be promoted to a higher tier
    pub fn should_promote(&self, handle: CacheHandle, access_frequency: f64) -> Option<CacheTier> {
        match handle.tier {
            CacheTier::L3Compressed => {
                if access_frequency > self.l2_demotion_threshold {
                    Some(CacheTier::L2Paged)
                } else {
                    None
                }
            },
            CacheTier::L2Paged => {
                if access_frequency > self.l1_promotion_threshold {
                    Some(CacheTier::L1Radix)
                } else {
                    None
                }
            },
            CacheTier::L1Radix => None, // Already at highest tier
        }
    }

    /// Determine if data should be demoted to a lower tier
    pub fn should_demote(&self, handle: CacheHandle, access_frequency: f64) -> Option<CacheTier> {
        match handle.tier {
            CacheTier::L1Radix => {
                if access_frequency < self.l2_demotion_threshold {
                    Some(CacheTier::L2Paged)
                } else {
                    None
                }
            },
            CacheTier::L2Paged => {
                if access_frequency < self.l3_compression_threshold {
                    Some(CacheTier::L3Compressed)
                } else {
                    None
                }
            },
            CacheTier::L3Compressed => None, // Already at lowest tier
        }
    }
}

/// Access pattern information
#[derive(Debug, Clone)]
pub struct AccessPattern {
    pub sequence_id: SequenceId,
    pub tokens: Vec<TokenId>,
    pub prefix_length: usize,
    pub access_time: Instant,
    pub frequency: f64,
}

/// Sequence information for policy decisions
#[derive(Debug, Clone)]
pub struct SequenceInfo {
    pub sequence_id: SequenceId,
    pub length: usize,
    pub has_common_prefix: bool,
    pub access_frequency: f64,
    pub last_access: Instant,
}

/// Main hybrid cache implementation
pub struct HybridKVCache {
    /// L1: Radix tree cache for token-level sharing
    l1_radix: Arc<Mutex<RadixCache>>,

    /// L2: Paged block cache for efficient allocation
    l2_paged: Arc<Mutex<PagedKvAllocator>>,

    /// L3: Compressed storage (placeholder for now)
    l3_compressed: Arc<Mutex<HashMap<u64, Vec<u8>>>>,

    /// Adaptive policy engine
    policy_engine: AdaptiveCachePolicy,

    /// Access pattern tracking
    access_patterns: VecDeque<AccessPattern>,

    /// Cache handle mapping
    handle_mapping: HashMap<CacheHandle, CacheLocation>,

    /// Next handle ID
    next_handle_id: u64,

    /// Cache statistics
    stats: HybridCacheStats,
}

#[derive(Debug, Clone)]
pub struct CacheLocation {
    pub tier: CacheTier,
    pub internal_id: u64,
    pub sequence_id: SequenceId,
}

#[derive(Debug, Default, Clone)]
pub struct HybridCacheStats {
    pub l1_hits: u64,
    pub l2_hits: u64,
    pub l3_hits: u64,
    pub total_misses: u64,
    pub promotions: u64,
    pub demotions: u64,
    pub memory_usage_l1: usize,
    pub memory_usage_l2: usize,
    pub memory_usage_l3: usize,
}

impl HybridKVCache {
    pub fn new(
        l1_max_nodes: usize,
        l2_total_pages: usize,
        l2_page_size: usize,
        l2_pages_per_block: usize,
        base_device_ptr: u64,
    ) -> Self {
        Self {
            l1_radix: Arc::new(Mutex::new(RadixCache::new(l1_max_nodes))),
            l2_paged: Arc::new(Mutex::new(PagedKvAllocator::new(
                l2_total_pages,
                l2_page_size,
                l2_pages_per_block,
                base_device_ptr,
            ))),
            l3_compressed: Arc::new(Mutex::new(HashMap::new())),
            policy_engine: AdaptiveCachePolicy::new(),
            access_patterns: VecDeque::new(),
            handle_mapping: HashMap::new(),
            next_handle_id: 1,
            stats: HybridCacheStats::default(),
        }
    }

    /// Allocate cache for a new sequence
    pub fn allocate_sequence(&mut self, tokens: &[TokenId], max_length: usize) -> Result<CacheHandle, Box<dyn std::error::Error>> {
        // Determine optimal tier based on policy
        let sequence_info = SequenceInfo {
            sequence_id: self.next_handle_id as SequenceId,
            length: tokens.len(),
            has_common_prefix: self.has_common_prefix(tokens),
            access_frequency: 1.0, // New sequence starts with base frequency
            last_access: Instant::now(),
        };

        let target_tier = self.policy_engine.select_tier(&sequence_info);
        let handle = CacheHandle {
            tier: target_tier,
            id: self.next_handle_id,
        };
        self.next_handle_id += 1;

        match target_tier {
            CacheTier::L1Radix => {
                // Try L1 radix cache first
                let kv_data = KVTensorPair::new(0, 0, tokens.len(), 128, 32); // Placeholder
                let mut l1_cache = self.l1_radix.lock().unwrap();
                match l1_cache.insert_sequence(tokens, kv_data) {
                    Ok(node_id) => {
                        let location = CacheLocation {
                            tier: CacheTier::L1Radix,
                            internal_id: node_id,
                            sequence_id: sequence_info.sequence_id,
                        };
                        self.handle_mapping.insert(handle, location);
                        self.stats.l1_hits += 1;
                        Ok(handle)
                    },
                    Err(_) => {
                        // Fallback to L2 if L1 fails
                        drop(l1_cache);
                        self.allocate_in_l2(handle, max_length)
                    }
                }
            },
            CacheTier::L2Paged => {
                self.allocate_in_l2(handle, max_length)
            },
            CacheTier::L3Compressed => {
                // TODO: Implement L3 allocation
                Err("L3 compression not yet implemented".into())
            }
        }
    }

    fn allocate_in_l2(&mut self, handle: CacheHandle, max_length: usize) -> Result<CacheHandle, Box<dyn std::error::Error>> {
        let mut l2_cache = self.l2_paged.lock().unwrap();
        match l2_cache.allocate_sequence(max_length) {
            Ok((seq_id, _pages)) => {
                let location = CacheLocation {
                    tier: CacheTier::L2Paged,
                    internal_id: seq_id as u64,
                    sequence_id: seq_id,
                };
                let new_handle = CacheHandle {
                    tier: CacheTier::L2Paged,
                    id: handle.id,
                };
                self.handle_mapping.insert(new_handle, location);
                self.stats.l2_hits += 1;
                Ok(new_handle)
            },
            Err(e) => Err(e)
        }
    }

    /// Check if tokens have a common prefix with existing sequences
    fn has_common_prefix(&self, tokens: &[TokenId]) -> bool {
        if let Ok(mut l1_cache) = self.l1_radix.try_lock() {
            let (_node_id, prefix_length) = l1_cache.find_longest_prefix(tokens);
            prefix_length > 0
        } else {
            false
        }
    }

    /// Get comprehensive cache statistics
    pub fn get_stats(&self) -> HybridCacheStats {
        let mut stats = self.stats.clone();

        if let Ok(l1_cache) = self.l1_radix.try_lock() {
            let l1_stats = l1_cache.get_stats();
            stats.memory_usage_l1 = l1_stats.memory_usage_bytes;
        }

        if let Ok(l2_cache) = self.l2_paged.try_lock() {
            let l2_stats = l2_cache.get_stats();
            stats.memory_usage_l2 = l2_stats.allocated_pages * 16 * 4096; // Approximate
        }

        stats
    }

    /// Record access pattern for policy optimization
    pub fn record_access(&mut self, handle: CacheHandle, tokens: &[TokenId]) {
        let access_pattern = AccessPattern {
            sequence_id: handle.id as SequenceId,
            tokens: tokens.to_vec(),
            prefix_length: if let Ok(mut l1_cache) = self.l1_radix.try_lock() {
                l1_cache.find_longest_prefix(tokens).1
            } else {
                0
            },
            access_time: Instant::now(),
            frequency: 1.0, // TODO: Calculate actual frequency
        };

        self.access_patterns.push_back(access_pattern);

        // Keep only recent access patterns (sliding window)
        if self.access_patterns.len() > 1000 {
            self.access_patterns.pop_front();
        }

        // Periodically update policy based on access patterns
        if self.access_patterns.len() % 100 == 0 {
            let patterns: Vec<_> = self.access_patterns.iter().cloned().collect();
            self.policy_engine.analyze_workload(&patterns);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radix_cache_basic() {
        let mut cache = RadixCache::new(100);
        let tokens = vec![1, 2, 3, 4];
        let kv_data = KVTensorPair::new(0x1000, 0x2000, 4, 128, 32);

        let node_id = cache.insert_sequence(&tokens, kv_data).unwrap();
        assert!(node_id > 0);

        let (found_id, length) = cache.find_longest_prefix(&tokens);
        assert_eq!(found_id, Some(node_id));
        assert_eq!(length, 4);
    }

    #[test]
    fn test_hybrid_cache_allocation() {
        let mut cache = HybridKVCache::new(100, 1000, 16, 16, 0x10000000);
        let tokens = vec![1, 2, 3];

        let handle = cache.allocate_sequence(&tokens, 64).unwrap();
        assert_eq!(handle.tier, CacheTier::L1Radix);

        let stats = cache.get_stats();
        assert!(stats.l1_hits > 0 || stats.l2_hits > 0);
    }
}

impl HybridKVCache {
    /// Analyze a request for cache optimization (stub implementation)
    pub fn analyze_request(&self, _prompt: &str, _max_tokens: usize) -> CacheAnalysis {
        CacheAnalysis {
            hit_probability: 0.5,
            shared_prefix_length: 0,
            optimal_tier: CacheTier::L1Radix,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CacheAnalysis {
    pub hit_probability: f64,
    pub shared_prefix_length: usize,
    pub optimal_tier: CacheTier,
}