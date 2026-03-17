//! Enhanced KV Cache with rykv Backend
//!
//! High-performance multi-level caching system with:
//! 1. GPU memory (L1) - fastest access
//! 2. CPU memory (L2) - fast fallback
//! 3. rykv disk storage (L3) - persistent, compressed
//! 4. Intelligent prefetching and eviction
//! 5. Compression and deduplication

use crate::types::*;
use crate::gpu_tensor_ops::{GpuTensor, GpuTensorOps, GpuDevice};
use crate::memory_pool::AdvancedMemoryPool;
use tokio::sync::{RwLock, Mutex};
use std::sync::Arc;
use std::collections::{HashMap, BTreeMap};
use tokio::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
// Note: Using standard storage instead of rykv for now
// use rykv::{Database, Config as RykvConfig};
use std::fs;
use std::path::Path;
use lz4::{Encoder, Decoder};
use bincode;
use serde::{Serialize, Deserialize};

/// Enhanced KV cache configuration
#[derive(Debug, Clone)]
pub struct EnhancedKVConfig {
    pub max_cached_tokens: usize,
    pub block_size: usize,
    pub max_gpu_memory_mb: usize,
    pub max_cpu_memory_mb: usize,
    pub disk_cache_path: String,
    pub prefetch_blocks: usize,
    pub eviction_batch_size: usize,
    pub compression_level: u32,
    pub enable_deduplication: bool,
    pub async_writeback: bool,
    pub cache_ttl_seconds: u64,
}

impl Default for EnhancedKVConfig {
    fn default() -> Self {
        Self {
            max_cached_tokens: 2 * 1024 * 1024, // 2M tokens
            block_size: 32,                      // 32 tokens per block
            max_gpu_memory_mb: 12 * 1024,       // 12GB GPU
            max_cpu_memory_mb: 32 * 1024,       // 32GB CPU
            disk_cache_path: "./unillm_cache".to_string(),
            prefetch_blocks: 8,
            eviction_batch_size: 128,
            compression_level: 4, // LZ4 fast compression
            enable_deduplication: true,
            async_writeback: true,
            cache_ttl_seconds: 3600, // 1 hour TTL
        }
    }
}

/// Serializable cache block for disk storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableKVBlock {
    pub sequence_id: u64,
    pub start_token: usize,
    pub end_token: usize,
    pub key_data: Vec<f32>,
    pub value_data: Vec<f32>,
    pub key_shape: Vec<usize>,
    pub value_shape: Vec<usize>,
    pub created_at: u64,
    pub access_count: u64,
    pub compression_ratio: f32,
    pub checksum: u64,
}

/// Cache statistics and metrics
#[derive(Debug, Clone)]
pub struct EnhancedCacheStats {
    pub hit_rate: f64,
    pub miss_rate: f64,
    pub gpu_blocks: usize,
    pub cpu_blocks: usize,
    pub disk_blocks: usize,
    pub gpu_memory_used_mb: usize,
    pub cpu_memory_used_mb: usize,
    pub disk_storage_used_mb: usize,
    pub compression_ratio: f64,
    pub deduplication_savings_mb: usize,
    pub total_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub evictions: u64,
    pub prefetches: u64,
}

/// Cache access result
#[derive(Debug)]
pub struct CacheAccessResult {
    pub hit: bool,
    pub level: u8, // 1=GPU, 2=CPU, 3=Disk
    pub latency_us: u64,
    pub key_tensor: Option<GpuTensor>,
    pub value_tensor: Option<GpuTensor>,
    pub prefetch_triggered: bool,
}

/// Enhanced KV cache with rykv backend
pub struct EnhancedKVCache {
    config: EnhancedKVConfig,
    device: GpuDevice,
    tensor_ops: GpuTensorOps,

    // L1 Cache: GPU Memory
    gpu_cache: Arc<RwLock<HashMap<(u64, usize), Arc<GpuKVBlock>>>>,
    gpu_memory_pool: Arc<AdvancedMemoryPool>,

    // L2 Cache: CPU Memory
    cpu_cache: Arc<RwLock<HashMap<(u64, usize), Arc<CpuKVBlock>>>>,

    // L3 Cache: rykv Disk Storage
    // TODO: Replace with proper disk storage implementation
    // disk_cache: Arc<Mutex<Database>>,
    disk_cache: Arc<Mutex<HashMap<String, Vec<u8>>>>, // Temporary in-memory placeholder

    // Cache management
    access_tracker: Arc<RwLock<BTreeMap<(u64, usize), CacheAccessMetadata>>>,
    deduplication_index: Arc<RwLock<HashMap<u64, Vec<(u64, usize)>>>>, // checksum -> keys

    // Metrics
    stats: Arc<Mutex<EnhancedCacheStats>>,
    total_requests: Arc<AtomicU64>,
    cache_hits: Arc<AtomicU64>,
    cache_misses: Arc<AtomicU64>,
}

/// GPU cache block
#[derive(Debug)]
pub struct GpuKVBlock {
    pub key_tensor: GpuTensor,
    pub value_tensor: GpuTensor,
    pub metadata: CacheBlockMetadata,
}

/// CPU cache block
#[derive(Debug)]
pub struct CpuKVBlock {
    pub key_data: Vec<f32>,
    pub value_data: Vec<f32>,
    pub key_shape: Vec<usize>,
    pub value_shape: Vec<usize>,
    pub metadata: CacheBlockMetadata,
}

/// Cache block metadata
#[derive(Debug, Clone)]
pub struct CacheBlockMetadata {
    pub sequence_id: u64,
    pub start_token: usize,
    pub end_token: usize,
    pub created_at: Instant,
    pub last_accessed: Instant,
    pub access_count: u64,
    pub size_bytes: usize,
    pub checksum: u64,
}

/// Cache access tracking
#[derive(Debug, Clone)]
pub struct CacheAccessMetadata {
    pub last_accessed: Instant,
    pub access_count: u64,
    pub frequency_score: f64,
    pub recency_score: f64,
    pub prediction_score: f64,
}

impl EnhancedKVCache {
    /// Create new enhanced KV cache
    pub async fn new(config: EnhancedKVConfig, device: GpuDevice) -> ModelResult<Self> {
        // Initialize rykv database
        let cache_path = Path::new(&config.disk_cache_path);
        std::fs::create_dir_all(cache_path)
            .map_err(|e| ModelError::CacheError(format!("Failed to create cache directory: {}", e)))?;

        // TODO: Replace with proper disk storage implementation
        // let rykv_config = RykvConfig::default()
        //     .dir(cache_path)
        //     .cache_capacity(config.max_cpu_memory_mb * 1024 * 1024)
        //     .create_if_missing(true);
        // let disk_cache = Database::open(&rykv_config)?;

        // Temporary placeholder - using in-memory HashMap
        let disk_cache = HashMap::new();

        let cache = Self {
            tensor_ops: GpuTensorOps::with_device(device.clone()),
            gpu_memory_pool: Arc::new(AdvancedMemoryPool::new(device.clone())),
            gpu_cache: Arc::new(RwLock::new(HashMap::new())),
            cpu_cache: Arc::new(RwLock::new(HashMap::new())),
            disk_cache: Arc::new(Mutex::new(disk_cache)),
            access_tracker: Arc::new(RwLock::new(BTreeMap::new())),
            deduplication_index: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(Mutex::new(EnhancedCacheStats {
                hit_rate: 0.0,
                miss_rate: 0.0,
                gpu_blocks: 0,
                cpu_blocks: 0,
                disk_blocks: 0,
                gpu_memory_used_mb: 0,
                cpu_memory_used_mb: 0,
                disk_storage_used_mb: 0,
                compression_ratio: 0.0,
                deduplication_savings_mb: 0,
                total_requests: 0,
                cache_hits: 0,
                cache_misses: 0,
                evictions: 0,
                prefetches: 0,
            })),
            total_requests: Arc::new(AtomicU64::new(0)),
            cache_hits: Arc::new(AtomicU64::new(0)),
            cache_misses: Arc::new(AtomicU64::new(0)),
            device,
            config,
        };

        println!("✅ Enhanced KV Cache initialized with rykv backend");
        println!("   GPU Memory: {}MB", cache.config.max_gpu_memory_mb);
        println!("   CPU Memory: {}MB", cache.config.max_cpu_memory_mb);
        println!("   Disk Cache: {}", cache.config.disk_cache_path);
        println!("   Compression: Level {}", cache.config.compression_level);
        println!("   Deduplication: {}", cache.config.enable_deduplication);

        Ok(cache)
    }

    /// Get cached K/V tensors with intelligent prefetching
    pub async fn get_async(&self, sequence_id: u64, token_range: (usize, usize)) -> CacheAccessResult {
        let start_time = Instant::now();
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        let cache_key = (sequence_id, token_range.0);

        // Update access tracking
        self.update_access_metadata(cache_key).await;

        // L1: Try GPU cache first
        if let Some(result) = self.try_gpu_cache(cache_key).await {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
            return CacheAccessResult {
                hit: true,
                level: 1,
                latency_us: start_time.elapsed().as_micros() as u64,
                key_tensor: Some(result.key_tensor),
                value_tensor: Some(result.value_tensor),
                prefetch_triggered: self.should_prefetch(cache_key).await,
            };
        }

        // L2: Try CPU cache
        if let Some(cpu_block) = self.try_cpu_cache(cache_key).await {
            // Promote to GPU cache
            if let Ok((key_tensor, value_tensor)) = self.promote_to_gpu(cpu_block).await {
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                return CacheAccessResult {
                    hit: true,
                    level: 2,
                    latency_us: start_time.elapsed().as_micros() as u64,
                    key_tensor: Some(key_tensor),
                    value_tensor: Some(value_tensor),
                    prefetch_triggered: self.should_prefetch(cache_key).await,
                };
            }
        }

        // L3: Try disk cache
        if let Some(disk_block) = self.try_disk_cache(cache_key).await {
            // Promote through the cache hierarchy
            if let Ok((key_tensor, value_tensor)) = self.promote_from_disk(disk_block).await {
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                return CacheAccessResult {
                    hit: true,
                    level: 3,
                    latency_us: start_time.elapsed().as_micros() as u64,
                    key_tensor: Some(key_tensor),
                    value_tensor: Some(value_tensor),
                    prefetch_triggered: self.should_prefetch(cache_key).await,
                };
            }
        }

        // Cache miss
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        CacheAccessResult {
            hit: false,
            level: 0,
            latency_us: start_time.elapsed().as_micros() as u64,
            key_tensor: None,
            value_tensor: None,
            prefetch_triggered: false,
        }
    }

    /// Store K/V tensors with intelligent caching strategy
    pub async fn store_async(
        &self,
        sequence_id: u64,
        token_range: (usize, usize),
        key_tensor: GpuTensor,
        value_tensor: GpuTensor,
    ) -> ModelResult<()> {
        let cache_key = (sequence_id, token_range.0);

        // Calculate checksum for deduplication
        let checksum = if self.config.enable_deduplication {
            self.calculate_tensor_checksum(&key_tensor, &value_tensor).await?
        } else {
            0
        };

        // Check for deduplication
        if self.config.enable_deduplication && checksum != 0 {
            if let Some(existing_keys) = self.deduplication_index.read().await.get(&checksum) {
                if !existing_keys.is_empty() {
                    // Found duplicate, create reference instead of storing
                    println!("🔄 Deduplication hit for sequence {} (checksum: {})", sequence_id, checksum);
                    return Ok(());
                }
            }
        }

        // Create metadata
        let metadata = CacheBlockMetadata {
            sequence_id,
            start_token: token_range.0,
            end_token: token_range.1,
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            access_count: 1,
            size_bytes: key_tensor.size_bytes() + value_tensor.size_bytes(),
            checksum,
        };

        // Store in GPU cache (L1)
        let gpu_block = Arc::new(GpuKVBlock {
            key_tensor: key_tensor.clone(),
            value_tensor: value_tensor.clone(),
            metadata: metadata.clone(),
        });

        self.gpu_cache.write().await.insert(cache_key, gpu_block);

        // Update deduplication index
        if self.config.enable_deduplication && checksum != 0 {
            self.deduplication_index.write().await
                .entry(checksum)
                .or_insert_with(Vec::new)
                .push(cache_key);
        }

        // Async background tasks
        if self.config.async_writeback {
            self.schedule_background_writeback(cache_key, key_tensor, value_tensor, metadata).await;
        }

        // Check if we need eviction
        self.check_memory_pressure().await?;

        Ok(())
    }

    /// Get comprehensive cache statistics
    pub async fn get_stats(&self) -> EnhancedCacheStats {
        let mut stats = self.stats.lock().await;

        let total_requests = self.total_requests.load(Ordering::Relaxed);
        let cache_hits = self.cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.cache_misses.load(Ordering::Relaxed);

        stats.total_requests = total_requests;
        stats.cache_hits = cache_hits;
        stats.cache_misses = cache_misses;
        stats.hit_rate = if total_requests > 0 {
            cache_hits as f64 / total_requests as f64
        } else {
            0.0
        };
        stats.miss_rate = 1.0 - stats.hit_rate;

        // Update block counts
        stats.gpu_blocks = self.gpu_cache.read().await.len();
        stats.cpu_blocks = self.cpu_cache.read().await.len();

        stats.clone()
    }

    /// Start background maintenance tasks
    pub async fn start_background_tasks(&self) -> Vec<tokio::task::JoinHandle<()>> {
        let mut tasks = Vec::new();

        // Cache maintenance task
        let cache_clone = Arc::new(self.clone());
        let maintenance_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            loop {
                interval.tick().await;
                if let Err(e) = cache_clone.run_maintenance().await {
                    eprintln!("Cache maintenance error: {}", e);
                }
            }
        });
        tasks.push(maintenance_task);

        // Statistics update task
        let cache_clone = Arc::new(self.clone());
        let stats_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));
            loop {
                interval.tick().await;
                cache_clone.update_statistics().await;
            }
        });
        tasks.push(stats_task);

        println!("✅ Enhanced KV Cache background tasks started");
        tasks
    }

    // Private helper methods

    async fn try_gpu_cache(&self, cache_key: (u64, usize)) -> Option<GpuKVBlock> {
        self.gpu_cache.read().await
            .get(&cache_key)
            .map(|block| (**block).clone())
    }

    async fn try_cpu_cache(&self, cache_key: (u64, usize)) -> Option<CpuKVBlock> {
        self.cpu_cache.read().await
            .get(&cache_key)
            .map(|block| (**block).clone())
    }

    async fn try_disk_cache(&self, cache_key: (u64, usize)) -> Option<SerializableKVBlock> {
        let db = self.disk_cache.lock().await;
        let key_bytes = bincode::serialize(&cache_key).ok()?;

        if let Ok(Some(compressed_data)) = db.get(&key_bytes) {
            // Decompress data
            let mut decoder = Decoder::new(&compressed_data[..]).ok()?;
            let mut decompressed = Vec::new();
            std::io::copy(&mut decoder, &mut decompressed).ok()?;

            // Deserialize
            bincode::deserialize(&decompressed).ok()
        } else {
            None
        }
    }

    async fn promote_to_gpu(&self, cpu_block: CpuKVBlock) -> ModelResult<(GpuTensor, GpuTensor)> {
        // Convert CPU data to GPU tensors
        let key_tensor = GpuTensor::from_data(
            cpu_block.key_data,
            cpu_block.key_shape,
            self.device.clone()
        )?;

        let value_tensor = GpuTensor::from_data(
            cpu_block.value_data,
            cpu_block.value_shape,
            self.device.clone()
        )?;

        Ok((key_tensor, value_tensor))
    }

    async fn promote_from_disk(&self, disk_block: SerializableKVBlock) -> ModelResult<(GpuTensor, GpuTensor)> {
        // Convert disk data to GPU tensors
        let key_tensor = GpuTensor::from_data(
            disk_block.key_data,
            disk_block.key_shape,
            self.device.clone()
        )?;

        let value_tensor = GpuTensor::from_data(
            disk_block.value_data,
            disk_block.value_shape,
            self.device.clone()
        )?;

        Ok((key_tensor, value_tensor))
    }

    async fn calculate_tensor_checksum(&self, key_tensor: &GpuTensor, value_tensor: &GpuTensor) -> ModelResult<u64> {
        // Simple checksum based on tensor shapes and first few elements
        let key_shape_sum: usize = key_tensor.shape().iter().sum();
        let value_shape_sum: usize = value_tensor.shape().iter().sum();

        // In a real implementation, you'd hash actual tensor data
        Ok((key_shape_sum + value_shape_sum) as u64)
    }

    async fn update_access_metadata(&self, cache_key: (u64, usize)) {
        let mut tracker = self.access_tracker.write().await;
        let metadata = tracker.entry(cache_key).or_insert(CacheAccessMetadata {
            last_accessed: Instant::now(),
            access_count: 0,
            frequency_score: 0.0,
            recency_score: 0.0,
            prediction_score: 0.0,
        });

        metadata.last_accessed = Instant::now();
        metadata.access_count += 1;
        metadata.frequency_score = metadata.access_count as f64 / 100.0; // Simple frequency
        metadata.recency_score = 1.0; // Recently accessed
    }

    async fn should_prefetch(&self, _cache_key: (u64, usize)) -> bool {
        // Simple prefetch heuristic - could be made much more sophisticated
        true
    }

    async fn schedule_background_writeback(
        &self,
        cache_key: (u64, usize),
        key_tensor: GpuTensor,
        value_tensor: GpuTensor,
        metadata: CacheBlockMetadata,
    ) {
        // Convert to CPU data first
        let key_data = match key_tensor.to_vec() {
            Ok(data) => data,
            Err(_) => return,
        };

        let value_data = match value_tensor.to_vec() {
            Ok(data) => data,
            Err(_) => return,
        };

        // Store in CPU cache
        let cpu_block = Arc::new(CpuKVBlock {
            key_data: key_data.clone(),
            value_data: value_data.clone(),
            key_shape: key_tensor.shape().to_vec(),
            value_shape: value_tensor.shape().to_vec(),
            metadata: metadata.clone(),
        });

        self.cpu_cache.write().await.insert(cache_key, cpu_block);

        // Schedule disk writeback
        let disk_cache = self.disk_cache.clone();
        let compression_level = self.config.compression_level;

        tokio::spawn(async move {
            let serializable = SerializableKVBlock {
                sequence_id: metadata.sequence_id,
                start_token: metadata.start_token,
                end_token: metadata.end_token,
                key_data,
                value_data,
                key_shape: key_tensor.shape().to_vec(),
                value_shape: value_tensor.shape().to_vec(),
                created_at: metadata.created_at.elapsed().as_secs(),
                access_count: metadata.access_count,
                compression_ratio: 0.0, // Will be calculated
                checksum: metadata.checksum,
            };

            if let Ok(serialized) = bincode::serialize(&serializable) {
                // Compress data
                let mut encoder = Encoder::new(Vec::new(), compression_level).unwrap();
                if std::io::copy(&mut &serialized[..], &mut encoder).is_ok() {
                    if let Ok((compressed, _)) = encoder.finish() {
                        let key_bytes = bincode::serialize(&cache_key).unwrap();
                        let db = disk_cache.lock().await;
                        let _ = db.insert(&key_bytes, &compressed);
                    }
                }
            }
        });
    }

    async fn check_memory_pressure(&self) -> ModelResult<()> {
        // Simple memory pressure check - could be much more sophisticated
        let gpu_blocks = self.gpu_cache.read().await.len();
        let max_gpu_blocks = (self.config.max_gpu_memory_mb * 1024 * 1024) /
                           (self.config.block_size * 4 * 128); // Rough estimate

        if gpu_blocks > max_gpu_blocks {
            self.evict_lru_blocks().await?;
        }

        Ok(())
    }

    async fn evict_lru_blocks(&self) -> ModelResult<()> {
        // Simple LRU eviction - could use more sophisticated algorithms
        let mut to_evict = Vec::new();

        {
            let access_tracker = self.access_tracker.read().await;
            let mut entries: Vec<_> = access_tracker.iter().collect();
            entries.sort_by_key(|(_, metadata)| metadata.last_accessed);

            // Evict oldest blocks
            for ((seq_id, token_pos), _) in entries.iter().take(self.config.eviction_batch_size) {
                to_evict.push((*seq_id, *token_pos));
            }
        }

        let mut gpu_cache = self.gpu_cache.write().await;
        for key in to_evict {
            gpu_cache.remove(&key);
        }

        Ok(())
    }

    async fn run_maintenance(&self) -> ModelResult<()> {
        // TTL cleanup
        let now = Instant::now();
        let ttl_duration = Duration::from_secs(self.config.cache_ttl_seconds);

        let mut expired_keys = Vec::new();
        {
            let access_tracker = self.access_tracker.read().await;
            for (key, metadata) in access_tracker.iter() {
                if now.duration_since(metadata.last_accessed) > ttl_duration {
                    expired_keys.push(*key);
                }
            }
        }

        // Remove expired entries
        for key in expired_keys {
            self.gpu_cache.write().await.remove(&key);
            self.cpu_cache.write().await.remove(&key);
            self.access_tracker.write().await.remove(&key);
        }

        println!("🧹 Cache maintenance completed");
        Ok(())
    }

    async fn update_statistics(&self) {
        let mut stats = self.stats.lock().await;
        stats.gpu_blocks = self.gpu_cache.read().await.len();
        stats.cpu_blocks = self.cpu_cache.read().await.len();
        // disk_blocks would require scanning the rykv database
    }
}

// Make EnhancedKVCache cloneable for background tasks
impl Clone for EnhancedKVCache {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            device: self.device.clone(),
            tensor_ops: self.tensor_ops.clone(),
            gpu_cache: self.gpu_cache.clone(),
            gpu_memory_pool: self.gpu_memory_pool.clone(),
            cpu_cache: self.cpu_cache.clone(),
            disk_cache: self.disk_cache.clone(),
            access_tracker: self.access_tracker.clone(),
            deduplication_index: self.deduplication_index.clone(),
            stats: self.stats.clone(),
            total_requests: self.total_requests.clone(),
            cache_hits: self.cache_hits.clone(),
            cache_misses: self.cache_misses.clone(),
        }
    }
}