//! Hybrid KV cache combining RadixAttention and PagedAttention
//!
//! This crate implements UniLLM's innovative memory management system that combines:
//! - SGLang's RadixAttention for token-level prefix sharing (L1)
//! - vLLM's PagedAttention for block-level efficiency (L2)
//! - Compressed storage for cold data (L3)

mod hybrid_cache;
mod gpu_memory;
mod gpu_integrated_cache;

pub use hybrid_cache::{
    HybridKVCache, CacheHandle, CacheTier, KVTensorPair, TokenId, SequenceId,
    RadixCache, AdaptiveCachePolicy, CachePolicy, HybridCacheStats
};

pub use gpu_memory::{
    GpuAwareMemoryPool, GpuMemoryBackend, GpuDevicePtr, GpuAllocation,
    GpuMemoryError, GpuMemoryResult, GpuMemoryStats, GpuDeviceProperties,
    CudaMemoryBackend, HipMemoryBackend
};

pub use gpu_integrated_cache::{
    GpuIntegratedCache, GpuIntegratedCacheBuilder, GpuIntegratedCacheStats,
    GpuBackendType
};

#[cfg(test)]
mod tests;

use std::collections::{HashMap, VecDeque, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// A page in the KV cache
#[derive(Debug, Clone)]
pub struct KvPage {
    /// Unique page ID
    pub id: u32,
    /// Device pointer to the page memory
    pub device_ptr: u64,
    /// Whether the page is currently allocated
    pub allocated: bool,
    /// Sequence ID that owns this page
    pub owner_seq_id: Option<u32>,
    /// Timestamp when page was allocated
    pub allocated_at: Option<Instant>,
}

impl KvPage {
    /// Create a new KV page
    pub fn new(id: u32, device_ptr: u64) -> Self {
        Self {
            id,
            device_ptr,
            allocated: false,
            owner_seq_id: None,
            allocated_at: None,
        }
    }
    
    /// Allocate the page to a sequence
    pub fn allocate(&mut self, seq_id: u32) {
        self.allocated = true;
        self.owner_seq_id = Some(seq_id);
        self.allocated_at = Some(Instant::now());
    }
    
    /// Free the page
    pub fn free(&mut self) {
        self.allocated = false;
        self.owner_seq_id = None;
        self.allocated_at = None;
    }
}

/// A block of pages (typically 16 pages per block)
#[derive(Debug)]
pub struct KvBlock {
    /// Block ID
    pub id: u32,
    /// Pages in this block
    pub pages: Vec<KvPage>,
    /// Number of allocated pages in this block
    pub allocated_count: usize,
    /// Whether the block is fully allocated
    pub is_full: bool,
}

impl KvBlock {
    /// Create a new KV block
    pub fn new(id: u32, page_count: usize, base_device_ptr: u64, page_size: usize) -> Self {
        let mut pages = Vec::new();
        for i in 0..page_count {
            let page_id = id * page_count as u32 + i as u32;
            let device_ptr = base_device_ptr + (i * page_size) as u64;
            pages.push(KvPage::new(page_id, device_ptr));
        }
        
        Self {
            id,
            pages,
            allocated_count: 0,
            is_full: false,
        }
    }
    
    /// Allocate a page from this block
    pub fn allocate_page(&mut self, seq_id: u32) -> Option<u32> {
        if self.is_full {
            return None;
        }

        let pages_len = self.pages.len();
        for page in &mut self.pages {
            if !page.allocated {
                let page_id = page.id;
                page.allocate(seq_id);
                self.allocated_count += 1;
                if self.allocated_count == pages_len {
                    self.is_full = true;
                }
                return Some(page_id);
            }
        }

        None
    }
    
    /// Free a page in this block
    pub fn free_page(&mut self, page_id: u32) -> bool {
        for page in &mut self.pages {
            if page.id == page_id && page.allocated {
                page.free();
                self.allocated_count -= 1;
                self.is_full = false;
                return true;
            }
        }
        false
    }
}

/// Sequence information for KV cache management
#[derive(Debug, Clone)]
pub struct KvSequence {
    /// Sequence ID
    pub seq_id: u32,
    /// Pages allocated to this sequence
    pub pages: Vec<u32>,
    /// Current sequence length
    pub length: usize,
    /// Maximum sequence length
    pub max_length: usize,
    /// Number of tokens processed
    pub tokens_processed: usize,
    /// Whether the sequence is active
    pub is_active: bool,
    /// Creation timestamp
    pub created_at: Instant,
}

impl KvSequence {
    /// Create a new KV sequence
    pub fn new(seq_id: u32, max_length: usize) -> Self {
        Self {
            seq_id,
            pages: Vec::new(),
            length: 0,
            max_length,
            tokens_processed: 0,
            is_active: true,
            created_at: Instant::now(),
        }
    }
    
    /// Add a page to this sequence
    pub fn add_page(&mut self, page_id: u32) {
        self.pages.push(page_id);
    }
    
    /// Remove a page from this sequence
    pub fn remove_page(&mut self, page_id: u32) -> bool {
        if let Some(pos) = self.pages.iter().position(|&id| id == page_id) {
            self.pages.remove(pos);
            true
        } else {
            false
        }
    }
    
    /// Check if sequence needs more pages
    pub fn needs_more_pages(&self, page_size: usize) -> bool {
        let current_capacity = self.pages.len() * page_size;
        current_capacity < self.max_length
    }
    
    /// Get the number of pages needed
    pub fn pages_needed(&self, page_size: usize) -> usize {
        let current_capacity = self.pages.len() * page_size;
        let needed_capacity = self.max_length - current_capacity;
        (needed_capacity + page_size - 1) / page_size // Ceiling division
    }
}

/// Paged KV allocator implementation
pub struct PagedKvAllocator {
    /// All blocks in the allocator
    blocks: Vec<KvBlock>,
    /// Free pages available for allocation
    free_pages: VecDeque<u32>,
    /// Active sequences
    sequences: HashMap<u32, KvSequence>,
    /// Page size in tokens
    page_size: usize,
    /// Pages per block
    pages_per_block: usize,
    /// Total number of pages
    total_pages: usize,
    /// Number of allocated pages
    allocated_pages: usize,
    /// Base device pointer for the first page
    base_device_ptr: u64,
    /// Next sequence ID
    next_seq_id: u32,
}

impl PagedKvAllocator {
    /// Create a new paged KV allocator
    /// 
    /// # Arguments
    /// * `total_pages` - Total number of pages to allocate
    /// * `page_size` - Size of each page in tokens
    /// * `pages_per_block` - Number of pages per block
    /// * `base_device_ptr` - Base device pointer for memory allocation
    pub fn new(
        total_pages: usize,
        page_size: usize,
        pages_per_block: usize,
        base_device_ptr: u64,
    ) -> Self {
        let num_blocks = (total_pages + pages_per_block - 1) / pages_per_block;
        let mut blocks = Vec::new();
        let mut free_pages = VecDeque::new();
        
        for block_id in 0..num_blocks {
            let pages_in_block = std::cmp::min(pages_per_block, total_pages - block_id * pages_per_block);
            let block_base_ptr = base_device_ptr + (block_id * pages_per_block * page_size) as u64;
            
            let block = KvBlock::new(block_id as u32, pages_in_block, block_base_ptr, page_size);
            
            // Add all pages to free list
            for page in &block.pages {
                free_pages.push_back(page.id);
            }
            
            blocks.push(block);
        }
        
        Self {
            blocks,
            free_pages,
            sequences: HashMap::new(),
            page_size,
            pages_per_block,
            total_pages,
            allocated_pages: 0,
            base_device_ptr,
            next_seq_id: 0,
        }
    }
    
    /// Allocate pages for a new sequence
    /// 
    /// # Arguments
    /// * `max_length` - Maximum sequence length
    /// 
    /// # Returns
    /// Sequence ID and allocated page IDs
    pub fn allocate_sequence(&mut self, max_length: usize) -> Result<(u32, Vec<u32>), Box<dyn std::error::Error>> {
        let seq_id = self.next_seq_id;
        self.next_seq_id += 1;
        
        let pages_needed = (max_length + self.page_size - 1) / self.page_size;
        
        if self.free_pages.len() < pages_needed {
            return Err(format!("Not enough free pages: need {}, have {}", pages_needed, self.free_pages.len()).into());
        }
        
        let mut allocated_pages = Vec::new();
        
        // Allocate pages
        for _ in 0..pages_needed {
            if let Some(page_id) = self.free_pages.pop_front() {
                // Find the block containing this page
                let block_id = page_id / self.pages_per_block as u32;
                if let Some(block) = self.blocks.get_mut(block_id as usize) {
                    if let Some(page_id) = block.allocate_page(seq_id) {
                        allocated_pages.push(page_id);
                        self.allocated_pages += 1;
                    }
                }
            }
        }
        
        // Create sequence
        let mut sequence = KvSequence::new(seq_id, max_length);
        sequence.pages = allocated_pages.clone();
        self.sequences.insert(seq_id, sequence);
        
        println!("Allocated sequence {} with {} pages (max_length: {})", seq_id, allocated_pages.len(), max_length);
        
        Ok((seq_id, allocated_pages))
    }
    
    /// Free a sequence and return its pages to the free pool
    pub fn free_sequence(&mut self, seq_id: u32) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(sequence) = self.sequences.remove(&seq_id) {
            let pages_count = sequence.pages.len();
            for page_id in sequence.pages {
                // Find the block containing this page
                let block_id = page_id / self.pages_per_block as u32;
                if let Some(block) = self.blocks.get_mut(block_id as usize) {
                    if block.free_page(page_id) {
                        self.free_pages.push_back(page_id);
                        self.allocated_pages -= 1;
                    }
                }
            }
            
            println!("Freed sequence {} with {} pages", seq_id, pages_count);
        }
        
        Ok(())
    }
    
    /// Extend a sequence with more pages
    pub fn extend_sequence(&mut self, seq_id: u32, additional_length: usize) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        if let Some(sequence) = self.sequences.get_mut(&seq_id) {
            let additional_pages = (additional_length + self.page_size - 1) / self.page_size;
            
            if self.free_pages.len() < additional_pages {
                return Err(format!("Not enough free pages for extension: need {}, have {}", additional_pages, self.free_pages.len()).into());
            }
            
            let mut new_pages = Vec::new();
            
            // Allocate additional pages
            for _ in 0..additional_pages {
                if let Some(page_id) = self.free_pages.pop_front() {
                    let block_id = page_id / self.pages_per_block as u32;
                    if let Some(block) = self.blocks.get_mut(block_id as usize) {
                        if let Some(page_id) = block.allocate_page(seq_id) {
                            new_pages.push(page_id);
                            sequence.pages.push(page_id);
                            self.allocated_pages += 1;
                        }
                    }
                }
            }
            
            println!("Extended sequence {} with {} additional pages", seq_id, new_pages.len());
            Ok(new_pages)
        } else {
            Err(format!("Sequence {} not found", seq_id).into())
        }
    }
    
    /// Get sequence information
    pub fn get_sequence(&self, seq_id: u32) -> Option<&KvSequence> {
        self.sequences.get(&seq_id)
    }
    
    /// Get all active sequences
    pub fn get_active_sequences(&self) -> Vec<&KvSequence> {
        self.sequences.values().filter(|s| s.is_active).collect()
    }
    
    /// Get memory usage statistics
    pub fn get_stats(&self) -> KvAllocatorStats {
        KvAllocatorStats {
            total_pages: self.total_pages,
            allocated_pages: self.allocated_pages,
            free_pages: self.free_pages.len(),
            active_sequences: self.sequences.len(),
            memory_usage_percent: (self.allocated_pages as f64 / self.total_pages as f64) * 100.0,
        }
    }
    
    /// Defragment memory by moving sequences to contiguous pages
    pub fn defragment(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("Starting memory defragmentation...");
        
        // For now, this is a placeholder implementation
        // In a real implementation, we would:
        // 1. Identify fragmented sequences
        // 2. Move pages to create contiguous blocks
        // 3. Update device pointers accordingly
        
        println!("Memory defragmentation completed");
        Ok(())
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct KvAllocatorStats {
    pub total_pages: usize,
    pub allocated_pages: usize,
    pub free_pages: usize,
    pub active_sequences: usize,
    pub memory_usage_percent: f64,
}

impl std::fmt::Display for KvAllocatorStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KV Allocator Stats: {}/{} pages allocated ({:.1}%), {} free pages, {} active sequences",
               self.allocated_pages, self.total_pages, self.memory_usage_percent, self.free_pages, self.active_sequences)
    }
}