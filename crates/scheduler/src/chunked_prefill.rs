//! Chunked prefill implementation for efficient prompt processing

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for chunked prefill
#[derive(Debug, Clone)]
pub struct ChunkedPrefillConfig {
    /// Maximum chunk size in tokens
    pub max_chunk_size: usize,
    /// Number of chunks to process in parallel
    pub parallel_chunks: usize,
    /// Overlap between chunks (for context continuity)
    pub chunk_overlap: usize,
    /// Maximum prefill length before switching to chunked mode
    pub chunking_threshold: usize,
}

impl Default for ChunkedPrefillConfig {
    fn default() -> Self {
        Self {
            max_chunk_size: 2048,      // 2K tokens per chunk
            parallel_chunks: 4,         // Process 4 chunks in parallel
            chunk_overlap: 64,         // 64 token overlap between chunks
            chunking_threshold: 4096,  // Start chunking at 4K tokens
        }
    }
}

/// A chunk of tokens to be processed
#[derive(Debug, Clone)]
pub struct PrefillChunk {
    /// Chunk ID
    pub chunk_id: u32,
    /// Request ID this chunk belongs to
    pub request_id: u32,
    /// Tokens in this chunk
    pub tokens: Vec<u32>,
    /// Start position in the original sequence
    pub start_pos: usize,
    /// End position in the original sequence
    pub end_pos: usize,
    /// Whether this chunk has been processed
    pub processed: bool,
    /// Processing timestamp
    pub processed_at: Option<Instant>,
    /// Chunk priority (higher priority chunks processed first)
    pub priority: u32,
}

impl PrefillChunk {
    /// Create a new prefill chunk
    pub fn new(
        chunk_id: u32,
        request_id: u32,
        tokens: Vec<u32>,
        start_pos: usize,
        end_pos: usize,
        priority: u32,
    ) -> Self {
        Self {
            chunk_id,
            request_id,
            tokens,
            start_pos,
            end_pos,
            processed: false,
            processed_at: None,
            priority,
        }
    }
    
    /// Get chunk size
    pub fn size(&self) -> usize {
        self.tokens.len()
    }
    
    /// Mark chunk as processed
    pub fn mark_processed(&mut self) {
        self.processed = true;
        self.processed_at = Some(Instant::now());
    }
}

/// Chunked prefill manager
pub struct ChunkedPrefillManager {
    /// Configuration
    config: ChunkedPrefillConfig,
    /// Active chunks being processed
    active_chunks: HashMap<u32, PrefillChunk>,
    /// Completed chunks
    completed_chunks: HashMap<u32, Vec<PrefillChunk>>,
    /// Next chunk ID
    next_chunk_id: u32,
    /// Statistics
    stats: PrefillStats,
}

impl ChunkedPrefillManager {
    /// Create a new chunked prefill manager
    pub fn new(config: ChunkedPrefillConfig) -> Self {
        Self {
            config,
            active_chunks: HashMap::new(),
            completed_chunks: HashMap::new(),
            next_chunk_id: 0,
            stats: PrefillStats::new(),
        }
    }
    
    /// Create chunks for a request's prompt
    /// 
    /// # Arguments
    /// * `request_id` - Request ID
    /// * `prompt_tokens` - Full prompt tokens
    /// 
    /// # Returns
    /// Vector of chunk IDs created
    pub fn create_chunks(&mut self, request_id: u32, prompt_tokens: &[u32]) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let prompt_length = prompt_tokens.len();
        
        // If prompt is short enough, don't chunk it
        if prompt_length <= self.config.chunking_threshold {
            let chunk_id = self.next_chunk_id;
            self.next_chunk_id += 1;
            
            let chunk = PrefillChunk::new(
                chunk_id,
                request_id,
                prompt_tokens.to_vec(),
                0,
                prompt_length,
                0, // High priority for short prompts
            );
            
            self.active_chunks.insert(chunk_id, chunk);
            self.stats.chunks_created += 1;
            
            println!("Created single chunk {} for request {} ({} tokens)", 
                     chunk_id, request_id, prompt_length);
            
            return Ok(vec![chunk_id]);
        }
        
        // Create multiple chunks
        let mut chunk_ids = Vec::new();
        let mut start_pos = 0;
        let mut chunk_priority = 0;
        
        while start_pos < prompt_length {
            let end_pos = std::cmp::min(
                start_pos + self.config.max_chunk_size,
                prompt_length
            );
            
            let chunk_id = self.next_chunk_id;
            self.next_chunk_id += 1;
            
            let chunk_tokens = prompt_tokens[start_pos..end_pos].to_vec();
            
            let chunk = PrefillChunk::new(
                chunk_id,
                request_id,
                chunk_tokens,
                start_pos,
                end_pos,
                chunk_priority,
            );
            
            self.active_chunks.insert(chunk_id, chunk);
            chunk_ids.push(chunk_id);
            
            self.stats.chunks_created += 1;
            
            // Move to next chunk with overlap
            start_pos = end_pos.saturating_sub(self.config.chunk_overlap);
            chunk_priority += 1;
        }
        
        println!("Created {} chunks for request {} ({} tokens)", 
                 chunk_ids.len(), request_id, prompt_length);
        
        Ok(chunk_ids)
    }
    
    /// Get the next chunk to process
    /// 
    /// # Returns
    /// Chunk ID and chunk data, or None if no chunks available
    pub fn get_next_chunk(&mut self) -> Option<(u32, &PrefillChunk)> {
        // Find the highest priority unprocessed chunk
        let mut best_chunk_id = None;
        let mut best_priority = u32::MAX;
        
        for (chunk_id, chunk) in &self.active_chunks {
            if !chunk.processed && chunk.priority < best_priority {
                best_chunk_id = Some(*chunk_id);
                best_priority = chunk.priority;
            }
        }
        
        if let Some(chunk_id) = best_chunk_id {
            self.active_chunks.get(&chunk_id).map(|chunk| (chunk_id, chunk))
        } else {
            None
        }
    }
    
    /// Mark a chunk as completed
    pub fn complete_chunk(&mut self, chunk_id: u32) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(mut chunk) = self.active_chunks.remove(&chunk_id) {
            let request_id = chunk.request_id; // Store before moving
            chunk.mark_processed();

            // Move to completed chunks
            self.completed_chunks
                .entry(chunk.request_id)
                .or_insert_with(Vec::new)
                .push(chunk);

            self.stats.chunks_completed += 1;

            println!("Completed chunk {} for request {}", chunk_id, request_id);
        }
        
        Ok(())
    }
    
    /// Check if all chunks for a request are completed
    pub fn is_request_complete(&self, request_id: u32) -> bool {
        // Check if there are any active chunks for this request
        let has_active_chunks = self.active_chunks
            .values()
            .any(|chunk| chunk.request_id == request_id && !chunk.processed);
        
        !has_active_chunks
    }
    
    /// Get completed chunks for a request
    pub fn get_completed_chunks(&self, request_id: u32) -> Option<&Vec<PrefillChunk>> {
        self.completed_chunks.get(&request_id)
    }
    
    /// Get all active chunks
    pub fn get_active_chunks(&self) -> Vec<&PrefillChunk> {
        self.active_chunks.values().collect()
    }
    
    /// Get prefill statistics
    pub fn get_stats(&self) -> &PrefillStats {
        &self.stats
    }
    
    /// Clear completed chunks for a request
    pub fn clear_completed_chunks(&mut self, request_id: u32) {
        self.completed_chunks.remove(&request_id);
    }
    
    /// Get the number of active chunks
    pub fn active_chunk_count(&self) -> usize {
        self.active_chunks.len()
    }
    
    /// Get the number of completed chunks
    pub fn completed_chunk_count(&self) -> usize {
        self.completed_chunks.values().map(|chunks| chunks.len()).sum()
    }
    
    /// Estimate processing time for a request
    pub fn estimate_processing_time(&self, prompt_length: usize) -> Duration {
        if prompt_length <= self.config.chunking_threshold {
            // Single chunk processing time
            Duration::from_millis(50) // 50ms baseline
        } else {
            // Multiple chunks processing time
            let num_chunks = (prompt_length + self.config.max_chunk_size - 1) / self.config.max_chunk_size;
            let parallel_chunks = std::cmp::min(num_chunks, self.config.parallel_chunks);
            let processing_rounds = (num_chunks + parallel_chunks - 1) / parallel_chunks;
            
            Duration::from_millis(50 * processing_rounds as u64)
        }
    }
}

/// Prefill processing statistics
#[derive(Debug, Clone)]
pub struct PrefillStats {
    pub chunks_created: usize,
    pub chunks_completed: usize,
    pub total_processing_time: Duration,
    pub average_chunk_size: f64,
    pub parallel_efficiency: f64,
}

impl PrefillStats {
    pub fn new() -> Self {
        Self {
            chunks_created: 0,
            chunks_completed: 0,
            total_processing_time: Duration::from_millis(0),
            average_chunk_size: 0.0,
            parallel_efficiency: 0.0,
        }
    }
}

impl std::fmt::Display for PrefillStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Prefill Stats: {} chunks created, {} completed, {:.2}ms total time",
               self.chunks_created, self.chunks_completed, self.total_processing_time.as_millis())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_chunked_prefill_short_prompt() {
        let config = ChunkedPrefillConfig::default();
        let mut manager = ChunkedPrefillManager::new(config);
        
        let prompt_tokens = vec![1, 2, 3, 4, 5]; // Short prompt
        let chunk_ids = manager.create_chunks(1, &prompt_tokens).unwrap();
        
        assert_eq!(chunk_ids.len(), 1);
        assert_eq!(manager.active_chunk_count(), 1);
    }
    
    #[test]
    fn test_chunked_prefill_long_prompt() {
        let config = ChunkedPrefillConfig {
            max_chunk_size: 100,
            chunking_threshold: 200,
            ..Default::default()
        };
        let mut manager = ChunkedPrefillManager::new(config);
        
        let prompt_tokens: Vec<u32> = (1..=500).collect(); // Long prompt
        let chunk_ids = manager.create_chunks(1, &prompt_tokens).unwrap();
        
        assert!(chunk_ids.len() > 1);
        assert_eq!(manager.active_chunk_count(), chunk_ids.len());
    }
    
    #[test]
    fn test_chunk_completion() {
        let config = ChunkedPrefillConfig::default();
        let mut manager = ChunkedPrefillManager::new(config);
        
        let prompt_tokens = vec![1, 2, 3, 4, 5];
        let chunk_ids = manager.create_chunks(1, &prompt_tokens).unwrap();
        
        let chunk_id = chunk_ids[0];
        manager.complete_chunk(chunk_id).unwrap();
        
        assert_eq!(manager.active_chunk_count(), 0);
        assert!(manager.is_request_complete(1));
    }
}