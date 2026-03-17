//! Chunked prefill implementation

use super::*;

pub struct ChunkedPrefillImpl {
    chunk_size: usize,
}

impl ChunkedPrefillImpl {
    pub fn new(chunk_size: usize) -> Self {
        Self { chunk_size }
    }
}