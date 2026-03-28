//! Tokenizer implementation using HuggingFace tokenizers
//!
//! This module provides tokenization using the HuggingFace tokenizers library,
//! with fallback to a basic tokenizer for simple use cases.

use crate::types::*;
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;

/// Special token IDs (defaults, may be overridden by loaded tokenizer)
pub const BOS_TOKEN_ID: u32 = 1;   // Beginning of sequence
pub const EOS_TOKEN_ID: u32 = 2;   // End of sequence
pub const UNK_TOKEN_ID: u32 = 0;   // Unknown token
pub const PAD_TOKEN_ID: u32 = 3;   // Padding token

/// Special tokens configuration
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub unk_token_id: u32,
    pub pad_token_id: u32,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos_token_id: BOS_TOKEN_ID,
            eos_token_id: EOS_TOKEN_ID,
            unk_token_id: UNK_TOKEN_ID,
            pad_token_id: PAD_TOKEN_ID,
        }
    }
}

/// Tokenizer implementation that wraps HuggingFace tokenizers
pub struct Tokenizer {
    inner: TokenizerBackend,
    special_tokens: SpecialTokens,
}

enum TokenizerBackend {
    HuggingFace(HfTokenizer),
    Basic(BasicTokenizer),
    GGUF(GGUFTokenizerImpl),
}

impl Tokenizer {
    /// Create a new tokenizer with basic vocabulary (fallback)
    pub fn new() -> Self {
        Self {
            inner: TokenizerBackend::Basic(BasicTokenizer::new()),
            special_tokens: SpecialTokens::default(),
        }
    }

    /// Load tokenizer from a tokenizer.json file (HuggingFace format)
    pub fn from_file<P: AsRef<Path>>(path: P) -> ModelResult<Self> {
        let hf_tokenizer = HfTokenizer::from_file(path.as_ref())
            .map_err(|e| ModelError::InitializationFailed(format!("Failed to load tokenizer: {}", e)))?;

        // Try to extract special token IDs from the tokenizer
        let special_tokens = Self::extract_special_tokens(&hf_tokenizer);

        Ok(Self {
            inner: TokenizerBackend::HuggingFace(hf_tokenizer),
            special_tokens,
        })
    }

    /// Load tokenizer from a model directory (looks for tokenizer.json)
    pub fn from_model_dir<P: AsRef<Path>>(model_dir: P) -> ModelResult<Self> {
        let tokenizer_path = model_dir.as_ref().join("tokenizer.json");
        if tokenizer_path.exists() {
            Self::from_file(tokenizer_path)
        } else {
            // Try tokenizer_config.json for special tokens, use basic tokenizer
            Ok(Self::new())
        }
    }

    /// Create tokenizer from GGUF tokenizer data
    pub fn from_gguf(gguf_tokenizer: &crate::weight_loader_core::GGUFTokenizer) -> ModelResult<Self> {
        let impl_tokenizer = GGUFTokenizerImpl::new(gguf_tokenizer);
        let special_tokens = impl_tokenizer.special_tokens.clone();

        Ok(Self {
            inner: TokenizerBackend::GGUF(impl_tokenizer),
            special_tokens,
        })
    }

    /// Create tokenizer from ModelWeights (prefers GGUF tokenizer if available)
    pub fn from_model_weights(weights: &crate::model_core::ModelWeights) -> ModelResult<Self> {
        // Prefer GGUF tokenizer if available
        if let Some(ref gguf_tok) = weights.gguf_tokenizer {
            return Self::from_gguf(gguf_tok);
        }

        // Fallback to basic tokenizer
        Ok(Self::new())
    }

    /// Create tokenizer from pretrained model name (downloads from HuggingFace)
    /// Note: This requires network access and HuggingFace Hub authentication for gated models
    pub fn from_pretrained(model_name: &str) -> ModelResult<Self> {
        // Try to load from local cache first
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("huggingface")
            .join("hub");

        // Convert model name to cache path format
        let model_path = cache_dir.join(format!("models--{}", model_name.replace('/', "--")));
        let tokenizer_path = model_path.join("snapshots").join("*").join("tokenizer.json");

        // Try to find tokenizer in cache
        if let Ok(entries) = glob::glob(tokenizer_path.to_str().unwrap_or("")) {
            for entry in entries.flatten() {
                if let Ok(tokenizer) = Self::from_file(entry) {
                    return Ok(tokenizer);
                }
            }
        }

        // Fall back to basic tokenizer
        Ok(Self::new())
    }

    /// Extract special token IDs from HuggingFace tokenizer
    fn extract_special_tokens(hf_tokenizer: &HfTokenizer) -> SpecialTokens {
        let mut special = SpecialTokens::default();

        // Try to get special token IDs from the tokenizer
        if let Some(id) = hf_tokenizer.token_to_id("<s>") {
            special.bos_token_id = id;
        } else if let Some(id) = hf_tokenizer.token_to_id("<bos>") {
            special.bos_token_id = id;
        }

        if let Some(id) = hf_tokenizer.token_to_id("</s>") {
            special.eos_token_id = id;
        } else if let Some(id) = hf_tokenizer.token_to_id("<eos>") {
            special.eos_token_id = id;
        }

        if let Some(id) = hf_tokenizer.token_to_id("<unk>") {
            special.unk_token_id = id;
        }

        if let Some(id) = hf_tokenizer.token_to_id("<pad>") {
            special.pad_token_id = id;
        }

        special
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        match &self.inner {
            TokenizerBackend::HuggingFace(hf) => {
                match hf.encode(text, false) {
                    Ok(encoding) => encoding.get_ids().to_vec(),
                    Err(_) => {
                        // Fallback to basic encoding
                        vec![self.special_tokens.bos_token_id, self.special_tokens.eos_token_id]
                    }
                }
            }
            TokenizerBackend::Basic(basic) => basic.encode(text),
            TokenizerBackend::GGUF(gguf) => gguf.encode(text),
        }
    }

    /// Encode text with special tokens
    pub fn encode_with_special_tokens(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<u32> {
        match &self.inner {
            TokenizerBackend::HuggingFace(hf) => {
                match hf.encode(text, add_bos) {
                    Ok(encoding) => {
                        let mut ids = encoding.get_ids().to_vec();
                        if add_eos && !ids.ends_with(&[self.special_tokens.eos_token_id]) {
                            ids.push(self.special_tokens.eos_token_id);
                        }
                        ids
                    }
                    Err(_) => {
                        let mut ids = if add_bos {
                            vec![self.special_tokens.bos_token_id]
                        } else {
                            vec![]
                        };
                        if add_eos {
                            ids.push(self.special_tokens.eos_token_id);
                        }
                        ids
                    }
                }
            }
            TokenizerBackend::Basic(basic) => {
                let mut ids = basic.encode_raw(text);
                if add_bos {
                    ids.insert(0, self.special_tokens.bos_token_id);
                }
                if add_eos {
                    ids.push(self.special_tokens.eos_token_id);
                }
                ids
            }
            TokenizerBackend::GGUF(gguf) => {
                let mut ids = gguf.encode_raw(text);
                if add_bos {
                    ids.insert(0, self.special_tokens.bos_token_id);
                }
                if add_eos {
                    ids.push(self.special_tokens.eos_token_id);
                }
                ids
            }
        }
    }

    /// Decode token IDs back to text
    pub fn decode(&self, token_ids: &[u32]) -> String {
        match &self.inner {
            TokenizerBackend::HuggingFace(hf) => {
                hf.decode(token_ids, true).unwrap_or_default()
            }
            TokenizerBackend::Basic(basic) => basic.decode(token_ids),
            TokenizerBackend::GGUF(gguf) => gguf.decode(token_ids),
        }
    }

    /// Decode token IDs without special token filtering
    pub fn decode_raw(&self, token_ids: &[u32]) -> String {
        match &self.inner {
            TokenizerBackend::HuggingFace(hf) => {
                hf.decode(token_ids, false).unwrap_or_default()
            }
            TokenizerBackend::Basic(basic) => basic.decode(token_ids),
            TokenizerBackend::GGUF(gguf) => gguf.decode_raw(token_ids),
        }
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        match &self.inner {
            TokenizerBackend::HuggingFace(hf) => hf.get_vocab_size(true),
            TokenizerBackend::Basic(basic) => basic.vocab_size(),
            TokenizerBackend::GGUF(gguf) => gguf.vocab_size(),
        }
    }

    /// Get special tokens
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    /// Get BOS token ID
    pub fn bos_token_id(&self) -> u32 {
        self.special_tokens.bos_token_id
    }

    /// Get EOS token ID
    pub fn eos_token_id(&self) -> u32 {
        self.special_tokens.eos_token_id
    }

    /// Get PAD token ID
    pub fn pad_token_id(&self) -> u32 {
        self.special_tokens.pad_token_id
    }

    /// Get UNK token ID
    pub fn unk_token_id(&self) -> u32 {
        self.special_tokens.unk_token_id
    }

    /// Check if token exists in vocabulary
    pub fn contains_token(&self, token: &str) -> bool {
        match &self.inner {
            TokenizerBackend::HuggingFace(hf) => hf.token_to_id(token).is_some(),
            TokenizerBackend::Basic(basic) => basic.contains_token(token),
            TokenizerBackend::GGUF(gguf) => gguf.contains_token(token),
        }
    }

    /// Get token ID for a string
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        match &self.inner {
            TokenizerBackend::HuggingFace(hf) => hf.token_to_id(token),
            TokenizerBackend::Basic(basic) => basic.token_to_id(token),
            TokenizerBackend::GGUF(gguf) => gguf.token_to_id(token),
        }
    }

    /// Get string for a token ID
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        match &self.inner {
            TokenizerBackend::HuggingFace(hf) => hf.id_to_token(id),
            TokenizerBackend::Basic(basic) => basic.id_to_token(id).map(|s| s.to_string()),
            TokenizerBackend::GGUF(gguf) => gguf.id_to_token(id).map(|s| s.to_string()),
        }
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Basic tokenizer for fallback (same as the original implementation)
struct BasicTokenizer {
    vocab: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    vocab_size: usize,
}

impl BasicTokenizer {
    fn new() -> Self {
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();

        // Add special tokens
        vocab.insert("<unk>".to_string(), UNK_TOKEN_ID);
        vocab.insert("<s>".to_string(), BOS_TOKEN_ID);
        vocab.insert("</s>".to_string(), EOS_TOKEN_ID);
        vocab.insert("<pad>".to_string(), PAD_TOKEN_ID);

        id_to_token.insert(UNK_TOKEN_ID, "<unk>".to_string());
        id_to_token.insert(BOS_TOKEN_ID, "<s>".to_string());
        id_to_token.insert(EOS_TOKEN_ID, "</s>".to_string());
        id_to_token.insert(PAD_TOKEN_ID, "<pad>".to_string());

        let mut next_id = 4;

        // Add basic vocabulary
        let basic_vocab = vec![
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "can", "may", "might", "must",
            "I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
            "this", "that", "these", "those", "here", "there", "where", "when", "why", "how",
            "what", "who", "which", "whose", "all", "some", "any", "no", "not", "yes",
            "hello", "world", "test", "example", "text", "model", "language", "AI",
            "quick", "brown", "fox", "dog", "cat", "house", "car", "tree", "book", "water",
            ".", "!", "?", ",", ";", ":", "'", "\"", "-", "_", "(", ")", "[", "]", "{", "}",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
            "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
            "ing", "ed", "er", "ly", "tion", "ment", "ness", "ity", "ous", "ful",
            " ", "\n", "\t",
        ];

        for word in basic_vocab {
            if !vocab.contains_key(word) {
                vocab.insert(word.to_string(), next_id);
                id_to_token.insert(next_id, word.to_string());
                next_id += 1;
            }
        }

        Self {
            vocab,
            id_to_token,
            vocab_size: next_id as usize,
        }
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = vec![BOS_TOKEN_ID];
        tokens.extend(self.encode_raw(text));
        tokens.push(EOS_TOKEN_ID);
        tokens
    }

    fn encode_raw(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        let words = self.tokenize_text(text);

        for word in words {
            if let Some(&token_id) = self.vocab.get(&word) {
                tokens.push(token_id);
            } else {
                for ch in word.chars() {
                    let ch_str = ch.to_string();
                    if let Some(&token_id) = self.vocab.get(&ch_str) {
                        tokens.push(token_id);
                    } else {
                        tokens.push(UNK_TOKEN_ID);
                    }
                }
            }
        }
        tokens
    }

    fn decode(&self, token_ids: &[u32]) -> String {
        let mut result = String::new();
        let mut last_was_char = false;

        for &token_id in token_ids {
            if token_id == BOS_TOKEN_ID || token_id == EOS_TOKEN_ID || token_id == PAD_TOKEN_ID {
                continue;
            }

            if let Some(token) = self.id_to_token.get(&token_id) {
                let is_single_char = token.len() == 1 && token.chars().next().unwrap().is_alphabetic();
                let is_whitespace = token.chars().all(|c| c.is_whitespace());
                let is_punctuation = token.chars().all(|c| c.is_ascii_punctuation());

                if !result.is_empty() && !is_whitespace && !is_punctuation {
                    if !last_was_char || !is_single_char {
                        result.push(' ');
                    }
                }

                result.push_str(token);
                last_was_char = is_single_char;
            } else {
                if !result.is_empty() {
                    result.push(' ');
                }
                result.push_str("<unk>");
                last_was_char = false;
            }
        }

        result.trim().to_string()
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn contains_token(&self, token: &str) -> bool {
        self.vocab.contains_key(token)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }

    fn tokenize_text(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current_word = String::new();

        for ch in text.chars() {
            if ch.is_whitespace() {
                if !current_word.is_empty() {
                    tokens.push(current_word.clone());
                    current_word.clear();
                }
                tokens.push(ch.to_string());
            } else if ch.is_ascii_punctuation() {
                if !current_word.is_empty() {
                    tokens.push(current_word.clone());
                    current_word.clear();
                }
                tokens.push(ch.to_string());
            } else {
                current_word.push(ch);
            }
        }

        if !current_word.is_empty() {
            tokens.push(current_word);
        }

        tokens
    }
}

/// GGUF-based tokenizer implementation
struct GGUFTokenizerImpl {
    tokens: Vec<String>,
    token_to_id: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    special_tokens: SpecialTokens,
    vocab_size: usize,
}

impl GGUFTokenizerImpl {
    fn new(gguf_tokenizer: &crate::weight_loader_core::GGUFTokenizer) -> Self {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();

        for (id, token) in gguf_tokenizer.tokens.iter().enumerate() {
            let id = id as u32;
            token_to_id.insert(token.clone(), id);
            id_to_token.insert(id, token.clone());
        }

        let gguf_special = &gguf_tokenizer.special_tokens;
        let special_tokens = SpecialTokens {
            bos_token_id: gguf_special.bos_token_id.unwrap_or(BOS_TOKEN_ID),
            eos_token_id: gguf_special.eos_token_id.unwrap_or(EOS_TOKEN_ID),
            unk_token_id: gguf_special.unk_token_id.unwrap_or(UNK_TOKEN_ID),
            pad_token_id: gguf_special.pad_token_id.unwrap_or(PAD_TOKEN_ID),
        };

        let vocab_size = gguf_tokenizer.tokens.len();

        Self {
            tokens: gguf_tokenizer.tokens.clone(),
            token_to_id,
            id_to_token,
            special_tokens,
            vocab_size,
        }
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = vec![self.special_tokens.bos_token_id];
        tokens.extend(self.encode_raw(text));
        tokens.push(self.special_tokens.eos_token_id);
        tokens
    }

    fn encode_raw(&self, text: &str) -> Vec<u32> {
        // SentencePiece uses ▁ (U+2581) to represent spaces
        // We need to convert spaces to this character for proper tokenization
        // The ▁ is prepended to tokens that follow a space (word boundaries)
        let processed = format!("▁{}", text.replace(' ', "▁"));

        let mut result = Vec::new();
        let bytes = processed.as_bytes();
        let mut i = 0;

        while i < bytes.len() {
            let mut matched = false;

            // Try to match longest token first (greedy)
            // Start with reasonable max length to avoid O(n^2)
            let max_len = std::cmp::min(bytes.len() - i, 32);

            for len in (1..=max_len).rev() {
                if let Ok(substr) = std::str::from_utf8(&bytes[i..i + len]) {
                    if let Some(&id) = self.token_to_id.get(substr) {
                        result.push(id);
                        i += len;
                        matched = true;
                        break;
                    }
                }
            }

            if !matched {
                // Try single byte as fallback
                let byte = bytes[i];

                // First try the character directly
                if let Ok(ch) = std::str::from_utf8(&bytes[i..i + 1]) {
                    if let Some(&id) = self.token_to_id.get(ch) {
                        result.push(id);
                        i += 1;
                        continue;
                    }
                }

                // Try byte-level token format: <0xNN>
                let byte_token = format!("<0x{:02X}>", byte);
                if let Some(&id) = self.token_to_id.get(&byte_token) {
                    result.push(id);
                } else {
                    // Unknown token
                    result.push(self.special_tokens.unk_token_id);
                }
                i += 1;
            }
        }

        result
    }

    fn decode(&self, token_ids: &[u32]) -> String {
        let mut result = String::new();

        for &id in token_ids {
            // Skip special tokens
            if id == self.special_tokens.bos_token_id
                || id == self.special_tokens.eos_token_id
                || id == self.special_tokens.pad_token_id
            {
                continue;
            }

            if let Some(token) = self.id_to_token.get(&id) {
                result.push_str(token);
            }
        }

        // Convert SentencePiece's ▁ back to spaces and trim leading space
        result.replace('▁', " ").trim_start().to_string()
    }

    fn decode_raw(&self, token_ids: &[u32]) -> String {
        let mut result = String::new();

        for &id in token_ids {
            if let Some(token) = self.id_to_token.get(&id) {
                result.push_str(token);
            }
        }

        // Convert SentencePiece's ▁ back to spaces and trim leading space
        result.replace('▁', " ").trim_start().to_string()
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn contains_token(&self, token: &str) -> bool {
        self.token_to_id.contains_key(token)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }
}

/// Batch tokenization for multiple texts
pub struct BatchTokenizer {
    tokenizer: Tokenizer,
}

impl BatchTokenizer {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self { tokenizer }
    }

    /// Encode multiple texts with padding
    pub fn encode_batch(&self, texts: &[&str], max_length: Option<usize>) -> (Vec<Vec<u32>>, Vec<Vec<bool>>) {
        let mut encoded_batch = Vec::new();
        let mut attention_masks = Vec::new();

        for text in texts {
            let tokens = self.tokenizer.encode(text);
            encoded_batch.push(tokens);
        }

        let max_len = max_length.unwrap_or_else(|| {
            encoded_batch.iter().map(|tokens| tokens.len()).max().unwrap_or(0)
        });

        for tokens in &mut encoded_batch {
            if tokens.len() > max_len {
                tokens.truncate(max_len);
            }

            let mut attention_mask = vec![true; tokens.len()];

            while tokens.len() < max_len {
                tokens.push(self.tokenizer.pad_token_id());
                attention_mask.push(false);
            }

            attention_masks.push(attention_mask);
        }

        (encoded_batch, attention_masks)
    }

    /// Decode batch of token sequences
    pub fn decode_batch(&self, token_sequences: &[Vec<u32>]) -> Vec<String> {
        token_sequences.iter()
            .map(|tokens| self.tokenizer.decode(tokens))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_creation() {
        let tokenizer = Tokenizer::new();
        assert!(tokenizer.vocab_size() > 100);
        assert!(tokenizer.contains_token("the"));
        assert!(tokenizer.contains_token("<s>"));
        assert!(tokenizer.contains_token("</s>"));
    }

    #[test]
    fn test_special_tokens() {
        let tokenizer = Tokenizer::new();
        assert_eq!(tokenizer.token_to_id("<unk>"), Some(UNK_TOKEN_ID));
        assert_eq!(tokenizer.token_to_id("<s>"), Some(BOS_TOKEN_ID));
        assert_eq!(tokenizer.token_to_id("</s>"), Some(EOS_TOKEN_ID));
        assert_eq!(tokenizer.token_to_id("<pad>"), Some(PAD_TOKEN_ID));
    }

    #[test]
    fn test_encode_decode_simple() {
        let tokenizer = Tokenizer::new();
        let text = "hello world";

        let tokens = tokenizer.encode(text);
        assert!(tokens.len() > 2);

        let decoded = tokenizer.decode(&tokens);
        assert!(decoded.contains("hello"));
        assert!(decoded.contains("world"));
    }

    #[test]
    fn test_special_token_ids() {
        let tokenizer = Tokenizer::new();
        assert_eq!(tokenizer.bos_token_id(), BOS_TOKEN_ID);
        assert_eq!(tokenizer.eos_token_id(), EOS_TOKEN_ID);
        assert_eq!(tokenizer.pad_token_id(), PAD_TOKEN_ID);
        assert_eq!(tokenizer.unk_token_id(), UNK_TOKEN_ID);
    }

    #[test]
    fn test_batch_tokenizer() {
        let tokenizer = Tokenizer::new();
        let batch_tokenizer = BatchTokenizer::new(tokenizer);

        let texts = vec!["hello", "hello world", "hello world test"];
        let (encoded_batch, attention_masks) = batch_tokenizer.encode_batch(&texts, Some(10));

        assert_eq!(encoded_batch.len(), 3);
        assert_eq!(attention_masks.len(), 3);

        for tokens in &encoded_batch {
            assert_eq!(tokens.len(), 10);
        }
    }
}
