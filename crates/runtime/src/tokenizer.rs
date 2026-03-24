//! Basic tokenizer implementation for text processing
//!
//! This module provides a simple tokenizer that can encode text to token IDs
//! and decode token IDs back to text. Starts with a basic implementation
//! that can be extended to support more sophisticated tokenization schemes.

use crate::types::*;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Special token IDs
pub const BOS_TOKEN_ID: u32 = 1;   // Beginning of sequence
pub const EOS_TOKEN_ID: u32 = 2;   // End of sequence
pub const UNK_TOKEN_ID: u32 = 0;   // Unknown token
pub const PAD_TOKEN_ID: u32 = 3;   // Padding token

/// Basic tokenizer implementation
pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    vocab_size: usize,
}

impl Tokenizer {
    /// Create a new tokenizer with basic vocabulary
    pub fn new() -> Self {
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

        // Add basic vocabulary - common English words and characters
        let basic_vocab = vec![
            // Common words
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "can", "may", "might", "must",
            "I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
            "this", "that", "these", "those", "here", "there", "where", "when", "why", "how",
            "what", "who", "which", "whose", "all", "some", "any", "no", "not", "yes",
            "hello", "world", "test", "example", "text", "model", "language", "AI",
            // Test words
            "quick", "brown", "fox", "dog", "cat", "house", "car", "tree", "book", "water",
            // Punctuation and symbols
            ".", "!", "?", ",", ";", ":", "'", "\"", "-", "_", "(", ")", "[", "]", "{", "}",
            // Numbers
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            // Letters (for character fallback)
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
            "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
            // Common subwords
            "ing", "ed", "er", "ly", "tion", "ment", "ness", "ity", "ous", "ful",
            " ", "\n", "\t", // Whitespace
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

    /// Create tokenizer from vocabulary file
    pub fn from_vocab_file<P: AsRef<Path>>(vocab_path: P) -> ModelResult<Self> {
        let vocab_content = fs::read_to_string(vocab_path.as_ref())
            .map_err(|e| ModelError::InitializationFailed(format!("Failed to read vocab file: {}", e)))?;

        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();

        for (id, line) in vocab_content.lines().enumerate() {
            let token = line.trim().to_string();
            vocab.insert(token.clone(), id as u32);
            id_to_token.insert(id as u32, token);
        }

        let vocab_size = vocab.len();

        Ok(Self {
            vocab,
            id_to_token,
            vocab_size,
        })
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = vec![BOS_TOKEN_ID]; // Start with BOS token

        // Simple tokenization: split by whitespace and punctuation
        let words = self.tokenize_text(text);

        for word in words {
            if let Some(&token_id) = self.vocab.get(&word) {
                tokens.push(token_id);
            } else {
                // Try character-level fallback
                let char_tokens = self.encode_as_chars(&word);
                tokens.extend(char_tokens);
            }
        }

        tokens.push(EOS_TOKEN_ID); // End with EOS token
        tokens
    }

    /// Decode token IDs back to text
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let mut result = String::new();
        let mut last_was_char = false;

        for &token_id in token_ids {
            if token_id == BOS_TOKEN_ID || token_id == EOS_TOKEN_ID || token_id == PAD_TOKEN_ID {
                continue; // Skip special tokens in output
            }

            if let Some(token) = self.id_to_token.get(&token_id) {
                let is_single_char = token.len() == 1 && token.chars().next().unwrap().is_alphabetic();
                let is_whitespace = token.chars().all(|c| c.is_whitespace());
                let is_punctuation = token.chars().all(|c| c.is_ascii_punctuation());

                // Add space logic
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

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Check if token exists in vocabulary
    pub fn contains_token(&self, token: &str) -> bool {
        self.vocab.contains_key(token)
    }

    /// Get token ID for a string
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// Get string for a token ID
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }

    /// Simple text tokenization
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

    /// Encode unknown word as character tokens
    fn encode_as_chars(&self, word: &str) -> Vec<u32> {
        let mut char_tokens = Vec::new();

        for ch in word.chars() {
            let ch_str = ch.to_string();
            if let Some(&token_id) = self.vocab.get(&ch_str) {
                char_tokens.push(token_id);
            } else {
                char_tokens.push(UNK_TOKEN_ID);
            }
        }

        char_tokens
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

        // Encode all texts
        for text in texts {
            let tokens = self.tokenizer.encode(text);
            encoded_batch.push(tokens);
        }

        // Determine max length
        let max_len = if let Some(max_len) = max_length {
            max_len
        } else {
            encoded_batch.iter().map(|tokens| tokens.len()).max().unwrap_or(0)
        };

        // Pad sequences and create attention masks
        for tokens in &mut encoded_batch {
            let _original_len = tokens.len();

            // Truncate if too long
            if tokens.len() > max_len {
                tokens.truncate(max_len);
            }

            // Create attention mask (true for real tokens, false for padding)
            let mut attention_mask = vec![true; tokens.len()];

            // Pad sequence
            while tokens.len() < max_len {
                tokens.push(PAD_TOKEN_ID);
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
        assert!(tokenizer.vocab_size() > 100); // Should have a reasonable vocabulary
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
        assert!(tokens.len() > 2); // Should have BOS, tokens, EOS
        assert_eq!(tokens[0], BOS_TOKEN_ID);
        assert_eq!(tokens[tokens.len() - 1], EOS_TOKEN_ID);

        let decoded = tokenizer.decode(&tokens);
        assert!(decoded.contains("hello"));
        assert!(decoded.contains("world"));
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let tokenizer = Tokenizer::new();
        let text = "the quick brown fox";

        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens);

        println!("Original: '{}'", text);
        println!("Tokens: {:?}", tokens);
        println!("Decoded: '{}'", decoded);

        // Should preserve the general meaning even if not exact
        assert!(decoded.contains("the"));
        assert!(decoded.contains("quick"));
        assert!(decoded.contains("brown"));
        assert!(decoded.contains("fox"));
    }

    #[test]
    fn test_unknown_word_handling() {
        let tokenizer = Tokenizer::new();
        let text = "supercalifragilisticexpialidocious"; // Likely unknown word

        let tokens = tokenizer.encode(text);
        assert!(tokens.len() > 2); // Should have some tokens even for unknown words

        let decoded = tokenizer.decode(&tokens);
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_punctuation_handling() {
        let tokenizer = Tokenizer::new();
        let text = "Hello, world! How are you?";

        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens);

        assert!(decoded.contains("Hello"));
        assert!(decoded.contains("world"));
        assert!(decoded.contains("How"));
    }

    #[test]
    fn test_empty_text() {
        let tokenizer = Tokenizer::new();
        let tokens = tokenizer.encode("");

        // Should have at least BOS and EOS
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0], BOS_TOKEN_ID);
        assert_eq!(tokens[1], EOS_TOKEN_ID);
    }

    #[test]
    fn test_batch_tokenizer() {
        let tokenizer = Tokenizer::new();
        let batch_tokenizer = BatchTokenizer::new(tokenizer);

        let texts = vec!["hello", "hello world", "hello world test"];
        let (encoded_batch, attention_masks) = batch_tokenizer.encode_batch(&texts, Some(10));

        assert_eq!(encoded_batch.len(), 3);
        assert_eq!(attention_masks.len(), 3);

        // All sequences should be same length
        for tokens in &encoded_batch {
            assert_eq!(tokens.len(), 10);
        }

        for mask in &attention_masks {
            assert_eq!(mask.len(), 10);
        }
    }

    #[test]
    fn test_batch_decode() {
        let tokenizer = Tokenizer::new();
        let batch_tokenizer = BatchTokenizer::new(tokenizer);

        let texts = vec!["hello", "world"];
        let (encoded_batch, _) = batch_tokenizer.encode_batch(&texts, None);
        let decoded_batch = batch_tokenizer.decode_batch(&encoded_batch);

        assert_eq!(decoded_batch.len(), 2);
        assert!(decoded_batch[0].contains("hello"));
        assert!(decoded_batch[1].contains("world"));
    }

    #[test]
    fn test_vocab_size_consistency() {
        let tokenizer = Tokenizer::new();

        // Vocab size should match the number of unique tokens
        assert_eq!(tokenizer.vocab.len(), tokenizer.vocab_size());
        assert_eq!(tokenizer.id_to_token.len(), tokenizer.vocab_size());
    }

    #[test]
    fn test_id_token_mapping_consistency() {
        let tokenizer = Tokenizer::new();

        // Every token in vocab should have a reverse mapping
        for (token, &id) in &tokenizer.vocab {
            assert_eq!(tokenizer.id_to_token.get(&id), Some(token));
        }

        // Every ID should have a reverse mapping
        for (&id, token) in &tokenizer.id_to_token {
            assert_eq!(tokenizer.vocab.get(token), Some(&id));
        }
    }
}