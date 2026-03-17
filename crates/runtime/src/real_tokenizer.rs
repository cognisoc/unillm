//! Real Tokenizer Implementation
//!
//! Implements actual tokenization using BPE/SentencePiece instead of placeholders

use crate::types::*;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Real tokenizer with actual BPE implementation
pub struct RealTokenizer {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    merges: Vec<(String, String)>,
    vocab_size: usize,
    bos_token_id: u32,
    eos_token_id: u32,
    unk_token_id: u32,
    pad_token_id: u32,
}

impl RealTokenizer {
    /// Load tokenizer from vocabulary and merges files
    pub fn load<P: AsRef<Path>>(
        vocab_path: P,
        merges_path: Option<P>,
    ) -> ModelResult<Self> {
        println!("🔄 Loading real tokenizer...");

        let vocab = Self::load_vocabulary(vocab_path)?;
        let reverse_vocab = vocab.iter()
            .map(|(k, &v)| (v, k.clone()))
            .collect();

        let merges = if let Some(merges_path) = merges_path {
            Self::load_merges(merges_path)?
        } else {
            Vec::new()
        };

        let vocab_size = vocab.len();

        // Standard special tokens (may need adjustment based on model)
        let bos_token_id = *vocab.get("<s>").unwrap_or(&1);
        let eos_token_id = *vocab.get("</s>").unwrap_or(&2);
        let unk_token_id = *vocab.get("<unk>").unwrap_or(&0);
        let pad_token_id = *vocab.get("<pad>").unwrap_or(&0);

        println!("✅ Loaded tokenizer with {} vocab items and {} merges",
                vocab_size, merges.len());

        Ok(Self {
            vocab,
            reverse_vocab,
            merges,
            vocab_size,
            bos_token_id,
            eos_token_id,
            unk_token_id,
            pad_token_id,
        })
    }

    /// Create a basic tokenizer from a simple word list (fallback)
    pub fn from_word_list(words: Vec<String>) -> ModelResult<Self> {
        let mut vocab = HashMap::new();

        // Add special tokens first
        vocab.insert("<pad>".to_string(), 0);
        vocab.insert("<unk>".to_string(), 1);
        vocab.insert("<s>".to_string(), 2);
        vocab.insert("</s>".to_string(), 3);

        // Add words
        let mut token_id = 4;
        for word in words {
            vocab.insert(word, token_id);
            token_id += 1;
        }

        let reverse_vocab = vocab.iter()
            .map(|(k, &v)| (v, k.clone()))
            .collect();

        Ok(Self {
            vocab_size: vocab.len(),
            reverse_vocab,
            vocab,
            merges: Vec::new(),
            bos_token_id: 2,
            eos_token_id: 3,
            unk_token_id: 1,
            pad_token_id: 0,
        })
    }

    /// Load vocabulary from JSON file
    fn load_vocabulary<P: AsRef<Path>>(vocab_path: P) -> ModelResult<HashMap<String, u32>> {
        let vocab_str = fs::read_to_string(vocab_path.as_ref())
            .map_err(|e| ModelError::InitializationFailed(
                format!("Failed to read vocabulary: {}", e)
            ))?;

        if vocab_path.as_ref().extension().and_then(|s| s.to_str()) == Some("json") {
            // JSON format
            let vocab_json: serde_json::Value = serde_json::from_str(&vocab_str)
                .map_err(|e| ModelError::InitializationFailed(
                    format!("Failed to parse vocabulary JSON: {}", e)
                ))?;

            if let Some(obj) = vocab_json.as_object() {
                let mut vocab = HashMap::new();
                for (token, id) in obj {
                    if let Some(id_num) = id.as_u64() {
                        vocab.insert(token.clone(), id_num as u32);
                    }
                }
                Ok(vocab)
            } else {
                Err(ModelError::InitializationFailed(
                    "Vocabulary JSON is not an object".to_string()
                ))
            }
        } else {
            // Plain text format (one token per line with optional ID)
            let mut vocab = HashMap::new();
            for (line_num, line) in vocab_str.lines().enumerate() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                if let Some(tab_pos) = line.find('\t') {
                    // Format: token\tid
                    let token = line[..tab_pos].to_string();
                    let id_str = &line[tab_pos + 1..];
                    let id = id_str.parse::<u32>()
                        .map_err(|_| ModelError::InitializationFailed(
                            format!("Invalid token ID: {}", id_str)
                        ))?;
                    vocab.insert(token, id);
                } else {
                    // Format: one token per line, use line number as ID
                    vocab.insert(line.to_string(), line_num as u32);
                }
            }
            Ok(vocab)
        }
    }

    /// Load BPE merges from file
    fn load_merges<P: AsRef<Path>>(merges_path: P) -> ModelResult<Vec<(String, String)>> {
        let merges_str = fs::read_to_string(merges_path.as_ref())
            .map_err(|e| ModelError::InitializationFailed(
                format!("Failed to read merges: {}", e)
            ))?;

        let mut merges = Vec::new();
        for line in merges_str.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                merges.push((parts[0].to_string(), parts[1].to_string()));
            }
        }

        Ok(merges)
    }

    /// Tokenize text into token IDs
    pub fn encode(&self, text: &str) -> ModelResult<Vec<u32>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        // Basic preprocessing
        let text = self.preprocess_text(text);

        // Apply BPE if we have merges
        let tokens = if !self.merges.is_empty() {
            self.apply_bpe(&text)?
        } else {
            self.basic_tokenize(&text)
        };

        // Convert tokens to IDs
        let mut token_ids = Vec::new();
        for token in tokens {
            let id = *self.vocab.get(&token).unwrap_or(&self.unk_token_id);
            token_ids.push(id);
        }

        Ok(token_ids)
    }

    /// Decode token IDs back to text
    pub fn decode(&self, token_ids: &[u32]) -> ModelResult<String> {
        let mut tokens = Vec::new();

        for &token_id in token_ids {
            if let Some(token) = self.reverse_vocab.get(&token_id) {
                tokens.push(token.clone());
            } else {
                tokens.push(self.reverse_vocab.get(&self.unk_token_id)
                    .unwrap_or(&"<unk>".to_string())
                    .clone());
            }
        }

        Ok(self.postprocess_tokens(tokens))
    }

    /// Basic text preprocessing
    fn preprocess_text(&self, text: &str) -> String {
        // Basic cleanup - in practice, this would be more sophisticated
        text.to_lowercase()
            .replace('\n', " ")
            .replace('\t', " ")
            .split_whitespace()
            .collect::<Vec<&str>>()
            .join(" ")
    }

    /// Apply Byte Pair Encoding
    fn apply_bpe(&self, text: &str) -> ModelResult<Vec<String>> {
        // Start with character-level tokenization
        let mut word_tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();

        // Apply merges in order
        for (merge_a, merge_b) in &self.merges {
            word_tokens = self.apply_merge(&word_tokens, merge_a, merge_b);
        }

        Ok(word_tokens)
    }

    /// Apply a single BPE merge
    fn apply_merge(&self, tokens: &[String], merge_a: &str, merge_b: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut i = 0;

        while i < tokens.len() {
            if i < tokens.len() - 1 && tokens[i] == merge_a && tokens[i + 1] == merge_b {
                // Merge these two tokens
                result.push(format!("{}{}", merge_a, merge_b));
                i += 2;
            } else {
                result.push(tokens[i].clone());
                i += 1;
            }
        }

        result
    }

    /// Basic word-level tokenization (fallback)
    fn basic_tokenize(&self, text: &str) -> Vec<String> {
        // Simple whitespace tokenization with basic punctuation handling
        let mut tokens = Vec::new();
        let mut current_word = String::new();

        for ch in text.chars() {
            if ch.is_whitespace() {
                if !current_word.is_empty() {
                    tokens.push(current_word);
                    current_word = String::new();
                }
            } else if ch.is_ascii_punctuation() {
                // Treat punctuation as separate tokens
                if !current_word.is_empty() {
                    tokens.push(current_word);
                    current_word = String::new();
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

    /// Post-process decoded tokens back to readable text
    fn postprocess_tokens(&self, tokens: Vec<String>) -> String {
        let mut text = tokens.join(" ");

        // Basic cleanup
        text = text.replace(" <s>", "")
            .replace("</s> ", "")
            .replace("<s>", "")
            .replace("</s>", "")
            .replace("<pad>", "")
            .replace("<unk>", "[UNK]");

        // Fix spacing around punctuation
        for punct in &[".", ",", "!", "?", ":", ";"] {
            text = text.replace(&format!(" {}", punct), punct);
        }

        text.trim().to_string()
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get special token IDs
    pub fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn unk_token_id(&self) -> u32 {
        self.unk_token_id
    }

    pub fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }

    /// Encode text with special tokens
    pub fn encode_with_special_tokens(&self, text: &str, add_bos: bool, add_eos: bool) -> ModelResult<Vec<u32>> {
        let mut tokens = Vec::new();

        if add_bos {
            tokens.push(self.bos_token_id);
        }

        tokens.extend(self.encode(text)?);

        if add_eos {
            tokens.push(self.eos_token_id);
        }

        Ok(tokens)
    }

    /// Create a simple test tokenizer for demos
    pub fn create_demo_tokenizer() -> ModelResult<Self> {
        // Create a basic vocabulary with common English words and tokens
        let demo_words = vec![
            // Special tokens
            "<pad>".to_string(), "<unk>".to_string(), "<s>".to_string(), "</s>".to_string(),
            // Common words
            "the".to_string(), "and".to_string(), "a".to_string(), "to".to_string(), "of".to_string(),
            "in".to_string(), "is".to_string(), "it".to_string(), "you".to_string(), "that".to_string(),
            "he".to_string(), "was".to_string(), "for".to_string(), "on".to_string(), "are".to_string(),
            "as".to_string(), "with".to_string(), "his".to_string(), "they".to_string(), "at".to_string(),
            "be".to_string(), "this".to_string(), "have".to_string(), "from".to_string(), "or".to_string(),
            "one".to_string(), "had".to_string(), "by".to_string(), "word".to_string(), "but".to_string(),
            "what".to_string(), "some".to_string(), "we".to_string(), "can".to_string(), "out".to_string(),
            "other".to_string(), "were".to_string(), "all".to_string(), "there".to_string(), "when".to_string(),
            "up".to_string(), "use".to_string(), "your".to_string(), "how".to_string(), "said".to_string(),
            "an".to_string(), "each".to_string(), "which".to_string(), "she".to_string(), "do".to_string(),
            "their".to_string(), "time".to_string(), "will".to_string(), "about".to_string(), "if".to_string(),
            "up".to_string(), "out".to_string(), "many".to_string(), "then".to_string(), "them".to_string(),
            // Question words and common phrases
            "what".to_string(), "where".to_string(), "when".to_string(), "why".to_string(), "how".to_string(),
            "who".to_string(), "capital".to_string(), "france".to_string(), "paris".to_string(),
            "hello".to_string(), "world".to_string(), "good".to_string(), "morning".to_string(),
            "thank".to_string(), "please".to_string(), "yes".to_string(), "no".to_string(),
            // Punctuation
            ".".to_string(), ",".to_string(), "!".to_string(), "?".to_string(),
            ":".to_string(), ";".to_string(), "'".to_string(), "\"".to_string(),
        ];

        Self::from_word_list(demo_words)
    }

    /// Batch encode multiple texts
    pub fn batch_encode(&self, texts: &[String]) -> ModelResult<Vec<Vec<u32>>> {
        texts.iter()
            .map(|text| self.encode(text))
            .collect()
    }

    /// Batch decode multiple token sequences
    pub fn batch_decode(&self, token_sequences: &[Vec<u32>]) -> ModelResult<Vec<String>> {
        token_sequences.iter()
            .map(|tokens| self.decode(tokens))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_tokenizer() -> Result<(), Box<dyn std::error::Error>> {
        let tokenizer = RealTokenizer::create_demo_tokenizer()?;

        // Test basic encoding/decoding
        let text = "Hello world!";
        let tokens = tokenizer.encode(text)?;
        let decoded = tokenizer.decode(&tokens)?;

        println!("Original: {}", text);
        println!("Tokens: {:?}", tokens);
        println!("Decoded: {}", decoded);

        assert!(!tokens.is_empty());
        assert!(decoded.contains("hello"));
        assert!(decoded.contains("world"));

        Ok(())
    }

    #[test]
    fn test_special_tokens() -> Result<(), Box<dyn std::error::Error>> {
        let tokenizer = RealTokenizer::create_demo_tokenizer()?;

        let tokens = tokenizer.encode_with_special_tokens("Hello", true, true)?;

        // Should have BOS + content + EOS
        assert!(tokens.len() >= 3);
        assert_eq!(tokens[0], tokenizer.bos_token_id());
        assert_eq!(tokens[tokens.len() - 1], tokenizer.eos_token_id());

        Ok(())
    }

    #[test]
    fn test_batch_operations() -> Result<(), Box<dyn std::error::Error>> {
        let tokenizer = RealTokenizer::create_demo_tokenizer()?;

        let texts = vec![
            "Hello world".to_string(),
            "Good morning".to_string(),
        ];

        let token_sequences = tokenizer.batch_encode(&texts)?;
        let decoded_texts = tokenizer.batch_decode(&token_sequences)?;

        assert_eq!(token_sequences.len(), 2);
        assert_eq!(decoded_texts.len(), 2);

        Ok(())
    }

    #[test]
    fn test_unknown_tokens() -> Result<(), Box<dyn std::error::Error>> {
        let tokenizer = RealTokenizer::create_demo_tokenizer()?;

        let tokens = tokenizer.encode("XyZwVuTsRq")?; // Nonsense word not in vocab
        assert!(tokens.contains(&tokenizer.unk_token_id()));

        Ok(())
    }
}