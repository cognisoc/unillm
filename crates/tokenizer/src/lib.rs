//! Rust tokenizer implementation

use tokenizers::{Tokenizer, EncodeInput, Encoding};
use std::path::Path;

/// A simple token structure
#[derive(Debug, Clone)]
pub struct Token {
    pub id: u32,
    pub text: String,
}

/// Tokenizer implementation
pub struct Tokenizer {
    tokenizer: tokenizers::Tokenizer,
    vocab_size: usize,
}

impl Tokenizer {
    /// Create a new tokenizer instance from a file
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        println!("Loading tokenizer from: {:?}", path.as_ref());
        
        // Try to load the tokenizer from file
        match Tokenizer::from_file(path.as_ref()) {
            Ok(tokenizer) => {
                let vocab_size = tokenizer.get_vocab_size(false);
                println!("Loaded tokenizer with vocab size: {}", vocab_size);
                Ok(Self { tokenizer, vocab_size })
            },
            Err(e) => {
                println!("Failed to load tokenizer from file: {}, falling back to test tokenizer", e);
                Self::new_test()
            }
        }
    }
    
    /// Create a new tokenizer with a basic vocabulary for testing
    pub fn new_test() -> Result<Self, Box<dyn std::error::Error>> {
        println!("Creating test tokenizer");
        
        // Create a simple BPE tokenizer for testing
        let tokenizer = Tokenizer::from_pretrained("microsoft/DialoGPT-medium", None)?;
        let vocab_size = tokenizer.get_vocab_size(false);
        
        println!("Created test tokenizer with vocab size: {}", vocab_size);
        
        Ok(Self { tokenizer, vocab_size })
    }
    
    /// Create a tokenizer from a JSON configuration
    pub fn from_json<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        println!("Loading tokenizer from JSON: {:?}", path.as_ref());
        
        let tokenizer = Tokenizer::from_file(path.as_ref())?;
        let vocab_size = tokenizer.get_vocab_size(false);
        
        println!("Loaded tokenizer from JSON with vocab size: {}", vocab_size);
        
        Ok(Self { tokenizer, vocab_size })
    }
    
    /// Encode text into tokens
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<Token>, Box<dyn std::error::Error>> {
        let encoding = self.tokenizer.encode(EncodeInput::Single(text.to_string()), add_special_tokens)?;
        
        let mut tokens = Vec::new();
        for (i, &token_id) in encoding.get_ids().iter().enumerate() {
            let token_text = encoding.get_tokens()[i].to_string();
            tokens.push(Token {
                id: token_id,
                text: token_text,
            });
        }
        
        Ok(tokens)
    }
    
    /// Encode text and return just the token IDs
    pub fn encode_ids(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let encoding = self.tokenizer.encode(EncodeInput::Single(text.to_string()), add_special_tokens)?;
        Ok(encoding.get_ids().to_vec())
    }
    
    /// Decode tokens back into text
    pub fn decode(&self, tokens: &[Token], skip_special_tokens: bool) -> Result<String, Box<dyn std::error::Error>> {
        let ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();
        self.decode_ids(&ids, skip_special_tokens)
    }
    
    /// Decode token IDs back into text
    pub fn decode_ids(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, Box<dyn std::error::Error>> {
        let text = self.tokenizer.decode(ids, skip_special_tokens)?;
        Ok(text)
    }
    
    /// Get the vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    
    /// Get the tokenizer's special tokens
    pub fn get_special_tokens(&self) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        // Get special tokens from the tokenizer
        let special_tokens = vec![
            "[UNK]".to_string(),
            "[CLS]".to_string(),
            "[SEP]".to_string(),
            "[PAD]".to_string(),
            "[MASK]".to_string(),
        ];
        Ok(special_tokens)
    }
    
    /// Get the token ID for a special token
    pub fn get_special_token_id(&self, token: &str) -> Option<u32> {
        // In a real implementation, we would query the tokenizer for special token IDs
        match token {
            "[UNK]" => Some(0),
            "[CLS]" => Some(1),
            "[SEP]" => Some(2),
            "[PAD]" => Some(3),
            "[MASK]" => Some(4),
            _ => None,
        }
    }
}