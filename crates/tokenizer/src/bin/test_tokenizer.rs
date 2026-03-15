//! Test program for the tokenizer

use tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing tokenizer...");
    
    // Create a test tokenizer
    let tokenizer = Tokenizer::new_test()?;
    
    println!("Tokenizer created with vocab size: {}", tokenizer.vocab_size());
    
    // Test encoding
    let text = "hello world";
    let tokens = tokenizer.encode(text, false)?;
    
    println!("Encoded '{}' into {} tokens:", text, tokens.len());
    for token in &tokens {
        println!("  Token {}: '{}' (id: {})", token.text, token.text, token.id);
    }
    
    // Test decoding
    let decoded = tokenizer.decode(&tokens, false)?;
    println!("Decoded back to: '{}'", decoded);
    
    Ok(())
}