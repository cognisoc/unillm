//! Test client for UniLLM Production API
//!
//! Demonstrates loading models and generating completions.

use reqwest::Client;
use serde_json::json;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, error};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("debug")
        .init();

    let client = Client::new();
    let base_url = "http://localhost:8000";

    info!("🧪 Testing UniLLM Production API");

    // Test health endpoint
    info!("🏥 Testing health endpoint...");
    let response = client
        .get(&format!("{}/unillm/health", base_url))
        .send()
        .await?;

    if response.status().is_success() {
        let health: serde_json::Value = response.json().await?;
        info!("✅ Health check passed: {}", health);
    } else {
        error!("❌ Health check failed: {}", response.status());
        return Ok(());
    }

    // Test metrics endpoint
    info!("📊 Testing metrics endpoint...");
    let response = client
        .get(&format!("{}/unillm/metrics", base_url))
        .send()
        .await?;

    if response.status().is_success() {
        let metrics: serde_json::Value = response.json().await?;
        info!("✅ Metrics retrieved: {}", metrics);
    } else {
        error!("❌ Metrics failed: {}", response.status());
    }

    // Test models list endpoint
    info!("📚 Testing models list endpoint...");
    let response = client
        .get(&format!("{}/v1/models", base_url))
        .send()
        .await?;

    if response.status().is_success() {
        let models: serde_json::Value = response.json().await?;
        info!("✅ Models list: {}", models);
    } else {
        error!("❌ Models list failed: {}", response.status());
    }

    // Test model loading (this would fail without actual model files)
    info!("🔄 Testing model loading endpoint (will likely fail without model files)...");
    let load_request = json!({
        "model_id": "test-model",
        "model_type": "gguf",
        "device": "cpu"
    });

    let response = client
        .post(&format!("{}/unillm/models/load", base_url))
        .json(&load_request)
        .send()
        .await?;

    match response.status().as_u16() {
        200 => {
            let result: serde_json::Value = response.json().await?;
            info!("✅ Model loaded successfully: {}", result);

            // Test chat completion with loaded model
            info!("💬 Testing chat completion...");
            let chat_request = json!({
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"}
                ],
                "max_tokens": 50
            });

            let response = client
                .post(&format!("{}/v1/chat/completions", base_url))
                .json(&chat_request)
                .send()
                .await?;

            if response.status().is_success() {
                let completion: serde_json::Value = response.json().await?;
                info!("✅ Chat completion successful: {}", completion);
            } else {
                let error_text = response.text().await?;
                error!("❌ Chat completion failed: {}", error_text);
            }
        }
        _ => {
            let error_text = response.text().await?;
            info!("⚠️  Model loading failed as expected (no model files): {}", error_text);
        }
    }

    // Test chat completion without model (should fail)
    info!("💬 Testing chat completion without model (should fail)...");
    let chat_request = json!({
        "model": "nonexistent-model",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "max_tokens": 50
    });

    let response = client
        .post(&format!("{}/v1/chat/completions", base_url))
        .json(&chat_request)
        .send()
        .await?;

    if response.status().is_success() {
        let completion: serde_json::Value = response.json().await?;
        error!("🚨 Unexpected success with nonexistent model: {}", completion);
    } else {
        info!("✅ Chat completion correctly failed for nonexistent model: {}", response.status());
    }

    info!("🎉 API test completed!");

    Ok(())
}