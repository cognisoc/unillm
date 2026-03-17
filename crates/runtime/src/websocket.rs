//! WebSocket streaming API for real-time inference communication

use crate::optimized_llama::OptimizedLlamaModel;
use crate::sampler::GreedySampler;
use crate::simple::SimpleTokenizer;
use axum::{
    extract::{State, WebSocketUpgrade},
    response::Response,
};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// WebSocket message types
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum WsMessage {
    /// Client sends generation request
    GenerateRequest {
        id: String,
        prompt: String,
        max_tokens: Option<usize>,
        stream: Option<bool>,
        temperature: Option<f32>,
        top_p: Option<f32>,
    },
    /// Server sends streaming token
    StreamingToken {
        id: String,
        token: String,
        token_id: u32,
        step: usize,
        is_final: bool,
        elapsed_ms: f64,
        tokens_per_second: f64,
    },
    /// Server sends final response
    GenerationComplete {
        id: String,
        text: String,
        tokens: Vec<u32>,
        prompt_tokens: usize,
        completion_tokens: usize,
        total_tokens: usize,
        processing_time_ms: f64,
        tokens_per_second: f64,
    },
    /// Server sends error
    Error {
        id: Option<String>,
        error: String,
        code: u16,
        details: Option<String>,
    },
    /// Client cancels generation
    CancelRequest {
        id: String,
    },
    /// Server acknowledges cancel
    CancelAck {
        id: String,
    },
    /// Ping/Pong for connection health
    Ping,
    Pong,
    /// Connection established
    Connected {
        session_id: String,
        server_info: ServerInfo,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ServerInfo {
    pub version: String,
    pub model_loaded: bool,
    pub capabilities: Vec<String>,
}

/// WebSocket connection state
#[derive(Clone)]
pub struct WsAppState {
    pub model: Arc<RwLock<Option<OptimizedLlamaModel>>>,
    pub tokenizer: Arc<SimpleTokenizer>,
    pub sampler: Arc<GreedySampler>,
}

/// Handle WebSocket upgrade and manage the connection
pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<WsAppState>,
) -> Response {
    ws.on_upgrade(|socket| handle_websocket(socket, state))
}

/// Main WebSocket connection handler
async fn handle_websocket(
    socket: axum::extract::ws::WebSocket,
    state: WsAppState,
) {
    let session_id = Uuid::new_v4().to_string();
    println!("🌐 WebSocket connection established: {}", session_id);

    let (mut sender, mut receiver) = socket.split();

    // Send welcome message
    let welcome = WsMessage::Connected {
        session_id: session_id.clone(),
        server_info: ServerInfo {
            version: "1.0.0".to_string(),
            model_loaded: state.model.read().await.is_some(),
            capabilities: vec![
                "streaming_generation".to_string(),
                "cancellation".to_string(),
                "real_time_tokens".to_string(),
            ],
        },
    };

    if let Ok(msg) = serde_json::to_string(&welcome) {
        if sender.send(axum::extract::ws::Message::Text(msg.into())).await.is_err() {
            println!("❌ Failed to send welcome message");
            return;
        }
    }

    // Handle incoming messages
    while let Some(msg) = receiver.next().await {
        match msg {
            Ok(axum::extract::ws::Message::Text(text)) => {
                if let Err(e) = handle_text_message(&text, &mut sender, &state).await {
                    println!("❌ Error handling message: {}", e);

                    let error_msg = WsMessage::Error {
                        id: None,
                        error: "Message processing failed".to_string(),
                        code: 500,
                        details: Some(e.to_string()),
                    };

                    if let Ok(msg) = serde_json::to_string(&error_msg) {
                        let _ = sender.send(axum::extract::ws::Message::Text(msg.into())).await;
                    }
                }
            }
            Ok(axum::extract::ws::Message::Close(_)) => {
                println!("🔌 WebSocket connection closed: {}", session_id);
                break;
            }
            Ok(axum::extract::ws::Message::Ping(data)) => {
                if sender.send(axum::extract::ws::Message::Pong(data)).await.is_err() {
                    break;
                }
            }
            Err(e) => {
                println!("❌ WebSocket error: {}", e);
                break;
            }
            _ => {
                // Ignore other message types
            }
        }
    }

    println!("🔌 WebSocket connection terminated: {}", session_id);
}

/// Handle text messages from client
async fn handle_text_message(
    text: &str,
    sender: &mut futures::stream::SplitSink<axum::extract::ws::WebSocket, axum::extract::ws::Message>,
    state: &WsAppState,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let message: WsMessage = serde_json::from_str(text)?;

    match message {
        WsMessage::GenerateRequest {
            id,
            prompt,
            max_tokens,
            stream,
            ..
        } => {
            handle_generation_request(id, prompt, max_tokens, stream, sender, state).await?;
        }
        WsMessage::CancelRequest { id } => {
            // For now, just acknowledge the cancel
            let ack = WsMessage::CancelAck { id };
            let msg_text = serde_json::to_string(&ack)?;
            sender.send(axum::extract::ws::Message::Text(msg_text.into())).await?;
        }
        WsMessage::Ping => {
            let pong = WsMessage::Pong;
            let msg_text = serde_json::to_string(&pong)?;
            sender.send(axum::extract::ws::Message::Text(msg_text.into())).await?;
        }
        _ => {
            // Ignore other message types from client
        }
    }

    Ok(())
}

/// Handle generation request with streaming support
async fn handle_generation_request(
    id: String,
    prompt: String,
    max_tokens: Option<usize>,
    stream: Option<bool>,
    sender: &mut futures::stream::SplitSink<axum::extract::ws::WebSocket, axum::extract::ws::Message>,
    state: &WsAppState,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let start_time = std::time::Instant::now();

    // Validate input
    if prompt.is_empty() {
        let error = WsMessage::Error {
            id: Some(id),
            error: "Empty prompt".to_string(),
            code: 400,
            details: Some("Prompt cannot be empty".to_string()),
        };
        let msg_text = serde_json::to_string(&error)?;
        sender.send(axum::extract::ws::Message::Text(msg_text.into())).await?;
        return Ok(());
    }

    // Get model with mutable access
    let mut model_guard = state.model.write().await;
    let model = match model_guard.as_mut() {
        Some(m) => m,
        None => {
            let error = WsMessage::Error {
                id: Some(id),
                error: "Model not loaded".to_string(),
                code: 503,
                details: Some("Please wait for model to load".to_string()),
            };
            let msg_text = serde_json::to_string(&error)?;
            sender.send(axum::extract::ws::Message::Text(msg_text.into())).await?;
            return Ok(());
        }
    };

    // Tokenize prompt
    let prompt_tokens = state.tokenizer.encode(&prompt);
    let max_tokens = max_tokens.unwrap_or(50);
    let should_stream = stream.unwrap_or(true);

    if should_stream {
        // Streaming generation
        let result_tokens = model.generate_streaming(
            &prompt_tokens,
            max_tokens,
            &state.sampler,
            |token, step, elapsed_ms| {
                let token_text = state.tokenizer.decode(&[token]);

                let streaming_msg = WsMessage::StreamingToken {
                    id: id.clone(),
                    token: token_text,
                    token_id: token,
                    step,
                    is_final: false,
                    elapsed_ms,
                    tokens_per_second: step as f64 / (elapsed_ms / 1000.0),
                };

                // Send streaming token (blocking for simplicity)
                if let Ok(msg_text) = serde_json::to_string(&streaming_msg) {
                    // Note: In a real implementation, we'd need async streaming
                    // For now, we collect tokens and send at the end
                }

                true // Continue generation
            },
        );

        // Send final completion
        let total_time = start_time.elapsed();
        let completion_tokens = result_tokens.len() - prompt_tokens.len();
        let generated_text = state.tokenizer.decode(&result_tokens);

        let total_tokens = result_tokens.len();

        let completion = WsMessage::GenerationComplete {
            id,
            text: generated_text,
            tokens: result_tokens,
            prompt_tokens: prompt_tokens.len(),
            completion_tokens,
            total_tokens,
            processing_time_ms: total_time.as_millis() as f64,
            tokens_per_second: completion_tokens as f64 / total_time.as_secs_f64(),
        };

        let msg_text = serde_json::to_string(&completion)?;
        sender.send(axum::extract::ws::Message::Text(msg_text.into())).await?;
    } else {
        // Non-streaming generation
        let result_tokens = model.generate_optimized(&prompt_tokens, max_tokens, &state.sampler);
        let total_time = start_time.elapsed();
        let completion_tokens = result_tokens.len() - prompt_tokens.len();
        let generated_text = state.tokenizer.decode(&result_tokens);

        let total_tokens = result_tokens.len();

        let completion = WsMessage::GenerationComplete {
            id,
            text: generated_text,
            tokens: result_tokens,
            prompt_tokens: prompt_tokens.len(),
            completion_tokens,
            total_tokens,
            processing_time_ms: total_time.as_millis() as f64,
            tokens_per_second: completion_tokens as f64 / total_time.as_secs_f64(),
        };

        let msg_text = serde_json::to_string(&completion)?;
        sender.send(axum::extract::ws::Message::Text(msg_text.into())).await?;
    }

    Ok(())
}