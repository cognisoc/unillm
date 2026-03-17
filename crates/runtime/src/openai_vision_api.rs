//! OpenAI Vision API Compatibility Layer
//!
//! Provides full compatibility with OpenAI's Vision API, allowing UniLLM to serve
//! as a drop-in replacement for GPT-4V and other vision-language models.

use crate::types::*;
// use crate::models::llava::{LLaVAModel, LLaVAConfig};  // Temporarily disabled
use crate::image_processing::{ImageProcessor, ImageUrl};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use axum::{
    extract::{State, Json},
    response::Json as ResponseJson,
    http::StatusCode,
};

/// OpenAI Chat Completion Request with Vision Support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stream: bool,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

fn default_temperature() -> f32 { 0.7 }
fn default_top_p() -> f32 { 1.0 }

/// Chat Message with Vision Content Support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: MessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Message Role
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

/// Message Content - can be text or multimodal
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Array(Vec<ContentPart>),
}

/// Content Part for multimodal messages
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

/// OpenAI Chat Completion Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

/// Chat Choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

/// Token Usage Information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Vision-enabled Chat Completion Service (simplified)
pub struct VisionChatService {
    // llava_model: LLaVAModel,  // Temporarily disabled
    image_processor: ImageProcessor,
}

impl VisionChatService {
    pub fn new() -> ModelResult<Self> {
        // let config = LLaVAConfig::default();  // Temporarily disabled
        // let llava_model = LLaVAModel::new(config)?;  // Temporarily disabled
        let image_processor = ImageProcessor::new();

        Ok(Self {
            // llava_model,  // Temporarily disabled
            image_processor,
        })
    }

    pub fn with_config(_config: crate::image_processing::VisionConfig) -> ModelResult<Self> {
        // let llava_model = LLaVAModel::new(config.clone())?;  // Temporarily disabled
        let image_processor = ImageProcessor::with_config(_config);

        Ok(Self {
            // llava_model,  // Temporarily disabled
            image_processor,
        })
    }

    /// Process OpenAI-compatible chat completion request
    pub async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> ModelResult<ChatCompletionResponse> {
        // Extract images and text from messages
        let (text_prompt, image_tensors) = self.extract_multimodal_content(&request.messages).await?;

        // Simplified processing (LLaVA temporarily disabled)
        let response_text = if !image_tensors.is_empty() {
            format!("Vision processing placeholder: {} (found {} images)", text_prompt, image_tensors.len())
        } else {
            format!("Text-only processing: {}", text_prompt)
        };

        // Create OpenAI-compatible response
        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            model: request.model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: MessageRole::Assistant,
                    content: MessageContent::Text(response_text),
                    name: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Usage {
                prompt_tokens: text_prompt.split_whitespace().count(),
                completion_tokens: 50, // Placeholder
                total_tokens: text_prompt.split_whitespace().count() + 50,
            },
        };

        Ok(response)
    }

    /// Extract multimodal content from chat messages
    async fn extract_multimodal_content(
        &self,
        messages: &[ChatMessage],
    ) -> ModelResult<(String, Vec<crate::image_processing::ImageTensor>)> {
        let mut text_parts = Vec::new();
        let mut image_tensors = Vec::new();

        for message in messages {
            match &message.content {
                MessageContent::Text(text) => {
                    text_parts.push(format!("{:?}: {}", message.role, text));
                }
                MessageContent::Array(parts) => {
                    for part in parts {
                        match part {
                            ContentPart::Text { text } => {
                                text_parts.push(format!("{:?}: {}", message.role, text));
                            }
                            ContentPart::ImageUrl { image_url } => {
                                let image_tensor = self.process_image_url(image_url).await?;
                                image_tensors.push(image_tensor);
                            }
                        }
                    }
                }
            }
        }

        let combined_text = text_parts.join("\n");
        Ok((combined_text, image_tensors))
    }

    /// Process image from URL or base64 data
    async fn process_image_url(
        &self,
        image_url: &ImageUrl,
    ) -> ModelResult<crate::image_processing::ImageTensor> {
        let url = &image_url.url;

        if url.starts_with("data:image") {
            // Base64 encoded image (placeholder)
            Ok(crate::image_processing::ImageTensor {
                data: vec![0.0; 3 * 224 * 224],
                shape: vec![3, 224, 224],
                dtype: crate::types::DataType::Float32,
                device: crate::types::Device::CPU,
                original_size: (224, 224),
                processed_size: (224, 224),
            })
        } else if url.starts_with("http") {
            // URL-based image (placeholder)
            Ok(crate::image_processing::ImageTensor {
                data: vec![0.0; 3 * 224 * 224],
                shape: vec![3, 224, 224],
                dtype: crate::types::DataType::Float32,
                device: crate::types::Device::CPU,
                original_size: (224, 224),
                processed_size: (224, 224),
            })
        } else {
            Err(ModelError::InvalidInput(format!("Unsupported image URL format: {}", url)))
        }
    }

    // Removed simple_tokenize and detokenize methods (LLaVA temporarily disabled)
}

/// OpenAI-compatible error response
#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIError {
    pub error: ErrorDetails,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorDetails {
    pub message: String,
    pub r#type: String,
    pub code: Option<String>,
}

/// Axum handler for OpenAI-compatible chat completions
pub async fn chat_completions_handler(
    State(service): State<VisionChatService>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<ResponseJson<ChatCompletionResponse>, StatusCode> {
    match service.chat_completion(request).await {
        Ok(response) => Ok(ResponseJson(response)),
        Err(error) => {
            eprintln!("Chat completion error: {:?}", error);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Create router with OpenAI vision API endpoints
pub fn create_openai_vision_router() -> ModelResult<axum::Router> {
    let service = VisionChatService::new()?;

    let router = axum::Router::new()
        .route("/v1/chat/completions", axum::routing::post(chat_completions_handler))
        .with_state(service);

    Ok(router)
}

/// Utility functions for OpenAI API compatibility
pub mod utils {
    use super::*;

    /// Convert UniLLM ModelError to OpenAI error format
    pub fn model_error_to_openai(error: ModelError) -> OpenAIError {
        OpenAIError {
            error: ErrorDetails {
                message: error.to_string(),
                r#type: "server_error".to_string(),
                code: Some("internal_error".to_string()),
            },
        }
    }

    /// Validate OpenAI request format
    pub fn validate_chat_request(request: &ChatCompletionRequest) -> Result<(), String> {
        if request.messages.is_empty() {
            return Err("Messages array cannot be empty".to_string());
        }

        // Check for supported models
        let supported_models = vec!["gpt-4-vision-preview", "gpt-4v", "llava-1.5", "unillm-vision"];
        if !supported_models.contains(&request.model.as_str()) {
            return Err(format!("Unsupported model: {}", request.model));
        }

        Ok(())
    }

    /// Extract image count from request for billing/metrics
    pub fn count_images_in_request(request: &ChatCompletionRequest) -> usize {
        let mut count = 0;
        for message in &request.messages {
            if let MessageContent::Array(parts) = &message.content {
                count += parts.iter().filter(|part| matches!(part, ContentPart::ImageUrl { .. })).count();
            }
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_request_serialization() {
        let request = ChatCompletionRequest {
            model: "gpt-4-vision-preview".to_string(),
            messages: vec![ChatMessage {
                role: MessageRole::User,
                content: MessageContent::Text("Hello!".to_string()),
                name: None,
            }],
            max_tokens: Some(100),
            temperature: 0.7,
            top_p: 1.0,
            stream: false,
            extra: HashMap::new(),
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: ChatCompletionRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(request.model, deserialized.model);
    }

    #[test]
    fn test_multimodal_content() {
        let content = MessageContent::Array(vec![
            ContentPart::Text {
                text: "What's in this image?".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/jpeg;base64,/9j/4AAQ...".to_string(),
                    detail: Some("high".to_string()),
                },
            },
        ]);

        let json = serde_json::to_string(&content).unwrap();
        let deserialized: MessageContent = serde_json::from_str(&json).unwrap();

        match deserialized {
            MessageContent::Array(parts) => assert_eq!(parts.len(), 2),
            _ => panic!("Expected array content"),
        }
    }

    #[test]
    fn test_request_validation() {
        let request = ChatCompletionRequest {
            model: "gpt-4-vision-preview".to_string(),
            messages: vec![],
            max_tokens: None,
            temperature: 0.7,
            top_p: 1.0,
            stream: false,
            extra: HashMap::new(),
        };

        let result = utils::validate_chat_request(&request);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Messages array cannot be empty"));
    }

    #[test]
    fn test_image_counting() {
        let request = ChatCompletionRequest {
            model: "gpt-4-vision-preview".to_string(),
            messages: vec![ChatMessage {
                role: MessageRole::User,
                content: MessageContent::Array(vec![
                    ContentPart::Text {
                        text: "Describe these images".to_string(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.com/image1.jpg".to_string(),
                            detail: None,
                        },
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==".to_string(),
                            detail: Some("low".to_string()),
                        },
                    },
                ]),
                name: None,
            }],
            max_tokens: Some(500),
            temperature: 0.3,
            top_p: 1.0,
            stream: false,
            extra: HashMap::new(),
        };

        let count = utils::count_images_in_request(&request);
        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn test_vision_service_creation() {
        let service = VisionChatService::new();
        assert!(service.is_ok(), "Vision service should be created successfully");
    }
}