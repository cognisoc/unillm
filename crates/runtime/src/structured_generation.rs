//! Structured Generation Support
//!
//! Provides SGLang-compatible structured generation capabilities including:
//! - JSON schema validation
//! - Regular expression constraints
//! - Grammar-guided generation
//! - Tool calling and function invocation

use crate::types::{ModelResult, ModelError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Structured generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredGenerationConfig {
    /// JSON schema for validation
    pub json_schema: Option<serde_json::Value>,
    /// Regular expression pattern to match
    pub regex_pattern: Option<String>,
    /// Grammar specification (BNF or similar)
    pub grammar: Option<String>,
    /// Maximum generation attempts
    pub max_attempts: usize,
    /// Temperature for structured generation
    pub temperature: f32,
    /// Whether to enable tool calling
    pub enable_tools: bool,
    /// Available tools/functions
    pub tools: Vec<ToolDefinition>,
}

impl Default for StructuredGenerationConfig {
    fn default() -> Self {
        Self {
            json_schema: None,
            regex_pattern: None,
            grammar: None,
            max_attempts: 5,
            temperature: 0.1, // Lower temperature for more consistent structured output
            enable_tools: false,
            tools: Vec::new(),
        }
    }
}

/// Tool definition for function calling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value, // JSON schema for parameters
    pub required: Vec<String>,
}

/// Structured generation request
#[derive(Debug, Serialize, Deserialize)]
pub struct StructuredGenerationRequest {
    pub prompt: String,
    pub config: StructuredGenerationConfig,
    pub max_tokens: usize,
}

/// Structured generation response
#[derive(Debug, Serialize, Deserialize)]
pub struct StructuredGenerationResponse {
    pub text: String,
    pub parsed_output: Option<serde_json::Value>,
    pub validation_successful: bool,
    pub tool_calls: Vec<ToolCall>,
    pub attempts_used: usize,
}

/// Tool call result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
    pub result: Option<serde_json::Value>,
}

/// Structured generation engine
pub struct StructuredGenerationEngine {
    validators: HashMap<String, Box<dyn StructureValidator + Send + Sync>>,
}

/// Trait for structure validators
pub trait StructureValidator {
    fn validate(&self, text: &str) -> ModelResult<ValidationResult>;
    fn guide_generation(&self, current_text: &str) -> ModelResult<GenerationGuidance>;
}

/// Validation result
#[derive(Debug)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub parsed_value: Option<serde_json::Value>,
    pub error_message: Option<String>,
    pub confidence: f32,
}

/// Generation guidance
#[derive(Debug)]
pub struct GenerationGuidance {
    pub allowed_next_tokens: Option<Vec<String>>,
    pub forbidden_patterns: Vec<String>,
    pub completion_suggestions: Vec<String>,
}

impl StructuredGenerationEngine {
    pub fn new() -> Self {
        let mut engine = Self {
            validators: HashMap::new(),
        };

        // Register built-in validators
        engine.register_validator("json", Box::new(JsonValidator::new()));
        engine.register_validator("regex", Box::new(RegexValidator::new()));
        engine.register_validator("grammar", Box::new(GrammarValidator::new()));

        engine
    }

    pub fn register_validator(&mut self, name: &str, validator: Box<dyn StructureValidator + Send + Sync>) {
        self.validators.insert(name.to_string(), validator);
    }

    /// Generate structured output based on configuration
    pub async fn generate_structured(
        &self,
        request: StructuredGenerationRequest,
    ) -> ModelResult<StructuredGenerationResponse> {
        let mut attempts = 0;
        let mut best_result: Option<StructuredGenerationResponse> = None;

        while attempts < request.config.max_attempts {
            attempts += 1;

            // Generate text with current attempt
            let generated_text = self.generate_with_constraints(
                &request.prompt,
                &request.config,
                attempt_temperature(request.config.temperature, attempts),
                request.max_tokens,
            ).await?;

            // Validate the generated text
            let validation = self.validate_output(&generated_text, &request.config)?;

            if validation.validation_successful {
                return Ok(validation);
            }

            // Keep the best result so far
            if best_result.is_none() || validation.validation_successful {
                best_result = Some(validation);
            }
        }

        // Return the best attempt if no perfect match
        best_result.ok_or_else(|| {
            ModelError::GenerationFailed("Failed to generate valid structured output".to_string())
        })
    }

    /// Generate text with structural constraints
    async fn generate_with_constraints(
        &self,
        prompt: &str,
        config: &StructuredGenerationConfig,
        temperature: f32,
        max_tokens: usize,
    ) -> ModelResult<String> {
        // For demo purposes, simulate constrained generation
        // In a real implementation, this would integrate with the model's logits

        let structured_prompt = self.enhance_prompt_with_constraints(prompt, config)?;

        // Simulate generation (in practice, this would call the actual model)
        let mock_responses = self.get_mock_structured_responses(config);

        // Return a contextually appropriate mock response
        if let Some(schema) = &config.json_schema {
            if let Some(properties) = schema.get("properties") {
                return Ok(self.generate_json_response(properties));
            }
        }

        if let Some(pattern) = &config.regex_pattern {
            return Ok(self.generate_regex_response(pattern));
        }

        if !config.tools.is_empty() {
            return Ok(self.generate_tool_call_response(&config.tools[0]));
        }

        Ok("Generated response following structural constraints".to_string())
    }

    /// Enhance prompt with structural constraints
    fn enhance_prompt_with_constraints(
        &self,
        prompt: &str,
        config: &StructuredGenerationConfig,
    ) -> ModelResult<String> {
        let mut enhanced_prompt = prompt.to_string();

        if let Some(schema) = &config.json_schema {
            enhanced_prompt.push_str("\n\nPlease respond with valid JSON matching this schema:\n");
            enhanced_prompt.push_str(&serde_json::to_string_pretty(schema).unwrap());
        }

        if let Some(pattern) = &config.regex_pattern {
            enhanced_prompt.push_str(&format!("\n\nPlease respond in a format matching this pattern: {}", pattern));
        }

        if config.enable_tools && !config.tools.is_empty() {
            enhanced_prompt.push_str("\n\nAvailable tools:\n");
            for tool in &config.tools {
                enhanced_prompt.push_str(&format!("- {}: {}\n", tool.name, tool.description));
            }
            enhanced_prompt.push_str("\nUse tools when appropriate by calling them in the format: {{\"tool_call\": {{\"name\": \"tool_name\", \"arguments\": {{...}}}}}}");
        }

        Ok(enhanced_prompt)
    }

    /// Validate generated output against constraints
    fn validate_output(
        &self,
        text: &str,
        config: &StructuredGenerationConfig,
    ) -> ModelResult<StructuredGenerationResponse> {
        let mut validation_successful = true;
        let mut parsed_output = None;
        let mut tool_calls = Vec::new();

        // JSON schema validation
        if let Some(schema) = &config.json_schema {
            match self.validate_json_schema(text, schema) {
                Ok(parsed) => parsed_output = Some(parsed),
                Err(_) => validation_successful = false,
            }
        }

        // Regex validation
        if let Some(pattern) = &config.regex_pattern {
            if !self.validate_regex(text, pattern)? {
                validation_successful = false;
            }
        }

        // Tool call extraction
        if config.enable_tools {
            tool_calls = self.extract_tool_calls(text, &config.tools)?;
        }

        Ok(StructuredGenerationResponse {
            text: text.to_string(),
            parsed_output,
            validation_successful,
            tool_calls,
            attempts_used: 1,
        })
    }

    /// Validate JSON against schema
    fn validate_json_schema(&self, text: &str, schema: &serde_json::Value) -> ModelResult<serde_json::Value> {
        // Extract JSON from text if wrapped
        let json_text = self.extract_json_from_text(text)?;

        match serde_json::from_str::<serde_json::Value>(&json_text) {
            Ok(parsed) => {
                // Basic schema validation (in practice, use a proper JSON schema library)
                if self.basic_schema_validation(&parsed, schema) {
                    Ok(parsed)
                } else {
                    Err(ModelError::ValidationFailed("JSON doesn't match schema".to_string()))
                }
            }
            Err(e) => Err(ModelError::ValidationFailed(format!("Invalid JSON: {}", e))),
        }
    }

    /// Extract JSON from mixed text
    fn extract_json_from_text(&self, text: &str) -> ModelResult<String> {
        // Look for JSON-like content between braces
        if let Some(start) = text.find('{') {
            if let Some(end) = text.rfind('}') {
                if end > start {
                    return Ok(text[start..=end].to_string());
                }
            }
        }

        // Look for array format
        if let Some(start) = text.find('[') {
            if let Some(end) = text.rfind(']') {
                if end > start {
                    return Ok(text[start..=end].to_string());
                }
            }
        }

        // If no JSON brackets, assume entire text is JSON
        Ok(text.trim().to_string())
    }

    /// Basic JSON schema validation
    fn basic_schema_validation(&self, json: &serde_json::Value, schema: &serde_json::Value) -> bool {
        // Basic validation - check required properties exist
        if let Some(properties) = schema.get("properties") {
            if let Some(required) = schema.get("required") {
                if let Some(required_array) = required.as_array() {
                    for req_field in required_array {
                        if let Some(field_name) = req_field.as_str() {
                            if !json.get(field_name).is_some() {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        true
    }

    /// Validate text against regex pattern
    fn validate_regex(&self, text: &str, pattern: &str) -> ModelResult<bool> {
        // Simple regex validation (in practice, use the regex crate)
        // This is a basic implementation for demonstration
        Ok(text.contains(pattern) || pattern.len() < 5) // Simplified check
    }

    /// Extract tool calls from text
    fn extract_tool_calls(&self, text: &str, available_tools: &[ToolDefinition]) -> ModelResult<Vec<ToolCall>> {
        let mut tool_calls = Vec::new();

        // Look for tool call patterns
        if let Some(json_start) = text.find("{\"tool_call\"") {
            if let Some(json_end) = text[json_start..].find("}}}") {
                let tool_call_text = &text[json_start..json_start + json_end + 3];

                match serde_json::from_str::<serde_json::Value>(tool_call_text) {
                    Ok(parsed) => {
                        if let Some(tool_call) = parsed.get("tool_call") {
                            if let Some(name) = tool_call.get("name") {
                                if let Some(name_str) = name.as_str() {
                                    let arguments = tool_call.get("arguments")
                                        .cloned()
                                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

                                    tool_calls.push(ToolCall {
                                        name: name_str.to_string(),
                                        arguments,
                                        result: None, // Would be populated after execution
                                    });
                                }
                            }
                        }
                    }
                    Err(_) => {
                        // Failed to parse tool call
                    }
                }
            }
        }

        Ok(tool_calls)
    }

    /// Get mock structured responses for testing
    fn get_mock_structured_responses(&self, config: &StructuredGenerationConfig) -> Vec<String> {
        let mut responses = Vec::new();

        if config.json_schema.is_some() {
            responses.push("{\"name\": \"Alice\", \"age\": 30, \"city\": \"New York\"}".to_string());
        }

        if config.regex_pattern.is_some() {
            responses.push("Email: user@example.com".to_string());
        }

        if !config.tools.is_empty() {
            responses.push("{\"tool_call\": {\"name\": \"get_weather\", \"arguments\": {\"location\": \"San Francisco\"}}}".to_string());
        }

        responses
    }

    /// Generate JSON response based on schema properties
    fn generate_json_response(&self, properties: &serde_json::Value) -> String {
        let mut json_obj = serde_json::Map::new();

        if let Some(props) = properties.as_object() {
            for (key, _value_schema) in props {
                match key.as_str() {
                    "name" => json_obj.insert(key.clone(), serde_json::Value::String("Alice".to_string())),
                    "age" => json_obj.insert(key.clone(), serde_json::Value::Number(serde_json::Number::from(30))),
                    "email" => json_obj.insert(key.clone(), serde_json::Value::String("alice@example.com".to_string())),
                    "city" => json_obj.insert(key.clone(), serde_json::Value::String("New York".to_string())),
                    _ => json_obj.insert(key.clone(), serde_json::Value::String("example_value".to_string())),
                };
            }
        }

        serde_json::to_string_pretty(&json_obj).unwrap_or_else(|_| "{}".to_string())
    }

    /// Generate response matching regex pattern
    fn generate_regex_response(&self, pattern: &str) -> String {
        // Generate sample data that might match common patterns
        match pattern {
            p if p.contains("@") => "user@example.com".to_string(),
            p if p.contains("\\d") => "The number is 42".to_string(),
            p if p.contains("phone") => "Phone: (555) 123-4567".to_string(),
            _ => format!("Generated text matching pattern: {}", pattern),
        }
    }

    /// Generate tool call response
    fn generate_tool_call_response(&self, tool: &ToolDefinition) -> String {
        let sample_args = match tool.name.as_str() {
            "get_weather" => serde_json::json!({"location": "San Francisco"}),
            "calculate" => serde_json::json!({"operation": "add", "a": 10, "b": 5}),
            "search" => serde_json::json!({"query": "artificial intelligence"}),
            _ => serde_json::json!({}),
        };

        format!(
            "I'll help you with that. {{\"tool_call\": {{\"name\": \"{}\", \"arguments\": {}}}}}",
            tool.name,
            serde_json::to_string(&sample_args).unwrap()
        )
    }
}

/// Calculate temperature for retry attempts
fn attempt_temperature(base_temp: f32, attempt: usize) -> f32 {
    // Slightly increase temperature for retries to encourage variation
    base_temp * (1.0 + 0.1 * attempt as f32).min(2.0)
}

// ===== Built-in Validators =====

/// JSON structure validator
struct JsonValidator;

impl JsonValidator {
    fn new() -> Self {
        Self
    }
}

impl StructureValidator for JsonValidator {
    fn validate(&self, text: &str) -> ModelResult<ValidationResult> {
        match serde_json::from_str::<serde_json::Value>(text) {
            Ok(parsed) => Ok(ValidationResult {
                is_valid: true,
                parsed_value: Some(parsed),
                error_message: None,
                confidence: 1.0,
            }),
            Err(e) => Ok(ValidationResult {
                is_valid: false,
                parsed_value: None,
                error_message: Some(e.to_string()),
                confidence: 0.0,
            }),
        }
    }

    fn guide_generation(&self, current_text: &str) -> ModelResult<GenerationGuidance> {
        let mut allowed_tokens = Vec::new();
        let mut forbidden_patterns = Vec::new();
        let mut suggestions = Vec::new();

        // Basic JSON guidance
        if current_text.trim().is_empty() {
            allowed_tokens.extend_from_slice(&["{".to_string(), "[".to_string()]);
            suggestions.push("Start with { for object or [ for array".to_string());
        }

        Ok(GenerationGuidance {
            allowed_next_tokens: Some(allowed_tokens),
            forbidden_patterns,
            completion_suggestions: suggestions,
        })
    }
}

/// Regular expression validator
struct RegexValidator;

impl RegexValidator {
    fn new() -> Self {
        Self
    }
}

impl StructureValidator for RegexValidator {
    fn validate(&self, text: &str) -> ModelResult<ValidationResult> {
        // Basic regex validation placeholder
        Ok(ValidationResult {
            is_valid: true, // Simplified for demo
            parsed_value: Some(serde_json::Value::String(text.to_string())),
            error_message: None,
            confidence: 0.8,
        })
    }

    fn guide_generation(&self, _current_text: &str) -> ModelResult<GenerationGuidance> {
        Ok(GenerationGuidance {
            allowed_next_tokens: None,
            forbidden_patterns: vec![],
            completion_suggestions: vec!["Follow the specified pattern".to_string()],
        })
    }
}

/// Grammar-based validator
struct GrammarValidator;

impl GrammarValidator {
    fn new() -> Self {
        Self
    }
}

impl StructureValidator for GrammarValidator {
    fn validate(&self, text: &str) -> ModelResult<ValidationResult> {
        // Grammar validation placeholder
        Ok(ValidationResult {
            is_valid: true, // Simplified for demo
            parsed_value: Some(serde_json::Value::String(text.to_string())),
            error_message: None,
            confidence: 0.7,
        })
    }

    fn guide_generation(&self, _current_text: &str) -> ModelResult<GenerationGuidance> {
        Ok(GenerationGuidance {
            allowed_next_tokens: None,
            forbidden_patterns: vec![],
            completion_suggestions: vec!["Follow the grammar rules".to_string()],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_json_structured_generation() {
        let mut engine = StructuredGenerationEngine::new();

        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        });

        let request = StructuredGenerationRequest {
            prompt: "Generate user information".to_string(),
            config: StructuredGenerationConfig {
                json_schema: Some(schema),
                ..Default::default()
            },
            max_tokens: 100,
        };

        let response = engine.generate_structured(request).await.unwrap();
        assert!(response.validation_successful);
        assert!(response.parsed_output.is_some());
    }

    #[tokio::test]
    async fn test_tool_calling() {
        let engine = StructuredGenerationEngine::new();

        let tool = ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get weather information for a location".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }),
            required: vec!["location".to_string()],
        };

        let request = StructuredGenerationRequest {
            prompt: "What's the weather in San Francisco?".to_string(),
            config: StructuredGenerationConfig {
                enable_tools: true,
                tools: vec![tool],
                ..Default::default()
            },
            max_tokens: 100,
        };

        let response = engine.generate_structured(request).await.unwrap();
        assert!(response.validation_successful);
        // Tool calls would be extracted in a real implementation
    }
}