//! Chat Template Support
//!
//! Renders chat messages into model-specific prompt formats.
//! Supports common templates: LLaMA instruct, ChatML, Llama3.

/// A chat message with role and content
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Known chat template formats
#[derive(Debug, Clone)]
pub enum ChatTemplateFormat {
    /// LLaMA 2 instruct format: [INST] ... [/INST]
    Llama2,
    /// LLaMA 3 instruct format: <|start_header_id|>...<|end_header_id|>
    Llama3,
    /// ChatML format: <|im_start|>role\ncontent<|im_end|>
    ChatML,
    /// Raw format: just concatenate messages
    Raw,
}

impl ChatTemplateFormat {
    /// Detect template format from model architecture or metadata
    pub fn detect(architecture: &str, model_name: &str) -> Self {
        let name_lower = model_name.to_lowercase();
        if name_lower.contains("llama-3") || name_lower.contains("llama3") {
            ChatTemplateFormat::Llama3
        } else if name_lower.contains("qwen") || name_lower.contains("mistral") || name_lower.contains("chatml") {
            ChatTemplateFormat::ChatML
        } else if architecture == "llama" {
            ChatTemplateFormat::Llama2
        } else {
            ChatTemplateFormat::ChatML
        }
    }

    /// Render messages into a prompt string
    pub fn render(&self, messages: &[ChatMessage]) -> String {
        match self {
            ChatTemplateFormat::Llama2 => render_llama2(messages),
            ChatTemplateFormat::Llama3 => render_llama3(messages),
            ChatTemplateFormat::ChatML => render_chatml(messages),
            ChatTemplateFormat::Raw => render_raw(messages),
        }
    }
}

fn render_llama2(messages: &[ChatMessage]) -> String {
    let mut result = String::new();
    let mut system_prompt = String::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                system_prompt = msg.content.clone();
            }
            "user" => {
                result.push_str("<s>[INST] ");
                if !system_prompt.is_empty() {
                    result.push_str(&format!("<<SYS>>\n{}\n<</SYS>>\n\n", system_prompt));
                    system_prompt.clear();
                }
                result.push_str(&msg.content);
                result.push_str(" [/INST]");
            }
            "assistant" => {
                result.push(' ');
                result.push_str(&msg.content);
                result.push_str(" </s>");
            }
            _ => {}
        }
    }
    result
}

fn render_llama3(messages: &[ChatMessage]) -> String {
    let mut result = String::from("<|begin_of_text|>");

    for msg in messages {
        result.push_str(&format!(
            "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
            msg.role, msg.content
        ));
    }
    // Add assistant header for generation
    result.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    result
}

fn render_chatml(messages: &[ChatMessage]) -> String {
    let mut result = String::new();

    for msg in messages {
        result.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            msg.role, msg.content
        ));
    }
    result.push_str("<|im_start|>assistant\n");
    result
}

fn render_raw(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chatml_render() {
        let messages = vec![
            ChatMessage { role: "system".into(), content: "You are helpful.".into() },
            ChatMessage { role: "user".into(), content: "Hello".into() },
        ];
        let result = ChatTemplateFormat::ChatML.render(&messages);
        assert!(result.contains("<|im_start|>system"));
        assert!(result.contains("<|im_start|>user"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_llama3_render() {
        let messages = vec![
            ChatMessage { role: "user".into(), content: "Hello".into() },
        ];
        let result = ChatTemplateFormat::Llama3.render(&messages);
        assert!(result.contains("<|begin_of_text|>"));
        assert!(result.contains("<|start_header_id|>user"));
        assert!(result.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_detect_template() {
        assert!(matches!(
            ChatTemplateFormat::detect("llama", "llama-3-8b"),
            ChatTemplateFormat::Llama3
        ));
        assert!(matches!(
            ChatTemplateFormat::detect("qwen2", "qwen2-7b"),
            ChatTemplateFormat::ChatML
        ));
        assert!(matches!(
            ChatTemplateFormat::detect("llama", "tinyllama"),
            ChatTemplateFormat::Llama2
        ));
    }
}
