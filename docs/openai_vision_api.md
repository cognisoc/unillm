# OpenAI Vision API Compatibility

UniLLM provides **full compatibility** with OpenAI's GPT-4 Vision API, allowing you to use it as a drop-in replacement for OpenAI's vision-language models.

## 🎯 Key Features

- ✅ **100% OpenAI API Compatible** - Same request/response format
- ✅ **Base64 Image Support** - Send images directly in requests
- ✅ **URL Image Support** - Process images from web URLs
- ✅ **Multimodal Conversations** - Mix text and images naturally
- ✅ **Batch Image Processing** - Multiple images per request
- ✅ **Streaming Support** - Real-time response streaming
- ✅ **Production Ready** - Enterprise-grade performance

## 🚀 Quick Start

### 1. Start the Vision API Server

```bash
cargo run --bin openai_vision_server
```

The server will start on `http://localhost:8000` with the endpoint `/v1/chat/completions`.

### 2. Make Your First Vision API Call

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4-vision-preview",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What'\''s in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
          }
        }
      ]
    }],
    "max_tokens": 300
  }'
```

## 📡 API Reference

### Chat Completions Endpoint

**POST** `/v1/chat/completions`

### Request Format

```json
{
  "model": "gpt-4-vision-preview",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What's in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,<base64_image>",
            "detail": "high"
          }
        }
      ]
    }
  ],
  "max_tokens": 300,
  "temperature": 0.7,
  "top_p": 1.0,
  "stream": false
}
```

### Response Format

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-4-vision-preview",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The image shows a beautiful landscape with mountains in the background..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 85,
    "completion_tokens": 45,
    "total_tokens": 130
  }
}
```

## 🖼️ Image Input Formats

### Base64 Encoded Images

```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...",
    "detail": "high"
  }
}
```

**Supported formats:** JPEG, PNG, WebP, TIFF

### URL Images

```json
{
  "type": "image_url",
  "image_url": {
    "url": "https://example.com/image.jpg",
    "detail": "high"
  }
}
```

### Detail Levels

- `"high"` - High resolution analysis (default)
- `"low"` - Lower resolution, faster processing
- `"auto"` - Automatic selection based on image

## 🔧 Integration Examples

### Python with OpenAI Client

```python
import openai

# Configure to use UniLLM server
openai.api_base = "http://localhost:8000/v1"
openai.api_key = "dummy-key"  # Not required for UniLLM

response = openai.ChatCompletion.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,..."
                    }
                }
            ]
        }
    ],
    max_tokens=300
)

print(response.choices[0].message.content)
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

const response = await axios.post('http://localhost:8000/v1/chat/completions', {
  model: 'gpt-4-vision-preview',
  messages: [{
    role: 'user',
    content: [
      { type: 'text', text: "Describe this image" },
      {
        type: 'image_url',
        image_url: {
          url: 'data:image/jpeg;base64,...',
          detail: 'high'
        }
      }
    ]
  }],
  max_tokens: 300
}, {
  headers: { 'Content-Type': 'application/json' }
});

console.log(response.data.choices[0].message.content);
```

### cURL

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4-vision-preview",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What do you see?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
      ]
    }]
  }'
```

## 📊 Supported Models

| Model ID | Description | Vision Support |
|----------|-------------|----------------|
| `gpt-4-vision-preview` | GPT-4V compatible mode | ✅ Full |
| `gpt-4v` | Alias for vision mode | ✅ Full |
| `llava-1.5` | LLaVA 1.5 architecture | ✅ Full |
| `unillm-vision` | UniLLM native vision | ✅ Full |

## 🔒 Authentication

UniLLM currently runs without authentication for development. For production:

```json
{
  "headers": {
    "Authorization": "Bearer your-api-key",
    "Content-Type": "application/json"
  }
}
```

## ⚡ Performance Tips

1. **Image Optimization**
   - Use JPEG for photos (smaller size)
   - Use PNG for text/diagrams (better quality)
   - Resize large images to 1024x1024 max

2. **Batch Processing**
   - Send multiple images in one request
   - Use lower detail for faster processing
   - Cache processed results

3. **Memory Management**
   - Base64 images use ~33% more memory
   - Use URLs when possible for large images
   - Process images sequentially for memory-constrained environments

## 🚨 Error Handling

### Common Error Codes

```json
{
  "error": {
    "message": "Invalid image format. Supported: JPEG, PNG, WebP, TIFF",
    "type": "invalid_request_error",
    "code": "invalid_image_format"
  }
}
```

### Status Codes

- `200` - Success
- `400` - Bad Request (invalid format)
- `413` - Image too large
- `429` - Rate limit exceeded
- `500` - Internal server error

## 🔄 Migration from OpenAI

### 1. Update API Base URL

```python
# Before (OpenAI)
openai.api_base = "https://api.openai.com/v1"

# After (UniLLM)
openai.api_base = "http://localhost:8000/v1"
```

### 2. No Code Changes Required

All existing OpenAI GPT-4V code works without modification!

### 3. Model Compatibility

```python
# These all work with UniLLM
models = [
    "gpt-4-vision-preview",
    "gpt-4v",
    "llava-1.5",
    "unillm-vision"
]
```

## 🧪 Testing

Run the included test suite:

```bash
# Start server
cargo run --bin openai_vision_server

# Run Python demo
python examples/vision_api_demo.py

# Run integration tests
cargo test openai_vision_api
```

## 📈 Monitoring & Metrics

UniLLM provides built-in metrics:

```bash
curl http://localhost:8000/metrics
```

Key metrics:
- Request count and latency
- Image processing time
- Memory usage
- Error rates
- Model performance

## 🆚 OpenAI Comparison

| Feature | OpenAI GPT-4V | UniLLM Vision |
|---------|---------------|---------------|
| API Compatibility | ✅ Reference | ✅ 100% Compatible |
| Image Formats | JPEG, PNG, WebP | JPEG, PNG, WebP, TIFF |
| Max Image Size | 20MB | Configurable |
| Batch Images | 1-4 per request | Unlimited |
| Streaming | ✅ | ✅ |
| Self-hosted | ❌ | ✅ |
| Cost | $0.01/1K tokens | Free |

## 🎯 Use Cases

### 1. **Document Analysis**
```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Extract all text from this document and format it as JSON"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]
}
```

### 2. **Medical Imaging**
```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Analyze this X-ray and identify any abnormalities"},
    {"type": "image_url", "image_url": {"url": "https://hospital.com/xray.jpg"}}
  ]
}
```

### 3. **E-commerce**
```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Generate a product description for this item"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
  ]
}
```

### 4. **Content Moderation**
```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Check if this image contains inappropriate content"},
    {"type": "image_url", "image_url": {"url": "https://cdn.example.com/user-upload.jpg"}}
  ]
}
```

## 🛠️ Advanced Configuration

### Custom Vision Models

```rust
use runtime::openai_vision_api::*;
use runtime::models::llava::LLaVAConfig;

let config = LLaVAConfig {
    language_config: ModelConfig { /* custom config */ },
    vision_config: VisionConfig {
        image_size: 336,  // Higher resolution
        patch_size: 14,   // Smaller patches
        hidden_size: 1024,
        ..Default::default()
    },
    projection_dim: 4096,
    freeze_vision_encoder: false,  // Fine-tuning enabled
    freeze_language_model: false,
};

let service = VisionChatService::with_config(config)?;
```

### Performance Tuning

```toml
[runtime.vision]
image_cache_size = "1GB"
max_batch_size = 8
processing_timeout = 30
enable_gpu_acceleration = true
```

## 🤝 Contributing

We welcome contributions! Areas of focus:

1. **Model Architecture Improvements**
2. **Performance Optimizations**
3. **Additional Image Formats**
4. **Streaming Enhancements**
5. **Documentation & Examples**

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/your-org/unillm/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-org/unillm/discussions)
- 📧 **Email**: support@unillm.ai
- 🔗 **Discord**: [UniLLM Community](https://discord.gg/unillm)

---

**🎉 You now have a fully functional OpenAI GPT-4 Vision API replacement!**

The UniLLM Vision API provides enterprise-grade multimodal capabilities with complete OpenAI compatibility, giving you the freedom to self-host while maintaining seamless integration with existing applications.