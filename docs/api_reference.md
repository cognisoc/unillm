# 📡 UniLLM API Reference

Complete API documentation for UniLLM inference engine with examples in multiple programming languages.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Base URLs](#base-urls)
4. [Core Endpoints](#core-endpoints)
5. [Request/Response Formats](#requestresponse-formats)
6. [Error Handling](#error-handling)
7. [Rate Limiting](#rate-limiting)
8. [Client Libraries](#client-libraries)
9. [WebSocket Streaming](#websocket-streaming)
10. [Examples](#examples)

## 🌐 Overview

UniLLM provides a REST API for high-performance LLM inference with support for both traditional and streaming responses. The API is designed to be compatible with OpenAI's format while providing additional UniLLM-specific features and optimizations.

### Key Features
- **OpenAI Compatibility**: Drop-in replacement for OpenAI API
- **Streaming Support**: Real-time token generation
- **Batch Processing**: Multiple requests in single call
- **Cache Analytics**: Detailed cache hit/miss statistics
- **Performance Metrics**: Real-time performance monitoring
- **Runtime Detection**: Container vs unikernel mode information

## 🔐 Authentication

UniLLM supports multiple authentication methods:

### API Key Authentication
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     http://localhost:8080/v1/generate
```

### Basic Authentication
```bash
curl -u username:password \
     -H "Content-Type: application/json" \
     http://localhost:8080/v1/generate
```

### No Authentication (Development)
```bash
curl -H "Content-Type: application/json" \
     http://localhost:8080/v1/generate
```

## 🌍 Base URLs

### Local Development
```
http://localhost:8080
```

### Production Deployment
```
https://api.yourdomain.com
```

### Container Mode
```
http://container-host:8080
```

### Unikernel Mode
```
http://unikernel-ip:8080
```

## 🔧 Core Endpoints

### Health Check

**GET** `/health`

Check server health and runtime information.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "gpu_target": "cuda",
  "runtime_mode": "container",
  "memory_usage_mb": 1024.5,
  "gpu_memory_usage_mb": 2048.0,
  "uptime_seconds": 3600,
  "total_requests": 1500
}
```

**Example:**
```bash
curl http://localhost:8080/health
```

### Text Generation

**POST** `/v1/generate`

Generate text completion for a given prompt.

**Request Body:**
```json
{
  "prompt": "The future of AI is",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "stop": ["\n", "END"],
  "stream": false,
  "echo": false
}
```

**Response:**
```json
{
  "id": "gen-abc123def456",
  "object": "text_completion",
  "created": 1699564800,
  "model": "unillm",
  "choices": [
    {
      "text": "bright and full of possibilities. With advances in machine learning...",
      "index": 0,
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 87,
    "total_tokens": 92
  },
  "unillm_stats": {
    "inference_time_ms": 245,
    "cache_hits": 15,
    "gpu_utilization": 0.85,
    "memory_efficiency": 0.78
  }
}
```

### OpenAI Compatible Completions

**POST** `/v1/completions`

OpenAI-compatible text completion endpoint.

**Request Body:**
```json
{
  "model": "unillm",
  "prompt": "Once upon a time",
  "max_tokens": 50,
  "temperature": 0.8,
  "top_p": 1.0,
  "n": 1,
  "stream": false,
  "logprobs": null,
  "echo": false,
  "stop": null,
  "presence_penalty": 0,
  "frequency_penalty": 0,
  "best_of": 1,
  "logit_bias": {},
  "user": "user123"
}
```

### Chat Completions

**POST** `/v1/chat/completions`

Chat-based completions with conversation history.

**Request Body:**
```json
{
  "model": "unillm",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello! How can you help me today?"
    }
  ],
  "max_tokens": 150,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1699564800,
  "model": "unillm",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm here to help you with a wide variety of tasks..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 68,
    "total_tokens": 93
  }
}
```

### Batch Processing

**POST** `/v1/batch`

Process multiple requests in a single call for improved throughput.

**Request Body:**
```json
{
  "requests": [
    {
      "id": "req-1",
      "prompt": "Translate to French: Hello",
      "max_tokens": 50
    },
    {
      "id": "req-2",
      "prompt": "Summarize: AI is transforming...",
      "max_tokens": 100
    }
  ],
  "batch_size": 32,
  "parallel": true
}
```

**Response:**
```json
{
  "id": "batch-abc123",
  "object": "batch_completion",
  "responses": [
    {
      "id": "req-1",
      "generated_text": "Bonjour",
      "tokens_generated": 2,
      "status": "completed"
    },
    {
      "id": "req-2",
      "generated_text": "AI is revolutionizing industries...",
      "tokens_generated": 45,
      "status": "completed"
    }
  ],
  "batch_stats": {
    "total_requests": 2,
    "successful": 2,
    "failed": 0,
    "total_time_ms": 180,
    "throughput_rps": 11.1
  }
}
```

### Statistics

**GET** `/stats`

Get detailed server statistics and performance metrics.

**Response:**
```json
{
  "server_info": {
    "version": "0.1.0",
    "runtime_mode": "unikernel",
    "gpu_target": "cuda",
    "uptime_seconds": 7200
  },
  "performance": {
    "total_requests": 5000,
    "total_tokens_generated": 500000,
    "average_latency_ms": 185.5,
    "throughput_rps": 27.8,
    "p50_latency_ms": 160.0,
    "p95_latency_ms": 320.0,
    "p99_latency_ms": 480.0
  },
  "cache_stats": {
    "hit_rate": 0.78,
    "miss_rate": 0.22,
    "eviction_rate": 0.05,
    "memory_usage_mb": 2048.0,
    "l1_hit_rate": 0.45,
    "l2_hit_rate": 0.33,
    "l3_hit_rate": 0.22
  },
  "gpu_stats": {
    "utilization": 0.85,
    "memory_usage_mb": 18432.0,
    "memory_total_mb": 24576.0,
    "temperature_celsius": 72,
    "power_usage_watts": 350
  },
  "memory_stats": {
    "total_memory_mb": 16384.0,
    "used_memory_mb": 8192.0,
    "cache_memory_mb": 2048.0,
    "available_memory_mb": 6144.0
  }
}
```

### Models

**GET** `/v1/models`

List available models and their capabilities.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "unillm",
      "object": "model",
      "created": 1699564800,
      "owned_by": "unillm",
      "capabilities": [
        "text-generation",
        "chat",
        "streaming",
        "batch-processing"
      ],
      "context_length": 8192,
      "supported_features": [
        "hybrid-cache",
        "gpu-optimization",
        "unikernel-mode"
      ]
    }
  ]
}
```

## 📝 Request/Response Formats

### Standard Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `prompt` | string | Input text prompt | Required |
| `max_tokens` | integer | Maximum tokens to generate | 100 |
| `temperature` | float | Sampling temperature (0.0-2.0) | 1.0 |
| `top_p` | float | Nucleus sampling threshold | 1.0 |
| `top_k` | integer | Top-k sampling limit | 50 |
| `stop` | array | Stop sequences | [] |
| `stream` | boolean | Enable streaming response | false |
| `echo` | boolean | Include prompt in response | false |

### UniLLM-Specific Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `cache_policy` | string | Cache behavior (auto/force/disable) | "auto" |
| `batch_priority` | string | Batch priority (low/normal/high) | "normal" |
| `gpu_preference` | string | GPU allocation preference | "auto" |
| `precision` | string | Inference precision (fp16/fp32/int8) | "fp16" |

### Response Headers

```http
HTTP/1.1 200 OK
Content-Type: application/json
X-UniLLM-Version: 0.1.0
X-UniLLM-Runtime: unikernel
X-UniLLM-GPU: cuda
X-UniLLM-Cache-Hit-Rate: 0.78
X-UniLLM-Inference-Time-Ms: 245
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1699564860
```

## ❌ Error Handling

### Error Response Format

```json
{
  "error": {
    "type": "invalid_request_error",
    "code": "invalid_parameter",
    "message": "Invalid value for 'temperature': must be between 0.0 and 2.0",
    "param": "temperature",
    "request_id": "req-abc123def456"
  },
  "unillm_debug": {
    "timestamp": "2023-11-09T12:00:00Z",
    "runtime_mode": "container",
    "gpu_status": "healthy",
    "memory_available_mb": 1024
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `invalid_request_error` | 400 | Invalid request format or parameters |
| `authentication_error` | 401 | Invalid or missing authentication |
| `permission_error` | 403 | Insufficient permissions |
| `not_found_error` | 404 | Endpoint or resource not found |
| `rate_limit_error` | 429 | Rate limit exceeded |
| `server_error` | 500 | Internal server error |
| `gpu_error` | 503 | GPU unavailable or error |
| `memory_error` | 503 | Insufficient memory |

### Retry Logic

```python
import time
import requests

def make_request_with_retry(url, data, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=data)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limited, wait and retry
                retry_after = int(response.headers.get('Retry-After', 60))
                time.sleep(retry_after)
                continue
            elif response.status_code >= 500:
                # Server error, retry with exponential backoff
                time.sleep(2 ** attempt)
                continue
            else:
                # Client error, don't retry
                response.raise_for_status()

        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)

    raise Exception("Max retries exceeded")
```

## 🚦 Rate Limiting

UniLLM implements adaptive rate limiting based on resource usage:

### Rate Limit Headers

```http
X-RateLimit-Limit: 1000          # Requests per hour
X-RateLimit-Remaining: 999       # Remaining requests
X-RateLimit-Reset: 1699564860    # Reset timestamp
X-RateLimit-Type: adaptive       # Rate limit type
```

### Rate Limit Tiers

| Tier | RPM | Tokens/Min | Concurrent | Price |
|------|-----|------------|------------|-------|
| Free | 100 | 10,000 | 3 | $0 |
| Pro | 1,000 | 100,000 | 10 | $20/month |
| Enterprise | 10,000 | 1,000,000 | 50 | Custom |

## 📚 Client Libraries

### Python Client

```python
# pip install unillm-client

import unillm

client = unillm.Client(api_key="your-api-key", base_url="http://localhost:8080")

# Simple generation
response = client.generate(
    prompt="The future of AI is",
    max_tokens=100,
    temperature=0.7
)
print(response.text)

# Streaming generation
for token in client.generate_stream(
    prompt="Tell me a story about",
    max_tokens=200
):
    print(token, end="", flush=True)

# Chat completion
response = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=50
)
print(response.choices[0].message.content)

# Batch processing
responses = client.batch_generate([
    {"prompt": "Translate: Hello", "max_tokens": 10},
    {"prompt": "Summarize: AI is...", "max_tokens": 50}
])
```

### JavaScript/Node.js Client

```javascript
// npm install unillm-client

const UniLLM = require('unillm-client');

const client = new UniLLM({
  apiKey: 'your-api-key',
  baseURL: 'http://localhost:8080'
});

// Simple generation
async function generate() {
  const response = await client.generate({
    prompt: 'The future of AI is',
    maxTokens: 100,
    temperature: 0.7
  });

  console.log(response.text);
}

// Streaming generation
async function generateStream() {
  const stream = await client.generateStream({
    prompt: 'Tell me a story about',
    maxTokens: 200
  });

  for await (const chunk of stream) {
    process.stdout.write(chunk.text);
  }
}

// Chat completion
async function chat() {
  const response = await client.chat.completions.create({
    messages: [
      { role: 'user', content: 'Hello!' }
    ],
    maxTokens: 50
  });

  console.log(response.choices[0].message.content);
}
```

### Go Client

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/unillm/go-client"
)

func main() {
    client := unillm.NewClient("your-api-key", "http://localhost:8080")

    // Simple generation
    response, err := client.Generate(context.Background(), &unillm.GenerateRequest{
        Prompt:      "The future of AI is",
        MaxTokens:   100,
        Temperature: 0.7,
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(response.Text)

    // Streaming generation
    stream, err := client.GenerateStream(context.Background(), &unillm.GenerateRequest{
        Prompt:    "Tell me a story about",
        MaxTokens: 200,
    })
    if err != nil {
        log.Fatal(err)
    }

    for {
        chunk, err := stream.Recv()
        if err != nil {
            break
        }
        fmt.Print(chunk.Text)
    }
}
```

### cURL Examples

```bash
# Simple generation
curl -X POST http://localhost:8080/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Streaming generation
curl -X POST http://localhost:8080/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Tell me a story",
    "max_tokens": 200,
    "stream": true
  }'

# Chat completion
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 50
  }'
```

## 🔌 WebSocket Streaming

For real-time streaming with lower latency:

### WebSocket Connection

```javascript
const ws = new WebSocket('ws://localhost:8080/v1/stream');

ws.onopen = function() {
    // Send generation request
    ws.send(JSON.stringify({
        type: 'generate',
        prompt: 'The future of AI is',
        max_tokens: 100,
        temperature: 0.7
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);

    if (data.type === 'token') {
        // New token generated
        console.log(data.token);
    } else if (data.type === 'complete') {
        // Generation complete
        console.log('Generation finished');
        console.log('Stats:', data.stats);
    } else if (data.type === 'error') {
        // Error occurred
        console.error('Error:', data.error);
    }
};
```

### Server-Sent Events (SSE)

```javascript
const eventSource = new EventSource('http://localhost:8080/v1/generate-sse', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        prompt: 'The future of AI is',
        max_tokens: 100,
        stream: true
    })
});

eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);

    if (data.choices && data.choices[0].delta) {
        process.stdout.write(data.choices[0].delta.content || '');
    }

    if (data.choices && data.choices[0].finish_reason) {
        eventSource.close();
        console.log('\nGeneration complete');
    }
};

eventSource.onerror = function(event) {
    console.error('SSE error:', event);
    eventSource.close();
};
```

## 📊 Advanced Examples

### Performance Monitoring

```python
import unillm
import time

client = unillm.Client(base_url="http://localhost:8080")

# Monitor inference performance
def benchmark_inference(num_requests=100):
    latencies = []

    for i in range(num_requests):
        start_time = time.time()

        response = client.generate(
            prompt=f"Request {i}: Tell me about AI",
            max_tokens=50
        )

        latency = (time.time() - start_time) * 1000
        latencies.append(latency)

        print(f"Request {i}: {latency:.1f}ms, "
              f"Cache hits: {response.unillm_stats.cache_hits}")

    print(f"\nBenchmark Results:")
    print(f"Average latency: {sum(latencies) / len(latencies):.1f}ms")
    print(f"P95 latency: {sorted(latencies)[int(0.95 * len(latencies))]:.1f}ms")

    # Get server stats
    stats = client.get_stats()
    print(f"Cache hit rate: {stats.cache_stats.hit_rate:.2%}")
    print(f"GPU utilization: {stats.gpu_stats.utilization:.2%}")

benchmark_inference()
```

### Adaptive Batch Processing

```python
import asyncio
import unillm

client = unillm.AsyncClient(base_url="http://localhost:8080")

async def adaptive_batch_processing(requests):
    # Start with small batch size
    batch_size = 4
    max_batch_size = 32

    while requests:
        # Take current batch
        current_batch = requests[:batch_size]
        requests = requests[batch_size:]

        start_time = time.time()

        # Process batch
        responses = await client.batch_generate(current_batch)

        batch_time = time.time() - start_time
        throughput = len(current_batch) / batch_time

        print(f"Batch size: {batch_size}, "
              f"Throughput: {throughput:.1f} req/s")

        # Adapt batch size based on performance
        if throughput > 10 and batch_size < max_batch_size:
            batch_size = min(batch_size * 2, max_batch_size)
        elif throughput < 5 and batch_size > 1:
            batch_size = max(batch_size // 2, 1)

        # Process responses
        for response in responses:
            print(f"Generated: {response.text[:50]}...")

# Example usage
requests = [
    {"prompt": f"Question {i}: What is AI?", "max_tokens": 50}
    for i in range(100)
]

asyncio.run(adaptive_batch_processing(requests))
```

### Cache Optimization

```python
import unillm

client = unillm.Client(base_url="http://localhost:8080")

# Optimize for cache hits with common prefixes
def cache_optimized_generation():
    # Use common prefixes that can be cached
    common_prefixes = [
        "Translate the following text to French:",
        "Summarize the following article:",
        "Answer the following question:",
        "Generate a creative story about"
    ]

    requests = []
    for prefix in common_prefixes:
        for i in range(10):
            requests.append({
                "prompt": f"{prefix} {generate_content(i)}",
                "max_tokens": 100,
                "cache_policy": "force"  # Force caching of prefix
            })

    # Process requests to build cache
    print("Building cache...")
    for request in requests[:20]:  # Prime the cache
        client.generate(**request)

    # Now process remaining requests with cache hits
    print("Processing with cache...")
    total_cache_hits = 0
    for request in requests[20:]:
        response = client.generate(**request)
        total_cache_hits += response.unillm_stats.cache_hits
        print(f"Cache hits: {response.unillm_stats.cache_hits}")

    print(f"Total cache hits: {total_cache_hits}")

def generate_content(i):
    return f"Example content number {i} for testing cache optimization."

cache_optimized_generation()
```

---

This API reference provides comprehensive documentation for integrating with UniLLM. For additional examples and advanced usage patterns, please refer to the client library documentation and example repositories.