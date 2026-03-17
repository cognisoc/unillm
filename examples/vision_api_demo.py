#!/usr/bin/env python3
"""
UniLLM OpenAI Vision API Demo

Demonstrates how to use UniLLM as a drop-in replacement for OpenAI's GPT-4 Vision API.
This script shows various use cases including base64 images, URLs, and multimodal conversations.
"""

import requests
import base64
import json
from pathlib import Path

# UniLLM server endpoint (compatible with OpenAI API)
API_BASE = "http://localhost:8000/v1"

def encode_image_to_base64(image_path):
    """Convert a local image file to base64 format for API usage."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_vision_request(prompt, image_data=None, image_url=None, model="gpt-4-vision-preview"):
    """Create an OpenAI-compatible vision request."""
    content = [{"type": "text", "text": prompt}]

    if image_data:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_data}",
                "detail": "high"
            }
        })
    elif image_url:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": image_url,
                "detail": "high"
            }
        })

    return {
        "model": model,
        "messages": [{
            "role": "user",
            "content": content
        }],
        "max_tokens": 300,
        "temperature": 0.7
    }

def call_vision_api(request_data):
    """Make a request to the UniLLM vision API."""
    headers = {
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            f"{API_BASE}/chat/completions",
            headers=headers,
            json=request_data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
        return None

def demo_text_only():
    """Demo 1: Text-only conversation."""
    print("🔤 Demo 1: Text-only conversation")

    request = {
        "model": "gpt-4-vision-preview",
        "messages": [{
            "role": "user",
            "content": "Hello! Can you tell me about the capabilities of vision-language models?"
        }],
        "max_tokens": 200
    }

    response = call_vision_api(request)
    if response:
        print(f"✅ Response: {response['choices'][0]['message']['content']}")
    print()

def demo_base64_image():
    """Demo 2: Base64 encoded image analysis."""
    print("🖼️  Demo 2: Base64 image analysis")

    # Create a simple 1x1 pixel image in base64 (for demo purposes)
    tiny_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

    request = create_vision_request(
        "What do you see in this image? Describe it in detail.",
        image_data=tiny_image_b64
    )

    response = call_vision_api(request)
    if response:
        print(f"✅ Vision Analysis: {response['choices'][0]['message']['content']}")
    print()

def demo_image_url():
    """Demo 3: URL-based image analysis."""
    print("🌐 Demo 3: URL-based image analysis")

    # Example with a public image URL
    request = create_vision_request(
        "Analyze this image and tell me what you observe.",
        image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    )

    response = call_vision_api(request)
    if response:
        print(f"✅ Vision Analysis: {response['choices'][0]['message']['content']}")
    print()

def demo_multimodal_conversation():
    """Demo 4: Multi-turn conversation with images."""
    print("💬 Demo 4: Multi-turn multimodal conversation")

    # First message with image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here's an image. Can you describe what you see?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
                        "detail": "low"
                    }
                }
            ]
        },
        {
            "role": "assistant",
            "content": "I can see a very small image that appears to be a single pixel."
        },
        {
            "role": "user",
            "content": "That's correct! Now tell me about the applications of such minimal images in computer graphics."
        }
    ]

    request = {
        "model": "gpt-4-vision-preview",
        "messages": messages,
        "max_tokens": 250,
        "temperature": 0.5
    }

    response = call_vision_api(request)
    if response:
        print(f"✅ Conversation Response: {response['choices'][0]['message']['content']}")
    print()

def demo_batch_images():
    """Demo 5: Multiple images in one request."""
    print("🖼️🖼️ Demo 5: Multiple images analysis")

    # Create request with multiple images
    content = [
        {"type": "text", "text": "Compare and contrast these two images:"}
    ]

    # Add two identical tiny images for demo
    tiny_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

    for i in range(2):
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{tiny_image}",
                "detail": "low"
            }
        })

    request = {
        "model": "gpt-4-vision-preview",
        "messages": [{
            "role": "user",
            "content": content
        }],
        "max_tokens": 300
    }

    response = call_vision_api(request)
    if response:
        print(f"✅ Multi-image Analysis: {response['choices'][0]['message']['content']}")
    print()

def main():
    """Run all demos."""
    print("🚀 UniLLM OpenAI Vision API Demo")
    print("=" * 50)
    print()

    # Check if server is running
    try:
        health_response = requests.get(f"{API_BASE.replace('/v1', '')}/health", timeout=5)
        print("✅ UniLLM server is running!")
    except:
        print("❌ UniLLM server is not running. Please start it first:")
        print("   cargo run --bin openai_vision_server")
        return

    print()

    # Run all demos
    demo_text_only()
    demo_base64_image()
    demo_image_url()
    demo_multimodal_conversation()
    demo_batch_images()

    print("🎉 All demos completed!")
    print()
    print("💡 Integration Tips:")
    print("- Replace 'openai.api_base' with your UniLLM server URL")
    print("- Use the same request format as OpenAI GPT-4 Vision")
    print("- Supports base64 images and URLs")
    print("- Compatible with existing OpenAI client libraries")

if __name__ == "__main__":
    main()