# Anthropic to OpenAI API Proxy Server

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README_EN.md) [![中文](https://img.shields.io/badge/lang-中文-red.svg)](README_CN.md)

A lightweight Python proxy server that converts Anthropic Messages API format requests to OpenAI Chat Completions API format and forwards them to local OpenAI-style services.

## ✨ Features

### 🟢 Supported Features
- `POST /v1/messages` - Anthropic Messages API endpoint
- Text messages (`type: text`)
- Single/multi-turn conversations
- System prompts
- Streaming responses (stream=true/false)
- Basic parameters (model, temperature, top_p, max_tokens)
- Content array format (multiple text blocks)
- Internationalization support (Chinese/English)
- Error handling and parameter validation
- No external dependencies, uses only Python native modules

### 🔴 Unsupported Features
- Tools/function calling
- Images/files/audio
- Multimodal content

## 🚀 Quick Start

### Requirements
- Python 3.6 or higher
- No additional packages required (uses only Python standard library)

### Installation and Running

1. **Download the code**
   ```bash
   git clone https://github.com/yourusername/anthropic-openai-proxy.git
   cd anthropic-openai-proxy
   ```

2. **Run the proxy server**
   ```bash
   python3 proxy_server_i18n.py
   ```

   The server will start on `http://localhost:8082`

3. **Configure backend service**

   Default configuration:
   - OpenAI backend URL: `http://10.108.201.163:1025`
   - Endpoint: `/v1/chat/completions`

   To modify, edit the `Config` class in `proxy_server_i18n.py`.

## 📖 Usage Examples

### Using curl

#### Non-streaming request
```bash
curl -X POST http://localhost:8082/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "DeepSeek32B",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

#### Streaming request
```bash
curl -X POST http://localhost:8082/v1/messages \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "DeepSeek32B",
    "max_tokens": 100,
    "messages": [
      {"role": "user", "content": "Tell me about spring"}
    ],
    "stream": true
  }' --no-buffer
```

#### Request with system prompt
```bash
curl -X POST http://localhost:8082/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek32B",
    "system": "You are a helpful assistant",
    "max_tokens": 100,
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

### Using Python

```python
import anthropic

# Create client pointing to proxy server
client = anthropic.Anthropic(
    base_url="http://localhost:8082",
    api_key="dummy-key"  # Proxy server doesn't validate API key
)

# Send request
message = client.messages.create(
    model="DeepSeek32B",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Tell me about Python"}
    ]
)

print(message.content[0].text)
```

### Using Node.js

```javascript
import Anthropic from '@anthropic-ai/sdk';

const anthropic = new Anthropic({
    baseURL: 'http://localhost:8082',
    apiKey: 'dummy-key', // Proxy server doesn't validate API key
});

const message = await anthropic.messages.create({
    model: 'DeepSeek32B',
    max_tokens: 1024,
    messages: [
        { role: 'user', content: 'Tell me about JavaScript' }
    ]
});

console.log(message.content[0].text);
```

## 🔄 Request/Response Conversion

### Anthropic Request → OpenAI Request

```json
// Anthropic format
{
  "model": "DeepSeek32B",
  "system": "You are a helpful assistant",
  "max_tokens": 100,
  "temperature": 0.7,
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": " world!"}
      ]
    }
  ]
}

// Converted to OpenAI format
{
  "model": "DeepSeek32B",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello world!"}
  ],
  "max_tokens": 100,
  "temperature": 0.7,
  "stream": false
}
```

### OpenAI Response → Anthropic Response

```json
// OpenAI format
{
  "choices": [
    {
      "message": {
        "content": "Hello! How can I help you?",
        "role": "assistant"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8
  }
}

// Converted to Anthropic format
{
  "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
  "type": "message",
  "role": "assistant",
  "model": "DeepSeek32B",
  "content": [
    {
      "type": "text",
      "text": "Hello! How can I help you?"
    }
  ],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 10,
    "output_tokens": 8
  }
}
```

## 🌍 Internationalization

The proxy server supports both Chinese and English:

- Automatically detects client language (based on User-Agent)
- Server startup messages display in detected language
- Log messages support bilingual display

## 📝 Configuration

You can configure the following parameters in the `Config` class:

```python
class Config:
    # OpenAI backend service address
    OPENAI_BASE_URL = "http://10.108.201.163:1025"
    OPENAI_CHAT_ENDPOINT = "/v1/chat/completions"

    # Proxy server configuration
    HOST = "0.0.0.0"  # Listen address
    PORT = 8082       # Listen port
```

## ❗ Error Handling

The proxy server returns standard Anthropic error format:

```json
{
  "error": {
    "type": "invalid_request_error",
    "message": "messages field is required"
  }
}
```

Common error types:
- `invalid_request_error` - Request format error
- `feature_not_supported` - Unsupported feature
- `api_error` - Upstream service error

## 🔧 Troubleshooting

### Issue: Connection refused
- Ensure the proxy server is running
- Check if port 8082 is occupied

### Issue: Upstream service error
- Check if the OpenAI-style service is running normally
- Verify network connection
- Check log file

### Issue: Request format error
- Ensure the request complies with Anthropic Messages API specification
- Verify JSON format is correct

## 📋 Development Roadmap

- [ ] Support more OpenAI parameters (e.g., presence_penalty, frequency_penalty)
- [ ] Add request/response logging
- [ ] Support authentication and API key validation
- [ ] Add request rate limiting
- [ ] Support WebSocket connections

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📄 License

MIT License

## 🙏 Acknowledgments

- Thanks to Anthropic and OpenAI for their excellent APIs
- Thanks to all contributors for their support

## 📧 Contact

For questions or suggestions:
- Submit an Issue: https://github.com/yourusername/anthropic-openai-proxy/issues
- Send an email: your.email@example.com
