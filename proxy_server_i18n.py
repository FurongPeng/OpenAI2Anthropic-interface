#!/usr/bin/env python3
"""
Anthropic to OpenAI API Proxy Server (International Version)
Converts Anthropic-style API requests to OpenAI format and forwards to local service

将Anthropic风格的API请求转换为OpenAI风格并发送到本地服务
"""

import http.server
import socketserver
import json
import urllib.request
import urllib.parse
import urllib.error
from typing import Dict, Any, List, Optional
import uuid
import logging
import re

# Configure logging / 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class I18N:
    """Internationalization class / 国际化类"""

    MESSAGES = {
        'en': {
            'server_starting': 'Anthropic proxy server starting on {host}:{port}',
            'proxying_to': 'Proxying requests to: {url}',
            'supported_endpoints': 'Supported endpoints:',
            'endpoint_messages': '  POST /v1/messages',
            'supported_features': 'Supported features:',
            'feature_text': '  - Text messages',
            'feature_streaming': '  - Streaming responses',
            'feature_system': '  - System prompts',
            'feature_multi_turn': '  - Multi-turn conversations',
            'feature_params': '  - Basic parameters (model, temperature, top_p, max_tokens)',
            'unsupported_features': 'Unsupported features:',
            'feature_tools': '  - Tools/function calling',
            'feature_media': '  - Images/files/audio',
            'feature_multimodal': '  - Multimodal content',
            'server_stopped': 'Server stopped by user',
            'received_request': 'Received Anthropic request',
            'converted_to_openai': 'Converted to OpenAI request',
            'converted_to_anthropic': 'Converted to Anthropic response',
            'handling_stream': 'Handling stream response',
            'validation_error': 'Validation error',
            'unexpected_error': 'Unexpected error',
            'stream_error': 'Stream error'
        },
        'zh': {
            'server_starting': 'Anthropic代理服务器启动在 {host}:{port}',
            'proxying_to': '代理请求到：{url}',
            'supported_endpoints': '支持的端点：',
            'endpoint_messages': '  POST /v1/messages',
            'supported_features': '支持的功能：',
            'feature_text': '  - 文本消息',
            'feature_streaming': '  - 流式响应',
            'feature_system': '  - 系统提示词',
            'feature_multi_turn': '  - 多轮对话',
            'feature_params': '  - 基本参数（model, temperature, top_p, max_tokens）',
            'unsupported_features': '不支持的功能：',
            'feature_tools': '  - 工具/函数调用',
            'feature_media': '  - 图片/文件/音频',
            'feature_multimodal': '  - 多模态内容',
            'server_stopped': '服务器被用户停止',
            'received_request': '接收到Anthropic请求',
            'converted_to_openai': '转换为OpenAI请求',
            'converted_to_anthropic': '转换为Anthropic响应',
            'handling_stream': '处理流式响应',
            'validation_error': '验证错误',
            'unexpected_error': '意外错误',
            'stream_error': '流错误'
        }
    }

    @staticmethod
    def detect_language(text: str) -> str:
        """Detect language from text / 从文本检测语言"""
        # Simple heuristic: if contains Chinese characters, use Chinese / 简单启发式：如果包含中文字符，使用中文
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh'
        return 'en'

    @classmethod
    def get_message(cls, key: str, lang: str = 'en', **kwargs) -> str:
        """Get localized message / 获取本地化消息"""
        if lang not in cls.MESSAGES:
            lang = 'en'

        message = cls.MESSAGES[lang].get(key, cls.MESSAGES['en'].get(key, key))

        if kwargs:
            try:
                return message.format(**kwargs)
            except:
                return message

        return message

class Config:
    """Configuration class / 配置类"""
    # OpenAI backend service address / OpenAI后端服务地址
    OPENAI_BASE_URL = "http://10.108.201.163:1025"
    OPENAI_CHAT_ENDPOINT = "/v1/chat/completions"

    # Proxy server configuration / 代理服务器配置
    HOST = "0.0.0.0"
    PORT = 8082

    # Anthropic API version / Anthropic API 版本
    ANTHROPIC_VERSION = "2023-06-01"

class RequestConverter:
    """Request converter: Converts Anthropic requests to OpenAI format / 请求转换器：将Anthropic请求转换为OpenAI格式"""

    @staticmethod
    def convert_anthropic_to_openai(anthropic_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Anthropic Messages API request to OpenAI Chat Completions format
        将Anthropic Messages API请求转换为OpenAI Chat Completions格式

        Args:
            anthropic_request: Anthropic format request body / Anthropic格式的请求体

        Returns:
            OpenAI format request body / OpenAI格式的请求体
        """
        openai_request = {}

        # Basic field mapping / 基本字段映射
        openai_request["model"] = anthropic_request.get("model", "DeepSeek32B")

        # Build messages array / 构建messages数组
        openai_messages = []

        # If there's a system prompt, add as the first message / 如果有system提示，添加为第一条消息
        if "system" in anthropic_request and anthropic_request["system"]:
            openai_messages.append({
                "role": "system",
                "content": anthropic_request["system"]
            })

        # Convert messages / 转换messages
        for msg in anthropic_request.get("messages", []):
            role = msg.get("role")
            content = msg.get("content")

            # Handle content as array (Anthropic format) / 处理content为数组的情况（Anthropic格式）
            if isinstance(content, list):
                # Extract and concatenate all text content / 提取所有text内容并拼接
                text_parts = []
                for item in content:
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    else:
                        # Currently only support text type / 目前不支持非text类型
                        raise ValueError(f"Unsupported content type: {item.get('type')}")
                content = "".join(text_parts)

            if role and content:
                openai_messages.append({
                    "role": role,
                    "content": content
                })

        openai_request["messages"] = openai_messages

        # Parameter mapping - only map OpenAI supported parameters / 参数映射 - 只映射OpenAI支持的参数
        param_mapping = {
            "max_tokens": "max_tokens",
            "temperature": "temperature",
            "top_p": "top_p"
        }

        for anthropic_param, openai_param in param_mapping.items():
            if anthropic_param in anthropic_request:
                openai_request[openai_param] = anthropic_request[anthropic_param]

        # If max_tokens is not set, set default value / 如果没有设置max_tokens，设置默认值
        if "max_tokens" not in openai_request:
            openai_request["max_tokens"] = 1024

        # OpenAI specific parameter - set stream based on request / OpenAI特有参数 - 根据请求设置stream
        openai_request["stream"] = anthropic_request.get("stream", False)

        return openai_request

class ResponseConverter:
    """Response converter: Converts OpenAI responses to Anthropic format / 响应转换器：将OpenAI响应转换为Anthropic格式"""

    @staticmethod
    def convert_openai_to_anthropic(openai_response: Dict[str, Any], model: str) -> Dict[str, Any]:
        """
        Convert OpenAI Chat Completions response to Anthropic Messages format
        将OpenAI Chat Completions响应转换为Anthropic Messages格式

        Args:
            openai_response: OpenAI format response / OpenAI格式的响应
            model: Model name used / 使用的模型名称

        Returns:
            Anthropic format response / Anthropic格式的响应
        """
        # Extract OpenAI response content / 提取OpenAI响应内容
        choice = openai_response.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        finish_reason = choice.get("finish_reason", "stop")

        # Map finish_reason / 映射finish_reason
        stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "content_filter": "stop_sequence"
        }
        stop_reason = stop_reason_map.get(finish_reason, "end_turn")

        # Extract usage information / 提取usage信息
        usage = openai_response.get("usage", {})

        # Build Anthropic response / 构建Anthropic响应
        anthropic_response = {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [
                {
                    "type": "text",
                    "text": content
                }
            ],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0)
            }
        }

        return anthropic_response

class AnthropicProxyHandler(http.server.BaseHTTPRequestHandler):
    """Proxy server request handler / 代理服务器请求处理器"""

    def __init__(self, *args, **kwargs):
        self.language = 'en'  # Default language / 默认语言
        super().__init__(*args, **kwargs)

    def do_POST(self):
        """Handle POST requests / 处理POST请求"""
        # Detect language from User-Agent or headers / 从User-Agent或headers检测语言
        user_agent = self.headers.get('User-Agent', '')
        self.language = I18N.detect_language(user_agent)

        if self.path == "/v1/messages":
            self._handle_messages()
        else:
            error_msg = I18N.get_message('not_found', self.language)
            self._send_error(404, "Not Found", {"type": "invalid_request_error", "message": error_msg})

    def _handle_messages(self):
        """Handle /v1/messages endpoint / 处理 /v1/messages 端点"""
        try:
            # Read request body / 读取请求体
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                error_msg = I18N.get_message('missing_body', self.language)
                self._send_error(400, "Bad Request", {"type": "invalid_request_error", "message": error_msg})
                return

            request_body = self.rfile.read(content_length).decode('utf-8')

            # Parse JSON / 解析JSON
            try:
                anthropic_request = json.loads(request_body)
            except json.JSONDecodeError as e:
                error_msg = I18N.get_message('invalid_json', self.language).format(str(e))
                self._send_error(400, "Bad Request", {"type": "invalid_request_error", "message": error_msg})
                return

            # Validate required fields / 验证必要字段
            if not anthropic_request.get("messages"):
                error_msg = I18N.get_message('missing_messages', self.language)
                self._send_error(400, "Bad Request", {"type": "invalid_request_error", "message": error_msg})
                return

            # Stream response handling / 流式响应处理
            is_stream = anthropic_request.get("stream", False)

            logger.info(I18N.get_message('received_request', self.language))
            logger.info(json.dumps(anthropic_request, indent=2))

            # Convert request format / 转换请求格式
            openai_request = RequestConverter.convert_anthropic_to_openai(anthropic_request)

            logger.info(I18N.get_message('converted_to_openai', self.language))
            logger.info(json.dumps(openai_request, indent=2))

            if is_stream:
                # Stream response handling / 流式响应处理
                self._handle_stream_response(openai_request, anthropic_request.get("model", "DeepSeek32B"))
            else:
                # Non-stream response handling / 非流式响应处理
                openai_response = self._send_to_openai(openai_request)

                # Convert response format / 转换响应格式
                anthropic_response = ResponseConverter.convert_openai_to_anthropic(
                    openai_response,
                    anthropic_request.get("model", "DeepSeek32B")
                )

                logger.info(I18N.get_message('converted_to_anthropic', self.language))
                logger.info(json.dumps(anthropic_response, indent=2))

                # Return response / 返回响应
                self._send_json(200, anthropic_response)

        except ValueError as e:
            logger.error(f"{I18N.get_message('validation_error', self.language)}: {str(e)}")
            error_msg = str(e) if self.language == 'en' else str(e)
            self._send_error(400, "Bad Request", {"type": "invalid_request_error", "message": error_msg})
        except Exception as e:
            logger.error(f"{I18N.get_message('unexpected_error', self.language)}: {str(e)}")
            error_msg = I18N.get_message('internal_error', self.language)
            self._send_error(500, "Internal Server Error", {"type": "api_error", "message": error_msg})

    def _handle_stream_response(self, openai_request: Dict[str, Any], model: str):
        """Handle streaming response / 处理流式响应"""
        url = f"{Config.OPENAI_BASE_URL}{Config.OPENAI_CHAT_ENDPOINT}"

        # Prepare request data / 准备请求数据
        data = json.dumps(openai_request).encode('utf-8')

        # Create request / 创建请求
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream',
                'Cache-Control': 'no-cache'
            }
        )

        # Set response headers / 设置响应头
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.end_headers()

        try:
            # Send request / 发送请求
            with urllib.request.urlopen(req, timeout=120) as response:
                logger.info(I18N.get_message('handling_stream', self.language))

                # Create event stream ID / 创建事件流ID
                event_id = f"evt_{uuid.uuid4().hex[:24]}"

                # Send start event / 发送开始事件
                self._send_sse_event(event_id, "message_start", {
                    "type": "message_start",
                    "message": {
                        "id": f"msg_{uuid.uuid4().hex[:24]}",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": 0,
                            "output_tokens": 0
                        }
                    }
                })

                # Process streaming data / 处理流式数据
                buffer = ""
                for chunk in response:
                    buffer += chunk.decode('utf-8')

                    # Process each complete JSON line / 处理每个完整的JSON行
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()

                        if line.startswith('data: '):
                            data_str = line[6:]  # Remove 'data: ' prefix / 移除 'data: ' 前缀

                            if data_str == '[DONE]':
                                # If there's remaining buffered content, send it first / 如果还有缓冲的内容，先发送
                                if hasattr(self, '_content_buffer') and self._content_buffer:
                                    self._send_sse_event(event_id, "content_block_delta", {
                                        "type": "content_block_delta",
                                        "index": 0,
                                        "delta": {
                                            "type": "text_delta",
                                            "text": self._content_buffer
                                        }
                                    })
                                    self._content_buffer = ""

                                # If content block has started, send stop event / 如果内容块已经开始，发送停止事件
                                if hasattr(self, '_content_started') and self._content_started:
                                    self._send_sse_event(event_id, "content_block_stop", {
                                        "type": "content_block_stop",
                                        "index": 0
                                    })
                                    self._content_started = False

                                # Send end events / 发送结束事件
                                self._send_sse_event(event_id, "message_delta", {
                                    "type": "message_delta",
                                    "delta": {
                                        "stop_reason": "end_turn",
                                        "stop_sequence": None
                                    },
                                    "usage": {
                                        "output_tokens": 0
                                    }
                                })
                                self._send_sse_event(event_id, "message_stop", {
                                    "type": "message_stop"
                                })
                                return

                            try:
                                chunk_data = json.loads(data_str)
                                self._process_openai_stream_chunk(chunk_data, event_id, model)
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            logger.error(f"{I18N.get_message('stream_error', self.language)}: {str(e)}")
            # Send error event / 发送错误事件
            self._send_sse_event(event_id, "error", {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "Streaming failed"
                }
            })

    def _process_openai_stream_chunk(self, chunk_data: Dict[str, Any], event_id: str, model: str):
        """Process OpenAI streaming response chunk / 处理OpenAI流式响应块"""
        choices = chunk_data.get("choices", [])
        if not choices:
            return

        delta = choices[0].get("delta", {})

        # Handle content block / 处理内容块
        if "content" in delta and delta["content"]:
            if not hasattr(self, '_content_started'):
                self._content_started = False
                self._content_buffer = ""

            # If content block hasn't started, send start event / 如果还没有开始内容块，发送开始事件
            if not self._content_started:
                self._send_sse_event(event_id, "content_block_start", {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "text",
                        "text": ""
                    }
                })
                self._content_started = True

            # Accumulate content / 累积内容
            self._content_buffer += delta["content"]

            # Send when buffer reaches certain size or encounters punctuation / 当缓冲区达到一定大小或遇到标点符号时发送
            if len(self._content_buffer) >= 10 or delta["content"] in "。！？，；.!?,":
                self._send_sse_event(event_id, "content_block_delta", {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "text_delta",
                        "text": self._content_buffer
                    }
                })
                self._content_buffer = ""

    def _send_to_openai(self, openai_request: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to OpenAI backend service / 发送请求到OpenAI后端服务"""
        url = f"{Config.OPENAI_BASE_URL}{Config.OPENAI_CHAT_ENDPOINT}"

        # Prepare request data / 准备请求数据
        data = json.dumps(openai_request).encode('utf-8')

        # Create request / 创建请求
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        )

        try:
            # Send request / 发送请求
            with urllib.request.urlopen(req, timeout=120) as response:
                response_data = response.read().decode('utf-8')
                return json.loads(response_data)

        except urllib.error.HTTPError as e:
            error_message = e.read().decode('utf-8')
            logger.error(f"OpenAI service returned HTTP {e.code}: {error_message}")
            raise Exception(f"Upstream OpenAI service error: {e.code}")
        except urllib.error.URLError as e:
            logger.error(f"Failed to connect to OpenAI service: {str(e)}")
            raise Exception("Upstream OpenAI service unavailable")
        except Exception as e:
            logger.error(f"Error calling OpenAI service: {str(e)}")
            raise Exception("Upstream OpenAI service error")

    def _send_json(self, status_code: int, data: Dict[str, Any]):
        """Send JSON response / 发送JSON响应"""
        response_data = json.dumps(data).encode('utf-8')

        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response_data)))
        self.end_headers()

        self.wfile.write(response_data)

    def _send_error(self, status_code: int, message: str, error: Dict[str, str]):
        """Send error response / 发送错误响应"""
        error_response = {
            "error": error
        }
        self._send_json(status_code, error_response)

    def _send_sse_event(self, event_id: str, event_type: str, data: Dict[str, Any]):
        """Send Server-Sent Event / 发送Server-Sent Event"""
        # Build SSE format data / 构建SSE格式的数据
        sse_data = f"event: {event_type}\n"
        sse_data += f"data: {json.dumps(data)}\n\n"

        # Send data / 发送数据
        self.wfile.write(sse_data.encode('utf-8'))
        self.wfile.flush()

    def log_message(self, format, *args):
        """Override log method / 重写日志方法"""
        logger.info(f"{self.client_address[0]} - {format % args}")

def run_server():
    """Start proxy server / 启动代理服务器"""
    # Detect system language / 检测系统语言
    import locale
    try:
        system_lang = locale.getdefaultlocale()[0]
        if system_lang and system_lang.startswith('zh'):
            lang = 'zh'
        else:
            lang = 'en'
    except:
        lang = 'en'

    with socketserver.TCPServer((Config.HOST, Config.PORT), AnthropicProxyHandler) as httpd:
        # Print startup messages / 打印启动信息
        print(I18N.get_message('server_starting', lang, host=Config.HOST, port=Config.PORT))
        print(I18N.get_message('proxying_to', lang, url=Config.OPENAI_BASE_URL))
        print("")
        print(I18N.get_message('supported_endpoints', lang))
        print(I18N.get_message('endpoint_messages', lang))
        print("")
        print(I18N.get_message('supported_features', lang))
        print(I18N.get_message('feature_text', lang))
        print(I18N.get_message('feature_streaming', lang))
        print(I18N.get_message('feature_system', lang))
        print(I18N.get_message('feature_multi_turn', lang))
        print(I18N.get_message('feature_params', lang))
        print("")
        print(I18N.get_message('unsupported_features', lang))
        print(I18N.get_message('feature_tools', lang))
        print(I18N.get_message('feature_media', lang))
        print(I18N.get_message('feature_multimodal', lang))

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("")
            print(I18N.get_message('server_stopped', lang))

if __name__ == "__main__":
    run_server()
