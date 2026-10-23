import os
from typing import Any, Optional, List, Dict, Union, Tuple
from dotenv import load_dotenv
import logging
import json
import yaml

load_dotenv()
logger = logging.getLogger(__name__)

try:
    from smolagents import AmazonBedrockModel, ChatMessage, MessageRole
    from smolagents.models import TokenUsage
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    logger.warning("smolagents not available, using mock implementation for compatibility")
    SMOLAGENTS_AVAILABLE = False

    # Mock classes for compatibility when smolagents is not available
    class MessageRole:
        USER = "user"
        ASSISTANT = "assistant"
        SYSTEM = "system"

    class ChatMessage:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    class TokenUsage:
        def __init__(self, input_tokens=0, output_tokens=0):
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens

    class AmazonBedrockModel:
        def __init__(self, *args, **kwargs):
            logger.warning("Mock AmazonBedrockModel created - smolagents not available")

        def generate(self, messages):
            logger.warning("Mock generate called - returning placeholder response")
            return ChatMessage(MessageRole.ASSISTANT, "Mock response: smolagents not available")


class BedrocksmolagentsModelWrapper:
    """Wrapper for Bedrock that implements smolagents Model interface using boto3 directly.

    This wrapper uses boto3 to call Bedrock API directly, avoiding the tool_calls bug
    in the smolagents library while still being compatible with CodeAgent.
    """

    def __init__(
        self,
        model_id: str,
        region: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs
    ):
        self.model_id = model_id
        self.region = region
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.kwargs = kwargs

        # Token usage tracking (cumulative across all calls)
        self._cumulative_input_tokens = 0
        self._cumulative_output_tokens = 0

        # Initialize boto3 client
        import boto3
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region)

        logger.info(f"Initialized BedrocksmolagentsModelWrapper for {model_id}")

    def _is_anthropic_model(self) -> bool:
        """Check if the model is an Anthropic Claude model"""
        return 'anthropic' in self.model_id.lower() or 'claude' in self.model_id.lower()

    def _supports_stop_sequences(self) -> bool:
        """Check if the model supports stopSequences in Converse API"""
        # Qwen models don't support stopSequences
        if 'qwen' in self.model_id.lower():
            return False
        # Most other models support it (Mistral, Llama, etc.)
        return True

    def generate(
        self,
        messages: List,
        stop_sequences: Optional[List[str]] = None,
        response_format: Optional[Dict[str, str]] = None,
        tools_to_call_from: Optional[List] = None,
        **kwargs
    ):
        """Generate response using boto3 directly (implements smolagents Model interface)"""

        # Separate system messages from user/assistant messages
        system_prompt = None
        api_messages = []

        for msg in messages:
            role = str(msg.role) if hasattr(msg, 'role') else 'user'

            # Extract content
            if hasattr(msg, 'content'):
                content = msg.content
            else:
                content = str(msg)

            # Handle list content (multimodal)
            if isinstance(content, list):
                content_text = ""
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        content_text += item["text"] + "\n"
                    else:
                        content_text += str(item) + "\n"
                content = content_text.strip()
            else:
                content = str(content)

            # Map MessageRole enum to string
            if 'SYSTEM' in role.upper():
                # Bedrock requires system as separate parameter
                system_prompt = content
            elif 'USER' in role.upper():
                api_messages.append({"role": "user", "content": content})
            elif 'ASSISTANT' in role.upper():
                api_messages.append({"role": "assistant", "content": content})
            else:
                # Default to user for unknown roles
                api_messages.append({"role": "user", "content": content})

        try:
            # Use different API formats based on model type
            if self._is_anthropic_model():
                # Anthropic models use Messages API format
                # Note: Claude 4.x models don't allow both temperature and top_p
                # Only include temperature (skip top_p) to avoid ValidationException
                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": kwargs.get('max_tokens', self.max_tokens),
                    "temperature": kwargs.get('temperature', self.temperature),
                    "messages": api_messages
                }

                # Add system prompt if present
                if system_prompt:
                    request_body["system"] = system_prompt

                # Add stop sequences if provided
                if stop_sequences:
                    request_body["stop_sequences"] = stop_sequences

                # Call Bedrock API using invoke_model
                response = self.bedrock_runtime.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body)
                )

                response_body = json.loads(response['body'].read())
                content = response_body['content'][0]['text']

                # Extract token usage from Anthropic response
                if 'usage' in response_body:
                    self._cumulative_input_tokens += response_body['usage'].get('input_tokens', 0)
                    self._cumulative_output_tokens += response_body['usage'].get('output_tokens', 0)

            else:
                # Non-Anthropic models (Qwen, Mistral, etc.) use Converse API
                # Convert messages to Converse API format
                converse_messages = []
                for msg in api_messages:
                    converse_messages.append({
                        "role": msg["role"],
                        "content": [{"text": msg["content"]}]
                    })

                # Build inference config
                inference_config = {
                    "maxTokens": kwargs.get('max_tokens', self.max_tokens),
                    "temperature": kwargs.get('temperature', self.temperature),
                    "topP": kwargs.get('top_p', self.top_p)
                }

                # Add stop sequences if provided and model supports it
                if stop_sequences and self._supports_stop_sequences():
                    inference_config["stopSequences"] = stop_sequences

                # Prepare converse call parameters
                converse_params = {
                    "modelId": self.model_id,
                    "messages": converse_messages,
                    "inferenceConfig": inference_config
                }

                # Add system prompt if present
                if system_prompt:
                    converse_params["system"] = [{"text": system_prompt}]

                # Call Bedrock Converse API
                response = self.bedrock_runtime.converse(**converse_params)

                # Extract content from Converse API response
                content = response['output']['message']['content'][0]['text']

                # Extract token usage from Converse API response
                if 'usage' in response:
                    self._cumulative_input_tokens += response['usage'].get('inputTokens', 0)
                    self._cumulative_output_tokens += response['usage'].get('outputTokens', 0)

            # Return ChatMessage in smolagents format
            if SMOLAGENTS_AVAILABLE:
                return ChatMessage(MessageRole.ASSISTANT, content)
            else:
                # Fallback for when smolagents is not available
                class SimpleChatMessage:
                    def __init__(self, role, content):
                        self.role = role
                        self.content = content
                return SimpleChatMessage('assistant', content)

        except Exception as e:
            logger.error(f"Error calling Bedrock via boto3: {e}")
            raise

    def __call__(self, *args, **kwargs):
        """Allow calling the model directly"""
        return self.generate(*args, **kwargs)

    def get_cumulative_token_usage(self) -> Dict[str, int]:
        """Get cumulative token usage across all API calls.

        Returns:
            Dictionary with 'input_tokens', 'output_tokens', 'total_tokens'
        """
        return {
            'input_tokens': self._cumulative_input_tokens,
            'output_tokens': self._cumulative_output_tokens,
            'total_tokens': self._cumulative_input_tokens + self._cumulative_output_tokens
        }

    def reset_token_usage(self):
        """Reset cumulative token counters."""
        self._cumulative_input_tokens = 0
        self._cumulative_output_tokens = 0


class BedrockSmolagentsModel:
    """Bedrock model compatible with smolagents framework while maintaining backward compatibility"""

    def __init__(
        self,
        model: str,
        region_name: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.region = region_name or os.environ.get("AWS_REGION", "us-east-2")

        # Always create a custom smolagents-compatible model that uses boto3 directly
        # to avoid the tool_calls bug in smolagents library
        self.smolagents_model = BedrocksmolagentsModelWrapper(
            model_id=model,
            region=self.region,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )

    def get_cumulative_token_usage(self) -> Dict[str, int]:
        """Get cumulative token usage from the wrapped model.

        Returns:
            Dictionary with 'input_tokens', 'output_tokens', 'total_tokens'
        """
        if hasattr(self.smolagents_model, 'get_cumulative_token_usage'):
            return self.smolagents_model.get_cumulative_token_usage()
        return {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}

    def reset_token_usage(self):
        """Reset cumulative token counters in the wrapped model."""
        if hasattr(self.smolagents_model, 'reset_token_usage'):
            self.smolagents_model.reset_token_usage()

    def _is_anthropic_model(self) -> bool:
        """Check if the model is an Anthropic Claude model"""
        return 'anthropic' in self.model.lower() or 'claude' in self.model.lower()

    def _supports_stop_sequences(self) -> bool:
        """Check if the model supports stopSequences in Converse API"""
        # Qwen models don't support stopSequences
        if 'qwen' in self.model.lower():
            return False
        # Most other models support it (Mistral, Llama, etc.)
        return True

    async def __call__(self, prompt: str) -> str:
        """Simple call: prompt -> output (maintains backward compatibility)"""
        # Always use boto3 directly to avoid smolagents tool_calls bug
        import boto3

        if not hasattr(self, 'bedrock_runtime'):
            self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=self.region)

        try:
            if self._is_anthropic_model():
                # Anthropic models use Messages API format
                # Note: Claude 4.x models don't allow both temperature and top_p
                response = self.bedrock_runtime.invoke_model(
                    modelId=self.model,
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    })
                )

                response_body = json.loads(response['body'].read())
                content = response_body['content'][0]['text']
            else:
                # Non-Anthropic models (Qwen, Mistral, etc.) use Converse API
                response = self.bedrock_runtime.converse(
                    modelId=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [{"text": prompt}]
                        }
                    ],
                    inferenceConfig={
                        "maxTokens": self.max_tokens,
                        "temperature": self.temperature,
                        "topP": self.top_p
                    }
                )

                content = response['output']['message']['content'][0]['text']

            return content

        except Exception as e:
            logger.error(f"Error calling Bedrock via boto3: {e}")
            raise

    def generate(self, messages: List[ChatMessage], **kwargs) -> ChatMessage:
        """smolagents interface: generate method"""
        # Always use boto3 directly to avoid smolagents tool_calls bug
        import boto3

        if not hasattr(self, 'bedrock_runtime'):
            self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=self.region)

        # Convert messages to prompt
        prompt = ""
        for msg in messages:
            if isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict) and "text" in item:
                        prompt += item["text"] + "\n"
                    else:
                        prompt += str(item) + "\n"
            else:
                prompt += str(msg.content) + "\n"

        # Call via boto3 (synchronous wrapper for async __call__)
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        content = loop.run_until_complete(self(prompt))
        return ChatMessage(MessageRole.ASSISTANT, content)

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        tool_specs: Optional[List[Dict[str, Any]]] = None,  # Kept for backward compatibility
        system_prompt: Optional[str] = None
    ):
        """Stream events (maintains backward compatibility for Strands-based code)"""
        # tool_specs parameter is kept for backward compatibility but not used in smolagents
        _ = tool_specs  # Suppress unused parameter warning

        if not SMOLAGENTS_AVAILABLE:
            logger.warning(f"smolagents not available, returning mock stream response")
            mock_content = f"Mock streaming response from {self.model} (smolagents not available)"
            chunk_size = 20
            for i in range(0, len(mock_content), chunk_size):
                chunk = mock_content[i:i+chunk_size]
                yield {
                    "contentBlockDelta": {
                        "delta": {"text": chunk},
                        "contentBlockIndex": 0
                    }
                }
            return

        try:
            # Convert messages to ChatMessage format
            chat_messages = []

            # Add system message if provided
            if system_prompt:
                system_content = [{"type": "text", "text": system_prompt}]
                chat_messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_content))

            # Convert the messages
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                # Format content correctly for Bedrock API with required "type" field
                if isinstance(content, list):
                    # Already in list format, ensure each element has "type" field
                    formatted_content = []
                    for item in content:
                        if isinstance(item, dict):
                            if "type" not in item:
                                # Add type field if missing
                                item = {"type": "text", **item}
                            formatted_content.append(item)
                        else:
                            formatted_content.append({"type": "text", "text": str(item)})
                else:
                    # Convert string to list format with type field
                    formatted_content = [{"type": "text", "text": str(content)}]

                # Map role to MessageRole
                if role == "system":
                    msg_role = MessageRole.SYSTEM
                elif role == "assistant":
                    msg_role = MessageRole.ASSISTANT
                else:
                    msg_role = MessageRole.USER

                chat_messages.append(ChatMessage(role=msg_role, content=formatted_content))

            # For now, use generate() since smolagents doesn't expose streaming in the same way
            # This maintains compatibility but without actual streaming
            response = self.smolagents_model.generate(chat_messages)

            # Simulate streaming by yielding the response in chunks
            content = response.content
            chunk_size = 100
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i+chunk_size]
                yield {
                    "contentBlockDelta": {
                        "delta": {"text": chunk},
                        "contentBlockIndex": 0
                    }
                }

        except Exception as e:
            logger.error(f"Error streaming from Bedrock via smolagents: {e}")
            raise


class OpenAIModel:
    """OpenAI API model compatible with smolagents framework"""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        # Token usage tracking (cumulative across all calls)
        self._cumulative_input_tokens = 0
        self._cumulative_output_tokens = 0
        self._cumulative_reasoning_tokens = 0

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables or parameters")

        # Try to import openai
        try:
            import openai
            self.openai = openai
            self.client = openai.OpenAI(api_key=self.api_key, max_retries=10, timeout=120.0)
            logger.info(f"Initialized OpenAI model: {model}")
        except ImportError:
            raise ImportError("openai library is required for OpenAI models. Install with: pip install openai")

        # Create smolagents-compatible interface
        self.smolagents_model = self._create_mock_smolagents_model()

    def _create_mock_smolagents_model(self):
        """Create a mock smolagents interface for compatibility"""
        class MockSmolagentsModel:
            def __init__(self, parent):
                self.parent = parent
                self.model_id = parent.model
                self.max_tokens = parent.max_tokens
                self.temperature = parent.temperature
                self.top_p = parent.top_p

            def generate(self, messages, **kwargs):
                # Convert smolagents messages to OpenAI format
                api_messages = []
                for msg in messages:
                    role = msg.role
                    content = msg.content

                    # Handle different content formats
                    if isinstance(content, list):
                        text = ""
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                text += item["text"]
                            else:
                                text += str(item)
                    else:
                        text = str(content)

                    # Map role to OpenAI format
                    if 'SYSTEM' in str(role).upper():
                        api_role = "system"
                    elif 'ASSISTANT' in str(role).upper():
                        api_role = "assistant"
                    else:
                        api_role = "user"

                    api_messages.append({"role": api_role, "content": text})

                # Call OpenAI API
                response_text = self.parent._call_openai_api(api_messages)

                # Create token usage (OpenAI provides this in response)
                token_usage = TokenUsage(
                    input_tokens=getattr(self.parent, '_last_input_tokens', 0),
                    output_tokens=getattr(self.parent, '_last_output_tokens', 0)
                )

                # Return ChatMessage
                response_message = ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=response_text
                )
                response_message.token_usage = token_usage

                return response_message

            def __call__(self, messages, **kwargs):
                return self.generate(messages, **kwargs)

        return MockSmolagentsModel(self)

    def _call_openai_api(self, messages: List[Dict[str, str]]) -> str:
        """Call the OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )

            # Store token usage for this call
            self._last_input_tokens = response.usage.prompt_tokens
            self._last_output_tokens = response.usage.completion_tokens

            # Accumulate token usage
            self._cumulative_input_tokens += response.usage.prompt_tokens
            self._cumulative_output_tokens += response.usage.completion_tokens

            # Handle reasoning tokens for models like o1
            if hasattr(response.usage, 'completion_tokens_details'):
                details = response.usage.completion_tokens_details
                if hasattr(details, 'reasoning_tokens') and details.reasoning_tokens:
                    self._cumulative_reasoning_tokens += details.reasoning_tokens

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise

    async def __call__(self, prompt: str) -> str:
        """Simple call: prompt -> output"""
        messages = [{"role": "user", "content": prompt}]
        return self._call_openai_api(messages)

    def get_cumulative_token_usage(self) -> Dict[str, int]:
        """Get cumulative token usage across all API calls.

        Returns:
            Dictionary with 'input_tokens', 'output_tokens', 'total_tokens', and optionally 'reasoning_tokens'
        """
        usage = {
            'input_tokens': self._cumulative_input_tokens,
            'output_tokens': self._cumulative_output_tokens,
            'total_tokens': self._cumulative_input_tokens + self._cumulative_output_tokens + self._cumulative_reasoning_tokens
        }

        if self._cumulative_reasoning_tokens > 0:
            usage['reasoning_tokens'] = self._cumulative_reasoning_tokens

        return usage

    def reset_token_usage(self):
        """Reset cumulative token counters."""
        self._cumulative_input_tokens = 0
        self._cumulative_output_tokens = 0
        self._cumulative_reasoning_tokens = 0

    def generate(self, messages: List[ChatMessage], **kwargs) -> ChatMessage:
        """smolagents interface: generate method"""
        return self.smolagents_model.generate(messages, **kwargs)

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        tool_specs: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None
    ):
        """Stream events"""
        try:
            # Convert to OpenAI API messages format
            api_messages = []

            # Add system message if provided
            if system_prompt:
                api_messages.append({"role": "system", "content": system_prompt})

            # Convert the messages
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                # Handle content formatting
                if isinstance(content, list):
                    content_text = ""
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            content_text += item["text"]
                        else:
                            content_text += str(item)
                else:
                    content_text = str(content)

                api_messages.append({"role": role, "content": content_text})

            # Stream from OpenAI API
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield {
                        "contentBlockDelta": {
                            "delta": {"text": chunk.choices[0].delta.content},
                            "contentBlockIndex": 0
                        }
                    }

        except Exception as e:
            logger.error(f"Error streaming from OpenAI API: {e}")
            raise


class AnthropicAPIModel:
    """Anthropic API model (direct API, not Bedrock) compatible with smolagents framework"""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        # Token usage tracking (cumulative across all calls)
        self._cumulative_input_tokens = 0
        self._cumulative_output_tokens = 0

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables or parameters")

        # Try to import anthropic
        try:
            import anthropic
            self.anthropic = anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            logger.info(f"Initialized Anthropic API model: {model}")
        except ImportError:
            raise ImportError("anthropic library is required for Anthropic API models. Install with: pip install anthropic")

        # Create smolagents-compatible interface
        self.smolagents_model = self._create_mock_smolagents_model()

    def _create_mock_smolagents_model(self):
        """Create a mock smolagents interface for compatibility"""
        class MockSmolagentsModel:
            def __init__(self, parent):
                self.parent = parent
                self.model_id = parent.model
                self.max_tokens = parent.max_tokens
                self.temperature = parent.temperature
                self.top_p = parent.top_p

            def generate(self, messages, **kwargs):
                # Convert smolagents messages to Anthropic format
                api_messages = []
                system_prompt = None

                for msg in messages:
                    role = msg.role
                    content = msg.content

                    # Handle different content formats
                    if isinstance(content, list):
                        text = ""
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                text += item["text"]
                            else:
                                text += str(item)
                    else:
                        text = str(content)

                    # Map role to Anthropic format
                    if 'SYSTEM' in str(role).upper():
                        system_prompt = text
                    elif 'ASSISTANT' in str(role).upper():
                        api_messages.append({"role": "assistant", "content": text})
                    else:
                        api_messages.append({"role": "user", "content": text})

                # Call Anthropic API
                response_text = self.parent._call_anthropic_api(api_messages, system_prompt)

                # Create token usage
                token_usage = TokenUsage(
                    input_tokens=getattr(self.parent, '_last_input_tokens', 0),
                    output_tokens=getattr(self.parent, '_last_output_tokens', 0)
                )

                # Return ChatMessage
                response_message = ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=response_text
                )
                response_message.token_usage = token_usage

                return response_message

            def __call__(self, messages, **kwargs):
                return self.generate(messages, **kwargs)

        return MockSmolagentsModel(self)

    def _call_anthropic_api(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """Call the Anthropic API"""
        try:
            # Build request parameters
            request_params = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p
            }

            # Add system prompt if provided
            if system_prompt:
                request_params["system"] = system_prompt

            response = self.client.messages.create(**request_params)

            # Store token usage for this call
            self._last_input_tokens = response.usage.input_tokens
            self._last_output_tokens = response.usage.output_tokens

            # Accumulate token usage
            self._cumulative_input_tokens += response.usage.input_tokens
            self._cumulative_output_tokens += response.usage.output_tokens

            return response.content[0].text

        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            raise

    async def __call__(self, prompt: str) -> str:
        """Simple call: prompt -> output"""
        messages = [{"role": "user", "content": prompt}]
        return self._call_anthropic_api(messages)

    def get_cumulative_token_usage(self) -> Dict[str, int]:
        """Get cumulative token usage across all API calls.

        Returns:
            Dictionary with 'input_tokens', 'output_tokens', 'total_tokens'
        """
        return {
            'input_tokens': self._cumulative_input_tokens,
            'output_tokens': self._cumulative_output_tokens,
            'total_tokens': self._cumulative_input_tokens + self._cumulative_output_tokens
        }

    def reset_token_usage(self):
        """Reset cumulative token counters."""
        self._cumulative_input_tokens = 0
        self._cumulative_output_tokens = 0

    def generate(self, messages: List[ChatMessage], **kwargs) -> ChatMessage:
        """smolagents interface: generate method"""
        return self.smolagents_model.generate(messages, **kwargs)

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        tool_specs: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None
    ):
        """Stream events"""
        try:
            # Convert to Anthropic API messages format
            api_messages = []

            # Convert the messages
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                # Skip system messages (handled separately)
                if role == "system":
                    if not system_prompt:
                        system_prompt = content
                    continue

                # Handle content formatting
                if isinstance(content, list):
                    content_text = ""
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            content_text += item["text"]
                        else:
                            content_text += str(item)
                else:
                    content_text = str(content)

                api_messages.append({"role": role, "content": content_text})

            # Build request parameters
            request_params = {
                "model": self.model,
                "messages": api_messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p
            }

            # Add system prompt if provided
            if system_prompt:
                request_params["system"] = system_prompt

            # Stream from Anthropic API
            with self.client.messages.stream(**request_params) as stream:
                for text in stream.text_stream:
                    yield {
                        "contentBlockDelta": {
                            "delta": {"text": text},
                            "contentBlockIndex": 0
                        }
                    }

        except Exception as e:
            logger.error(f"Error streaming from Anthropic API: {e}")
            raise


class AzureOpenAIModel:
    """Azure OpenAI API model compatible with smolagents framework.

    Supports two Azure resources via separate env vars:
        OpenAI models (gpt-*):   AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY
        Anthropic models (claude-*): AZURE_Anthropic_ENDPOINT + AZURE_Anthropic_API_KEY

    Falls back to explicit api_key/endpoint params if provided.
    """

    # Models that require max_completion_tokens instead of max_tokens
    _USE_MAX_COMPLETION_TOKENS = {"gpt-5.2", "gpt-5", "o1", "o1-mini", "o3", "o3-mini"}

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.api_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

        # Auto-select endpoint and key based on model name
        is_claude = "claude" in model.lower()
        if api_key:
            self.api_key = api_key
        elif is_claude:
            self.api_key = os.environ.get("AZURE_Anthropic_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY")
        else:
            self.api_key = os.environ.get("AZURE_OPENAI_API_KEY")

        if endpoint:
            self.endpoint = endpoint
        elif is_claude:
            self.endpoint = os.environ.get("AZURE_Anthropic_ENDPOINT") or os.environ.get("AZURE_OPENAI_ENDPOINT")
        else:
            self.endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")

        self._cumulative_input_tokens = 0
        self._cumulative_output_tokens = 0

        if not self.api_key:
            key_name = "AZURE_Anthropic_API_KEY" if is_claude else "AZURE_OPENAI_API_KEY"
            raise ValueError(f"{key_name} not found in environment variables or parameters")
        if not self.endpoint:
            ep_name = "AZURE_Anthropic_ENDPOINT" if is_claude else "AZURE_OPENAI_ENDPOINT"
            raise ValueError(f"{ep_name} not found in environment variables or parameters")

        self._is_claude = is_claude
        if is_claude:
            # Claude on Azure uses the Anthropic REST API path
            import requests as _requests
            self._requests = _requests
            logger.info(f"Initialized Azure Anthropic model: {model} at {self.endpoint}")
        else:
            try:
                import openai
                self.openai = openai
                self.client = openai.AzureOpenAI(
                    api_key=self.api_key,
                    azure_endpoint=self.endpoint,
                    api_version=self.api_version,
                    max_retries=10,
                    timeout=120.0,
                )
                logger.info(f"Initialized Azure OpenAI model: {model} at {self.endpoint}")
            except ImportError:
                raise ImportError("openai library is required for Azure OpenAI models. Install with: pip install openai")

        self.smolagents_model = self._create_mock_smolagents_model()

    def _create_mock_smolagents_model(self):
        class MockSmolagentsModel:
            def __init__(self, parent):
                self.parent = parent
                self.model_id = parent.model
                self.max_tokens = parent.max_tokens
                self.temperature = parent.temperature
                self.top_p = parent.top_p

            def generate(self, messages, **kwargs):
                api_messages = []
                for msg in messages:
                    role = msg.role
                    content = msg.content
                    if isinstance(content, list):
                        text = ""
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                text += item["text"]
                            else:
                                text += str(item)
                    else:
                        text = str(content)
                    if 'SYSTEM' in str(role).upper():
                        api_role = "system"
                    elif 'ASSISTANT' in str(role).upper():
                        api_role = "assistant"
                    else:
                        api_role = "user"
                    api_messages.append({"role": api_role, "content": text})
                response_text = self.parent._call_api(api_messages)
                token_usage = TokenUsage(
                    input_tokens=getattr(self.parent, '_last_input_tokens', 0),
                    output_tokens=getattr(self.parent, '_last_output_tokens', 0)
                )
                response_message = ChatMessage(role=MessageRole.ASSISTANT, content=response_text)
                response_message.token_usage = token_usage
                return response_message

            def __call__(self, messages, **kwargs):
                return self.generate(messages, **kwargs)

        return MockSmolagentsModel(self)

    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        if self._is_claude:
            return self._call_anthropic_api(messages)
        return self._call_openai_api(messages)

    def _call_anthropic_api(self, messages: List[Dict[str, str]]) -> str:
        """Call Claude on Azure via the Anthropic REST API path."""
        try:
            # Separate system prompt from messages
            system_prompt = None
            api_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    api_messages.append(msg)

            url = f"{self.endpoint.rstrip('/')}/anthropic/v1/messages?api-version=2025-04-01"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            }
            body = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": api_messages,
                "temperature": self.temperature,
            }
            if system_prompt:
                body["system"] = system_prompt

            resp = self._requests.post(url, headers=headers, json=body, timeout=300)
            resp.raise_for_status()
            data = resp.json()

            self._last_input_tokens = data.get("usage", {}).get("input_tokens", 0)
            self._last_output_tokens = data.get("usage", {}).get("output_tokens", 0)
            self._cumulative_input_tokens += self._last_input_tokens
            self._cumulative_output_tokens += self._last_output_tokens

            return data["content"][0]["text"]
        except Exception as e:
            logger.error(f"Error calling Azure Anthropic API: {e}")
            raise

    def _call_openai_api(self, messages: List[Dict[str, str]]) -> str:
        """Call OpenAI models on Azure via the OpenAI SDK."""
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
            if self.model in self._USE_MAX_COMPLETION_TOKENS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens

            response = self.client.chat.completions.create(**params)
            self._last_input_tokens = response.usage.prompt_tokens
            self._last_output_tokens = response.usage.completion_tokens
            self._cumulative_input_tokens += response.usage.prompt_tokens
            self._cumulative_output_tokens += response.usage.completion_tokens
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling Azure OpenAI API: {e}")
            raise

    async def __call__(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self._call_api(messages)

    def get_cumulative_token_usage(self) -> Dict[str, int]:
        return {
            'input_tokens': self._cumulative_input_tokens,
            'output_tokens': self._cumulative_output_tokens,
            'total_tokens': self._cumulative_input_tokens + self._cumulative_output_tokens,
        }

    def reset_token_usage(self):
        self._cumulative_input_tokens = 0
        self._cumulative_output_tokens = 0

    def generate(self, messages: List[ChatMessage], **kwargs) -> ChatMessage:
        return self.smolagents_model.generate(messages, **kwargs)


class GoogleGeminiModel:
    """Google Gemini API model compatible with smolagents framework.

    Authentication via environment variables (typically set in .env):
        GOOGLE_API_KEY: API key for Google Gemini
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("Google_Gemini_API_KEY")

        self._cumulative_input_tokens = 0
        self._cumulative_output_tokens = 0

        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables or parameters")

        try:
            from google import genai
            self.genai = genai
            self.client = genai.Client(api_key=self.api_key)
            logger.info(f"Initialized Google Gemini model: {model}")
        except ImportError:
            raise ImportError("google-genai library is required for Gemini models. Install with: pip install google-genai")

        self.smolagents_model = self._create_mock_smolagents_model()

    def _create_mock_smolagents_model(self):
        class MockSmolagentsModel:
            def __init__(self, parent):
                self.parent = parent
                self.model_id = parent.model
                self.max_tokens = parent.max_tokens
                self.temperature = parent.temperature
                self.top_p = parent.top_p

            def generate(self, messages, **kwargs):
                api_messages = []
                for msg in messages:
                    role = msg.role
                    content = msg.content
                    if isinstance(content, list):
                        text = ""
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                text += item["text"]
                            else:
                                text += str(item)
                    else:
                        text = str(content)
                    if 'SYSTEM' in str(role).upper():
                        api_role = "system"
                    elif 'ASSISTANT' in str(role).upper():
                        api_role = "model"
                    else:
                        api_role = "user"
                    api_messages.append({"role": api_role, "content": text})
                response_text = self.parent._call_api(api_messages)
                token_usage = TokenUsage(
                    input_tokens=getattr(self.parent, '_last_input_tokens', 0),
                    output_tokens=getattr(self.parent, '_last_output_tokens', 0)
                )
                response_message = ChatMessage(role=MessageRole.ASSISTANT, content=response_text)
                response_message.token_usage = token_usage
                return response_message

            def __call__(self, messages, **kwargs):
                return self.generate(messages, **kwargs)

        return MockSmolagentsModel(self)

    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        try:
            from google.genai import types

            # Separate system instruction from messages
            system_instruction = None
            contents = []
            for msg in messages:
                if msg["role"] == "system":
                    system_instruction = msg["content"]
                else:
                    role = "model" if msg["role"] == "model" else "user"
                    contents.append(types.Content(
                        role=role,
                        parts=[types.Part(text=msg["content"])]
                    ))

            config = types.GenerateContentConfig(
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                system_instruction=system_instruction,
            )

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

            # Track token usage
            if response.usage_metadata:
                self._last_input_tokens = response.usage_metadata.prompt_token_count or 0
                self._last_output_tokens = response.usage_metadata.candidates_token_count or 0
                self._cumulative_input_tokens += self._last_input_tokens
                self._cumulative_output_tokens += self._last_output_tokens
            else:
                self._last_input_tokens = 0
                self._last_output_tokens = 0

            return response.text

        except Exception as e:
            logger.error(f"Error calling Google Gemini API: {e}")
            raise

    async def __call__(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self._call_api(messages)

    def get_cumulative_token_usage(self) -> Dict[str, int]:
        return {
            'input_tokens': self._cumulative_input_tokens,
            'output_tokens': self._cumulative_output_tokens,
            'total_tokens': self._cumulative_input_tokens + self._cumulative_output_tokens,
        }

    def reset_token_usage(self):
        self._cumulative_input_tokens = 0
        self._cumulative_output_tokens = 0

    def generate(self, messages: List[ChatMessage], **kwargs) -> ChatMessage:
        return self.smolagents_model.generate(messages, **kwargs)


def get_model(model_id: str, **kwargs) -> Union[BedrockSmolagentsModel, OpenAIModel, AnthropicAPIModel, AzureOpenAIModel, GoogleGeminiModel]:
    """Get model instance for smolagents.

    Args:
        model_id:
            - "bedrock:model_name" for AWS Bedrock models
            - "openai:model_name" for OpenAI API models (e.g., "openai:gpt-4o")
            - "anthropic:model_name" for Anthropic API models (e.g., "anthropic:claude-sonnet-4-20250514")
            - "azure:model_name" for Azure OpenAI models (e.g., "azure:gpt-4.1-mini")
            - "google:model_name" for Google Gemini models (e.g., "google:gemini-2.5-flash")
        **kwargs: temperature, max_tokens, api_key, etc.

    Returns:
        Model instance compatible with smolagents

    Examples:
        model = get_model("bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0", temperature=0.7)
        model = get_model("openai:gpt-4o", temperature=0.7)
        model = get_model("azure:gpt-4.1-mini", temperature=0.7)
        model = get_model("google:gemini-2.5-flash", temperature=0.7, max_tokens=4096)
        model = get_model("anthropic:claude-sonnet-4-20250514", temperature=0.7)
    """
    if ":" not in model_id:
        raise ValueError(f"Invalid model_id: {model_id}. Format: provider:model_name")

    provider, model_name = model_id.split(":", 1)

    if provider == "bedrock":
        return BedrockSmolagentsModel(model=model_name, **kwargs)

    elif provider == "openai":
        logger.info(f"Creating OpenAI model: {model_name}")
        return OpenAIModel(model=model_name, **kwargs)

    elif provider == "anthropic":
        logger.info(f"Creating Anthropic API model: {model_name}")
        return AnthropicAPIModel(model=model_name, **kwargs)

    elif provider == "azure":
        logger.info(f"Creating Azure OpenAI model: {model_name}")
        return AzureOpenAIModel(model=model_name, **kwargs)

    elif provider == "google":
        logger.info(f"Creating Google Gemini model: {model_name}")
        return GoogleGeminiModel(model=model_name, **kwargs)

    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: 'bedrock', 'openai', 'anthropic', 'azure', 'google'"
        )
