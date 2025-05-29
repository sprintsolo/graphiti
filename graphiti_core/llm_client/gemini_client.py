"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import logging
import typing
import os
from google.auth import credentials
from google.oauth2 import service_account

from google import genai  # type: ignore
from google.genai import types  # type: ignore
from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import RateLimitError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'gemini-2.0-flash'


class GeminiClient(LLMClient):
    """
    GeminiClient is a client class for interacting with Google's Gemini language models.

    This class extends the LLMClient and provides methods to initialize the client
    and generate responses from the Gemini language model.

    Attributes:
        model (str): The model name to use for generating responses.
        temperature (float): The temperature to use for generating responses.
        max_tokens (int): The maximum number of tokens to generate in a response.

    Methods:
        __init__(config: LLMConfig | None = None, cache: bool = False):
            Initializes the GeminiClient with the provided configuration and cache setting.

        _generate_response(messages: list[Message]) -> dict[str, typing.Any]:
            Generates a response from the language model based on the provided messages.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """
        Initialize the GeminiClient with the provided configuration and cache setting.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, temperature, and max tokens.
            cache (bool): Whether to use caching for responses. Defaults to False.
        """
        if config is None:
            config = LLMConfig()

        super().__init__(config, cache)

        self.model = config.model
        
        # Configure credentials for Vertex AI or API key
        credentials_obj = None
        use_vertexai = False
        
        # Define required scopes for Vertex AI
        scopes = ['https://www.googleapis.com/auth/cloud-platform']
        
        # Check for service account credentials
        if hasattr(config, 'service_account_key_json') and config.service_account_key_json:
            credentials_obj = service_account.Credentials.from_service_account_info(
                config.service_account_key_json,
                scopes=scopes
            )
            use_vertexai = True
        elif hasattr(config, 'service_account_key_path') and config.service_account_key_path:
            credentials_obj = service_account.Credentials.from_service_account_file(
                config.service_account_key_path,
                scopes=scopes
            )
            use_vertexai = True
        elif os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            credentials_obj = service_account.Credentials.from_service_account_file(
                os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
                scopes=scopes
            )
            use_vertexai = True
        
        # Ensure credentials are properly scoped
        if credentials_obj and not credentials_obj.valid:
            try:
                from google.auth.transport.requests import Request
                credentials_obj.refresh(Request())
            except Exception:
                # If refresh fails, try without explicit credentials
                credentials_obj = None
        
        # Configure the Gemini API
        if use_vertexai and credentials_obj:
            # Use Vertex AI with service account
            project_id = getattr(config, 'project_id', None) or os.getenv('GOOGLE_CLOUD_PROJECT', 'gen-lang-client-0768783796')
            location = getattr(config, 'location', 'us-central1')
            
            self.client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
                credentials=credentials_obj
            )
        else:
            # Use API key
            self.client = genai.Client(
                api_key=config.api_key,
            )
        
        self.max_tokens = max_tokens

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """
        Generate a response from the Gemini language model.

        Args:
            messages (list[Message]): A list of messages to send to the language model.
            response_model (type[BaseModel] | None): An optional Pydantic model to parse the response into.
            max_tokens (int): The maximum number of tokens to generate in the response.

        Returns:
            dict[str, typing.Any]: The response from the language model.

        Raises:
            RateLimitError: If the API rate limit is exceeded.
            RefusalError: If the content is blocked by the model.
            Exception: If there is an error generating the response.
        """
        import asyncio
        
        # 재시도 로직을 위한 설정
        max_retries = 5
        retry_count = 0
        consecutive_errors = 0
        
        while retry_count < max_retries:
            try:
                gemini_messages: list[types.Content] = []
                # If a response model is provided, add schema for structured output
                system_prompt = ''
                if response_model is not None:
                    # Get the schema from the Pydantic model

                    # Create instruction to output in the desired JSON format
                    system_prompt += (
                        f'Do not include any explanatory text before or after the JSON. '
                        'Any extracted information should be returned in the same language as it was written in.'
                    )

                # Add messages content
                # First check for a system message
                if messages and messages[0].role == 'system':
                    system_prompt = f'{messages[0].content}\n\n {system_prompt}'
                    messages = messages[1:]

                # Add the rest of the messages
                for m in messages:
                    m.content = self._clean_input(m.content)
                    gemini_messages.append(
                        types.Content(role=m.role, parts=[types.Part.from_text(text=m.content)])
                    )


                # Create generation config
                generation_config = types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=max_tokens or self.max_tokens,
                    response_mime_type='application/json' if response_model else None,
                    response_schema=response_model if response_model else None,
                    system_instruction=system_prompt,
                )

                # Generate content using the simple string approach
                response = await self.client.aio.models.generate_content(
                    model=self.model or DEFAULT_MODEL,
                    contents=gemini_messages,
                    config=generation_config,
                )
                
                # 성공시 연속 오류 카운터 초기화
                consecutive_errors = 0
                
                # If this was a structured output request, parse the response into the Pydantic model
                if response_model is not None:
                    try:
                        validated_model = response_model.model_validate(json.loads(response.text))

                        # Return as a dictionary for API consistency
                        return validated_model.model_dump()
                    except Exception as e:
                        raise Exception(f'Failed to parse structured response: {e}') from e

                # Otherwise, return the response text as a dictionary
                return {'content': response.text}

            except Exception as e:
                # 429 오류나 rate limit, quota 관련 메시지 처리
                error_message = str(e).lower()
                if ('429' in error_message or 
                    'rate limit' in error_message or 
                    'quota' in error_message or 
                    'resource_exhausted' in error_message):
                    
                    retry_count += 1
                    consecutive_errors += 1
                    
                    # 지수 백오프 적용 (30초, 60초, 120초 등)
                    wait_time = 30 * (2 ** (consecutive_errors - 1))
                    if wait_time > 300:  # 최대 5분까지만 대기
                        wait_time = 300
                    
                    logger.warning(f"API rate limit reached. Waiting for {wait_time} seconds before retry {retry_count}/{max_retries}. Error: {str(e)}")
                    await asyncio.sleep(wait_time)
                else:
                    # 다른 오류는 즉시 예외 발생
                    logger.error(f'Error in generating LLM response: {e}')
                    raise
        
        # 최대 재시도 횟수 초과 시 RateLimitError 발생
        if retry_count == max_retries:
            logger.error(f"Failed to generate response after {max_retries} retries due to rate limits")
            raise RateLimitError(f"Maximum retry attempts ({max_retries}) exceeded for rate limit")

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """
        Generate a response from the Gemini language model.
        This method overrides the parent class method to provide a direct implementation.

        Args:
            messages (list[Message]): A list of messages to send to the language model.
            response_model (type[BaseModel] | None): An optional Pydantic model to parse the response into.
            max_tokens (int): The maximum number of tokens to generate in the response.

        Returns:
            dict[str, typing.Any]: The response from the language model.
        """
        if max_tokens is None:
            max_tokens = self.max_tokens

        # Call the internal _generate_response method
        return await self._generate_response(
            messages=messages,
            response_model=response_model,
            max_tokens=max_tokens,
            model_size=model_size,
        )
