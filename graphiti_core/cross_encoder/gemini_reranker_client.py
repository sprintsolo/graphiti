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

import asyncio
import logging
import os
from typing import Any, List

import numpy as np
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from google.auth import credentials
from google.oauth2 import service_account

from ..helpers import semaphore_gather
from ..llm_client import LLMConfig, RateLimitError
from ..prompts import Message
from .client import CrossEncoderClient

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'gemini-2.5-flash-preview-05-20'

class GeminiRerankerClient(CrossEncoderClient):
    def __init__(
        self,
        config: LLMConfig | None = None,
        client: Any = None,
    ):
        """
        Initialize the GeminiRerankerClient with the provided configuration and client.

        This reranker uses the Gemini API to run a simple boolean classifier prompt concurrently
        for each passage. Confidence scores are used to rank the passages.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, etc.
            client (Any | None): An optional client instance to use. If not provided, a new genai Client is created.
        """
        if config is None:
            config = LLMConfig()

        self.config = config
        if client is None:
            # Configure credentials for Vertex AI or API key
            credentials_obj = None
            use_vertexai = False
            
            # Check for service account credentials
            if hasattr(config, 'service_account_key_json') and config.service_account_key_json:
                credentials_obj = service_account.Credentials.from_service_account_info(
                    config.service_account_key_json
                )
                use_vertexai = True
            elif hasattr(config, 'service_account_key_path') and config.service_account_key_path:
                credentials_obj = service_account.Credentials.from_service_account_file(
                    config.service_account_key_path
                )
                use_vertexai = True
            elif os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
                credentials_obj = service_account.Credentials.from_service_account_file(
                    os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
                )
                use_vertexai = True
            
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
                self.client = genai.Client(api_key=config.api_key)
        else:
            self.client = client
            
        self.model = config.model or DEFAULT_MODEL
        
    async def _call_gemini_with_retry(self, content: types.Content, max_retries: int = 5) -> dict:
        """Helper method to call Gemini API with retry logic for rate limits"""
        retry_count = 0
        consecutive_errors = 0
        
        while retry_count < max_retries:
            try:
                generation_config = types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=1,
                    # Gemini doesn't have direct logit_bias equivalent
                    # We'll rely on the prompt to constrain to True/False
                )
                
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=[content],
                    config=generation_config,
                )
                
                # Successful response
                return {"text": response.text, "success": True}
                
            except ClientError as e:
                error_message = str(e).lower()
                # Check for rate limit related errors
                if ('429' in error_message or 
                    'rate limit' in error_message or 
                    'quota' in error_message or 
                    'resource_exhausted' in error_message):
                    
                    retry_count += 1
                    consecutive_errors += 1
                    
                    # Exponential backoff (30s, 60s, 120s, etc.)
                    wait_time = 30 * (2 ** (consecutive_errors - 1))
                    if wait_time > 300:  # Max 5 minutes wait
                        wait_time = 300
                    
                    logger.warning(f"API rate limit reached. Waiting for {wait_time} seconds before retry {retry_count}/{max_retries}. Error: {str(e)}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f'Error in Gemini API call: {e}')
                    raise
            except Exception as e:
                logger.error(f'Unexpected error in Gemini API call: {e}')
                raise
        
        # If we've exhausted retries
        raise RateLimitError(f"Maximum retry attempts ({max_retries}) exceeded for rate limit")

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        """
        Rank passages based on their relevance to the query using Gemini API.
        
        Args:
            query (str): The query to compare passages against
            passages (list[str]): List of passages to rank
            
        Returns:
            list[tuple[str, float]]: Ranked passages with relevance scores
        """
        # Prepare contents list for each passage
        gemini_contents_list = []
        
        for passage in passages:
            content = types.Content(
                role="user",
                parts=[types.Part.from_text(
                    text=f"""
                    Respond with ONLY "True" or "False" (single word).
                    Is the following PASSAGE relevant to the QUERY?
                    
                    <PASSAGE>
                    {passage}
                    </PASSAGE>
                    
                    <QUERY>
                    {query}
                    </QUERY>
                    """
                )]
            )
            gemini_contents_list.append(content)
        
        try:
            # Call Gemini API for each passage in parallel
            responses = await semaphore_gather(
                *[self._call_gemini_with_retry(content) for content in gemini_contents_list]
            )
            
            # Process responses and calculate scores
            scores: List[float] = []
            for response in responses:
                if not response.get("success", False):
                    continue
                
                # Clean the response text and check if it's "True"
                response_text = response["text"].strip().lower()
                
                # Assign score based on the response
                if "true" in response_text:
                    # High confidence for relevant passages
                    scores.append(0.95)
                elif "false" in response_text:
                    # Low score for irrelevant passages
                    scores.append(0.05)
                else:
                    # Middle score for ambiguous responses
                    scores.append(0.5)
            
            # Create results with passages and scores
            results = [(passage, score) for passage, score in zip(passages, scores, strict=True)]
            
            # Sort by score in descending order
            results.sort(reverse=True, key=lambda x: x[1])
            return results
            
        except RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error(f'Error in ranking passages: {e}')
            raise
