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

from collections.abc import Iterable
import os
import json
from google.auth import credentials
from google.oauth2 import service_account

from google import genai  # type: ignore
from google.genai import types  # type: ignore
from pydantic import Field

from .client import EmbedderClient, EmbedderConfig

DEFAULT_EMBEDDING_MODEL = 'embedding-001'


class GeminiEmbedderConfig(EmbedderConfig):
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    api_key: str | None = None
    service_account_key_path: str | None = None
    service_account_key_json: dict | None = None
    project_id: str | None = None
    location: str = Field(default='us-central1')


class GeminiEmbedder(EmbedderClient):
    """
    Google Gemini Embedder Client
    """

    def __init__(self, config: GeminiEmbedderConfig | None = None):
        if config is None:
            config = GeminiEmbedderConfig()
        self.config = config

        # Configure credentials for Vertex AI
        credentials_obj = None
        
        # Define required scopes for Vertex AI
        scopes = ['https://www.googleapis.com/auth/cloud-platform']
        
        if config.service_account_key_json:
            # Use service account key JSON directly
            credentials_obj = service_account.Credentials.from_service_account_info(
                config.service_account_key_json,
                scopes=scopes
            )
        elif config.service_account_key_path:
            # Use service account key file path
            credentials_obj = service_account.Credentials.from_service_account_file(
                config.service_account_key_path,
                scopes=scopes
            )
        elif os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            # Use environment variable for service account key file
            credentials_obj = service_account.Credentials.from_service_account_file(
                os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
                scopes=scopes
            )
        
        # Ensure credentials are properly scoped
        if credentials_obj and not credentials_obj.valid:
            try:
                from google.auth.transport.requests import Request
                credentials_obj.refresh(Request())
            except Exception:
                # If refresh fails, try without explicit credentials
                credentials_obj = None
        
        # Get project ID
        project_id = config.project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
        
        # Configure the Gemini API with Vertex AI
        if credentials_obj:
            self.client = genai.Client(
                vertexai=True, 
                project=project_id, 
                location=config.location,
                credentials=credentials_obj
            )
        else:
            # Fallback to default credentials (ADC)
            self.client = genai.Client(
                vertexai=True, 
                project=project_id, 
                location=config.location
            )

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        """
        Create embeddings for the given input data using Google's Gemini embedding model.

        Args:
            input_data: The input data to create embeddings for. Can be a string, list of strings,
                       or an iterable of integers or iterables of integers.

        Returns:
            A list of floats representing the embedding vector.
        """
        # Generate embeddings
        result = await self.client.aio.models.embed_content(
            model=self.config.embedding_model or DEFAULT_EMBEDDING_MODEL,
            contents=[input_data],
            config=types.EmbedContentConfig(output_dimensionality=self.config.embedding_dim,task_type="RETRIEVAL_DOCUMENT"),
        )

        return result.embeddings[0].values

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        # Generate embeddings
        result = await self.client.aio.models.embed_content(
            model=self.config.embedding_model or DEFAULT_EMBEDDING_MODEL,
            contents=input_data_list,
            config=types.EmbedContentConfig(output_dimensionality=self.config.embedding_dim),
        )

        return [embedding.values for embedding in result.embeddings]

