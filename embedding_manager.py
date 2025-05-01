"""
Embedding Manager for GitHub Stars Analyzer.
This module provides embedding functionality using different providers (OpenAI, Azure OpenAI, Ollama).
"""
from typing import List, Dict, Any, Optional, Union
import json
import os
import requests
import numpy as np
from abc import ABC, abstractmethod

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def get_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings for the given text
        
        Args:
            text: The text to embed
            
        Returns:
            A list of floating point numbers representing the embedding
        """
        pass
    
    @abstractmethod
    def get_dimensions(self) -> int:
        """
        Get the dimensionality of the embeddings
        
        Returns:
            Number of dimensions in the embedding vector
        """
        pass

class SentenceTransformerProvider(EmbeddingProvider):
    """Use sentence-transformers for embeddings (CPU-only mode)"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the sentence-transformer provider
        
        Args:
            model_name: Name of the sentence transformer model
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            # Force CPU mode to prevent GPU usage
            self.model = self.model.to('cpu')
        except ImportError:
            raise ImportError(
                "sentence-transformers package is required for SentenceTransformerProvider. "
                "Install it with 'pip install sentence-transformers'."
            )
    
    def get_embeddings(self, text: str) -> List[float]:
        """Generate embeddings using sentence-transformers"""
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def get_dimensions(self) -> int:
        """Get the dimensionality of the embeddings"""
        return self.model.get_sentence_embedding_dimension()

class OpenAIProvider(EmbeddingProvider):
    """Use OpenAI API for embeddings"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-ada-002"):
        """
        Initialize the OpenAI provider
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            model: Model name to use for embeddings
        """
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.model = model
            
            # Check dimensions based on model
            if "ada" in model:
                self._dimensions = 1536
            elif "3-small" in model:
                self._dimensions = 1536
            elif "3-large" in model:
                self._dimensions = 3072
            else:
                # Default for text-embedding-ada-002
                self._dimensions = 1536
                
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAIProvider. "
                "Install it with 'pip install openai>=1.0.0'."
            )
    
    def get_embeddings(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI API"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
    
    def get_dimensions(self) -> int:
        """Get the dimensionality of the embeddings"""
        return self._dimensions

class AzureOpenAIProvider(EmbeddingProvider):
    """Use Azure OpenAI API for embeddings"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        api_version: str = "2023-05-15"
    ):
        """
        Initialize the Azure OpenAI provider
        
        Args:
            api_key: Azure OpenAI API key (defaults to AZURE_OPENAI_API_KEY environment variable)
            endpoint: Azure OpenAI endpoint (defaults to AZURE_OPENAI_ENDPOINT environment variable)
            deployment: Azure OpenAI deployment name (defaults to AZURE_OPENAI_EMBEDDING_DEPLOYMENT environment variable)
            api_version: Azure OpenAI API version
        """
        try:
            from azure.identity import DefaultAzureCredential
            from openai import AzureOpenAI
            
            self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
            self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            self.deployment = deployment or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
              # Check for specific missing parameters and provide clear error messages
            missing_params = []
            if not self.endpoint:
                missing_params.append("endpoint")
            if not self.deployment:
                missing_params.append("deployment")
                
            if missing_params:
                missing_str = ", ".join(missing_params)
                raise ValueError(
                    f"Azure OpenAI requires the following parameter(s): {missing_str}. "
                    f"Please provide these values either as arguments or set the corresponding "
                    f"environment variables (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_EMBEDDING_DEPLOYMENT)."
                )
            
            if self.api_key:
                # Use API key authentication
                self.client = AzureOpenAI(
                    api_key=self.api_key,
                    api_version=api_version,
                    azure_endpoint=self.endpoint
                )
            else:
                # Try to use Azure DefaultAzureCredential
                credentials = DefaultAzureCredential()
                self.client = AzureOpenAI(
                    api_version=api_version,
                    azure_endpoint=self.endpoint,
                    azure_ad_token_provider=credentials
                )
                
            # Default dimensions for text-embedding-ada-002
            self._dimensions = 1536
                
        except ImportError:
            raise ImportError(
                "openai and azure-identity packages are required for AzureOpenAIProvider. "
                "Install them with 'pip install openai>=1.0.0 azure-identity'."
            )
    
    def get_embeddings(self, text: str) -> List[float]:
        """Generate embeddings using Azure OpenAI API"""
        response = self.client.embeddings.create(
            deployment_id=self.deployment,
            input=text
        )
        return response.data[0].embedding
    
    def get_dimensions(self) -> int:
        """Get the dimensionality of the embeddings"""
        return self._dimensions

class OllamaProvider(EmbeddingProvider):
    """Use Ollama API for embeddings"""
    
    def __init__(self, host: str = "http://localhost:11434", model: str = "mxbai-embed-large"):
        """
        Initialize the Ollama provider
        
        Args:
            host: Ollama API host URL
            model: Ollama model name (default: mxbai-embed-large, a specialized embedding model)
        """
        self.host = host
        self.model = model
        self.api_url = f"{host.rstrip('/')}/api/embeddings"
        # Set dimensions based on the model
        if model == "mxbai-embed-large":
            self._dimensions = 1024  # mxbai-embed-large produces 1024 dimensions
        else:
            # Default for other Ollama models
            self._dimensions = 4096
        
        # Test connection
        try:
            self._test_connection()
        except requests.RequestException as e:
            print(f"Warning: Could not connect to Ollama at {host}: {e}")
    
    def _test_connection(self):
        """Test connection to Ollama API"""
        response = requests.get(f"{self.host.rstrip('/')}/api/version")
        if response.status_code != 200:
            raise ConnectionError(f"Could not connect to Ollama at {self.host}")
    
    def get_embeddings(self, text: str) -> List[float]:
        """Generate embeddings using Ollama API"""
        payload = {
            "model": self.model,
            "prompt": text
        }
        
        response = requests.post(self.api_url, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error: {response.text}")
        
        return response.json()["embedding"]
    
    def get_dimensions(self) -> int:
        """Get the dimensionality of the embeddings"""
        return self._dimensions

class EmbeddingManager:
    """
    Manager class for handling embeddings from different providers
    """
    
    def __init__(self, provider_type: str = "sentence_transformer", **kwargs):
        """
        Initialize the embedding manager
        
        Args:
            provider_type: Type of provider to use ('openai', 'azure', 'ollama', 'sentence_transformer')
            **kwargs: Additional arguments to pass to the provider constructor
        """
        self.provider_type = provider_type.lower()
        self.provider = self._create_provider(**kwargs)
    
    def _create_provider(self, **kwargs) -> EmbeddingProvider:
        """Create and return the appropriate embedding provider"""
        if self.provider_type == "openai":
            return OpenAIProvider(**kwargs)
        elif self.provider_type in ("azure", "azure_openai"):
            return AzureOpenAIProvider(**kwargs)
        elif self.provider_type == "ollama":
            return OllamaProvider(**kwargs)
        elif self.provider_type == "sentence_transformer":
            return SentenceTransformerProvider(**kwargs)
        else:
            raise ValueError(f"Unknown provider type: {self.provider_type}")
    
    def get_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings for the given text
        
        Args:
            text: The text to embed
            
        Returns:
            A list of floating point numbers representing the embedding
        """
        return self.provider.get_embeddings(text)
    
    def get_dimensions(self) -> int:
        """
        Get the dimensionality of the embeddings
        
        Returns:
            Number of dimensions in the embedding vector
        """
        return self.provider.get_dimensions()
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        vec1 = np.array(embedding1).reshape(1, -1)
        vec2 = np.array(embedding2).reshape(1, -1)
        
        return float(cosine_similarity(vec1, vec2)[0][0])
