"""
Embeddings Module
Handles generation of vector embeddings using Google Generative AI.
"""

from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import get_settings

# Load settings for API key and model configuration
settings = get_settings()


# EmbeddingService - Converts text to vector embeddings for similarity search
# Uses Singleton pattern to avoid creating multiple API client instances
class EmbeddingService:
    """
    Service for generating text embeddings using Google's embedding model.
    """
    
    # Class-level variables for Singleton pattern
    # Shared across all instances to maintain single state
    _instance = None
    _embeddings = None
    
    # __new__() - Controls object creation, implements Singleton pattern
    # Returns existing instance if already created, otherwise creates new one
    def __new__(cls, model_name: str = None):
        """Singleton pattern to reuse embeddings instance."""
        if cls._instance is None:
            # super().__new__(cls) - Creates actual new instance only once
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    # __init__() - Initializes instance attributes, runs on every call
    # _initialized flag prevents re-initialization on subsequent calls
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding service with specified model.
        """
        # Skip if already initialized (Singleton guard)
        if self._initialized:
            return
        
        # "or" pattern - Uses settings default if no model specified    
        self.model_name = model_name or settings.embedding_model
        self._initialized = True
    
    # _get_embeddings() - Lazy loads the embedding model on first use
    # Defers expensive API client creation until actually needed
    def _get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Lazy load embeddings model."""
        if EmbeddingService._embeddings is None:
            # GoogleGenerativeAIEmbeddings - LangChain wrapper for Google's embedding API
            # Converts text to high-dimensional vectors for semantic similarity
            EmbeddingService._embeddings = GoogleGenerativeAIEmbeddings(
                model=self.model_name,
                google_api_key=settings.google_api_key
            )
        return EmbeddingService._embeddings
    
    # embed_text() - Generates embedding for single text (used for queries)
    # Returns list of floats representing semantic meaning as vector
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single text.
        """
        embeddings = self._get_embeddings()
        # embed_query() - Optimized for search queries
        return embeddings.embed_query(text)
    
    # embed_texts() - Batch embedding for multiple texts (used for documents)
    # More efficient than calling embed_text() in a loop
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for multiple texts.
        """
        embeddings = self._get_embeddings()
        # embed_documents() - Optimized for document indexing
        return embeddings.embed_documents(texts)
    
    # get_embeddings_model() - Exposes underlying model for external use
    # Allows direct integration with vector stores like ChromaDB
    def get_embeddings_model(self) -> GoogleGenerativeAIEmbeddings:
        """
        Return the underlying embeddings model for direct use.
        """
        return self._get_embeddings()