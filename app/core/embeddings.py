"""
Embeddings Module
Handles generation of vector embeddings using Google Generative AI.
"""

from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import get_settings

settings = get_settings()


class EmbeddingService:
    """
    Service for generating text embeddings using Google's embedding model.
    """
    
    _instance = None
    _embeddings = None
    
    def __new__(cls, model_name: str = None):
        """Singleton pattern to reuse embeddings instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding service with specified model.
        """
        if self._initialized:
            return
            
        self.model_name = model_name or settings.embedding_model
        self._initialized = True
    
    def _get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Lazy load embeddings model."""
        if EmbeddingService._embeddings is None:
            EmbeddingService._embeddings = GoogleGenerativeAIEmbeddings(
                model=self.model_name,
                google_api_key=settings.google_api_key
            )
        return EmbeddingService._embeddings
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single text.
        """
        embeddings = self._get_embeddings()
        return embeddings.embed_query(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for multiple texts.
        """
        embeddings = self._get_embeddings()
        return embeddings.embed_documents(texts)
    
    def get_embeddings_model(self) -> GoogleGenerativeAIEmbeddings:
        """
        Return the underlying embeddings model for direct use.
        """
        return self._get_embeddings()