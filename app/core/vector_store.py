"""
Vector Store Module
Handles storage and retrieval of document embeddings using ChromaDB.
"""

import os
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from app.config import get_settings

settings = get_settings()


class VectorStoreService:
    """
    Service for managing vector embeddings storage and similarity search.
    Uses ChromaDB as the underlying vector database.
    """
    
    _instance = None
    _vector_store = None
    
    def __new__(cls, persist_directory: str = None):
        """Singleton pattern to reuse vector store instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, persist_directory: str = None):
        """
        Initialize the vector store service.
        """
        if self._initialized:
            return
            
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        self.collection_name = settings.collection_name
        
        # Ensure persist directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize embedding service
        self._embedding_function = None
        
        self._initialized = True
    
    def _get_embedding_function(self):
        """Lazy load embedding function."""
        if self._embedding_function is None:
            from app.core.embeddings import EmbeddingService
            embedding_service = EmbeddingService()
            self._embedding_function = embedding_service.get_embeddings_model()
        return self._embedding_function
    
    def _get_vector_store(self) -> Chroma:
        """Get or create the vector store instance."""
        if VectorStoreService._vector_store is None:
            VectorStoreService._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self._get_embedding_function(),
                persist_directory=self.persist_directory
            )
        return VectorStoreService._vector_store
    
    def add_documents(self, documents: List[Document]) -> int:
        """
        Add documents to the vector store.
        """
        if not documents:
            return 0
        
        vector_store = self._get_vector_store()
        
        # Add documents to vector store
        vector_store.add_documents(documents)
        
        return len(documents)
    
    def similarity_search(
        self,
        query: str,
        k: int = None
    ) -> List[Document]:
        """
        Perform similarity search to find relevant documents.
        """
        k = k or settings.top_k_results
        
        vector_store = self._get_vector_store()
        results = vector_store.similarity_search(
            query=query,
            k=k
        )
        
        return results
    
    def similarity_search_with_scores(
        self,
        query: str,
        k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with relevance scores.
        """
        k = k or settings.top_k_results
        
        vector_store = self._get_vector_store()
        results = vector_store.similarity_search_with_score(
            query=query,
            k=k
        )
        
        return results
    
    def get_document_count(self) -> int:
        """
        Get the total number of documents in the vector store.
        """
        try:
            vector_store = self._get_vector_store()
            collection = vector_store._collection
            return collection.count()
        except Exception:
            return 0
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        """
        try:
            vector_store = self._get_vector_store()
            vector_store._client.delete_collection(self.collection_name)
            VectorStoreService._vector_store = None
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False