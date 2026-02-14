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


# VectorStoreService - Manages document embeddings storage and retrieval
# Uses Singleton pattern to maintain single ChromaDB connection
class VectorStoreService:
    """
    Service for managing vector embeddings storage and similarity search.
    Uses ChromaDB as the underlying vector database.
    """
    
    # Class-level variables for Singleton pattern
    # Ensures single database connection across application
    _instance = None
    _vector_store = None
    
    # __new__() - Implements Singleton pattern for vector store
    # Prevents multiple database connections and ensures data consistency
    def __new__(cls, persist_directory: str = None):
        """Singleton pattern to reuse vector store instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    # __init__() - Sets up directory paths and collection name
    # _initialized flag prevents re-running setup on subsequent calls
    def __init__(self, persist_directory: str = None):
        """
        Initialize the vector store service.
        """
        if self._initialized:
            return
        
        # "or" pattern - Uses settings default if not specified
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        self.collection_name = settings.collection_name
        
        # os.makedirs() - Creates storage directory if not exists
        # exist_ok=True prevents error if directory already exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Placeholder for lazy-loaded embedding function
        self._embedding_function = None
        
        self._initialized = True
    
    # _get_embedding_function() - Lazy loads embedding model
    # Avoids circular imports and defers API initialization
    def _get_embedding_function(self):
        """Lazy load embedding function."""
        if self._embedding_function is None:
            from app.core.embeddings import EmbeddingService
            embedding_service = EmbeddingService()
            self._embedding_function = embedding_service.get_embeddings_model()
        return self._embedding_function
    
    # _get_vector_store() - Creates or returns existing ChromaDB instance
    # Lazy initialization ensures database is created only when needed
    def _get_vector_store(self) -> Chroma:
        """Get or create the vector store instance."""
        if VectorStoreService._vector_store is None:
            # Chroma() - LangChain wrapper for ChromaDB vector database
            # persist_directory enables data persistence across restarts
            VectorStoreService._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self._get_embedding_function(),
                persist_directory=self.persist_directory
            )
        return VectorStoreService._vector_store
    
    # add_documents() - Embeds and stores documents in vector database
    # Returns count of documents added for confirmation
    def add_documents(self, documents: List[Document]) -> int:
        """
        Add documents to the vector store.
        """
        # Early return pattern - Handles empty input gracefully
        if not documents:
            return 0
        
        vector_store = self._get_vector_store()
        
        # add_documents() - Auto-embeds text and stores vectors + metadata
        vector_store.add_documents(documents)
        
        return len(documents)
    
    # similarity_search() - Finds semantically similar documents to query
    # Returns top-k most relevant documents based on vector distance
    def similarity_search(
        self,
        query: str,
        k: int = None
    ) -> List[Document]:
        """
        Perform similarity search to find relevant documents.
        """
        # Default to settings if k not provided
        k = k or settings.top_k_results
        
        vector_store = self._get_vector_store()
        # similarity_search() - Embeds query and finds nearest vectors
        results = vector_store.similarity_search(
            query=query,
            k=k
        )
        
        return results
    
    # similarity_search_with_scores() - Returns documents with relevance scores
    # Scores help rank results and filter low-confidence matches
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
        # Returns list of (Document, score) tuples for transparency
        results = vector_store.similarity_search_with_score(
            query=query,
            k=k
        )
        
        return results
    
    # get_document_count() - Returns total indexed documents
    # Useful for health checks and monitoring dashboard
    def get_document_count(self) -> int:
        """
        Get the total number of documents in the vector store.
        """
        # try-except - Handles case when collection doesn't exist yet
        try:
            vector_store = self._get_vector_store()
            # _collection - Accesses underlying ChromaDB collection directly
            collection = vector_store._collection
            return collection.count()
        except Exception:
            return 0
    
    # clear_collection() - Deletes all documents from collection
    # Resets _vector_store to None to force fresh initialization
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        """
        try:
            vector_store = self._get_vector_store()
            # _client.delete_collection() - ChromaDB native delete operation
            vector_store._client.delete_collection(self.collection_name)
            # Reset singleton to allow recreation of empty collection
            VectorStoreService._vector_store = None
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False