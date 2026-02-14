"""
Pydantic schemas for API request and response validation.
Ensures type safety and automatic documentation generation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


# DocumentChunk - Schema representing a piece of retrieved document
# Used in query responses to show source documents with relevance scores
class DocumentChunk(BaseModel):
    """Represents a chunk of processed document."""
    content: str                              # Text content of the chunk
    metadata: dict                            # Source file, chunk index, etc.
    relevance_score: Optional[float] = None   # Similarity score (0-1), optional for flexibility


# UploadResponse - Schema for document upload API response
# Confirms successful processing with details about created chunks
class UploadResponse(BaseModel):
    """Response model for document upload endpoint."""
    success: bool
    message: str
    filename: str
    chunks_created: int
    # Field(default_factory=) - Calls function at runtime for each instance
    # Ensures each response gets current timestamp, not import-time value
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Config inner class - Provides example for Swagger documentation
    # json_schema_extra populates the "Example Value" in API docs
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Document processed successfully",
                "filename": "products.pdf",
                "chunks_created": 15,
                "timestamp": "2024-01-15T10:30:00"
            }
        }


# QueryRequest - Schema for validating incoming query requests
# Field() adds validation constraints and documentation
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    # Field(...) - Ellipsis makes field required; min/max enforce length limits
    query: str = Field(..., min_length=3, max_length=500)
    # ge=greater/equal, le=less/equal - Limits top_k between 1 and 10
    top_k: Optional[int] = Field(default=3, ge=1, le=10)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the best smartphones under $500?",
                "top_k": 3
            }
        }


# QueryResponse - Schema for RAG query results
# Includes answer, source documents, and performance metrics
class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    query: str                                # Original user query (echo back)
    answer: str                               # LLM-generated response
    source_documents: List[DocumentChunk]     # Retrieved docs for transparency
    processing_time: float                    # Total execution time in seconds
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the best smartphones under $500?",
                "answer": "Based on our product catalog...",
                "source_documents": [],
                "processing_time": 1.23
            }
        }


# HealthResponse - Schema for system health check endpoint
# Reports status of key components for monitoring/debugging
class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str              # Overall system status (healthy/unhealthy)
    version: str             # Application version for debugging
    vector_db_status: str    # ChromaDB connection status
    documents_indexed: int   # Total docs in vector store