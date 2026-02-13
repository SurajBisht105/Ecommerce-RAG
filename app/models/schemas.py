"""
Pydantic schemas for API request and response validation.
Ensures type safety and automatic documentation generation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class DocumentChunk(BaseModel):
    """Represents a chunk of processed document."""
    content: str
    metadata: dict
    relevance_score: Optional[float] = None


class UploadResponse(BaseModel):
    """Response model for document upload endpoint."""
    success: bool
    message: str
    filename: str
    chunks_created: int
    timestamp: datetime = Field(default_factory=datetime.now)
    
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


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., min_length=3, max_length=500)
    top_k: Optional[int] = Field(default=3, ge=1, le=10)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the best smartphones under $500?",
                "top_k": 3
            }
        }


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    query: str
    answer: str
    source_documents: List[DocumentChunk]
    processing_time: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the best smartphones under $500?",
                "answer": "Based on our product catalog...",
                "source_documents": [],
                "processing_time": 1.23
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    version: str
    vector_db_status: str
    documents_indexed: int