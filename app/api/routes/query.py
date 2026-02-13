"""
Query API Endpoint
Handles user queries and product recommendations using RAG.
"""

import time
import traceback
from fastapi import APIRouter, HTTPException, status, Body
from pydantic import BaseModel
from typing import Optional
from app.models.schemas import QueryRequest, QueryResponse, DocumentChunk

router = APIRouter(prefix="/query", tags=["Query & Recommendations"])


def get_rag_pipeline():
    """Lazy load RAG pipeline to avoid import-time errors."""
    from app.core.rag_pipeline import RAGPipeline
    return RAGPipeline()


class RecommendRequest(BaseModel):
    """Request model for recommendations."""
    requirements: str
    budget: Optional[str] = None
    category: Optional[str] = None


@router.post(
    "",
    response_model=QueryResponse,
    summary="Query the product knowledge base",
    description="Ask questions or get product recommendations based on indexed documents."
)
async def query_products(request: QueryRequest) -> QueryResponse:
    """
    Process a user query using the RAG pipeline.
    """
    start_time = time.time()
    
    try:
        # Initialize RAG pipeline
        rag_pipeline = get_rag_pipeline()
        
        # Execute RAG pipeline
        answer, retrieved_docs = rag_pipeline.query(
            query=request.query,
            top_k=request.top_k
        )
        
        # Format source documents for response
        source_documents = []
        for doc, score in retrieved_docs:
            source_documents.append(
                DocumentChunk(
                    content=doc.page_content[:500],  # Truncate for response
                    metadata=doc.metadata,
                    relevance_score=round(float(score), 4)
                )
            )
        
        processing_time = round(time.time() - start_time, 3)
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            source_documents=source_documents,
            processing_time=processing_time
        )
        
    except Exception as e:
        print(f"Error processing query: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@router.post(
    "/recommend",
    response_model=QueryResponse,
    summary="Get product recommendations",
    description="Get personalized product recommendations based on requirements."
)
async def recommend_products(request: RecommendRequest = Body(...)) -> QueryResponse:
    """
    Get product recommendations based on user requirements.
    """
    # Build enhanced query for recommendations
    query_parts = [f"I'm looking for: {request.requirements}"]
    
    if request.budget:
        query_parts.append(f"My budget is: {request.budget}")
    if request.category:
        query_parts.append(f"Category preference: {request.category}")
    
    query_parts.append("Please recommend suitable products with explanations.")
    
    enhanced_query = " ".join(query_parts)
    
    # Use the standard query endpoint logic
    query_request = QueryRequest(query=enhanced_query)
    return await query_products(query_request)