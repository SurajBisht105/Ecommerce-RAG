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

# APIRouter() - Creates modular router for query-related endpoints
# Groups all endpoints under "/query" prefix in Swagger docs
router = APIRouter(prefix="/query", tags=["Query & Recommendations"])


# get_rag_pipeline() - Lazy loads RAG pipeline on first call
# Prevents circular imports and defers heavy initialization until needed
def get_rag_pipeline():
    """Lazy load RAG pipeline to avoid import-time errors."""
    from app.core.rag_pipeline import RAGPipeline
    return RAGPipeline()


# RecommendRequest - Pydantic model for request body validation
# Optional fields allow flexible queries with or without budget/category
class RecommendRequest(BaseModel):
    """Request model for recommendations."""
    requirements: str
    budget: Optional[str] = None
    category: Optional[str] = None


# @router.post("") - Creates POST endpoint at "/query"
# response_model ensures output conforms to QueryResponse schema
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
    # time.time() - Captures start time to measure total processing duration
    start_time = time.time()
    
    try:
        # Initialize RAG pipeline instance
        rag_pipeline = get_rag_pipeline()
        
        # rag_pipeline.query() - Executes RAG: retrieves docs + generates answer
        # Returns tuple of (generated_answer, list_of_retrieved_documents)
        answer, retrieved_docs = rag_pipeline.query(
            query=request.query,
            top_k=request.top_k
        )
        
        # for loop - Transforms retrieved docs into API response format
        # Truncates content to 500 chars and rounds relevance scores
        source_documents = []
        for doc, score in retrieved_docs:
            source_documents.append(
                DocumentChunk(
                    content=doc.page_content[:500],  # Truncate for response
                    metadata=doc.metadata,
                    relevance_score=round(float(score), 4)
                )
            )
        
        # Calculate total processing time in seconds
        processing_time = round(time.time() - start_time, 3)
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            source_documents=source_documents,
            processing_time=processing_time
        )
        
    except Exception as e:
        # traceback.format_exc() - Logs full stack trace for debugging
        print(f"Error processing query: {traceback.format_exc()}")
        # HTTPException - Returns proper HTTP error response to client
        # status.HTTP_500 indicates server-side error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


# @router.post("/recommend") - Creates POST endpoint at "/query/recommend"
# Body(...) makes request body required (... = Ellipsis means mandatory)
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
    # Build enhanced query - Constructs detailed prompt from user inputs
    # List used to conditionally append optional parts
    query_parts = [f"I'm looking for: {request.requirements}"]
    
    # if conditions - Append budget/category only if provided
    # Allows flexible queries without requiring all fields
    if request.budget:
        query_parts.append(f"My budget is: {request.budget}")
    if request.category:
        query_parts.append(f"Category preference: {request.category}")
    
    query_parts.append("Please recommend suitable products with explanations.")
    
    # " ".join() - Combines list into single space-separated string
    enhanced_query = " ".join(query_parts)
    
    # Reuses query_products() - DRY principle, avoids code duplication
    # await needed since query_products is async function
    query_request = QueryRequest(query=enhanced_query)
    return await query_products(query_request)