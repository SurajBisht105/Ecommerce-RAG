"""
Health Check API Endpoint
Provides system health and status information.
"""

import traceback
from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app import __version__

# APIRouter() - Creates a modular router for grouping related endpoints
# prefix adds "/health" before all routes, tags groups them in Swagger docs
router = APIRouter(prefix="/health", tags=["Health Check"])


# get_rag_pipeline() - Lazy loading function for RAG pipeline
# Import inside function avoids circular imports and speeds up app startup
def get_rag_pipeline():
    """Lazy load RAG pipeline."""
    from app.core.rag_pipeline import RAGPipeline
    return RAGPipeline()


# @router.get("") - Creates GET endpoint at "/health" (prefix + empty string)
# response_model validates and serializes output to HealthResponse schema
@router.get(
    "",
    response_model=HealthResponse,
    summary="Check system health",
    description="Returns the health status of the RAG system and its components."
)
async def health_check() -> HealthResponse:
    """
    Check the health status of the RAG system.
    """
    # try-except block - Attempts to connect to RAG pipeline
    # Catches failures gracefully and reports unhealthy status instead of crashing
    try:
        rag_pipeline = get_rag_pipeline()
        stats = rag_pipeline.get_stats()
        vector_db_status = "healthy"
        documents_indexed = stats["documents_indexed"]
    except Exception as e:
        # traceback.format_exc() - Gets full error stack trace for debugging
        print(f"Health check error: {traceback.format_exc()}")
        vector_db_status = "unhealthy"
        documents_indexed = 0
    
    # Returns Pydantic model - Auto-serialized to JSON by FastAPI
    return HealthResponse(
        status="healthy",
        version=__version__,
        vector_db_status=vector_db_status,
        documents_indexed=documents_indexed
    )


# @router.get("/stats") - Creates GET endpoint at "/health/stats"
# Returns dict directly - FastAPI auto-converts to JSON response
@router.get(
    "/stats",
    summary="Get system statistics",
    description="Returns detailed statistics about the RAG system."
)
async def get_stats() -> dict:
    """
    Get detailed system statistics.
    """
    # try-except block - Fetches stats or returns error info
    # **stats unpacks dictionary to merge with other fields
    try:
        rag_pipeline = get_rag_pipeline()
        stats = rag_pipeline.get_stats()
        return {
            "status": "operational",
            "version": __version__,
            **stats  # Spread operator - merges stats dict into response
        }
    except Exception as e:
        return {
            "status": "error",
            "version": __version__,
            "error": str(e)
        }