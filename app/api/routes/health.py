"""
Health Check API Endpoint
Provides system health and status information.
"""

import traceback
from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app import __version__

router = APIRouter(prefix="/health", tags=["Health Check"])


def get_rag_pipeline():
    """Lazy load RAG pipeline."""
    from app.core.rag_pipeline import RAGPipeline
    return RAGPipeline()


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
    try:
        rag_pipeline = get_rag_pipeline()
        stats = rag_pipeline.get_stats()
        vector_db_status = "healthy"
        documents_indexed = stats["documents_indexed"]
    except Exception as e:
        print(f"Health check error: {traceback.format_exc()}")
        vector_db_status = "unhealthy"
        documents_indexed = 0
    
    return HealthResponse(
        status="healthy",
        version=__version__,
        vector_db_status=vector_db_status,
        documents_indexed=documents_indexed
    )


@router.get(
    "/stats",
    summary="Get system statistics",
    description="Returns detailed statistics about the RAG system."
)
async def get_stats() -> dict:
    """
    Get detailed system statistics.
    """
    try:
        rag_pipeline = get_rag_pipeline()
        stats = rag_pipeline.get_stats()
        return {
            "status": "operational",
            "version": __version__,
            **stats
        }
    except Exception as e:
        return {
            "status": "error",
            "version": __version__,
            "error": str(e)
        }