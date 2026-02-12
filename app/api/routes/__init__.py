"""Routes package - Contains all API endpoint definitions."""

from app.api.routes.upload import router as upload_router
from app.api.routes.query import router as query_router
from app.api.routes.health import router as health_router

__all__ = ["upload_router", "query_router", "health_router"]