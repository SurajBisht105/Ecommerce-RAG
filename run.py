"""
Application Runner
Starts the FastAPI server with uvicorn.
"""

import uvicorn
from app.config import get_settings

# get_settings() - Fetches app configuration (debug mode, DB urls, etc.)
# Often implements caching to avoid repeated env variable reads
settings = get_settings()

# Entry point guard - Runs only when script is executed directly, not imported
if __name__ == "__main__":
    
    # uvicorn.run() - Starts the ASGI server to serve FastAPI application
    # Configures host, port, hot-reload (dev), and logging level
    uvicorn.run(
        "app.main:app",  # Import string pointing to FastAPI instance
        host="0.0.0.0",  # Binds to all interfaces (needed for Docker)
        port=8000,
        reload=settings.debug,  # Auto-restart on code changes in dev mode
        log_level="info"
    )