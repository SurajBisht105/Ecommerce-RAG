"""
Main FastAPI Application
Entry point for the E-Commerce RAG System.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import upload_router, query_router, health_router
from app.config import get_settings
from app import __version__

# Load application settings from environment variables
settings = get_settings()

# FastAPI() - Creates the main application instance
# Configures metadata for auto-generated Swagger/ReDoc documentation
app = FastAPI(
    title=settings.app_name,
    description="""
    ## E-Commerce Product Recommendation System using RAG
    
    This API provides intelligent product Q&A and recommendations using 
    Retrieval-Augmented Generation (RAG) architecture.
    
    ### Features:
    - **Document Upload**: Index product documents (PDF, TXT, Markdown)
    - **Semantic Search**: Find relevant products using natural language
    - **AI-Powered Responses**: Get accurate answers grounded in product data
    - **Product Recommendations**: Get personalized product suggestions
    
    ### Architecture:
    - **Vector Database**: ChromaDB for efficient similarity search
    - **Embeddings**: Google Generative AI Embeddings
    - **LLM**: Google Gemini Pro for response generation
    """,
    version=__version__,
    docs_url="/docs",      # Swagger UI endpoint
    redoc_url="/redoc"     # ReDoc documentation endpoint
)

# add_middleware() - Adds CORS middleware to handle cross-origin requests
# Allows frontend apps from different domains to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allows all origins (restrict in production)
    allow_credentials=True,    # Allows cookies/auth headers
    allow_methods=["*"],       # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],       # Allows all request headers
)

# include_router() - Registers route modules with the main app
# Organizes endpoints into separate files for better code structure
app.include_router(upload_router)   # Document upload endpoints
app.include_router(query_router)    # RAG query endpoints
app.include_router(health_router)   # Health check endpoints


# @app.get() - Decorator that creates a GET endpoint at root path "/"
# async def allows non-blocking I/O operations
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": __version__,
        "description": "E-Commerce RAG System API",
        "docs": "/docs",
        "health": "/health"
    }


# @app.on_event("startup") - Runs once when application starts
# Used for initializing DB connections, loading models, etc.
@app.on_event("startup")
async def startup_event():
    """Initialize components on application startup."""
    print(f"🚀 {settings.app_name} v{__version__} starting...")
    print(f"📚 Vector DB: {settings.chroma_persist_dir}")
    print(f"🤖 LLM Model: {settings.llm_model}")
    print(f"📖 Docs available at: /docs")


# @app.on_event("shutdown") - Runs once when application stops
# Used for cleanup tasks like closing DB connections, releasing resources
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    print("👋 Shutting down gracefully...")