"""
Main FastAPI Application
Entry point for the E-Commerce RAG System.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import upload_router, query_router, health_router
from app.config import get_settings
from app import __version__

settings = get_settings()

# Initialize FastAPI application
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
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload_router)
app.include_router(query_router)
app.include_router(health_router)


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


# Startup event
@app.lifespan("startup")
async def startup_event():
    """Initialize components on application startup."""
    print(f"🚀 {settings.app_name} v{__version__} starting...")
    print(f"📚 Vector DB: {settings.chroma_persist_dir}")
    print(f"🤖 LLM Model: {settings.llm_model}")
    print(f"📖 Docs available at: /docs")


# Shutdown event
@app.lifespan("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    print("👋 Shutting down gracefully...")