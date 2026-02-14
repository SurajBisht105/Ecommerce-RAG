"""
Upload API Endpoint
Handles document upload and indexing operations.
"""

import os
import shutil
import traceback
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Body
from pydantic import BaseModel
from app.models.schemas import UploadResponse
from app.utils.helpers import get_file_extension, sanitize_filename

# APIRouter() - Creates modular router for upload-related endpoints
# Groups all endpoints under "/upload" prefix
router = APIRouter(prefix="/upload", tags=["Document Upload"])

# ALLOWED_EXTENSIONS - Set of valid file types for upload validation
# Using set for O(1) lookup performance
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'md'}

# UPLOAD_DIR - Temporary storage path for uploaded files
# os.makedirs() creates directory if not exists, exist_ok prevents errors
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# TextUploadRequest - Pydantic model for direct text input validation
# Allows indexing text without file upload
class TextUploadRequest(BaseModel):
    """Request model for text upload."""
    content: str
    source_name: str = "direct_input"


# get_document_processor() - Lazy loads processor to avoid circular imports
# Defers heavy imports until actually needed
def get_document_processor():
    """Lazy load document processor to avoid import-time errors."""
    from app.core.document_processor import DocumentProcessor
    return DocumentProcessor()


# get_rag_pipeline() - Lazy loads RAG pipeline for vector store operations
# Same lazy loading pattern for consistency
def get_rag_pipeline():
    """Lazy load RAG pipeline to avoid import-time errors."""
    from app.core.rag_pipeline import RAGPipeline
    return RAGPipeline()


# @router.post("") - Creates POST endpoint at "/upload"
# status_code=201 indicates resource creation (REST convention)
@router.post(
    "",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a document",
    description="Upload a product document (PDF, TXT, or MD) to be processed and indexed for RAG."
)
async def upload_document(
    # File(...) - Expects multipart form data, ... makes it required
    file: UploadFile = File(..., description="Document file to upload")
) -> UploadResponse:
    """
    Upload and process a document for the RAG system.
    """
    # Validate file extension against allowed types
    # Raises 400 error if file type not supported
    extension = get_file_extension(file.filename)
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: {extension}. "
                   f"Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # sanitize_filename() - Removes dangerous characters to prevent path traversal
    # os.path.join() - Creates platform-independent file path
    safe_filename = sanitize_filename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    try:
        # with open() - Context manager ensures file is properly closed
        # shutil.copyfileobj() - Efficiently copies file stream to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Initialize processing components
        document_processor = get_document_processor()
        rag_pipeline = get_rag_pipeline()
        
        # process_file() - Loads document and splits into chunks
        # Returns list of document chunks for indexing
        chunks = document_processor.process_file(file_path)
        
        # if not chunks - Validates extraction produced content
        # Returns 422 if document is empty or unreadable
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No content could be extracted from the document"
            )
        
        # add_documents() - Embeds chunks and stores in vector database
        # Returns count of chunks successfully indexed
        chunks_added = rag_pipeline.add_documents(chunks)
        
        return UploadResponse(
            success=True,
            message="Document processed and indexed successfully",
            filename=safe_filename,
            chunks_created=chunks_added,
            timestamp=datetime.now()
        )
    
    # except HTTPException: raise - Re-raises HTTP errors without wrapping
    # Prevents double-wrapping of validation errors
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing document: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )
    
    # finally block - Always executes, even after exceptions
    # Cleans up temp file to prevent disk space issues
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# @router.post("/text") - Creates POST endpoint at "/upload/text"
# Alternative upload method for raw text without file handling
@router.post(
    "/text",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload text content directly",
    description="Upload raw text content to be indexed for RAG."
)
async def upload_text(
    # Body(...) - Expects JSON body, ... makes it required
    request: TextUploadRequest = Body(...)
) -> UploadResponse:
    """
    Upload raw text content directly without a file.
    """
    # Import inside function - Lazy loading for LangChain dependency
    from langchain_core.documents import Document
    
    try:
        # Initialize processing components
        document_processor = get_document_processor()
        rag_pipeline = get_rag_pipeline()
        
        # Document() - LangChain document wrapper with content and metadata
        # Metadata tracks source for citation in RAG responses
        doc = Document(
            page_content=request.content,
            metadata={"source_file": request.source_name}
        )
        
        # chunk_documents() - Splits document into smaller pieces
        # Smaller chunks improve retrieval relevance
        chunks = document_processor.chunk_documents([doc])
        
        # add_documents() - Stores chunks in vector database
        chunks_added = rag_pipeline.add_documents(chunks)
        
        return UploadResponse(
            success=True,
            message="Text content indexed successfully",
            filename=request.source_name,
            chunks_created=chunks_added,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        print(f"Error processing text: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing text: {str(e)}"
        )