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

router = APIRouter(prefix="/upload", tags=["Document Upload"])

# Supported file formats
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'md'}

# Upload directory
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class TextUploadRequest(BaseModel):
    """Request model for text upload."""
    content: str
    source_name: str = "direct_input"


def get_document_processor():
    """Lazy load document processor to avoid import-time errors."""
    from app.core.document_processor import DocumentProcessor
    return DocumentProcessor()


def get_rag_pipeline():
    """Lazy load RAG pipeline to avoid import-time errors."""
    from app.core.rag_pipeline import RAGPipeline
    return RAGPipeline()


@router.post(
    "",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a document",
    description="Upload a product document (PDF, TXT, or MD) to be processed and indexed for RAG."
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload")
) -> UploadResponse:
    """
    Upload and process a document for the RAG system.
    """
    # Validate file extension
    extension = get_file_extension(file.filename)
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: {extension}. "
                   f"Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Sanitize and save file
    safe_filename = sanitize_filename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Initialize components
        document_processor = get_document_processor()
        rag_pipeline = get_rag_pipeline()
        
        # Process document: load and chunk
        chunks = document_processor.process_file(file_path)
        
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No content could be extracted from the document"
            )
        
        # Add chunks to vector store
        chunks_added = rag_pipeline.add_documents(chunks)
        
        return UploadResponse(
            success=True,
            message="Document processed and indexed successfully",
            filename=safe_filename,
            chunks_created=chunks_added,
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing document: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )
    
    finally:
        # Clean up: remove temporary file
        if os.path.exists(file_path):
            os.remove(file_path)


@router.post(
    "/text",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload text content directly",
    description="Upload raw text content to be indexed for RAG."
)
async def upload_text(
    request: TextUploadRequest = Body(...)
) -> UploadResponse:
    """
    Upload raw text content directly without a file.
    
    Args:
        request: TextUploadRequest with content and source_name
        
    Returns:
        UploadResponse with processing results
    """
    from langchain_core.documents import Document
    
    try:
        # Initialize components
        document_processor = get_document_processor()
        rag_pipeline = get_rag_pipeline()
        
        # Create document from text
        doc = Document(
            page_content=request.content,
            metadata={"source_file": request.source_name}
        )
        
        # Chunk the document
        chunks = document_processor.chunk_documents([doc])
        
        # Add to vector store
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