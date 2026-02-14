"""
Document Processing Module
Handles loading, parsing, and chunking of various document formats.
"""

import os
from typing import List
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader
)
from app.config import get_settings
from app.utils.helpers import get_file_extension

# Load application settings for chunk configuration
settings = get_settings()


# DocumentProcessor - Handles document loading and chunking for RAG
# Supports multiple file formats (PDF, TXT, MD) with unified interface
class DocumentProcessor:
    """
    Processes documents by loading and splitting them into chunks.
    Supports PDF, TXT, and Markdown file formats.
    """
    
    # LOADER_MAPPING - Class-level dict mapping extensions to loader classes
    # Strategy pattern: selects appropriate loader based on file type
    LOADER_MAPPING = {
        'pdf': PyPDFLoader,
        'txt': TextLoader,
        'md': UnstructuredMarkdownLoader
    }
    
    # __init__() - Constructor initializes chunking parameters and text splitter
    # Uses settings defaults if no custom values provided
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        # "or" pattern - Uses settings value if parameter is None
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # RecursiveCharacterTextSplitter - Splits text while preserving context
        # Tries separators in order: paragraphs → lines → sentences → words
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,     # Overlap prevents context loss at boundaries
            length_function=len,                   # Uses character count for sizing
            separators=["\n\n", "\n", ". ", " ", ""]  # Priority order for splitting
        )
    
    # load_document() - Loads file content using appropriate loader
    # Returns list of Document objects with content and metadata
    def load_document(self, file_path: str) -> List[Document]:
        # Validate file exists before processing
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get extension and validate against supported formats
        extension = get_file_extension(file_path)
        
        if extension not in self.LOADER_MAPPING:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {list(self.LOADER_MAPPING.keys())}"
            )
        
        # Dynamic loader instantiation - Gets class from mapping, then instantiates
        # Factory pattern: creates appropriate loader based on file type
        loader_class = self.LOADER_MAPPING[extension]
        loader = loader_class(file_path)
        
        # loader.load() - Parses file and returns Document objects
        return loader.load()
    
    # chunk_documents() - Splits large documents into smaller retrievable pieces
    # Smaller chunks improve semantic search accuracy
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        # split_documents() - Applies splitting logic to all documents
        chunks = self.text_splitter.split_documents(documents)
        
        # for loop with enumerate - Adds positional metadata to each chunk
        # Useful for reconstructing document order or debugging
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = i
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks
    
    # process_file() - Main pipeline method combining load + chunk operations
    # Single entry point for complete document processing
    def process_file(self, file_path: str) -> List[Document]:
        # Step 1: Load document content
        documents = self.load_document(file_path)
        
        # os.path.basename() - Extracts filename from full path
        # Adds source tracking for citation in RAG responses
        filename = os.path.basename(file_path)
        for doc in documents:
            doc.metadata['source_file'] = filename
        
        # Step 2: Split into chunks for indexing
        chunks = self.chunk_documents(documents)
        
        return chunks