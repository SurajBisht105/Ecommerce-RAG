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

settings = get_settings()


class DocumentProcessor:
    """
    Processes documents by loading and splitting them into chunks.
    Supports PDF, TXT, and Markdown file formats.
    """
    
    # Mapping of file extensions to their respective loaders
    LOADER_MAPPING = {
        'pdf': PyPDFLoader,
        'txt': TextLoader,
        'md': UnstructuredMarkdownLoader
    }
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize the document processor with chunking parameters.
        
        Args:
            chunk_size: Size of each text chunk (default from settings)
            chunk_overlap: Overlap between chunks (default from settings)
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # Initialize the text splitter with optimal parameters for RAG
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a document from the specified file path.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = get_file_extension(file_path)
        
        if extension not in self.LOADER_MAPPING:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {list(self.LOADER_MAPPING.keys())}"
            )
        
        loader_class = self.LOADER_MAPPING[extension]
        loader = loader_class(file_path)
        
        return loader.load()
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better retrieval.
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of chunked Document objects
        """
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk index to metadata for tracking
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = i
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks
    
    def process_file(self, file_path: str) -> List[Document]:
        """
        Complete processing pipeline: load and chunk a document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of processed and chunked Document objects
        """
        # Load the document
        documents = self.load_document(file_path)
        
        # Add source filename to metadata
        filename = os.path.basename(file_path)
        for doc in documents:
            doc.metadata['source_file'] = filename
        
        # Chunk the documents
        chunks = self.chunk_documents(documents)
        
        return chunks