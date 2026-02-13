"""
Utility functions for the RAG system.
Contains helper functions used across the application.
"""

import os
import re
import time
from typing import Callable, Any
from functools import wraps


def get_file_extension(filename: str) -> str:
    """
    Extract and return the file extension from a filename.
    
    Args:
        filename: Name of the file
        
    Returns:
        Lowercase file extension without the dot
    """
    return os.path.splitext(filename)[1].lower().lstrip('.')


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing special characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for filesystem operations
    """
    # Remove special characters, keep alphanumeric, dots, and underscores
    sanitized = re.sub(r'[^\w\-_\.]', '_', filename)
    return sanitized


def calculate_processing_time(func: Callable) -> Callable:
    """
    Decorator to calculate and log function execution time.
    
    Args:
        func: Function to be wrapped
        
    Returns:
        Wrapped function with timing capability
    """
    @wraps(func)
    async def wrapper(*args, **kwargs) -> tuple[Any, float]:
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        processing_time = round(end_time - start_time, 3)
        return result, processing_time
    return wrapper


def format_context(documents: list) -> str:
    """
    Format retrieved documents into a context string for the LLM.
    
    Args:
        documents: List of retrieved document chunks
        
    Returns:
        Formatted context string
    """
    context_parts = []
    for i, doc in enumerate(documents, 1):
        context_parts.append(f"[Document {i}]\n{doc.page_content}")
    return "\n\n".join(context_parts)