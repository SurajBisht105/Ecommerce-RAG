"""
Utility functions for the RAG system.
Contains helper functions used across the application.
"""

import os
import re
import time
from typing import Callable, Any
from functools import wraps


# get_file_extension() - Extracts file extension from filename
# Returns lowercase extension without dot (e.g., "pdf", "txt")
def get_file_extension(filename: str) -> str:
    # os.path.splitext() - Splits into (name, extension) tuple
    # [1] gets extension, lower() normalizes case, lstrip('.') removes dot
    return os.path.splitext(filename)[1].lower().lstrip('.')


# sanitize_filename() - Removes dangerous characters from filenames
# Prevents path traversal attacks and filesystem errors
def sanitize_filename(filename: str) -> str:
    # re.sub() - Replaces regex matches with underscore
    # [^\w\-_\.] matches anything NOT alphanumeric, dash, underscore, or dot
    sanitized = re.sub(r'[^\w\-_\.]', '_', filename)
    return sanitized


# calculate_processing_time() - Decorator that measures function execution time
# Returns tuple of (original_result, time_in_seconds)
def calculate_processing_time(func: Callable) -> Callable:
    # @wraps(func) - Preserves original function's metadata (name, docstring)
    # Prevents decorator from hiding the wrapped function's identity
    @wraps(func)
    async def wrapper(*args, **kwargs) -> tuple[Any, float]:
        # Capture start time before execution
        start_time = time.time()
        # await - Handles async functions properly
        result = await func(*args, **kwargs)
        # Calculate elapsed time after completion
        end_time = time.time()
        processing_time = round(end_time - start_time, 3)
        return result, processing_time
    return wrapper


# format_context() - Converts document list into formatted string for LLM prompt
# Adds numbering for clear reference in generated responses
def format_context(documents: list) -> str:
    context_parts = []
    # enumerate(documents, 1) - Iterates with index starting at 1
    # Creates human-readable numbering [Document 1], [Document 2], etc.
    for i, doc in enumerate(documents, 1):
        context_parts.append(f"[Document {i}]\n{doc.page_content}")
    # "\n\n".join() - Combines parts with double newlines for readability
    return "\n\n".join(context_parts)