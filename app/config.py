"""
Configuration module for the E-Commerce RAG System.
Handles all environment variables and application settings.
"""

import os
from pydantic_settings import BaseSettings
from functools import lru_cache


# Settings class - Inherits BaseSettings to auto-load env variables
# Provides type validation, default values, and .env file support
class Settings(BaseSettings):
    
    # API Keys - Required field, must be set in environment/.env
    google_api_key: str
    
    # Application Settings - Default values provided, can be overridden
    app_name: str = "E-Commerce RAG System"
    debug: bool = True
    
    # Vector Database Settings - ChromaDB storage path and collection name
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "ecommerce_products"
    
    # RAG Settings - Controls text chunking and retrieval behavior
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_results: int = 3
    
    # Model Settings - Gemini models for embeddings and text generation
    embedding_model: str = "models/gemini-embedding-001"
    llm_model: str = "gemini-2.5-flash"
    
    # Config inner class - Tells Pydantic where to find env variables
    # Reads from .env file with UTF-8 encoding
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# @lru_cache - Decorator that caches function result (singleton pattern)
# Ensures Settings() is instantiated only once, improving performance
@lru_cache()
def get_settings() -> Settings:
    return Settings()