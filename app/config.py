"""
Configuration module for the E-Commerce RAG System.
Handles all environment variables and application settings.
"""

import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Uses pydantic-settings for validation and type coercion.
    """
    
    # API Keys
    google_api_key: str
    
    # Application Settings
    app_name: str = "E-Commerce RAG System"
    debug: bool = True
    
    # Vector Database Settings
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "ecommerce_products"
    
    # RAG Settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_results: int = 3
    
    # Model Settings
    embedding_model: str = "models/gemini-embedding-001"
    llm_model: str = "gemini-2.5-flash"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """
    Returns cached settings instance.
    Using lru_cache ensures settings are loaded only once.
    """
    return Settings()