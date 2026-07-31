"""
Core Configuration Module for FinRisk AI Platform.

Uses Pydantic BaseSettings to handle environment variables cleanly and safely.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """
    System configuration specifications.
    Loaded automatically from .env or system environment variables.
    """
    PROJECT_NAME: str = "FinRisk AI – Intelligent Credit & Financial Risk Analysis Platform"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # API Keys & Database Connections
    GEMINI_API_KEY: Optional[str] = None
    DATABASE_URL: str = "sqlite:///./finrisk_platform.db"
    VECTOR_DB_DIR: str = "./data/vector_db"
    
    # Processing Configuration
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )


settings = Settings()