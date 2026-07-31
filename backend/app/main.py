"""
Main Execution Entry Point for FinRisk AI Platform Backend.

Author: Pragya Pant
Institute: iPEC Solutions
"""

import sys
from pathlib import Path
import os
from dotenv import load_dotenv
from fastapi import FastAPI

# Load environment variables from .env file
load_dotenv()

API_TITLE = os.getenv("API_TITLE", "FinRisk AI Engine")
API_VERSION = os.getenv("API_VERSION", "v1")

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION
)

# Explicitly add the 'backend' root directory to Python's search path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI
from app.core.config import settings
from app.core.logging import logger
from app.api.v1.router import api_router

# Initialize FastAPI app with metadata
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise Credit & Financial Risk Intelligence Platform",
    contact={
        "name": "Pragya Pant",
        "institute": "iPEC Solutions",
    },
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Mount API v1 router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["System Health"])
async def system_health_check() -> dict:
    """
    System status endpoint to confirm API operational state.
    
    Returns:
        dict: Operational metadata including author and platform status.
    """
    logger.info("System health check requested.")
    return {
        "status": "online",
        "platform": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "author": "Pragya Pant",
        "institute": "iPEC Solutions"
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FinRisk AI Backend Server by Pragya Pant (iPEC Solutions)...")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        app_dir=str(backend_dir)
    )