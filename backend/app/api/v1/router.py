"""
Central API v1 Router Aggregator.

Author: Pragya Pant
Institute: iPEC Solutions
"""

from fastapi import APIRouter
from app.api.v1.endpoints import documents, financials, risk_scoring, rag

api_router = APIRouter()

# Register endpoint routers
api_router.include_router(documents.router, prefix="/documents", tags=["Document Intelligence"])
api_router.include_router(financials.router, prefix="/financials", tags=["Financial Analysis"])
api_router.include_router(risk_scoring.router, prefix="/risk", tags=["ML Credit Risk Scoring"])
api_router.include_router(rag.router, prefix="/rag", tags=["Vector Search & RAG"])