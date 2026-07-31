"""
API Endpoints for Vector RAG Search and Context Retrieval.

Author: Pragya Pant
Institute: iPEC Solutions
"""

from fastapi import APIRouter, HTTPException, status
from app.schemas.rag import RAGQueryInput, RAGQueryResponse
from app.services.vector_store import VectorStore
from app.core.logging import logger

router = APIRouter()
vector_store = VectorStore()


@router.post(
    "/search",
    response_model=RAGQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic Vector Search for Financial Context Retrieval"
)
async def semantic_search(query_input: RAGQueryInput) -> RAGQueryResponse:
    """
    Executes semantic vector search against indexed financial documents,
    returning top-k most relevant context chunks with similarity scores.
    """
    try:
        results = vector_store.search(
            query=query_input.query,
            top_k=query_input.top_k,
            document_filter=query_input.document_filter
        )

        return RAGQueryResponse(
            query=query_input.query,
            retrieved_chunks=results,
            total_results=len(results)
        )

    except Exception as e:
        logger.error(f"Error during vector search: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic vector search failed: {str(e)}"
        )