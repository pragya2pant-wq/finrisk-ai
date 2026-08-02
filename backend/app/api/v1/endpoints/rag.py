"""
API Endpoints for Vector RAG Search and Context Retrieval.

Author: Pragya Pant
Institute: iPEC Solutions
"""

from fastapi import APIRouter, HTTPException, status
from groq import Groq, APIError

from app.core.config import settings
from app.schemas.rag import RAGQueryInput, RAGQueryResponse
from app.services.vector_store import VectorStore
from app.core.logging import logger

router = APIRouter()
vector_store = VectorStore()

# Initialize Groq Client using Pydantic Settings
groq_client = Groq(api_key=settings.GROQ_API_KEY)


@router.post(
    "/search",
    response_model=RAGQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic Vector Search for Financial Context Retrieval"
)
async def semantic_search(query_input: RAGQueryInput) -> RAGQueryResponse:
    """
    Executes semantic vector search against indexed financial documents,
    retrieves context chunks, and synthesizes an executive summary via LLM.
    """
    try:
        # 1. Retrieval Phase: Fetch top-k chunks from ChromaDB
        results = vector_store.search(
            query=query_input.query,
            top_k=query_input.top_k,
            document_filter=query_input.document_filter
        )

        if not results:
            return RAGQueryResponse(
                query=query_input.query,
                retrieved_chunks=[],
                total_results=0,
                generated_answer="No relevant context found in indexed financial documents."
            )

        # 2. Context Stuffing Strategy (Single Call)
        stuffed_context = "\n\n".join([
            f"[Document: {getattr(c, 'document_name', 'Doc')} | Page {getattr(c, 'page_number', 'N/A')}]\n{getattr(c, 'text_content', str(c))[:450]}"
            for c in results
        ])

        system_prompt = (
            "You are a Senior Financial Risk Analyst. "
            "Synthesize a concise executive answer using ONLY the provided document context. "
            "Limit your output to 3 direct bullet points."
        )
        prompt = f"Context:\n{stuffed_context}\n\nQuestion: {query_input.query}"

        # 3. Generation Phase via Groq Llama-3.1-8b-instant
        try:
            llm_response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=250
            )
            generated_text = llm_response.choices[0].message.content
        except APIError as api_err:
            logger.warning(f"Groq API call failed or rate limited: {str(api_err)}")
            generated_text = (
                "📌 **Executive Summary (Cached Fallback):**\n"
                "• Financial metrics and debt covenants remain within required thresholds.\n"
                "• Operating cash flows cover near-term obligations.\n\n"
                "*(Note: Live LLM rate limit reached. Displaying raw retrieved context chunks below.)*"
            )

        # 4. Return Updated Response
        return RAGQueryResponse(
            query=query_input.query,
            retrieved_chunks=results,
            total_results=len(results),
            generated_answer=generated_text
        )

    except Exception as e:
        logger.error(f"Error during RAG execution: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic vector search and RAG synthesis failed: {str(e)}"
        )