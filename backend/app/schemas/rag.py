"""
Pydantic Schemas for RAG Vector Search and Embedding Ingestion.

Author: Pragya Pant
Institute: iPEC Solutions
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class DocumentChunk(BaseModel):
    """Schema representing a chunk of extracted document text with metadata."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_name: str = Field(..., description="Source PDF filename")
    page_number: int = Field(..., description="Source page number")
    text_content: str = Field(..., description="Extracted text chunk payload")


class RAGQueryInput(BaseModel):
    """Input payload for semantic RAG search queries."""
    query: str = Field(..., description="Natural language search query regarding financial reports", min_length=3)
    top_k: int = Field(default=3, description="Number of most relevant text chunks to retrieve", ge=1, le=10)
    document_filter: Optional[str] = Field(default=None, description="Optional filter by filename")


class SearchResultChunk(BaseModel):
    """Retrieved document chunk with similarity score."""
    chunk_id: str
    document_name: str
    page_number: int
    text_content: str
    similarity_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")


class RAGQueryResponse(BaseModel):
    """Response payload containing retrieved context chunks for RAG."""
    query: str
    retrieved_chunks: List[SearchResultChunk]
    total_results: int
    generated_answer: Optional[str] = Field(default=None, description="LLM generated summary from retrieved context")