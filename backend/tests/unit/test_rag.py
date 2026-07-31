"""
Unit Tests for Vector Store and Embedding RAG Search API Endpoints.

Author: Pragya Pant
Institute: iPEC Solutions
"""

from fastapi.testclient import TestClient
from app.schemas.rag import DocumentChunk
from app.services.vector_store import VectorStore


def test_vector_store_indexing_and_search() -> None:
    """Verify that chunks can be indexed into ChromaDB and retrieved via semantic search."""
    store = VectorStore()

    sample_chunks = [
        DocumentChunk(
            chunk_id="test-chunk-1",
            document_name="annual_report_2025.pdf",
            page_number=3,
            text_content="The company reported total net revenue of 500 million USD in FY2025 with operating profit margin of 18 percent."
        ),
        DocumentChunk(
            chunk_id="test-chunk-2",
            document_name="annual_report_2025.pdf",
            page_number=5,
            text_content="Total debt stood at 120 million USD, yielding a healthy debt-to-equity ratio of 0.6x."
        )
    ]

    # Index chunks
    store.add_chunks(sample_chunks)

    # Search query
    results = store.search(query="What was the total revenue?", top_k=1)

    assert len(results) > 0
    assert "revenue" in results[0].text_content.lower()
    assert results[0].similarity_score > 0.0


def test_rag_search_endpoint_success(client: TestClient) -> None:
    """Verify POST /api/v1/rag/search returns 200 OK with retrieved chunks."""
    payload = {
        "query": "net revenue operating profit",
        "top_k": 2,
        "document_filter": None
    }

    response = client.post("/api/v1/rag/search", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "net revenue operating profit"
    assert "retrieved_chunks" in data
    assert "total_results" in data
    assert data["total_results"] >= 0