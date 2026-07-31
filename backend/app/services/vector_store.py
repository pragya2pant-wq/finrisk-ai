"""
ChromaDB Vector Store Service with Custom Embedding Function.

Manages persistent vector database storage, indexing, and similarity search for RAG.
Author: Pragya Pant
Institute: iPEC Solutions
"""

import sys
from types import ModuleType
from pathlib import Path
from typing import List, Dict, Any, Optional

# Mock onnxruntime module before chromadb import to neutralize ONNX loading
class DummyONNX(ModuleType):
    def __init__(self) -> None:
        super().__init__("onnxruntime")
        self.__spec__ = sys.modules[__name__].__spec__

    def InferenceSession(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("ONNXRuntime disabled in favor of SentenceTransformers.")

sys.modules["onnxruntime"] = DummyONNX()

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from app.schemas.rag import DocumentChunk, SearchResultChunk
from app.services.embedding_engine import EmbeddingEngine
from app.core.logging import logger

VECTOR_DB_DIR = Path("./data/vector_db")


class CustomSentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """
    Custom ChromaDB EmbeddingFunction wrapper utilizing SentenceTransformer.
    Bypasses ChromaDB default ONNX runtime dependency.
    """

    def __init__(self) -> None:
        self.engine = EmbeddingEngine()

    def __call__(self, input: Documents) -> Embeddings:
        """Computes embeddings for input document strings."""
        return self.engine.get_embeddings_batch(input)


# Instantiate custom embedding function
custom_ef = CustomSentenceTransformerEmbeddingFunction()


class VectorStore:
    """
    Local persistent vector store using ChromaDB.
    """

    def __init__(self) -> None:
        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initializing ChromaDB vector store at {VECTOR_DB_DIR}")

        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

        # Register collection with explicit custom embedding function
        self.collection = self.client.get_or_create_collection(
            name="finrisk_documents",
            embedding_function=custom_ef,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Indexes document chunks into ChromaDB vector database.
        """
        if not chunks:
            return

        logger.info(f"Indexing {len(chunks)} chunks into vector database...")
        texts = [c.text_content for c in chunks]
        embeddings = custom_ef.engine.get_embeddings_batch(texts)

        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {"document_name": c.document_name, "page_number": c.page_number}
            for c in chunks
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        logger.info("Successfully indexed chunks into vector database.")

    def search(self, query: str, top_k: int = 3, document_filter: Optional[str] = None) -> List[SearchResultChunk]:
        """
        Executes cosine similarity search against indexed vector collection.
        """
        query_embedding = custom_ef.engine.get_embedding(query)

        where_clause = {"document_name": document_filter} if document_filter else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause
        )

        search_results: List[SearchResultChunk] = []

        if results and results["ids"] and len(results["ids"][0]) > 0:
            for idx in range(len(results["ids"][0])):
                chunk_id = results["ids"][0][idx]
                doc_text = results["documents"][0][idx]
                meta = results["metadatas"][0][idx]
                distance = results["distances"][0][idx] if "distances" in results and results["distances"] else 0.0

                similarity = round(max(0.0, 1.0 - (distance / 2.0)), 4)

                search_results.append(
                    SearchResultChunk(
                        chunk_id=chunk_id,
                        document_name=meta.get("document_name", "Unknown"),
                        page_number=meta.get("page_number", 1),
                        text_content=doc_text,
                        similarity_score=similarity
                    )
                )

        return search_results