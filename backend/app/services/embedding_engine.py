"""
Embedding Engine Service.

Generates dense vector embeddings for financial text chunks using sentence-transformers.
Author: Pragya Pant
Institute: iPEC Solutions
"""

from typing import List
from sentence_transformers import SentenceTransformer
from app.core.logging import logger

MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingEngine:
    """
    Service for generating vector embeddings from financial text chunks.
    """

    def __init__(self) -> None:
        logger.info(f"Initializing SentenceTransformer model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)

    def get_embedding(self, text: str) -> List[float]:
        """Generates a single dense vector embedding for input text string."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates vector embeddings for a batch of text strings."""
        logger.info(f"Generating embeddings for batch of {len(texts)} text chunks.")
        embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=32)
        return embeddings.tolist()