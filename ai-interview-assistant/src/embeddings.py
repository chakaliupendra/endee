"""
embeddings.py — Embedding Generation Module

Handles conversion of text documents into dense vector embeddings
using HuggingFace sentence-transformers. The model is loaded once and
reused for both document ingestion and query-time encoding.
"""

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME: str = "all-MiniLM-L6-v2"
"""
Sentence-transformer model used for embedding generation.
- Output dimension : 384
- Speed            : ~14 000 sentences / sec on a single GPU
- Quality          : Excellent trade-off between size and accuracy
"""


# ---------------------------------------------------------------------------
# EmbeddingEngine
# ---------------------------------------------------------------------------
class EmbeddingEngine:
    """Encapsulates the sentence-transformer model and provides helper
    methods to embed single texts or batches of texts."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        """
        Initialise the embedding engine.

        Args:
            model_name: HuggingFace model identifier.  Defaults to
                        ``all-MiniLM-L6-v2`` (384-dim, cosine similarity).
        """
        logger.info("Loading sentence-transformer model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.dimension: int = self.model.get_sentence_embedding_dimension()
        logger.info(
            "Model loaded successfully — embedding dimension: %d",
            self.dimension,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding vector for a single piece of text.

        Args:
            text: The input text string.

        Returns:
            A list of floats representing the embedding vector.
        """
        if not text or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        embedding: np.ndarray = self.model.encode(
            text, show_progress_bar=False, convert_to_numpy=True
        )
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts in a single batched call.

        Args:
            texts: A list of non-empty input strings.

        Returns:
            A list of embedding vectors (each a list of floats).
        """
        if not texts:
            raise ValueError("Input texts list must not be empty.")

        embeddings: np.ndarray = self.model.encode(
            texts, show_progress_bar=True, convert_to_numpy=True, batch_size=32
        )
        return embeddings.tolist()

    def get_dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        return self.dimension
