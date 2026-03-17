"""
vector_store.py — Endee Vector Database Wrapper

Provides a clean abstraction over the Endee Python SDK for managing
indexes, upserting vectors, and performing semantic similarity search.
"""

import logging
from typing import Any, Dict, List, Optional

from endee import Endee, Precision

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INDEX_NAME: str = "interview_qa"
DEFAULT_SPACE_TYPE: str = "cosine"
DEFAULT_PRECISION = Precision.FLOAT32


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------
class VectorStore:
    """High-level wrapper around the Endee vector database.

    Responsibilities
    ----------------
    * Index lifecycle management (create / get).
    * Upserting document vectors with metadata.
    * Cosine-similarity search and result formatting.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080/api/v1",
        auth_token: Optional[str] = None,
        index_name: str = DEFAULT_INDEX_NAME,
        dimension: int = 384,
    ) -> None:
        """
        Connect to the Endee server and prepare the target index.

        Args:
            base_url:   Endee HTTP API endpoint.
            auth_token: Optional authentication token.
            index_name: Name of the vector index to use.
            dimension:  Dimensionality of embedding vectors.
        """
        self.index_name = index_name
        self.dimension = dimension

        # --- Initialise client -------------------------------------------
        logger.info("Connecting to Endee at %s …", base_url)
        if auth_token:
            self.client = Endee(auth_token)
        else:
            self.client = Endee()
        self.client.set_base_url(base_url)

        # --- Ensure the index exists -------------------------------------
        self._ensure_index()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_index(self) -> None:
        """Create the vector index if it does not already exist."""
        try:
            self.index = self.client.get_index(name=self.index_name)
            logger.info("Using existing Endee index: '%s'", self.index_name)
        except Exception:
            logger.info(
                "Index '%s' not found — creating (dim=%d, space=%s) …",
                self.index_name,
                self.dimension,
                DEFAULT_SPACE_TYPE,
            )
            self.client.create_index(
                name=self.index_name,
                dimension=self.dimension,
                space_type=DEFAULT_SPACE_TYPE,
                precision=DEFAULT_PRECISION,
            )
            self.index = self.client.get_index(name=self.index_name)
            logger.info("Index '%s' created successfully.", self.index_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def upsert_vectors(self, vectors) -> None:
        """
        Insert or update vectors into the Endee index.
        """

        if not vectors:
            logger.warning("Empty vector list — skipping upsert.")
            return

        logger.info("Upserting %d vectors into index '%s' …", len(vectors), self.index_name)

        payload = []

        for v in vectors:
            payload.append({
                "id": v.get("id"),
                "vector": v.get("vector"),
                "meta": v.get("metadata", v.get("meta", {}))
            })

        self.index.upsert(payload)
        logger.info("Upsert complete.")

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Perform a cosine-similarity search against the Endee index.

        Args:
            query_vector: The embedding vector of the user query.
            top_k:        Number of nearest neighbours to return.

        Returns:
            A list of result dicts, each containing:
              - ``id``         : document ID
              - ``score``      : similarity score
              - ``meta``       : stored metadata dict
        """
        logger.info(
            "Searching index '%s' for top-%d results …", self.index_name, top_k
        )

        raw_results = self.index.query(vector=query_vector, top_k=top_k)

        # Endee query() returns plain dicts, so use .get() for access
        results: List[Dict[str, Any]] = []
        for item in raw_results:
            results.append(
                {
                    "id": item.get("id"),
                    "score": item.get("similarity", item.get("score", 0.0)),
                    "metadata": item.get("meta", {}),
                }
            )

        logger.info("Search returned %d results.", len(results))
        return results

    def delete_index(self) -> None:
        """Delete the current index (useful for cleanup / re-ingestion)."""
        try:
            self.client.delete_index(name=self.index_name)
            logger.info("Deleted index '%s'.", self.index_name)
        except Exception as exc:
            logger.warning("Could not delete index '%s': %s", self.index_name, exc)
