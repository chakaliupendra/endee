"""
ingest.py — Data Ingestion Module

Responsible for loading raw interview Q&A documents from text files,
parsing them into structured records, and orchestrating the embedding +
vector-storage pipeline.
"""

import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import List

from src.embeddings import EmbeddingEngine
from src.vector_store import VectorStore

file_path = os.path.join("data", "interview-questions.txt")

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Document:
    """Represents a single interview Q&A pair."""

    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    question: str = ""
    answer: str = ""
    full_text: str = ""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def parse_qa_file(file_path: str) -> List[Document]:
    """
    Parse a plain-text file of interview Q&A pairs.

    Expected format
    ---------------
    Q: <question text>
    A: <answer text>
    <blank line>
    Q: ...

    Args:
        file_path: Absolute or relative path to the text file.

    Returns:
        A list of ``Document`` objects.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    logger.info("Parsing Q&A file: %s", file_path)

    documents: List[Document] = []
    current_question = ""
    current_answer = ""

    with open(file_path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()

            if line.startswith("Q:"):
                # If we already have a Q-A pair in progress, save it first
                if current_question and current_answer:
                    full = f"Question: {current_question}\nAnswer: {current_answer}"
                    documents.append(
                        Document(
                            question=current_question,
                            answer=current_answer,
                            full_text=full,
                        )
                    )
                current_question = line[2:].strip()
                current_answer = ""

            elif line.startswith("A:"):
                current_answer = line[2:].strip()

            elif line == "":
                # Blank line acts as a record separator
                continue

            else:
                # Continuation line — append to the current answer
                if current_answer:
                    current_answer += " " + line

    # Flush the last Q-A pair
    if current_question and current_answer:
        full = f"Question: {current_question}\nAnswer: {current_answer}"
        documents.append(
            Document(
                question=current_question,
                answer=current_answer,
                full_text=full,
            )
        )

    logger.info("Parsed %d Q&A documents.", len(documents))
    return documents


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------
def ingest_documents(
    file_path: str,
    embedding_engine: EmbeddingEngine,
    vector_store: VectorStore,
) -> int:
    """
    End-to-end ingestion pipeline:
      1. Parse the text file into Document records.
      2. Generate embeddings for every document.
      3. Store embeddings + metadata in the Endee vector database.

    Args:
        file_path:        Path to the interview questions text file.
        embedding_engine: Pre-initialised ``EmbeddingEngine`` instance.
        vector_store:     Pre-initialised ``VectorStore`` instance.

    Returns:
        The number of documents successfully ingested.
    """
    # Step 1 – Parse
    documents = parse_qa_file(file_path)

    if not documents:
        logger.warning("No documents found in %s — nothing to ingest.", file_path)
        return 0

    # Step 2 – Embed
    logger.info("Generating embeddings for %d documents …", len(documents))
    texts = [doc.full_text for doc in documents]
    embeddings = embedding_engine.embed_batch(texts)

    # Step 3 – Store
    logger.info("Upserting %d vectors into Endee …", len(documents))
    vectors_payload = []
    for doc, emb in zip(documents, embeddings):
        vectors_payload.append(
            {
                "id": doc.doc_id,
                "vector": emb,
                "metadata": {
                    "question": doc.question,
                    "answer": doc.answer,
                    "full_text": doc.full_text,
                },
            }
        )

    vector_store.upsert_vectors(vectors_payload)

    logger.info("Ingestion complete — %d documents stored.", len(documents))
    return len(documents)
