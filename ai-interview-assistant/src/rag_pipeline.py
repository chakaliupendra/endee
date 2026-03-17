"""
rag_pipeline.py — Retrieval-Augmented Generation Pipeline

Orchestrates the full RAG workflow:
  1. Encode the user query into an embedding.
  2. Retrieve the most relevant documents from Endee.
  3. Construct a prompt with retrieved context.
  4. Generate a final answer using an LLM (OpenAI or a local fallback).
"""

import logging
import os
from typing import Any, Dict, List, Optional

from src.embeddings import EmbeddingEngine
from src.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert technical interview coach. Your role is to
provide clear, comprehensive, and accurate answers to technical interview
questions. Use the retrieved context below to formulate your answer. If the
context is relevant, incorporate it into your response and expand upon it.
If the context is not relevant, answer based on your own knowledge but mention
that no directly matching content was found in the knowledge base."""

CONTEXT_TEMPLATE = """
--- Retrieved Context (from Endee Vector Database) ---
{context}
--- End of Retrieved Context ---

User Question: {question}

Please provide a detailed, well-structured answer suitable for a technical 
interview. Include examples where appropriate.
"""


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------
class RAGPipeline:
    """Retrieval-Augmented Generation pipeline backed by Endee and
    sentence-transformers."""

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: VectorStore,
        openai_api_key: Optional[str] = None,
        top_k: int = 3,
    ) -> None:
        """
        Initialise the RAG pipeline.

        Args:
            embedding_engine: Pre-loaded ``EmbeddingEngine`` for encoding.
            vector_store:     Connected ``VectorStore`` (Endee) instance.
            openai_api_key:   Optional OpenAI API key for GPT-based generation.
                              When not provided, a context-only fallback is used.
            top_k:            Number of documents to retrieve per query.
        """
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.top_k = top_k

        # --- Optional OpenAI integration ---------------------------------
        self.openai_client = None
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI

                self.openai_client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialised — GPT generation enabled.")
            except ImportError:
                logger.warning(
                    "openai package not installed. "
                    "Install with `pip install openai` for LLM generation."
                )
        else:
            logger.info(
                "No OpenAI API key provided — using context-retrieval-only mode."
            )

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def query(self, question: str) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline for a given question.

        Steps
        -----
        1. Generate an embedding for the question.
        2. Search Endee for the closest matching documents.
        3. Build a context-enriched prompt.
        4. Generate a response (via OpenAI or fallback).

        Args:
            question: The user's interview question.

        Returns:
            A dict with keys:
              - ``question``           : original user question
              - ``retrieved_documents``: list of retrieved doc dicts
              - ``generated_answer``   : final LLM-generated or fallback answer
        """
        logger.info("RAG query: %s", question[:80])

        # Step 1 — Encode the question
        query_vector = self.embedding_engine.embed_text(question)

        # Step 2 — Semantic search in Endee
        retrieved_docs = self.vector_store.search(
            query_vector=query_vector, top_k=self.top_k
        )

        # Step 3 — Build context string
        context_parts: List[str] = []
        for i, doc in enumerate(retrieved_docs, start=1):
            meta = doc.get("metadata", {})
            score = doc.get("score", 0.0)
            context_parts.append(
                f"[Document {i}] (Similarity: {score:.4f})\n"
                f"Q: {meta.get('question', 'N/A')}\n"
                f"A: {meta.get('answer', 'N/A')}\n"
            )
        context_str = "\n".join(context_parts) if context_parts else "No relevant documents found."

        # Step 4 — Generate answer
        generated_answer = self._generate_answer(question, context_str)

        return {
            "question": question,
            "retrieved_documents": retrieved_docs,
            "generated_answer": generated_answer,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_answer(self, question: str, context: str) -> str:
        """
        Generate a final answer using OpenAI GPT (if available) or fall
        back to a context-summary approach.
        """
        prompt = CONTEXT_TEMPLATE.format(context=context, question=question)

        if self.openai_client:
            return self._openai_generate(prompt)
        return self._fallback_generate(question, context)

    def _openai_generate(self, prompt: str) -> str:
        """Call the OpenAI Chat Completions API to generate an answer."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("OpenAI API call failed: %s", exc)
            return (
                "⚠️ OpenAI API call failed. Showing retrieved context instead.\n\n"
                + prompt
            )

    @staticmethod
    def _fallback_generate(question: str, context: str) -> str:
        """
        Fallback answer generation when no LLM API key is configured.

        Constructs a well-formatted response from the retrieved context,
        demonstrating the semantic search capability even without an LLM.
        """
        header = (
            "📚 **Answer based on retrieved context**\n"
            "*(No OpenAI API key configured — showing semantically "
            "retrieved knowledge base results)*\n\n"
        )
        body = (
            f"**Your Question:** {question}\n\n"
            f"**Relevant Knowledge Base Entries:**\n\n{context}\n\n"
            "💡 *Tip: Set the `OPENAI_API_KEY` environment variable to "
            "enable GPT-powered answer synthesis.*"
        )
        return header + body
