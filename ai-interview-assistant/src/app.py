"""
app.py — Streamlit Web Interface

Provides a polished, interactive web UI for the AI Interview
Preparation Assistant.  Users can ask technical interview questions
and receive semantically retrieved context + LLM-generated answers.
"""

import logging
import os
import sys

import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so `src.*` imports work when
# Streamlit is launched from inside the src/ directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.embeddings import EmbeddingEngine
from src.ingest import ingest_documents
from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-24s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Interview Prep Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for premium look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main header gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.05rem;
        font-weight: 300;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9ff 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.10);
    }

    /* Similarity badge */
    .sim-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }

    /* Status badges */
    .status-ok {
        background: #d4edda;
        color: #155724;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-warn {
        background: #fff3cd;
        color: #856404;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #999;
        font-size: 0.82rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #eee;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    """Initialise Streamlit session-state variables on first run."""
    if "initialised" not in st.session_state:
        st.session_state.initialised = False
        st.session_state.embedding_engine = None
        st.session_state.vector_store = None
        st.session_state.rag_pipeline = None
        st.session_state.ingested = False
        st.session_state.query_history = []


def _initialise_components(
    endee_url: str,
    auth_token: str,
    openai_key: str,
    top_k: int,
) -> None:
    """Create or re-create the core engine components."""
    try:
        with st.spinner("🔄 Loading embedding model …"):
            engine = EmbeddingEngine()
            st.session_state.embedding_engine = engine

        with st.spinner("🔄 Connecting to Endee vector database …"):
            vs = VectorStore(
                base_url=endee_url,
                auth_token=auth_token if auth_token else None,
                dimension=engine.get_dimension(),
            )
            st.session_state.vector_store = vs

        rag = RAGPipeline(
            embedding_engine=engine,
            vector_store=vs,
            openai_api_key=openai_key if openai_key else None,
            top_k=top_k,
        )
        st.session_state.rag_pipeline = rag
        st.session_state.initialised = True
        st.success("✅ All components initialised successfully!")
    except Exception as exc:
        st.error(f"❌ Initialisation failed: {exc}")
        logger.exception("Component initialisation error")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _render_sidebar() -> dict:
    """Render sidebar with configuration controls and return settings."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")

        endee_url = st.text_input(
            "Endee Server URL",
            value="http://localhost:8080/api/v1",
            help="HTTP endpoint of your running Endee instance.",
        )

        auth_token = st.text_input(
            "Endee Auth Token (optional)",
            type="password",
            help="Leave blank if authentication is disabled.",
        )

        openai_key = st.text_input(
            "OpenAI API Key (optional)",
            type="password",
            help="Enables GPT-powered answer generation. Without it, "
                 "the app shows retrieved context directly.",
        )

        top_k = st.slider(
            "Top-K Results",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of documents to retrieve per query.",
        )

        st.markdown("---")

        # --- Initialise / re-initialise button ---------------------------
        if st.button("🚀 Initialise System", use_container_width=True):
            _initialise_components(endee_url, auth_token, openai_key, top_k)

        # --- Ingest button ------------------------------------------------
        st.markdown("---")
        st.markdown("### 📥 Data Ingestion")

        data_path = st.text_input(
            "Dataset path",
            value="data/interview_questions.txt",
        )

        if st.button("📂 Ingest Documents", use_container_width=True):
            if not st.session_state.initialised:
                st.warning("⚠️ Initialise the system first.")
            else:
                try:
                    with st.spinner("📄 Ingesting documents …"):
                        count = ingest_documents(
                            file_path=data_path,
                            embedding_engine=st.session_state.embedding_engine,
                            vector_store=st.session_state.vector_store,
                        )
                    st.session_state.ingested = True
                    st.success(f"✅ Ingested {count} documents into Endee.")
                except Exception as exc:
                    st.error(f"❌ Ingestion failed: {exc}")
                    logger.exception("Ingestion error")

        # --- Status -------------------------------------------------------
        st.markdown("---")
        st.markdown("### 📊 System Status")

        if st.session_state.initialised:
            st.markdown('<span class="status-ok">✅ System Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-warn">⏳ Not Initialised</span>', unsafe_allow_html=True)

        if st.session_state.ingested:
            st.markdown('<span class="status-ok">✅ Data Ingested</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-warn">⏳ No Data</span>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            "### 🏗️ Architecture\n"
            "1. **Embedding** → sentence-transformers\n"
            "2. **Vector DB** → Endee\n"
            "3. **RAG** → Context + LLM\n"
            "4. **UI** → Streamlit"
        )

    return {
        "endee_url": endee_url,
        "auth_token": auth_token,
        "openai_key": openai_key,
        "top_k": top_k,
    }


# ---------------------------------------------------------------------------
# Main content area
# ---------------------------------------------------------------------------
def _render_main() -> None:
    """Render the primary query interface and results."""

    # --- Header -----------------------------------------------------------
    st.markdown(
        """
        <div class="main-header">
            <h1>🎯 AI Interview Preparation Assistant</h1>
            <p>Powered by Endee Vector Database &amp; Retrieval-Augmented Generation</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Query input ------------------------------------------------------
    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input(
            "💬 Ask a technical interview question",
            placeholder="e.g. What is polymorphism in Java?",
            label_visibility="visible",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("🔍 Search", use_container_width=True)

    # --- Sample questions -------------------------------------------------
    st.markdown("**💡 Try these sample questions:**")
    sample_cols = st.columns(4)
    samples = [
        "What is polymorphism in Java?",
        "Explain REST API and how it works",
        "What are SOLID principles?",
        "Difference between SQL and NoSQL",
    ]
    selected_sample = None
    for col, sample in zip(sample_cols, samples):
        if col.button(sample, use_container_width=True):
            selected_sample = sample

    # --- Determine effective query ----------------------------------------
    effective_query = selected_sample or (question if search_clicked else None)

    if effective_query:
        if not st.session_state.initialised:
            st.warning(
                "⚠️ Please initialise the system using the sidebar first."
            )
            return
        if not st.session_state.ingested:
            st.warning(
                "⚠️ Please ingest the knowledge base using the sidebar first."
            )
            return

        _execute_query(effective_query)

    # --- Query history ----------------------------------------------------
    if st.session_state.query_history:
        with st.expander("📜 Query History", expanded=False):
            for i, entry in enumerate(reversed(st.session_state.query_history)):
                st.markdown(f"**{i + 1}.** {entry['question']}")


def _execute_query(question: str) -> None:
    """Run the RAG pipeline and display results."""
    with st.spinner("🧠 Thinking …"):
        try:
            result = st.session_state.rag_pipeline.query(question)
        except Exception as exc:
            st.error(f"❌ Query failed: {exc}")
            logger.exception("RAG pipeline error")
            return

    st.session_state.query_history.append(result)

    # --- Generated answer -------------------------------------------------
    st.markdown("---")
    st.markdown("### 🤖 Generated Answer")
    st.markdown(result["generated_answer"])

    # --- Retrieved documents ----------------------------------------------
    st.markdown("---")
    st.markdown("### 📄 Retrieved Documents from Endee")

    docs = result.get("retrieved_documents", [])
    if not docs:
        st.info("No relevant documents found.")
        return

    for i, doc in enumerate(docs, start=1):
        meta = doc.get("metadata", {})
        score = doc.get("score", 0.0)

        st.markdown(
            f"""
            <div class="result-card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <strong>📌 Document {i}</strong>
                    <span class="sim-badge">Similarity: {score:.4f}</span>
                </div>
                <hr style="margin:8px 0;border-color:#eee;">
                <p><strong>Q:</strong> {meta.get('question', 'N/A')}</p>
                <p><strong>A:</strong> {meta.get('answer', 'N/A')[:300]}…</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    _init_session_state()
    _render_sidebar()
    _render_main()

    # Footer
    st.markdown(
        """
        <div class="footer">
            Built with ❤️ using <strong>Endee Vector Database</strong>,
            <strong>Sentence-Transformers</strong>, and
            <strong>Streamlit</strong> &nbsp;|&nbsp;
            AI Interview Preparation Assistant &copy; 2025
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
