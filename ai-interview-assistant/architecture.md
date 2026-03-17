# 🏗️ Architecture Documentation

> Detailed technical architecture of the AI Interview Preparation Assistant

---

## 📐 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SYSTEM ARCHITECTURE                               │
│                AI Interview Preparation Assistant (RAG)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐     ┌──────────────────┐     ┌────────────────────┐      │
│   │  Data Layer  │────►│  Embedding Layer │────►│  Storage Layer     │      │
│   │             │     │                  │     │  (Endee VectorDB)  │      │
│   │ .txt files  │     │ sentence-        │     │                    │      │
│   │ Q&A pairs   │     │ transformers     │     │ Index: interview_qa│      │
│   │ 20 docs     │     │ all-MiniLM-L6-v2 │     │ Dim: 384           │      │
│   │             │     │ 384-dim output   │     │ Space: cosine      │      │
│   └─────────────┘     └──────────────────┘     └────────┬───────────┘      │
│                                                          │                  │
│   ┌─────────────┐     ┌──────────────────┐              │                  │
│   │  User Query  │────►│  Query Embedding │──────────────┤                  │
│   │             │     │                  │              │                  │
│   │ "What is    │     │ Same model as    │     ┌────────▼───────────┐      │
│   │  Docker?"   │     │ ingestion        │     │  Similarity Search │      │
│   └─────────────┘     └──────────────────┘     │  Top-K retrieval   │      │
│                                                 └────────┬───────────┘      │
│                                                          │                  │
│   ┌─────────────────────────────────────────────────────▼──────────────┐   │
│   │                     RAG Pipeline                                    │   │
│   │                                                                     │   │
│   │   Retrieved Context ──► Prompt Construction ──► LLM Generation      │   │
│   │                                                  (OpenAI GPT /      │   │
│   │                                                   Fallback)         │   │
│   └─────────────────────────────────────────────────────┬──────────────┘   │
│                                                          │                  │
│   ┌─────────────────────────────────────────────────────▼──────────────┐   │
│   │                     Streamlit UI                                    │   │
│   │                                                                     │   │
│   │   ┌──────────┐   ┌───────────────┐   ┌────────────────────┐        │   │
│   │   │  Sidebar  │   │  Query Input  │   │  Results Display   │        │   │
│   │   │  Config   │   │  + Samples    │   │  Answer + Sources  │        │   │
│   │   └──────────┘   └───────────────┘   └────────────────────┘        │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Component Details

### 1. Data Ingestion (`src/ingest.py`)

**Purpose:** Load and preprocess technical interview Q&A data from text files.

**Process:**
1. **Parse** — Read the `data/interview_questions.txt` file.
2. **Tokenise** — Split content into individual Q&A pairs using the `Q:` / `A:` format.
3. **Structure** — Create `Document` objects with unique IDs, question text, answer text, and combined full text.
4. **Pipeline** — Pass structured documents to the embedding engine and vector store.

**Key Design Decisions:**
- Plain text format for the dataset — easy to edit, extend, and version-control.
- UUID-based document IDs for uniqueness and idempotent re-ingestion.
- Full text (`Question: ... Answer: ...`) is embedded rather than question-only, because embedding the full context produces richer semantic vectors.

```python
@dataclass
class Document:
    doc_id: str       # UUID
    question: str     # Raw question text
    answer: str       # Raw answer text
    full_text: str    # Combined "Question: ... Answer: ..." for embedding
```

---

### 2. Embedding Generation (`src/embeddings.py`)

**Purpose:** Convert text documents and queries into dense vector representations.

**Model:** `all-MiniLM-L6-v2` from Hugging Face sentence-transformers.

| Property | Value |
|----------|-------|
| Output dimension | 384 |
| Similarity metric | Cosine |
| Speed | ~14,000 sentences/sec (GPU) |
| Model size | ~80 MB |
| Quality | Excellent for semantic similarity |

**Process:**
1. Load the pre-trained transformer model (cached after first download).
2. For **ingestion**: batch-encode all documents into 384-dim vectors.
3. For **queries**: encode a single question into the same vector space.

**Why this model?**
- Compact enough to run on CPU without GPU requirements.
- Produces high-quality semantic embeddings.
- Widely adopted and well-tested in production retrieval systems.
- 384-dimensional output provides a good balance between quality and storage efficiency.

---

### 3. Vector Storage with Endee (`src/vector_store.py`)

**Purpose:** Store, manage, and query document embeddings using the Endee vector database.

**Endee Configuration:**

| Setting | Value | Rationale |
|---------|-------|-----------|
| Index name | `interview_qa` | Descriptive, domain-specific |
| Dimension | 384 | Matches `all-MiniLM-L6-v2` output |
| Space type | `cosine` | Standard similarity metric for NLP embeddings |
| Precision | `FLOAT32` | Full precision for maximum accuracy |

**Operations:**

```
┌─────────────────────────────────────────────┐
│              Endee VectorStore               │
│                                             │
│  create_index() ──► Prepare storage         │
│  upsert()       ──► Store vectors + meta    │
│  query()        ──► Cosine similarity search│
│  delete_index() ──► Cleanup                 │
└─────────────────────────────────────────────┘
```

**Metadata Schema per Vector:**

```json
{
  "id": "uuid-string",
  "vector": [0.032, -0.118, ..., 0.045],
  "meta": {
    "question": "What is polymorphism in Java?",
    "answer": "Polymorphism in Java is ...",
    "full_text": "Question: What is ... Answer: Polymorphism ..."
  }
}
```

**Error Handling:**
- Automatic index creation if it doesn't exist.
- Graceful handling of connection failures.
- Logging at every operation for observability.

---

### 4. Semantic Retrieval

**Purpose:** Find the most relevant documents for a user's question based on meaning, not keywords.

**How It Works:**

```
User: "Explain inheritance in OOP"
                    │
                    ▼
            ┌───────────────┐
            │ Encode Query  │
            │ → [0.04, ...] │
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────────────────────┐
            │  Cosine Similarity Search     │
            │  in Endee (Top-K = 3)         │
            │                               │
            │  Score 0.91 → OOP concepts    │
            │  Score 0.78 → Abstract vs     │
            │               Interface       │
            │  Score 0.65 → SOLID principles│
            └───────────────────────────────┘
```

**Key Properties:**
- **Semantic** — understands that "inheritance" is related to "OOP" and "abstract classes" even without keyword overlap.
- **Fast** — Endee is optimised for sub-millisecond retrieval at scale.
- **Configurable** — Top-K parameter lets users control precision vs. recall.

---

### 5. Response Generation (`src/rag_pipeline.py`)

**Purpose:** Combine retrieved context with an LLM to generate comprehensive, interview-ready answers.

**Two Generation Modes:**

#### Mode A: OpenAI GPT (when API key is provided)

```
Retrieved Context + Question
        │
        ▼
┌────────────────────────────┐
│  System Prompt:            │
│  "You are an expert        │
│   interview coach..."      │
│                            │
│  User Prompt:              │
│  "[Context] + [Question]"  │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  OpenAI GPT-3.5-turbo      │
│  Temperature: 0.7           │
│  Max tokens: 1024           │
└────────────┬───────────────┘
             │
             ▼
     Synthesised Answer
```

#### Mode B: Context Fallback (no API key)

```
Retrieved Context
        │
        ▼
┌────────────────────────────┐
│  Format retrieved Q&A      │
│  pairs with similarity     │
│  scores                    │
└────────────┬───────────────┘
             │
             ▼
   Formatted Context Display
```

**Design Rationale:**
- The fallback mode ensures the application is fully functional even without an OpenAI key.
- The system prompt is carefully engineered to produce interview-appropriate, structured answers.
- Temperature of 0.7 balances creativity with accuracy.

---

## 🔀 Data Flow

### Ingestion Flow

```
interview_questions.txt
        │
        ▼
    parse_qa_file()           ──► List[Document]
        │
        ▼
    embed_batch(texts)        ──► List[List[float]]  (384-dim each)
        │
        ▼
    vector_store.upsert()     ──► Stored in Endee
```

### Query Flow

```
User Question
        │
        ▼
    embed_text(question)      ──► List[float]  (384-dim)
        │
        ▼
    vector_store.search()     ──► Top-K results with scores
        │
        ▼
    _generate_answer()        ──► LLM or fallback
        │
        ▼
    Display in Streamlit
```

---

## 🖥️ User Interface Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                     Streamlit App Layout                       │
├────────────────┬──────────────────────────────────────────────┤
│   SIDEBAR      │               MAIN CONTENT                   │
│                │                                              │
│ ⚙️ Config      │  ┌──────────────────────────────────────┐   │
│ • Endee URL    │  │         Gradient Header               │   │
│ • Auth Token   │  │    "AI Interview Prep Assistant"      │   │
│ • OpenAI Key   │  └──────────────────────────────────────┘   │
│ • Top-K slider │                                              │
│                │  ┌──────────────────────────────────────┐   │
│ 🚀 Initialise  │  │         Query Input Bar              │   │
│                │  └──────────────────────────────────────┘   │
│ ──────────     │                                              │
│ 📥 Ingest      │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐          │
│ • Dataset path │  │ Samp│ │ Samp│ │ Samp│ │ Samp│          │
│ • Ingest btn   │  │ le 1│ │ le 2│ │ le 3│ │ le 4│          │
│                │  └─────┘ └─────┘ └─────┘ └─────┘          │
│ ──────────     │                                              │
│ 📊 Status      │  ┌──────────────────────────────────────┐   │
│ • System ✅/⏳  │  │      🤖 Generated Answer              │   │
│ • Data   ✅/⏳  │  │                                      │   │
│                │  └──────────────────────────────────────┘   │
│ ──────────     │                                              │
│ 🏗️ Architecture│  ┌──────────────────────────────────────┐   │
│   summary      │  │      📄 Retrieved Documents          │   │
│                │  │  [Card 1] [Card 2] [Card 3]          │   │
│                │  └──────────────────────────────────────┘   │
└────────────────┴──────────────────────────────────────────────┘
```

---

## 🔐 Security Considerations

- **API Keys** — OpenAI keys are passed via environment variables or password-masked UI fields; never stored in code.
- **Endee Auth** — Optional token-based authentication protects the vector database.
- **Input Validation** — All user inputs are validated before processing.
- **No Persistent Storage of Secrets** — `.env` files are git-ignored.

---

## 📈 Scalability

| Aspect | Current | Scalable To |
|--------|---------|-------------|
| Dataset | 20 Q&A pairs | 1B+ vectors (Endee single-node) |
| Embedding model | CPU-based | GPU-accelerated with sentence-transformers |
| LLM | GPT-3.5-turbo | GPT-4, Claude, or local models |
| Deployment | Local Docker | Cloud Kubernetes + Endee cluster |

---

## 🧪 Testing Strategy

```bash
# Verify Endee is running
curl http://localhost:8080/api/v1/health

# Test embedding generation
python -c "from src.embeddings import EmbeddingEngine; e = EmbeddingEngine(); print(len(e.embed_text('test')))"

# Test full pipeline
streamlit run src/app.py
```

---

*Architecture document for the AI Interview Preparation Assistant project.*
