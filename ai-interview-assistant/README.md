# 🎯 AI Interview Preparation Assistant

> A Retrieval-Augmented Generation (RAG) system for technical interview preparation, powered by the **Endee Vector Database** and **Sentence-Transformers**.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Endee](https://img.shields.io/badge/Vector%20DB-Endee-blueviolet)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [🎯 AI Interview Preparation Assistant](#-ai-interview-preparation-assistant)
  - [📋 Table of Contents](#-table-of-contents)
  - [🎯 Problem Statement](#-problem-statement)
  - [🏗️ Architecture Overview](#️-architecture-overview)
  - [🗄️ Endee Vector Database Usage](#️-endee-vector-database-usage)
    - [How Endee is Integrated](#how-endee-is-integrated)
    - [Key Configuration](#key-configuration)
    - [Why Endee?](#why-endee)
  - [📁 Project Structure](#-project-structure)
  - [🛠️ Setup Instructions](#️-setup-instructions)
    - [Prerequisites](#prerequisites)
    - [1. Start the Endee Vector Database](#1-start-the-endee-vector-database)
    - [2. Verify Endee is Healthy](#2-verify-endee-is-healthy)
  - [📦 Installation](#-installation)
    - [(Optional) Set OpenAI API Key](#optional-set-openai-api-key)
  - [▶️ Running the Application](#️-running-the-application)
    - [Quick Start Steps](#quick-start-steps)
  - [💡 Example Usage](#-example-usage)
    - [Input](#input)
    - [System Workflow](#system-workflow)
    - [Sample Output](#sample-output)
  - [📸 Screenshots](#-screenshots)
    - [Main Interface](#main-interface)
    - [Search Results](#search-results)
    - [Sidebar Configuration](#sidebar-configuration)
  - [🧰 Screenshots](#-screenshots-1)
  - [🧰 Technologies Used](#-technologies-used)
  - [📄 License](#-license)
  - [🤝 Acknowledgements](#-acknowledgements)

---

## 🎯 Problem Statement

Preparing for technical interviews requires access to a **comprehensive knowledge base** of questions and answers across multiple domains. Traditional keyword-based search fails to capture the *semantic meaning* behind questions — for example, searching for *"OOP concepts"* should surface results about *"polymorphism"*, *"inheritance"*, and *"encapsulation"* even if those exact words aren't in the query.

This project solves that problem by building a **semantic search engine** backed by the [Endee Vector Database](https://github.com/endee-io/endee) and a **RAG pipeline** that:

1. Converts interview Q&A pairs into dense vector embeddings.
2. Stores them in Endee for high-performance similarity search.
3. Retrieves the most relevant content for any user question.
4. Optionally synthesises a polished answer using an LLM (OpenAI GPT).

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                    (Streamlit Web App)                           │
└────────────────────────────┬────────────────────────────────────┘
                             │  User Question
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     EMBEDDING ENGINE                            │
│           (sentence-transformers / all-MiniLM-L6-v2)            │
│                                                                 │
│   "What is polymorphism?"  ──►  [0.032, -0.118, ..., 0.045]    │
│                                       384-dim vector            │
└────────────────────────────┬────────────────────────────────────┘
                             │  Query Vector
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 ENDEE VECTOR DATABASE                            │
│              (Cosine Similarity Search)                          │
│                                                                 │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐                     │
│   │ Vector 1 │  │ Vector 2 │  │ Vector N │  ... (stored docs)   │
│   │ + meta   │  │ + meta   │  │ + meta   │                     │
│   └──────────┘  └──────────┘  └──────────┘                     │
│                                                                 │
│   Returns: Top-K most similar documents                         │
└────────────────────────────┬────────────────────────────────────┘
                             │  Retrieved Context
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAG PIPELINE                                  │
│                                                                 │
│   Retrieved Context + User Question                              │
│          │                                                       │
│          ├──► OpenAI GPT (if API key provided)                   │
│          │       └──► Synthesised Answer                         │
│          │                                                       │
│          └──► Fallback: Display retrieved context directly        │
│                                                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │  Final Answer
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│           Display answer + retrieved source documents            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🗄️ Endee Vector Database Usage

[Endee](https://github.com/endee-io/endee) is a high-performance, open-source vector database designed for AI search and retrieval workloads. In this project, Endee is used as the **core storage and retrieval engine** for semantic search.

### How Endee is Integrated

| Step | Operation | Endee API Used |
|------|-----------|----------------|
| **1. Index Creation** | Create a vector index (`interview_qa`) with 384 dimensions and cosine similarity | `client.create_index(name, dimension, space_type, precision)` |
| **2. Vector Upsert** | Store document embeddings with metadata (question, answer, full text) | `index.upsert([{id, vector, meta}])` |
| **3. Similarity Search** | Find the top-K most semantically similar documents to a query vector | `index.query(vector, top_k)` |
| **4. Index Management** | Get or delete indexes as needed | `client.get_index()`, `client.delete_index()` |

### Key Configuration

```python
from endee import Endee, Precision

# Connect to Endee server
client = Endee()
client.set_base_url("http://localhost:8080/api/v1")

# Create index optimised for embedding search
client.create_index(
    name="interview_qa",
    dimension=384,              # Matches all-MiniLM-L6-v2 output
    space_type="cosine",        # Cosine similarity for semantic search
    precision=Precision.FLOAT32 # Full precision for accuracy
)
```

### Why Endee?

- **Purpose-built for AI** — optimised for vector retrieval workloads.
- **High performance** — handles up to 1B vectors on a single node.
- **Metadata filtering** — supports payload-based filtering for structured queries.
- **Simple SDK** — clean Python API for seamless integration.
- **Docker-ready** — single command to start the server.

---

## 📁 Project Structure

```
ai-interview-prep-assistant/
│
├── data/
│   └── interview_questions.txt    # Knowledge base (20 Q&A pairs)
│
├── src/
│   ├── __init__.py                # Package initialisation
│   ├── app.py                     # Streamlit web interface
│   ├── embeddings.py              # Embedding generation module
│   ├── ingest.py                  # Data ingestion pipeline
│   ├── rag_pipeline.py            # RAG orchestration
│   └── vector_store.py            # Endee vector database wrapper
│
├── .gitignore                     # Git ignore rules
├── architecture.md                # Detailed architecture documentation
├── docker-compose.yml             # Docker Compose for Endee server
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

---

## 🛠️ Setup Instructions

### Prerequisites

- **Python 3.9+** installed
- **Docker** installed and running (for the Endee server)
- **(Optional)** An OpenAI API key for GPT-powered answer generation

### 1. Start the Endee Vector Database

```bash
# Using Docker Compose (recommended)
docker compose up -d

# Verify it's running
docker ps
# You should see: endee-server  listening on port 8080
```

Alternatively, visit the [Endee Getting Started Guide](https://docs.endee.io/quick-start) for other installation methods.

### 2. Verify Endee is Healthy

```bash
curl http://localhost:8080/api/v1/health
```

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/ai-interview-prep-assistant.git
cd ai-interview-prep-assistant

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### (Optional) Set OpenAI API Key

```bash
# Linux / macOS
export OPENAI_API_KEY="sk-your-key-here"

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-your-key-here"
```

---

## ▶️ Running the Application

```bash
# Launch the Streamlit web UI
streamlit run src/app.py
```

The app will open in your browser at **http://localhost:8501**.

### Quick Start Steps

1. Click **🚀 Initialise System** in the sidebar.
2. Click **📂 Ingest Documents** to load the knowledge base into Endee.
3. Type a question or click a sample question to search.
4. View the generated answer and retrieved source documents.

---

## 💡 Example Usage

### Input

```
What is polymorphism in Java?
```

### System Workflow

1. **Embed** — Converts the question into a 384-dimensional vector using `all-MiniLM-L6-v2`.
2. **Search** — Queries the Endee vector database for the top-3 most similar documents (cosine similarity).
3. **Retrieve** — Returns matching Q&A pairs with similarity scores.
4. **Generate** — Passes retrieved context to OpenAI GPT (or displays context directly in fallback mode).
5. **Display** — Shows the final answer alongside source documents with similarity badges.

### Sample Output

```
🤖 Generated Answer:
Polymorphism in Java is the ability of an object to take on many forms.
It allows a single interface to represent different underlying data types.
There are two primary types:
  - Compile-time polymorphism (method overloading)
  - Runtime polymorphism (method overriding)
...

📄 Retrieved Documents:
  [1] Similarity: 0.9523 — "What is polymorphism in Java?"
  [2] Similarity: 0.7841 — "Explain the concept of OOP"
  [3] Similarity: 0.6912 — "What is the difference between abstract class and interface?"
```

---

## 📸 Screenshots

### Main Interface
> *Screenshot: Main query interface with gradient header and sample question buttons*

`[screenshot_main_interface.png]`

### Search Results
> *Screenshot: Retrieved documents with similarity badges and generated answer*

`[screenshot_search_results.png]`

### Sidebar Configuration
> *Screenshot: Sidebar showing Endee configuration, ingestion controls, and system status*

`[screenshot_sidebar.png]`

---

## 🧰 Screenshots
Results

![Screenshot](https://iili.io/qV4foQ4.md.png)
![Screenshot](<Screenshot (74).png>)

## 🧰 Technologies Used

| Technology | Purpose |
|------------|---------|
| [Endee](https://github.com/endee-io/endee) | High-performance vector database for semantic search |
| [Sentence-Transformers](https://www.sbert.net/) | Dense embedding generation (`all-MiniLM-L6-v2`) |
| [Streamlit](https://streamlit.io/) | Interactive web UI |
| [OpenAI GPT](https://openai.com/) | LLM-based answer generation (optional) |
| [NumPy](https://numpy.org/) | Numerical operations |
| [Docker](https://www.docker.com/) | Containerised Endee deployment |

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🤝 Acknowledgements

- [Endee.io](https://endee.io/) for the open-source vector database.
- [Hugging Face](https://huggingface.co/) for sentence-transformer models.
- [Streamlit](https://streamlit.io/) for the rapid UI framework.
