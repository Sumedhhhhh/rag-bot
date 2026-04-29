# RAG Question Answering System

## 🚀 Overview

This project implements a Retrieval-Augmented Generation (RAG) system that answers questions based on a provided document.

The system supports:
- PDF and JSON document ingestion
- Semantic chunking for PDFs (splits at topic boundaries, not character limits)
- JSON documents skip chunking — each key-value pair is already an atomic fact
- OpenAI embeddings (`text-embedding-3-small`) or local HuggingFace embeddings
- Answer generation using OpenAI (`gpt-4o-mini`), Ollama (Mistral), or a mock fallback

---

## 🧠 Architecture

1. Document ingestion:
   - Supports PDF and JSON formats
   - JSON documents are recursively flattened into `"key path: value"` documents

2. Preprocessing:
   - PDFs are split using `SemanticChunker` — finds topic boundaries using embedding similarity
   - JSON documents skip splitting entirely — already atomic after flattening

3. Embedding:
   - Chunks are embedded using OpenAI `text-embedding-3-small` (or local `all-MiniLM-L6-v2`)
   - Embedding model is loaded once at startup and cached for all requests

4. Storage:
   - Chunks are stored in a FAISS vector database, persisted to disk

5. Query pipeline:
   - Retrieve top-k relevant chunks via cosine similarity
   - Generate answers using:
     - OpenAI `gpt-4o-mini` (primary), or
     - Local LLM via Ollama (Mistral), or
     - Mock LLM (for lightweight testing)

---

## 🤖 LLM Options

### OpenAI (default)

Add your key to `.env`:

```
OPENAI_API_KEY=sk-...
```

Set in `app/config.py`:

```python
USE_OPENAI = True
USE_OLLAMA = False
```

### Local LLM via Ollama (fallback)

```bash
brew install ollama
ollama serve
ollama pull mistral
```

Set in `app/config.py`:

```python
USE_OPENAI = False
USE_OLLAMA = True
```

The system communicates with Ollama via a local API (`http://localhost:11434`).

---

## 🛠️ Tech Stack

- Python
- FastAPI
- LangChain (`langchain`, `langchain-openai`, `langchain-experimental`)
- FAISS (Vector Database)
- OpenAI `text-embedding-3-small` (embeddings) + `gpt-4o-mini` (LLM)
- Sentence Transformers (local embeddings fallback)
- Ollama / Mistral (local LLM fallback)

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open: http://127.0.0.1:8000/docs

Steps:
1. Add your `OPENAI_API_KEY` to `.env`
2. Upload a document (PDF or JSON) via `/upload`
3. Ask questions via `/ask`

> **macOS note:** If you see an OpenMP conflict error (`libomp.dylib already initialized`), add `KMP_DUPLICATE_LIB_OK=TRUE` to your `.env`. This is caused by `faiss-cpu` and `torch` each bundling their own OpenMP runtime.

---

## 🧪 API Endpoints

### Upload Document

`POST /upload`

Supports:
- PDF files — semantically chunked at topic boundaries
- JSON files — flattened into key-value documents, no chunking needed

---

### Ask Questions

`POST /ask`

Request:
```json
{
  "questions": [
    "What is SLA?",
    "Which cloud providers are used?"
  ]
}
```

Response:
```json
{
  "What is SLA?": "Clients are notified of security incidents within defined service level agreements (SLAs).",
  "Which cloud providers are used?": "Amazon Web Services (AWS), GitHub, and Microsoft Office 365."
}
```

---

## ⚙️ Configuration

All tunables live in `app/config.py`:

| Setting | Default | Description |
|---|---|---|
| `USE_OPENAI` | `True` | Use OpenAI for embeddings + answer generation |
| `USE_OLLAMA` | `False` | Use local Ollama (Mistral) for answer generation |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name |
| `MODEL_NAME` | `gpt-4o-mini` | OpenAI model for answer generation |
| `CHUNK_SIZE` | `600` | Max characters per chunk (fixed splitter fallback) |
| `CHUNK_OVERLAP` | `50` | Character overlap between consecutive chunks |
| `TOP_K` | `4` | Number of chunks retrieved per question |

---

## 💡 Design Decisions

- **Semantic chunking for PDFs**
  Uses `SemanticChunker` which embeds each sentence and splits where the topic meaningfully changes (top 5th percentile cosine distance jumps). Keeps related sentences together unlike fixed-size chunking.

- **JSON skips the splitter**
  Each flattened entry (`"company ceo name: Alice"`) is already a self-contained fact. Splitting it would break the key-value format and lose context.

- **Embedding model cached at startup**
  The embedding model is loaded once and reused across all requests via a singleton pattern, avoiding repeated initialization overhead on every `/upload` and `/ask` call.

- **Separation of ingestion and query layers**
  Documents are processed at upload time and the FAISS index is persisted to disk. Subsequent `/ask` calls load the index without re-embedding, making queries fast.

- **LLM priority chain**
  `USE_OPENAI` takes precedence over `USE_OLLAMA`, which takes precedence over the mock. Only one should be `True` at a time.

- **Strict prompt design**
  The LLM is instructed to use only retrieved chunks, include specific technical terms (e.g. TLS, AES-256, MFA), avoid vague phrases, and respond with "Not found" if the answer is absent — reducing hallucination.

---

## ⚠️ Limitations

- Retrieval quality depends on chunk size and embedding model — very long or poorly structured documents may produce incomplete answers
- FAISS stores a single index; uploading a new document overwrites the previous one
- System is optimized for small-to-medium sized documents

---

## 📌 Note

The system supports OpenAI, Ollama, and a mock LLM — switchable via `app/config.py` — allowing development and testing at any cost level without changing application code.
