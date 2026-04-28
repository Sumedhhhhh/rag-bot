# RAG Question Answering System

## 🚀 Overview

This project implements a Retrieval-Augmented Generation (RAG) system that answers questions based on a provided document.

The system supports:
- PDF and JSON document ingestion
- JSON-based question input
- Semantic retrieval using vector embeddings (FAISS)
- Answer generation using a local LLM (Ollama - Mistral) or a mock fallback

---

## 🧠 Architecture

1. Document ingestion:
   - Supports PDF and JSON formats
   - JSON documents are flattened into text for semantic processing

2. Preprocessing:
   - Documents are split into overlapping chunks
   - Embeddings are generated using a local model

3. Storage:
   - Chunks are stored in a FAISS vector database

4. Query pipeline:
   - Retrieve top-k relevant chunks
   - Generate answers using:
     - Local LLM (Ollama - Mistral), or
     - Mock LLM (for lightweight testing)

---

## 🤖 Local LLM (Ollama)

This project uses Ollama to run a local LLM (Mistral) for answer generation.

### Setup

```bash
brew install ollama
ollama serve
ollama pull mistral
```

The system communicates with Ollama via a local API (`http://localhost:11434`).

---

## 🛠️ Tech Stack

- Python
- FastAPI
- LangChain
- FAISS (Vector Database)
- Sentence Transformers (local embeddings)
- Ollama (Mistral - local LLM)

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open: http://127.0.0.1:8000/docs

Steps:
1. Upload a document (PDF or JSON)
2. Ask questions via `/ask`

---

## 🧪 API Endpoints

### Upload Document

`POST /upload`

Supports:
- PDF files
- JSON files

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

## 💡 Design Decisions

- **Separation of ingestion and query layers**  
  Avoids reprocessing documents for every request and improves efficiency.

- **Support for multiple input formats (PDF & JSON)**  
  JSON inputs are normalized into a unified document format.

- **Embedding model cached at startup**  
  The HuggingFace embedding model is loaded once and reused across all requests, avoiding repeated PyTorch initialization overhead on every `/upload` and `/ask` call.

- **Use of local embeddings during development**  
  Reduces dependency on external APIs and lowers cost.

- **Local LLM integration via Ollama**  
  Enables fully offline inference and avoids API quota limitations.

- **Strict prompt design**  
  Ensures answers are grounded in retrieved context and reduces hallucination.

---

## ⚠️ Limitations

- Local LLM (Mistral) may produce less accurate answers compared to larger cloud models
- Performance depends on retrieval quality and chunking strategy
- System is optimized for small-to-medium sized documents

---

## 📌 Note

The system supports both local (Ollama) and mock LLMs, allowing development and testing without reliance on external APIs.
