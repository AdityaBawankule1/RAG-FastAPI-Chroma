# Production-Grade RAG REST API 🚀

A robust, full-featured Retrieval-Augmented Generation (RAG) backend service built with **FastAPI**, **ChromaDB**, and **Sentence-Transformers**. Designed to solve common local prototyping limitations by implementing a true persistent vector schema, file ingestion pipelines (PDF/TXT), and context-verified responses.

---

## 🏗️ Architecture & Tech Stack

* **Backend Framework:** FastAPI (Asynchronous request handling, automated Swagger/OpenAPI documentation)
* **Vector Database:** ChromaDB (`PersistentClient` storage schema bound to disk, avoiding temporary in-memory session loss)
* **Embeddings:** `all-MiniLM-L6-v2` via `sentence-transformers` (Runs completely free locally on CPU)
* **Inference Support:** Flexible integration capability with local runtimes (e.g., Ollama / Llama 3)
* **Data Ingestion:** Sliding-window chunking engine supporting raw text and structured PDF parsing (`pypdf`)

---

## ⚙️ Project Structure

```text
rag-fastapi-chroma/
│
├── main.py             # FastAPI app code & vector database controller
├── requirements.txt    # Python dependencies list
├── README.md           # Documentation
└── chroma_db/          # Persistent local storage directory for database collections
```

## ⚙️ Getting Started & Installation

1. Clone the Repository

```bash
git clone [https://github.com/your-username/rag-fastapi-chroma.git](https://github.com/your-username/rag-fastapi-chroma.git)
cd rag-fastapi-chroma
```