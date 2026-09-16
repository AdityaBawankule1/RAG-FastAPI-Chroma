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

2. Create and Activate a Virtual Environment
It is recommended to use Python 3.11 or 3.12 for seamless binary compatibility:

```bash
# Create venv
python3.11 -m venv venv

# Activate venv (Mac/Linux)
source venv/bin/activate

# Activate venv (Windows)
# venv\Scripts\activate
```

3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Run the Application
Start the Uvicorn development server with hot-reloading enabled:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## 📖 API Documentation & Testing (Swagger UI)

📖 API Documentation & Testing (Swagger UI)
FastAPI automatically provisions an interactive documentation and testing interface. Once the server is running, open your browser and navigate to:

👉 http://127.0.0.1:8000/docs

## Core Endpoints Overview:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **POST** | `/documents` | Ingests a `.pdf` or `.txt` file, chunks the text, computes local dense embeddings, and stores records in ChromaDB. |
| **GET** | `/documents` | Returns metadata overview including total chunk counts and active unique document sources. |
| **POST** | `/query` | Embeds the user question, runs a top-k vector similarity search, and returns the grounded answer alongside source verification snippets. |
| **DELETE** | `/documents/{source_name}` | Purges all vector records and chunks associated with a specific file source from disk. |

## 💡 Quick Test Examples (cURL)

1. Upload & Ingest a Document

```bash
curl -X 'POST' \
  '[http://127.0.0.1:8000/documents](http://127.0.0.1:8000/documents)' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@path_to_your_document.pdf'
```

2. Query the RAG Pipeline

```bash
curl -X 'POST' \
  '[http://127.0.0.1:8000/query](http://127.0.0.1:8000/query)' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "What are the core technical achievements highlighted?",
  "top_k": 3,
  "use_llm": false
}'
```

## 🛠️ Future Improvements / Scaling Roadmap

* Hook up a local LLM runtime like Ollama (Llama 3) to the /query payload for fully autonomous synthesis.

* Introduce metadata filtering namespaces for multi-tenant user document isolation.

* Add Docker containerization (Dockerfile & docker-compose.yml) for seamless cloud deployment.
