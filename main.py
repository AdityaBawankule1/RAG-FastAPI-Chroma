import os
import io
import uuid
from typing import List, Optional
import chromadb
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import pypdf
import requests

# ---------------------------------------------------------
# 1. FastAPI App Initialization & Config
# ---------------------------------------------------------
app = FastAPI(
    title="Production-Grade RAG API",
    description="A robust Retrieval-Augmented Generation REST API built with FastAPI, ChromaDB, and Sentence-Transformers.",
    version="1.0.0"
)

# ---------------------------------------------------------
# 2. Embedding Model & Vector DB Setup
# ---------------------------------------------------------
# Free, local model that runs on CPU without API keys/costs
print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Persistent ChromaDB client (saves vectors to disk so they survive server restarts)
CHROMA_PATH = "./chroma_db"
os.makedirs(CHROMA_PATH, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

COLLECTION_NAME = "rag_documents"
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# Optional configuration for local LLM generation via Ollama (Make sure ollama is running locally if used)
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3" # or mistral, phi3, etc.


# ---------------------------------------------------------
# 3. Pydantic Models for Request Validation
# ---------------------------------------------------------
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3
    use_llm: Optional[bool] = False  # Toggle true if Ollama is running locally


# ---------------------------------------------------------
# 4. Helper Functions
# ---------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits long text documents into overlapping sliding-window chunks 
    to maintain context across chunk boundaries.
    """
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


# ---------------------------------------------------------
# 5. API Endpoints
# ---------------------------------------------------------

@app.post("/documents", summary="Upload and ingest a PDF or TXT document")
async def upload_document(file: UploadFile = File(...)):
    """
    Ingests a document:
    1. Extracts text based on format (.pdf or .txt)
    2. Chunks the text using a sliding window
    3. Generates dense embeddings using sentence-transformers
    4. Persists chunks, vectors, and source metadata into ChromaDB
    """
    filename = file.filename
    content_type = file.content_type
    file_bytes = await file.read()
    extracted_text = ""

    # Parse based on file type
    if content_type == "application/pdf" or filename.endswith(".pdf"):
        try:
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(e)}")
            
    elif content_type == "text/plain" or filename.endswith(".txt"):
        try:
            extracted_text = file_bytes.decode("utf-8")
        except Exception:
            extracted_text = file_bytes.decode("latin-1")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Only PDF and TXT are supported.")

    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="The uploaded document contains no readable text.")

    # Chunk text
    chunks = chunk_text(extracted_text)
    
    # Generate unique structural IDs and embeddings
    doc_id = str(uuid.uuid4())
    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

    # Save to persistent vector store
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    return {
        "status": "success",
        "document_id": doc_id,
        "filename": filename,
        "total_chunks_ingested": len(chunks)
    }


@app.get("/documents", summary="List overview of all ingested data")
def list_documents():
    """Returns total chunk counts and unique active sources present in the vector database."""
    data = collection.get()
    sources = set()
    if data and "metadatas" in data:
        for meta in data["metadatas"]:
            if meta and "source" in meta:
                sources.add(meta["source"])
                
    return {
        "total_chunks_stored": len(data["ids"]) if data and "ids" in data else 0,
        "unique_sources": list(sources)
    }


@app.post("/query", summary="Ask a question and receive context-grounded responses")
def query_rag(request: QueryRequest):
    """
    1. Embeds user query
    2. Performs similarity search against ChromaDB (top-k)
    3. Returns context source chunks for verification
    4. Optional: Sends context + question to local Ollama LLM to synthesize final answer
    """
    # 1. Embed query vector
    query_embedding = embedding_model.encode([request.question]).tolist()

    # 2. Retrieve top matching chunks
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=request.top_k
    )

    retrieved_documents = results.get("documents", [[]])[0]
    retrieved_metadatas = results.get("metadatas", [[]])[0]

    if not retrieved_documents:
        return {
            "question": request.question,
            "answer": "No relevant context was found in the database to answer your question.",
            "sources": []
        }

    # Format sources for debugging/transparency (essential for trust)
    formatted_sources = [
        {"chunk_text": doc, "metadata": meta} 
        for doc, meta in zip(retrieved_documents, retrieved_metadatas)
    ]

    answer = "Context successfully retrieved."

    # 3. Optional LLM integration (Ollama)
    if request.use_llm:
        context_str = "\n\n".join(retrieved_documents)
        prompt = f"""You are a helpful AI assistant. Answer the question accurately using ONLY the provided context below. If you do not know the answer based on the context, state that you don't know.

Context:
{context_str}

Question: {request.question}
Answer:"""

        try:
            llm_response = requests.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=30
            )
            if llm_response.status_code == 200:
                answer = llm_response.json().get("response", "Error parsing response from local LLM.")
            else:
                answer = "Failed to communicate with local Ollama instance."
        except requests.exceptions.RequestException:
            answer = "Ollama connection error. Ensure Ollama is running locally if 'use_llm' is true."

    return {
        "question": request.question,
        "answer": answer,
        "retrieved_sources": formatted_sources
    }


@app.delete("/documents/{source_name}", summary="Delete document chunks by source filename")
def delete_document(source_name: str):
    """Removes all embedded chunks associated with a specific file name from disk."""
    data = collection.get(where={"source": source_name})
    if not data or not data["ids"]:
        raise HTTPException(status_code=404, detail=f"Source '{source_name}' not found in database.")
    
    collection.delete(ids=data["ids"])
    return {
        "status": "success", 
        "message": f"Successfully removed {len(data['ids'])} chunks associated with '{source_name}'."
    }