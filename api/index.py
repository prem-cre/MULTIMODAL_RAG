import os
import sys
from dotenv import load_dotenv
import traceback
import json
import tempfile
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared, operations

# --- 1. Load Keys (Production & Local) ---
load_dotenv()

# Global keys with stripping for safety
GEMINI_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
UNSTRUCTURED_KEY = (os.getenv("UNSTRUCTURED_API_KEY") or "").strip()

print(f"DEBUG: STARTUP - Gemini Key: {'SET' if GEMINI_KEY else 'MISSING'}")
print(f"DEBUG: STARTUP - Unstructured Key: {'SET' if UNSTRUCTURED_KEY else 'MISSING'}")

# --- 2. FastAPI Setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared In-Memory State
# This persists for the life of the worker process (minutes on Vercel)
class VectorState:
    store: Optional[InMemoryVectorStore] = None

state = VectorState()

def get_embedding_model():
    if not GEMINI_KEY:
        raise ValueError("Missing GEMINI_API_KEY")
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_KEY
    )

def get_llm():
    if not GEMINI_KEY:
        raise ValueError("Missing GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=GEMINI_KEY,
        temperature=0
    )

# --- 3. Processing Core ---

def process_and_index(file_path: str):
    if not UNSTRUCTURED_KEY:
        raise ValueError("Missing UNSTRUCTURED_API_KEY. Please set it in .env or Vercel dashboard.")

    print(f"DEBUG: Processing {file_path} via Unstructured Cloud...")
    
    client = UnstructuredClient(
        api_key_auth=UNSTRUCTURED_KEY,
        server_url="https://api.unstructuredapp.io"
    )

    with open(file_path, "rb") as f:
        file_content = f.read()

    req = operations.PartitionRequest(
        partition_parameters=shared.PartitionParameters(
            files=shared.Files(content=file_content, file_name=file_path),
            strategy="fast",
            extract_image_block_types=["Image", "Table"],
            chunking_strategy="by_title",
            max_characters=2000
        )
    )

    try:
        res = client.general.partition(request=req)
        print(f"DEBUG: Partition Success. Elements: {len(res.elements)}")
        
        docs = []
        for i, el in enumerate(res.elements):
            metadata = el.get("metadata", {})
            mm_payload = {
                "text": el.get("text", ""),
                "tables": [metadata["text_as_html"]] if metadata.get("text_as_html") else [],
                "images": [metadata["image_base64"]] if metadata.get("image_base64") else []
            }
            docs.append(Document(
                page_content=el.get("text", ""),
                metadata={"chunk_id": i, "mm_payload": json.dumps(mm_payload)}
            ))
        
        # Initialize/Update in-memory store
        state.store = InMemoryVectorStore.from_documents(docs, get_embedding_model())
        return len(docs)
    
    except Exception as e:
        err_msg = str(e).lower()
        if "401" in err_msg or "unauthorized" in err_msg or "invalid_api_key" in err_msg:
             raise HTTPException(status_code=401, detail="API Key is invalid or expired. Check your .env file or Vercel dashboard.")
        if "429" in err_msg or "exhausted" in err_msg:
             raise HTTPException(status_code=429, detail="API Rate Limit reached (Free tier). Wait 60 seconds and try again.")
        print(f"CRITICAL ERROR in ingestion: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Backend Error: {str(e)}")

# --- 4. API Endpoints ---

@app.get("/api")
def health():
    return {
        "status": "online",
        "engine": "Gemini 2.0 Flash Lite",
        "keys": {
            "gemini": "Set" if GEMINI_KEY else "Missing",
            "unstructured": "Set" if UNSTRUCTURED_KEY else "Missing"
        }
    }

@app.post("/api/ingest")
def ingest(file: UploadFile = File(...)):
    if not GEMINI_KEY or not UNSTRUCTURED_KEY:
        raise HTTPException(
            status_code=401, 
            detail="Keys not found in environment. Add GEMINI_API_KEY and UNSTRUCTURED_API_KEY to your .env or Vercel dashboard."
        )
        
    try:
        suffix = os.path.splitext(file.filename or ".pdf")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name
        
        count = process_and_index(tmp_path)
        os.unlink(tmp_path)
        return {"status": "success", "count": count}
    except HTTPException:
        raise
    except Exception as e:
        err_msg = str(e).lower()
        if "401" in err_msg or "unauthorized" in err_msg:
             raise HTTPException(status_code=401, detail="Authentication failed. Verify your API keys.")
        if "429" in err_msg or "exhausted" in err_msg:
             raise HTTPException(status_code=429, detail="Quota exhausted. Please wait.")
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
def query_endpoint(query: str = Form(...)):
    if state.store is None:
        raise HTTPException(status_code=400, detail="Document index is missing. Please re-upload the PDF.")
        
    try:
        # Similarity Search
        related = state.store.similarity_search(query, k=3)
        
        # Multimodal Reasoning
        prompt = f"Using the context provided, answer the user query.\n\nQuery: {query}\n\nContext:\n"
        content = []
        
        for i, doc in enumerate(related):
            prompt += f"\n--- Context Block {i+1} ---\n{doc.page_content}\n"
            payload = json.loads(doc.metadata.get("mm_payload", "{}"))
            # High-performance multimodal: send ONLY the most relevant image
            if payload.get("images"):
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{payload['images'][0]}"}})
                break # Only one image to keep latency low on Vercel
        
        content.append({"type": "text", "text": prompt})
        
        llm = get_llm()
        response = llm.invoke([HumanMessage(content=content)])
        
        return {
            "answer": response.content,
            "chunks": [{"page_content": d.page_content} for d in related]
        }
    except Exception as e:
        err_msg = str(e).lower()
        if "429" in err_msg or "exhausted" in err_msg:
             raise HTTPException(status_code=429, detail="Gemini Rate Limit reached. Try again in 60s.")
        print(f"QUERY ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
