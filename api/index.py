import os
import sys
import json
import tempfile
import traceback
import math
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Initialize environment variables
load_dotenv()

# We only need standard libraries and google-generativeai/unstructured-client
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared, operations

# --- API Config ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Persistent file path (Local Windows vs Vercel Cloud)
if os.name == 'nt':
    INDEX_PATH = os.path.join(os.getcwd(), "multimodal_index.json")
else:
    INDEX_PATH = "/tmp/multimodal_index.json"

GEMINI_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
UNSTRUCTURED_KEY = (os.getenv("UNSTRUCTURED_API_KEY") or "").strip()

# Initialize Google AI for embeddings
genai.configure(api_key=GEMINI_KEY)

# --- Helper Logic ---

def dot_product(v1: List[float], v2: List[float]) -> float:
    return sum(a * b for a, b in zip(v1, v2))

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings from Gemini in batch using a stabilized model retry strategy."""
    if not texts: return []
    # Try the modern model first, fallback to the legacy name if needed
    for model_name in ["models/text-embedding-004", "models/gemini-embedding-001"]:
        try:
            res = genai.embed_content(
                model=model_name,
                content=texts,
                task_type="retrieval_document"
            )
            return res["embedding"]
        except Exception as e:
            if model_name == "models/gemini-embedding-001":
                 raise e
            print(f"INFO: {model_name} failed in batch mode, trying fallback...")
            continue
    return []

def get_single_embedding(text: str) -> List[float]:
    """Get a single embedding using a stabilized model retry strategy."""
    for model_name in ["models/text-embedding-004", "models/gemini-embedding-001"]:
        try:
            res = genai.embed_content(
                model=model_name,
                content=text,
                task_type="retrieval_query"
            )
            return res["embedding"]
        except Exception as e:
            if model_name == "models/gemini-embedding-001":
                 raise e
            continue
    return []

# --- Core RAG Logic ---

def process_and_persist(file_path: str):
    """
    Cloud-native partitioning and JSON persistence.
    Bypasses ANY heavy vector database binaries.
    """
    if not UNSTRUCTURED_KEY:
        raise ValueError("UNSTRUCTURED_API_KEY is missing.")

    client = UnstructuredClient(
        api_key_auth=UNSTRUCTURED_KEY,
        server_url="https://api.unstructuredapp.io"
    )

    with open(file_path, "rb") as f:
        file_content = f.read()

    # Partitioning (Cloud)
    req = operations.PartitionRequest(
        partition_parameters=shared.PartitionParameters(
            files=shared.Files(content=file_content, file_name=file_path),
            strategy="fast", # Speed up for Vercel
            extract_image_block_types=["Image", "Table"],
            chunking_strategy="by_title",
            max_characters=1500 # Smaller chunks for faster embedding
        )
    )

    # 1. Partition
    res = client.general.partition(request=req)
    elements = res.elements
    
    # 2. Extract Text
    texts = [el.get("text", "") for el in elements]
    
    # 3. Embed (Cloud)
    vectors = get_embeddings(texts)
    
    # 4. Save to JSON for cross-invocation persistence on Vercel
    index_data = []
    for i, el in enumerate(elements):
        metadata = el.get("metadata", {})
        index_data.append({
            "text": texts[i],
            "vector": vectors[i],
            "metadata": {
                "tables": [metadata["text_as_html"]] if metadata.get("text_as_html") else [],
                "images": [metadata["image_base64"]] if metadata.get("image_base64") else []
            }
        })
    
    with open(INDEX_PATH, "w") as f:
        json.dump(index_data, f)
    
    return len(index_data)

# --- Endpoints ---

@app.get("/api")
def health():
    return {"status": "online", "key_check": "ok" if (GEMINI_KEY and UNSTRUCTURED_KEY) else "fail"}

@app.post("/api/ingest")
def ingest(file: UploadFile = File(...)):
    try:
        suffix = os.path.splitext(file.filename or ".pdf")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name
        
        count = process_and_persist(tmp_path)
        os.unlink(tmp_path)
        return {"status": "success", "count": count}
    except Exception as e:
        print(f"INGEST ERROR: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
def query_endpoint(query: str = Form(...)):
    if not os.path.exists(INDEX_PATH):
        raise HTTPException(status_code=400, detail="No document indexed. Please re-upload.")
        
    try:
        # 1. Embed Query (Using stabilized helper)
        query_vector = get_single_embedding(query)
        
        # 2. Load Index
        with open(INDEX_PATH, "r") as f:
            index_data = json.load(f)
        
        # 3. Linear Scan (Zero Binary Search)
        scored_docs = []
        for doc in index_data:
            score = dot_product(query_vector, doc["vector"])
            scored_docs.append((score, doc))
        
        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_k = scored_docs[:3]
        
        # 4. Generate Answer using Gemini 2.0
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=GEMINI_KEY,
            temperature=0
        )
        
        prompt = f"Answer the user query based on the context provided.\n\nQuery: {query}\n\nContext:\n"
        content = []
        
        for i, (score, doc) in enumerate(top_k):
            prompt += f"\n--- BLOCK {i+1} ---\n{doc['text']}\n"
            # Support images
            if doc["metadata"]["images"]:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{doc['metadata']['images'][0]}"}})
                break # Send top 1 image for speed
        
        content.insert(0, {"type": "text", "text": prompt})
        
        ans = llm.invoke([HumanMessage(content=content)])
        
        return {
            "answer": ans.content,
            "chunks": [{"page_content": d[1]["text"]} for d in top_k]
        }
    except Exception as e:
        print(f"QUERY ERROR: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
