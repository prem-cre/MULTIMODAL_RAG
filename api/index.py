import os
import sys
import tempfile
import traceback
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add the CURRENT directory to the path so Vercel can find rag_pipeline.py
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import functions from the sibling file (rag_pipeline.py)
try:
    from rag_pipeline import run_complete_ingestion_pipeline, rag_query
except ImportError:
    # Fallback for local development if running from root folder
    from api.rag_pipeline import run_complete_ingestion_pipeline, rag_query

app = FastAPI()

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared persistent path (Use project-local on Windows, /tmp on Vercel)
if os.name == 'nt':
    PERSIST_DIR = os.path.join(os.getcwd(), "chroma_db")
else:
    PERSIST_DIR = "/tmp/chroma_db"

@app.get("/api")
def hello_world():
    return {"message": "Hello from the Multimodal RAG API!"}

@app.post("/api/ingest")
def ingest_document(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "upload.pdf")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = file.file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        run_complete_ingestion_pipeline(tmp_path, persist_directory=PERSIST_DIR)
        return {"status": "success", "message": "Document ingested successfully!"}
    except Exception as e:
        print(f"INGEST_ERROR: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.post("/api/query")
def query_document(query: str = Form(...)):
    try:
        # Check if index exists on disk
        if not os.path.exists(PERSIST_DIR):
             raise HTTPException(status_code=400, detail="No document index found. Please upload a PDF first.")
        
        answer, chunks = rag_query(query, persist_directory=PERSIST_DIR)
        
        # Format chunks for JSON response
        formatted_chunks = [
            {"page_content": c.page_content, "metadata": c.metadata}
            for c in chunks
        ]
            
        return {
            "answer": answer,
            "chunks": formatted_chunks
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"QUERY_ERROR: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
