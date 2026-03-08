from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import json
from api.rag_pipeline import run_complete_ingestion_pipeline, rag_query

app = FastAPI()

# Enable CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api")
def hello_world():
    return {"message": "Hello from the Multimodal RAG API!"}

@app.post("/api/ingest")
async def ingest_document(file: UploadFile = File(...)):
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Vercel's filesystem is read-only, but /tmp is writable.
        # We store the vector DB in /tmp for the session.
        # NOTE: For persistent RAG, we'd need a hosted vector DB like Pinecone.
        # For this demo, let's stick to /tmp/chroma_db.
        persist_dir = "/tmp/chroma_db"
        run_complete_ingestion_pipeline(tmp_path, persist_directory=persist_dir)
        return {"status": "success", "message": "Document ingested successfully!"}
    except Exception as e:
        import traceback
        print(f"INGEST_ERROR: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Backend Ingest Error: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.post("/api/query")
async def query_document(query: str = Form(...)):
    try:
        persist_dir = "/tmp/chroma_db"
        if not os.path.exists(persist_dir):
             raise HTTPException(status_code=400, detail="Please ingest a document first.")
        
        answer, chunks = rag_query(query, persist_directory=persist_dir)
        
        # Format chunks for JSON response
        formatted_chunks = []
        for c in chunks:
            formatted_chunks.append({
                "page_content": c.page_content,
                "metadata": c.metadata
            })
            
        return {
            "answer": answer,
            "chunks": formatted_chunks
        }
    except Exception as e:
        import traceback
        print(f"QUERY_ERROR: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Backend Query Error: {str(e)}")
