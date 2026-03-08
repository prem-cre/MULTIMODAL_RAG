import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.rag_pipeline import (
    run_complete_ingestion_pipeline,
    rag_query,
    is_document_indexed,
    DEFAULT_PERSIST_DIR,
)

app = FastAPI()

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
    suffix = os.path.splitext(file.filename or "upload")[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        run_complete_ingestion_pipeline(tmp_path, persist_directory=DEFAULT_PERSIST_DIR)
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
        # Use module-level flag instead of filesystem check
        if not is_document_indexed():
            raise HTTPException(
                status_code=400,
                detail="No document indexed yet. Please upload a PDF first.",
            )

        answer, chunks = rag_query(query, persist_directory=DEFAULT_PERSIST_DIR)

        formatted_chunks = [
            {"page_content": c.page_content, "metadata": c.metadata}
            for c in chunks
        ]

        return {"answer": answer, "chunks": formatted_chunks}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"QUERY_ERROR: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Backend Query Error: {str(e)}")
