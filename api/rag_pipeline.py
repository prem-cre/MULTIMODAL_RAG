import os
import json
import tempfile
import shutil
from typing import List, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

from unstructured_client import UnstructuredClient
from unstructured_client.models import shared

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
DEFAULT_PERSIST_DIR = os.path.join(tempfile.gettempdir(), "chroma_db")

# ─── NUKE stale chroma state at import time ───────────────────────────────────
# This runs once when uvicorn loads the module.
# It removes the old /tmp/chroma_db folder (which may carry conflicting settings
# from a previous paid-key session) AND wipes chromadb's in-process singleton
# registry, so nothing from a prior run can pollute this one.
def _nuke_chroma_state():
    # 1. Delete old persist folder
    if os.path.exists(DEFAULT_PERSIST_DIR):
        try:
            shutil.rmtree(DEFAULT_PERSIST_DIR, ignore_errors=True)
            print(f"STARTUP: Deleted stale chroma dir → {DEFAULT_PERSIST_DIR}")
        except Exception as e:
            print(f"STARTUP: Could not delete chroma dir: {e}")

    # 2. Wipe chromadb in-process singleton cache (0.4.x – 0.6.x)
    try:
        from chromadb.api.client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
        print("STARTUP: Cleared SharedSystemClient cache")
    except Exception:
        pass

    try:
        import chromadb.api as _api
        if hasattr(_api, "_client_cache"):
            _api._client_cache.clear()
    except Exception:
        pass

_nuke_chroma_state()   # ← runs immediately on import

# ─── Module-level in-memory store ─────────────────────────────────────────────
_chroma_client = None
_vector_db = None
_COLLECTION = "multimodal_rag_collection"


def _get_fresh_client():
    """
    Return a fresh EphemeralClient.
    EphemeralClient is 100 % in-memory:
      • no SQLite files → no settings hash → no 'different settings' error
      • works identically with free or paid Gemini keys
    """
    import chromadb
    return chromadb.EphemeralClient()


def is_document_indexed() -> bool:
    return _vector_db is not None


# ─── API helpers ──────────────────────────────────────────────────────────────

def get_api_key() -> Optional[str]:
    return os.getenv("GEMINI_API_KEY")

def get_unstructured_api_key() -> Optional[str]:
    return os.getenv("UNSTRUCTURED_API_KEY")

def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=get_api_key(),
    )


# ─── Unstructured cloud partitioning ─────────────────────────────────────────

def partition_and_chunk_document(file_path: str, status_callback: Optional[Callable] = None):
    if status_callback:
        status_callback("☁️ Requesting Unstructured Cloud (Remote OCR & Chunking)...")

    api_key = get_unstructured_api_key()
    if not api_key:
        raise ValueError("UNSTRUCTURED_API_KEY is missing. Add it to your .env file.")

    client = UnstructuredClient(
        api_key_auth=api_key,
        server_url=os.getenv(
            "UNSTRUCTURED_API_URL",
            "https://api.unstructuredapp.io/general/v0/general",
        ),
    )

    with open(file_path, "rb") as f:
        file_content = f.read()

    from unstructured_client.models import operations

    req = operations.PartitionRequest(
        partition_parameters=shared.PartitionParameters(
            files=shared.Files(
                content=file_content,
                file_name=os.path.basename(file_path),
            ),
            strategy="hi_res",
            extract_image_block_types=["Image", "Table"],
            chunking_strategy="by_title",
            max_characters=3000,
            combine_under_n_chars=500,
        )
    )

    print("DEBUG: Sending to Unstructured Cloud API...")
    try:
        res = client.general.partition(request=req)
        print(f"DEBUG: Received {len(res.elements)} chunks from Unstructured.")
        if status_callback:
            status_callback(f"✅ Received {len(res.elements)} chunks from remote API")
        return res.elements
    except Exception as e:
        if status_callback:
            status_callback(f"❌ Cloud Partition failed: {str(e)}")
        raise


# ─── Content helpers ──────────────────────────────────────────────────────────

def separate_content_types_from_dict(element: dict) -> dict:
    metadata = element.get("metadata", {})
    text = element.get("text", "")
    data = {"text": text, "tables": [], "images": [], "types": ["text"]}
    if metadata.get("text_as_html"):
        data["tables"].append(metadata["text_as_html"])
        data["types"].append("table")
    if metadata.get("image_base64"):
        data["images"].append(metadata["image_base64"])
        data["types"].append("image")
    data["types"] = list(set(data["types"]))
    return data


def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=0,
            google_api_key=get_api_key(),
        )
        prompt = (
            "Summarize this document chunk concisely for retrieval. "
            "Include key data from tables and describe any images.\n\n"
            f"TEXT:\n{text}\n\n"
        )
        if tables:
            prompt += "TABLES:\n" + "\n".join(tables)

        content = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})

        response = llm.invoke([HumanMessage(content=content)])
        return response.content
    except Exception as e:
        if "429" in str(e):
            return f"SUMMARY_PENDING: {text[:200]}... [Rate limit – raw text used]"
        return f"{text[:400]}... [Tables: {len(tables)}, Images: {len(images)}]"


def _process_chunk_dict_worker(args):
    i, element = args
    c = separate_content_types_from_dict(element)
    has_mm = "table" in c["types"] or "image" in c["types"]
    summary = create_ai_enhanced_summary(c["text"], c["tables"], c["images"]) if has_mm else c["text"]
    doc = Document(
        page_content=summary,
        metadata={
            "chunk_id": i + 1,
            "original_content": json.dumps({
                "raw_text": c["text"],
                "tables_html": c["tables"],
                "images_base64": c["images"],
            }),
        },
    )
    return i, doc


def summarise_chunks(chunk_dicts, status_callback=None):
    if status_callback:
        status_callback(f"🧠 Summarising {len(chunk_dicts)} chunks...")

    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_process_chunk_dict_worker, (i, el)): i for i, el in enumerate(chunk_dicts)}
        for future in as_completed(futures):
            idx, doc = future.result()
            results[idx] = doc
            if status_callback and (idx + 1) % 4 == 0:
                status_callback(f"  Processed {idx + 1}/{len(chunk_dicts)} segments...")

    return [results[i] for i in sorted(results)]


# ─── Vector store ─────────────────────────────────────────────────────────────

def create_vector_store(documents, persist_directory=None):
    """
    Build a fresh in-memory Chroma vector store.
    persist_directory is ignored (kept for API compatibility only).
    """
    global _chroma_client, _vector_db

    # Always start with a brand-new client so there is zero chance of
    # reusing a stale client that holds old settings references.
    _chroma_client = _get_fresh_client()

    embedding_model = get_embedding_model()

    _vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        client=_chroma_client,
        collection_name=_COLLECTION,
    )
    print(f"DEBUG: In-memory collection '{_COLLECTION}' created with {len(documents)} docs.")
    return _vector_db


# ─── Answer generation ────────────────────────────────────────────────────────

def generate_final_answer(chunks, query: str) -> str:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=0,
            google_api_key=get_api_key(),
        )
        prompt = (
            "Answer the query using the provided context "
            "(which may include text, tables, and image descriptions).\n\n"
            f"Query: {query}\n\nContext:\n"
        )
        for i, chunk in enumerate(chunks):
            prompt += f"\n--- CONTEXT {i + 1} ---\n{chunk.page_content}\n"

        content = [{"type": "text", "text": prompt}]
        for chunk in chunks:
            if "original_content" in chunk.metadata:
                d = json.loads(chunk.metadata["original_content"])
                for img in d.get("images_base64", []):
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})

        response = llm.invoke([HumanMessage(content=content)])
        return response.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"


# ─── Public entrypoints ───────────────────────────────────────────────────────

def run_complete_ingestion_pipeline(pdf_path, persist_directory=None, status_callback=None):
    chunk_dicts = partition_and_chunk_document(pdf_path, status_callback)
    summarised = summarise_chunks(chunk_dicts, status_callback)
    db = create_vector_store(summarised)
    return db


def rag_query(query: str, persist_directory=None):
    global _vector_db
    if _vector_db is None:
        raise ValueError("No document indexed yet. Please upload a PDF first.")
    chunks = _vector_db.as_retriever(search_kwargs={"k": 3}).invoke(query)
    answer = generate_final_answer(chunks, query)
    return answer, chunks
