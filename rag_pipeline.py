import os
import sys
import json
import base64
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# Windows PATH configuration for Unstructured
# ==========================================
if sys.platform == "win32":
    tesseract_path = r"C:\Program Files\Tesseract-OCR"
    if os.path.exists(tesseract_path) and tesseract_path not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + tesseract_path

    poppler_base = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages")
    if os.path.exists(poppler_base):
        for root, dirs, files in os.walk(poppler_base):
            if "pdfinfo.exe" in files:
                if root not in os.environ["PATH"]:
                    os.environ["PATH"] += os.pathsep + root
                break
# ==========================================

# Silence HuggingFace unauthenticated warning if token is available
_hf_token = os.getenv("HF_TOKEN")
if _hf_token:
    os.environ["HUGGINGFACE_TOKEN"] = _hf_token

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ── Singleton embedding model (loaded once per process, never again) ───────────
_embedding_model: Optional[HuggingFaceEmbeddings] = None

def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return a process-level cached HuggingFace embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_model


def get_api_key() -> Optional[str]:
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except (ImportError, FileNotFoundError):
        pass
    return os.getenv("GEMINI_API_KEY")


# ── Strategy auto-detection ────────────────────────────────────────────────────

def _detect_strategy(file_path: str) -> str:
    """
    Probe page 1 with the fast strategy.
    If it yields almost no text the PDF is likely scanned → fall back to hi_res.
    For normal text-based PDFs this probe takes < 1 second.
    """
    try:
        sample = partition_pdf(filename=file_path, strategy="fast", pages=[1])
        text = " ".join(el.text for el in sample if hasattr(el, "text"))
        return "hi_res" if len(text.strip()) < 80 else "auto"
    except Exception:
        return "auto"


# ── Step 1 – Partition ─────────────────────────────────────────────────────────

def partition_document(
    file_path: str,
    status_callback: Optional[Callable] = None,
    force_strategy: Optional[str] = None,
):
    strategy = force_strategy or _detect_strategy(file_path)
    label = "hi-res OCR 🔬" if strategy == "hi_res" else "fast text-extract ⚡"

    if status_callback:
        status_callback(f"📄 Partitioning document — strategy: **{label}**")

    elements = partition_pdf(
        filename=file_path,
        strategy=strategy,
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
    )

    if status_callback:
        type_counts: Dict[str, int] = {}
        for el in elements:
            t = type(el).__name__
            type_counts[t] = type_counts.get(t, 0) + 1
        breakdown = ", ".join(
            f"{v} {k}{'s' if v > 1 else ''}" for k, v in type_counts.items()
        )
        status_callback(f"✅ Extracted **{len(elements)} elements** — {breakdown}")

    return elements


# ── Step 2 – Chunk ─────────────────────────────────────────────────────────────

def create_chunks_by_title(elements, status_callback: Optional[Callable] = None):
    if status_callback:
        status_callback("✂️ Splitting into smart chunks by title…")

    chunks = chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500,
    )

    if status_callback:
        status_callback(f"✅ Created **{len(chunks)} chunks** (max 3 000 chars each)")

    return chunks


# ── Helpers ────────────────────────────────────────────────────────────────────

def separate_content_types(chunk) -> Dict:
    data: Dict[str, Any] = {
        "text": chunk.text,
        "tables": [],
        "images": [],
        "types": ["text"],
    }
    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        for el in chunk.metadata.orig_elements:
            etype = type(el).__name__
            if etype == "Table":
                data["types"].append("table")
                data["tables"].append(getattr(el.metadata, "text_as_html", el.text))
            elif etype == "Image":
                if hasattr(el, "metadata") and hasattr(el.metadata, "image_base64"):
                    data["types"].append("image")
                    data["images"].append(el.metadata.image_base64)
    data["types"] = list(set(data["types"]))
    return data


def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0,
            google_api_key=get_api_key(),
        )
        prompt = (
            "You are creating a searchable description for document content retrieval.\n\n"
            f"TEXT CONTENT:\n{text}\n\n"
        )
        if tables:
            prompt += "TABLES:\n" + "".join(f"Table {i+1}:\n{t}\n\n" for i, t in enumerate(tables))
        prompt += (
            "Generate a comprehensive, searchable description covering:\n"
            "1. Key facts, numbers, and data points\n"
            "2. Main topics and concepts\n"
            "3. Questions this content could answer\n"
            "4. Visual content analysis (if images present)\n"
            "5. Alternative search terms\n\n"
            "SEARCHABLE DESCRIPTION:"
        )
        content: List[Dict] = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
        response = llm.invoke([HumanMessage(content=content)])
        return response.content
    except Exception as e:
        fallback = f"{text[:300]}…"
        if tables:
            fallback += f" [Contains {len(tables)} table(s)]"
        if images:
            fallback += f" [Contains {len(images)} image(s)]"
        return fallback


# ── Step 3 – Parallel summarisation ───────────────────────────────────────────

def _process_chunk_worker(args):
    """Thread worker: summarise one chunk, return (index, Document)."""
    i, chunk = args
    c = separate_content_types(chunk)
    has_mm = "table" in c["types"] or "image" in c["types"]
    page_content = (
        create_ai_enhanced_summary(c["text"], c["tables"], c["images"])
        if has_mm else c["text"]
    )
    doc = Document(
        page_content=page_content,
        metadata={
            "chunk_id": i + 1,
            "original_content": json.dumps({
                "raw_text": c["text"],
                "tables_html": c["tables"],
                "images_base64": c["images"],
            }),
        },
    )
    return i, doc, has_mm


def summarise_chunks(
    chunks,
    status_callback: Optional[Callable] = None,
    progress_callback: Optional[Callable] = None,
    max_workers: int = 4,
):
    """
    Summarise all chunks.
    - Text-only chunks: processed instantly (no API call).
    - Multimodal chunks: Gemini calls run in parallel (max_workers threads).
    """
    total = len(chunks)

    # Pre-classify
    text_jobs, mm_jobs = [], []
    for i, chunk in enumerate(chunks):
        c = separate_content_types(chunk)
        if "table" in c["types"] or "image" in c["types"]:
            mm_jobs.append((i, chunk))
        else:
            text_jobs.append((i, chunk))

    if status_callback:
        status_callback(
            f"🧠 Summarising **{total} chunks** — "
            f"📝 {len(text_jobs)} text-only (instant) · "
            f"🤖 {len(mm_jobs)} multimodal (parallel AI, {max_workers} threads)"
        )

    results: Dict[int, Document] = {}
    completed = 0

    # Text-only — instant, no API
    for i, chunk in text_jobs:
        c = separate_content_types(chunk)
        results[i] = Document(
            page_content=c["text"],
            metadata={
                "chunk_id": i + 1,
                "original_content": json.dumps({
                    "raw_text": c["text"],
                    "tables_html": [],
                    "images_base64": [],
                }),
            },
        )
        completed += 1
        if progress_callback:
            progress_callback(completed / total)

    # Multimodal — parallel Gemini calls
    if mm_jobs:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_chunk_worker, job): job for job in mm_jobs}
            for future in as_completed(futures):
                job = futures[future]
                i = job[0]
                try:
                    idx, doc, _ = future.result()
                    results[idx] = doc
                    completed += 1
                    if status_callback:
                        status_callback(
                            f"🔍 Chunk {idx+1} summarised — "
                            f"**{completed}/{total}** done"
                        )
                except Exception as e:
                    if status_callback:
                        status_callback(f"⚠️ Chunk {i+1} failed ({e}) — using raw text")
                    c = separate_content_types(job[1])
                    results[i] = Document(
                        page_content=c["text"],
                        metadata={
                            "chunk_id": i + 1,
                            "original_content": json.dumps({
                                "raw_text": c["text"],
                                "tables_html": c["tables"],
                                "images_base64": c["images"],
                            }),
                        },
                    )
                    completed += 1
                finally:
                    if progress_callback:
                        progress_callback(completed / total)

    docs = [results[i] for i in sorted(results)]

    if status_callback:
        status_callback(
            f"✅ All **{total} chunks** processed — "
            f"{len(text_jobs)} text-only · {len(mm_jobs)} AI-enhanced"
        )

    return docs


# ── Step 4 – Vector store ──────────────────────────────────────────────────────

def create_vector_store(
    documents,
    persist_directory: str = "db_local/chroma_db",
    status_callback: Optional[Callable] = None,
):
    if status_callback:
        status_callback(
            "🔮 Loading embedding model — **cached after first load, instant on repeat runs**"
        )

    model = get_embedding_model()   # singleton — never re-downloads

    if status_callback:
        status_callback(f"💾 Embedding **{len(documents)} chunks** & writing to ChromaDB…")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )

    if status_callback:
        status_callback(
            f"✅ Vector store ready — **{len(documents)} chunks** indexed at `{persist_directory}`"
        )

    return vectorstore


# ── Answer generation ──────────────────────────────────────────────────────────

def generate_final_answer(chunks, query: str) -> str:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0,
            google_api_key=get_api_key(),
        )
        prompt = f"Based on the following documents, answer: {query}\n\nCONTENT:\n"
        for i, chunk in enumerate(chunks):
            prompt += f"--- Document {i+1} ---\n"
            if "original_content" in chunk.metadata:
                d = json.loads(chunk.metadata["original_content"])
                if d.get("raw_text"):
                    prompt += f"TEXT:\n{d['raw_text']}\n\n"
                for j, tbl in enumerate(d.get("tables_html", [])):
                    prompt += f"Table {j+1}:\n{tbl}\n\n"
            prompt += "\n"
        prompt += (
            "Provide a clear, comprehensive answer. "
            "If the documents don't have enough info, say so.\n\nANSWER:"
        )
        content: List[Dict] = [{"type": "text", "text": prompt}]
        for chunk in chunks:
            if "original_content" in chunk.metadata:
                d = json.loads(chunk.metadata["original_content"])
                for img in d.get("images_base64", []):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                    })
        response = llm.invoke([HumanMessage(content=content)])
        return response.content
    except Exception as e:
        return f"Sorry, I encountered an error: {e}"


# ── Public API ─────────────────────────────────────────────────────────────────

def run_complete_ingestion_pipeline(
    pdf_path: str,
    persist_directory: str = "db_local/chroma_db",
    status_callback: Optional[Callable] = None,
    progress_callback: Optional[Callable] = None,
    force_strategy: Optional[str] = None,
    max_workers: int = 4,
):
    """
    Full ingestion pipeline: partition → chunk → summarise → embed → store.

    Parameters
    ----------
    pdf_path          Path to the PDF.
    persist_directory ChromaDB storage path.
    status_callback   callable(str) — markdown status lines for the UI.
    progress_callback callable(float 0–1) — drives a progress bar.
    force_strategy    "auto" | "hi_res" | None (auto-detect from content).
    max_workers       Parallel threads for Gemini AI summary calls (default 4).
    """
    elements   = partition_document(pdf_path, status_callback, force_strategy)
    chunks     = create_chunks_by_title(elements, status_callback)
    summarised = summarise_chunks(chunks, status_callback, progress_callback, max_workers)
    db         = create_vector_store(summarised, persist_directory, status_callback)

    if status_callback:
        status_callback("🎉 **Pipeline complete!** Your document is ready to query.")

    return db


def rag_query(
    query: str,
    persist_directory: str = "db_local/chroma_db",
    top_k: int = 3,
):
    """
    Retrieve top-k chunks and generate a grounded answer.

    Returns
    -------
    answer : str
    chunks : list[Document]
    """
    model = get_embedding_model()   # reuses cached model — no reload
    db = Chroma(persist_directory=persist_directory, embedding_function=model)
    chunks = db.as_retriever(search_kwargs={"k": top_k}).invoke(query)
    answer = generate_final_answer(chunks, query)
    return answer, chunks