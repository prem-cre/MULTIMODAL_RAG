import os
import json
import base64
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# We use the unstructured-client to avoid needing local Tesseract/Poppler on Vercel
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared

load_dotenv()

def get_api_key() -> Optional[str]:
    # Check streamlit secrets (if running in hybrid) or local env
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except (ImportError, FileNotFoundError):
        pass
    return os.getenv("GEMINI_API_KEY")

def get_unstructured_api_key() -> Optional[str]:
    return os.getenv("UNSTRUCTURED_API_KEY")

def get_embedding_model():
    """Cloud-based embeddings for Vercel memory efficiency."""
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=get_api_key()
    )

def partition_document(
    file_path: str,
    status_callback: Optional[Callable] = None,
    force_strategy: Optional[str] = None
):
    """
    Cloud-native partitioning using Unstructured API.
    Zero local dependencies (no Tesseract/Poppler).
    """
    if status_callback:
        status_callback("☁️ Requesting Unstructured Cloud API (Hi-Res Remote OCR)...")

    # Fallback to local 'fast' strategy if no API key is provided, although it's limited
    api_key = get_unstructured_api_key()
    if not api_key:
        if status_callback:
            status_callback("⚠️ No UNSTRUCTURED_API_KEY found. Falling back to simple text extraction...")
        from unstructured.partition.pdf import partition_pdf
        return partition_pdf(filename=file_path, strategy="fast")

    client = UnstructuredClient(
        api_key_auth=api_key,
        server_url=os.getenv("UNSTRUCTURED_API_URL", "https://api.unstructured.io/general/v0/general")
    )

    with open(file_path, "rb") as f:
        file_content = f.read()

    req = shared.PartitionParameters(
        files=shared.Files(
            content=file_content,
            file_name=file_path,
        ),
        strategy="hi_res",
        extract_image_block_types=["Image", "Table"],
    )

    try:
        res = client.general.partition(req)
        # Convert API elements into local unstructured elements for chunking
        # This is a simplified transformation for demo purposes
        from unstructured.staging.base import dict_to_elements
        elements = dict_to_elements(res.elements)
        
        if status_callback:
            status_callback(f"✅ Extracted {len(elements)} elements from remote API")
        return elements
    except Exception as e:
        if status_callback:
            status_callback(f"❌ Remote API failed: {e}. Using fast local fallback.")
        from unstructured.partition.pdf import partition_pdf
        return partition_pdf(filename=file_path, strategy="fast")

def create_chunks_by_title(elements, status_callback: Optional[Callable] = None):
    from unstructured.chunking.title import chunk_by_title
    if status_callback:
        status_callback("✂️ Creating smart chunks...")
    return chunk_by_title(elements, max_characters=3000, new_after_n_chars=2400)

def separate_content_types(chunk) -> Dict:
    data: Dict[str, Any] = {
        "text": getattr(chunk, 'text', str(chunk)),
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
            elif etype == "Image" or etype == "Figure":
                if hasattr(el, "metadata") and hasattr(el.metadata, "image_base64"):
                    data["types"].append("image")
                    data["images"].append(el.metadata.image_base64)
    data["types"] = list(set(data["types"]))
    return data

def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            google_api_key=get_api_key(),
        )
        prompt = (
            "Summarize this document chunk for retrieval. Include data from tables and describe any images.\n\n"
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
        return f"{text[:400]}... [Contains {len(tables)} tables, {len(images)} images]"

def _process_chunk_worker(args):
    i, chunk = args
    c = separate_content_types(chunk)
    # Only use AI for multimodal chunks to save API usage
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

def summarise_chunks(chunks, status_callback=None):
    if status_callback:
        status_callback(f"🧠 Summarising {len(chunks)} chunks using Gemini 1.5 Flash...")
    
    results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_process_chunk_worker, (i, chunk)): i for i, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            idx, doc = future.result()
            results[idx] = doc
            if status_callback and (idx + 1) % 5 == 0:
                status_callback(f"  Processed {idx + 1}/{len(chunks)} chunks...")
    
    return [results[i] for i in sorted(results)]

def create_vector_store(documents, persist_directory="/tmp/chroma_db"):
    model = get_embedding_model()
    # Chroma in Vercel - persistent directory must be in /tmp
    return Chroma.from_documents(
        documents=documents,
        embedding=model,
        persist_directory=persist_directory
    )

def generate_final_answer(chunks, query: str) -> str:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            google_api_key=get_api_key(),
        )
        prompt = f"Answer the query using the provided context (text, tables, and images).\n\nQuery: {query}\n\nContext:\n"
        for i, chunk in enumerate(chunks):
            prompt += f"\n--- CHUNK {i+1} ---\n{chunk.page_content}\n"
        
        content = [{"type": "text", "text": prompt}]
        # Extract images from context for multimodal reasoning
        for chunk in chunks:
            if "original_content" in chunk.metadata:
                d = json.loads(chunk.metadata["original_content"])
                for img in d.get("images_base64", []):
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
        
        response = llm.invoke([HumanMessage(content=content)])
        return response.content
    except Exception as e:
        return f"Error generating answer: {e}"

def run_complete_ingestion_pipeline(pdf_path, persist_directory="/tmp/chroma_db", status_callback=None):
    elements = partition_document(pdf_path, status_callback)
    chunks = create_chunks_by_title(elements, status_callback)
    summarised = summarise_chunks(chunks, status_callback)
    db = create_vector_store(summarised, persist_directory)
    if status_callback:
        status_callback("🎉 Successfully indexed document in vector store.")
    return db

def rag_query(query: str, persist_directory="/tmp/chroma_db"):
    model = get_embedding_model()
    db = Chroma(persist_directory=persist_directory, embedding_function=model)
    chunks = db.as_retriever(search_kwargs={"k": 3}).invoke(query)
    answer = generate_final_answer(chunks, query)
    return answer, chunks