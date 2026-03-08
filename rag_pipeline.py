import os
import sys
import json
import base64
from typing import List, Dict, Any

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

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()


def get_api_key():
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except ImportError:
        pass
    except FileNotFoundError:
        pass
    return os.getenv("GEMINI_API_KEY")


def partition_document(file_path: str, status_callback=None):
    """Extract elements from PDF using unstructured"""
    if status_callback:
        status_callback("📄 Reading & partitioning document (hi-res mode)…")

    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True
    )

    if status_callback:
        # Count element types for a richer status message
        type_counts: Dict[str, int] = {}
        for el in elements:
            t = type(el).__name__
            type_counts[t] = type_counts.get(t, 0) + 1
        breakdown = ", ".join(f"{v} {k}{'s' if v > 1 else ''}" for k, v in type_counts.items())
        status_callback(f"✅ Extracted **{len(elements)} elements** — {breakdown}")

    return elements


def create_chunks_by_title(elements, status_callback=None):
    """Create intelligent chunks using title-based strategy"""
    if status_callback:
        status_callback("✂️ Splitting into smart chunks by title…")

    chunks = chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500
    )

    if status_callback:
        status_callback(f"✅ Created **{len(chunks)} chunks** (max 3 000 chars each)")

    return chunks


def separate_content_types(chunk):
    """Analyse what types of content are in a chunk"""
    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }

    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            if element_type == 'Table':
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)

            elif element_type == 'Image':
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)

    content_data['types'] = list(set(content_data['types']))
    return content_data


def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    """Create AI-enhanced summary for mixed content using Gemini"""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", temperature=0, google_api_key=get_api_key()
        )

        prompt_text = f"""You are creating a searchable description for document content retrieval.

CONTENT TO ANALYZE:
TEXT CONTENT:
{text}

"""

        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"

        prompt_text += """
YOUR TASK:
Generate a comprehensive, searchable description that covers:

1. Key facts, numbers, and data points from text and tables
2. Main topics and concepts discussed  
3. Questions this content could answer
4. Visual content analysis (charts, diagrams, patterns in images)
5. Alternative search terms users might use

Make it detailed and searchable - prioritize findability over brevity.

SEARCHABLE DESCRIPTION:"""

        message_content = [{"type": "text", "text": prompt_text}]

        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })

        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        return response.content

    except Exception as e:
        summary = f"{text[:300]}…"
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary


def summarise_chunks(chunks, status_callback=None, progress_callback=None):
    """Process all chunks with AI Summaries — with per-chunk live updates"""
    if status_callback:
        status_callback("🧠 Generating AI summaries for each chunk…")

    langchain_documents = []
    total = len(chunks)
    text_only = 0
    multimodal = 0

    for i, chunk in enumerate(chunks):
        content_types = separate_content_types(chunk)

        has_table = 'table' in content_types['types']
        has_image = 'image' in content_types['types']

        # Live per-chunk status
        if status_callback:
            parts = []
            if has_table:
                parts.append(f"📊 {len(content_types['tables'])} table(s)")
            if has_image:
                parts.append(f"🖼️ {len(content_types['images'])} image(s)")
            content_label = ", ".join(parts) if parts else "📝 text only"
            status_callback(
                f"🔍 Processing chunk {i+1}/{total} — {content_label}"
            )

        if progress_callback:
            progress_callback((i + 1) / total)

        metadata = {
            "chunk_id": i + 1,
            "original_content": json.dumps({
                "raw_text": content_types['text'],
                "tables_html": content_types['tables'],
                "images_base64": content_types['images']
            })
        }

        if has_image or has_table:
            multimodal += 1
            enhanced_content = create_ai_enhanced_summary(
                content_types['text'],
                content_types['tables'],
                content_types['images']
            )
            doc = Document(page_content=enhanced_content, metadata=metadata)
        else:
            text_only += 1
            doc = Document(page_content=content_types['text'], metadata=metadata)

        langchain_documents.append(doc)

    if status_callback:
        status_callback(
            f"✅ Summaries done — {text_only} text-only, {multimodal} multimodal (tables/images)"
        )

    return langchain_documents


def create_vector_store(documents, persist_directory="db_local/chroma_db", status_callback=None):
    """Create and persist ChromaDB vector store using local HuggingFace Embeddings"""
    if status_callback:
        status_callback("🔮 Loading embedding model (all-MiniLM-L6-v2)…")

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if status_callback:
        status_callback(f"💾 Embedding {len(documents)} chunks & saving to ChromaDB…")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    if status_callback:
        status_callback(f"✅ Vector store ready — {len(documents)} chunks indexed at `{persist_directory}`")

    return vectorstore


def generate_final_answer(chunks, query):
    """Generate final answer using multimodal content and Gemini"""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", temperature=0, google_api_key=get_api_key()
        )

        prompt_text = f"""Based on the following documents, please answer this question: {query}

CONTENT TO ANALYZE:
"""

        for i, chunk in enumerate(chunks):
            prompt_text += f"--- Document {i+1} ---\n"

            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])

                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text}\n\n"

                tables_html = original_data.get("tables_html", [])
                if tables_html:
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html):
                        prompt_text += f"Table {j+1}:\n{table}\n\n"

            prompt_text += "\n"

        prompt_text += """
Please provide a clear, comprehensive answer using the text, tables, and images above. If the documents don't contain sufficient information to answer the question, say "I don't have enough information to answer that question based on the provided documents."

ANSWER:"""

        message_content = [{"type": "text", "text": prompt_text}]

        for chunk in chunks:
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                for image_base64 in original_data.get("images_base64", []):
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    })

        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        return response.content

    except Exception as e:
        return "Sorry, I encountered an error while generating the answer."


# ─────────────────────────────────────────────
# Public API used by Streamlit UI
# ─────────────────────────────────────────────

def run_complete_ingestion_pipeline(
    pdf_path: str,
    persist_directory="db_local/chroma_db",
    status_callback=None,
    progress_callback=None,
):
    """
    Run the complete RAG ingestion pipeline.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    persist_directory : str
        Where to persist ChromaDB.
    status_callback : callable(str) | None
        Called with a markdown string on every status update.
        Use this to pipe messages into st.status / st.markdown.
    progress_callback : callable(float) | None
        Called with a 0–1 float during the chunking phase so you can
        drive an st.progress bar.
    """
    elements = partition_document(pdf_path, status_callback)
    chunks = create_chunks_by_title(elements, status_callback)
    summarised = summarise_chunks(chunks, status_callback, progress_callback)
    db = create_vector_store(summarised, persist_directory, status_callback)

    if status_callback:
        status_callback("🎉 **Pipeline complete!** Your document is ready to query.")

    return db


def rag_query(query: str, persist_directory="db_local/chroma_db", top_k: int = 3):
    """
    Perform RAG retrieval and generation.

    Returns
    -------
    answer : str
    chunks : list[Document]   — the top-k retrieved chunks (rich metadata intact)
    """
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

    retriever = db.as_retriever(search_kwargs={"k": top_k})
    chunks = retriever.invoke(query)

    final_answer = generate_final_answer(chunks, query)
    return final_answer, chunks