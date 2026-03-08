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

# We use the unstructured-client to avoid needing local Tesseract/Poppler or the huge 'unstructured' library on Vercel
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared

from langchain_community.embeddings import HuggingFaceInferenceEmbeddings

load_dotenv()

def get_api_key() -> Optional[str]:
    return os.getenv("GEMINI_API_KEY")

def get_unstructured_api_key() -> Optional[str]:
    return os.getenv("UNSTRUCTURED_API_KEY")

def get_embedding_model():
    """FREE Open Source Embeddings (No paid Gemini key required)."""
    return HuggingFaceInferenceEmbeddings(
        api_key=os.getenv("HF_TOKEN"), # Optional, works limited without
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def partition_and_chunk_document(
    file_path: str,
    status_callback: Optional[Callable] = None,
):
    """
    Cloud-native partitioning AND CHUNKING using Unstructured API.
    Does NOT use the heavy 'unstructured' library locally.
    """
    if status_callback:
        status_callback("☁️ Requesting Unstructured Cloud (Remote OCR & Chunking)...")

    api_key = get_unstructured_api_key()
    if not api_key:
        raise ValueError("UNSTRUCTURED_API_KEY missing. This is required for Vercel deployment.")

    # Using the more reliable endpoint
    client = UnstructuredClient(
        api_key_auth=api_key,
        server_url=os.getenv("UNSTRUCTURED_API_URL", "https://api.unstructuredapp.io/general/v0/general")
    )

    with open(file_path, "rb") as f:
        file_content = f.read()

    from unstructured_client.models import operations
    
    req = operations.PartitionRequest(
        partition_parameters=shared.PartitionParameters(
            files=shared.Files(
                content=file_content,
                file_name=file_path,
            ),
            strategy="hi_res",
            extract_image_block_types=["Image", "Table"],
            # ⚡ SERVER-SIDE CHUNKING! (Removes the need for local 'unstructured' package)
            chunking_strategy="by_title",
            max_characters=3000,
            combine_under_n_chars=500
        )
    )

    try:
        res = client.general.partition(request=req)
        # res.elements is a list of dicts representing the chunks
        if status_callback:
            status_callback(f"✅ Received {len(res.elements)} chunks from remote API")
        return res.elements
    except Exception as e:
        if status_callback:
            status_callback(f"❌ Cloud Partition failed: {str(e)}")
        raise e

def separate_content_types_from_dict(element: Dict) -> Dict:
    """Extract text, tables, and images from Unstructured API response dict."""
    metadata = element.get("metadata", {})
    text = element.get("text", "")
    
    data = {
        "text": text,
        "tables": [],
        "images": [],
        "types": ["text"]
    }
    
    # Check for table content in metadata
    if metadata.get("text_as_html"):
        data["tables"].append(metadata["text_as_html"])
        data["types"].append("table")
    
    # Unstructured API often puts image data in orig_elements or specialized fields
    # If the element itself is a 'CompositeElement' containing images, they'll be in there.
    # For now we'll check common fields:
    if metadata.get("image_base64"):
        data["images"].append(metadata["image_base64"])
        data["types"].append("image")
        
    data["types"] = list(set(data["types"]))
    return data

def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    # Use Flash for higher rate limits on free tier (15 RPM)
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
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
        # If rate limited (429), return a simpler snippet to avoid crashing
        if "429" in str(e):
             return f"SUMMARY_PENDING: {text[:200]}... [Rate limit reached]"
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
        status_callback(f"🧠 Summarising {len(chunk_dicts)} chunks (AI enhancement for tables/images)...")
    
    results = {}
    # Lower worker count to stay within free Gemini free limits (15 RPM)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_process_chunk_dict_worker, (i, element)): i for i, element in enumerate(chunk_dicts)}
        for future in as_completed(futures):
            idx, doc = future.result()
            results[idx] = doc
            if status_callback and (idx + 1) % 4 == 0:
                status_callback(f"  Processed {idx + 1}/{len(chunk_dicts)} segments...")
    
    return [results[i] for i in sorted(results)]

def create_vector_store(documents, persist_directory="/tmp/chroma_db"):
    # Clear old local db to prevent schema conflicts
    import shutil
    if os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory)
        except:
            pass

    model = get_embedding_model()
    return Chroma.from_documents(
        documents=documents,
        embedding=model,
        persist_directory=persist_directory
    )

def generate_final_answer(chunks, query: str) -> str:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", # Use Flash for reliability/speed on Vercel
            temperature=0,
            google_api_key=get_api_key(),
        )
        prompt = f"Answer the query using the provided context (text, tables, and images).\n\nQuery: {query}\n\nContext:\n"
        for i, chunk in enumerate(chunks):
            prompt += f"\n--- CONTEXT {i+1} ---\n{chunk.page_content}\n"
        
        content = [{"type": "text", "text": prompt}]
        # Multimodal reasoning: include images found in retrieved chunks
        for chunk in chunks:
            if "original_content" in chunk.metadata:
                d = json.loads(chunk.metadata["original_content"])
                for img in d.get("images_base64", []):
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
        
        response = llm.invoke([HumanMessage(content=content)])
        return response.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def run_complete_ingestion_pipeline(pdf_path, persist_directory="/tmp/chroma_db", status_callback=None):
    chunk_dicts = partition_and_chunk_document(pdf_path, status_callback)
    summarised = summarise_chunks(chunk_dicts, status_callback)
    db = create_vector_store(summarised, persist_directory)
    if status_callback:
        status_callback("🎉 Multimodal Knowledge Base ready.")
    return db

def rag_query(query: str, persist_directory="/tmp/chroma_db"):
    model = get_embedding_model()
    db = Chroma(persist_directory=persist_directory, embedding_function=model)
    chunks = db.as_retriever(search_kwargs={"k": 3}).invoke(query)
    answer = generate_final_answer(chunks, query)
    return answer, chunks