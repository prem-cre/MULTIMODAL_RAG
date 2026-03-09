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

# Cloud-native partitioning (Zero local OCR binaries)
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared, operations

load_dotenv()

_COLLECTION_NAME = "multimodal_rag_collection"

def get_api_key() -> Optional[str]:
    return os.getenv("GEMINI_API_KEY")

def get_unstructured_api_key() -> Optional[str]:
    return os.getenv("UNSTRUCTURED_API_KEY")

def get_embedding_model():
    """Gemini Embeddings (Free, 1500 RPM, Zero Memory)."""
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=get_api_key()
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
        raise ValueError("UNSTRUCTURED_API_KEY missing.")

    client = UnstructuredClient(
        api_key_auth=api_key,
        server_url=os.getenv("UNSTRUCTURED_API_URL", "https://api.unstructuredapp.io/general/v0/general")
    )

    with open(file_path, "rb") as f:
        file_content = f.read()

    req = operations.PartitionRequest(
        partition_parameters=shared.PartitionParameters(
            files=shared.Files(
                content=file_content,
                file_name=file_path,
            ),
            strategy="hi_res",
            extract_image_block_types=["Image", "Table"],
            chunking_strategy="by_title",
            max_characters=3000,
            combine_under_n_chars=500
        )
    )

    try:
        res = client.general.partition(request=req)
        if status_callback:
            status_callback(f"✅ Received {len(res.elements)} segments.")
        return res.elements
    except Exception as e:
        if status_callback:
            status_callback(f"❌ Cloud Partition failed: {str(e)}")
        raise e

def separate_content_types_from_dict(element: Dict) -> Dict:
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
            "Summarize this document chunk for retrieval. Include table data and describe images.\n\n"
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
        futures = {executor.submit(_process_chunk_dict_worker, (i, element)): i for i, element in enumerate(chunk_dicts)}
        for future in as_completed(futures):
            idx, doc = future.result()
            results[idx] = doc
    return [results[i] for i in sorted(results)]

def create_vector_store(documents, persist_directory=None):
    import shutil
    import chromadb
    from chromadb.config import Settings
    
    if persist_directory is None:
        persist_directory = os.path.join(os.getcwd(), "chroma_db") if os.name == 'nt' else "/tmp/chroma_db"

    # Clean the directory to ensure no state conflicts
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory, ignore_errors=True)
    os.makedirs(persist_directory, exist_ok=True)

    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(allow_reset=True, anonymized_telemetry=False)
    )
    
    return Chroma.from_documents(
        documents=documents,
        embedding=get_embedding_model(),
        client=client,
        collection_name=_COLLECTION_NAME
    )

def generate_final_answer(chunks, query: str) -> str:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=0,
            google_api_key=get_api_key(),
        )
        prompt = f"Answer the query using the provided context.\n\nQuery: {query}\n\nContext:\n"
        for i, chunk in enumerate(chunks):
            prompt += f"\n--- CONTEXT {i+1} ---\n{chunk.page_content}\n"
        
        content = [{"type": "text", "text": prompt}]
        # Multimodal reasoning: include images found in retrieved chunks (limit to top 3)
        image_count = 0
        for chunk in chunks:
            if "original_content" in chunk.metadata and image_count < 3:
                try:
                    d = json.loads(chunk.metadata["original_content"])
                    for img in d.get("images_base64", []):
                        if image_count < 3:
                            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
                            image_count += 1
                except:
                    continue
        
        response = llm.invoke([HumanMessage(content=content)])
        return response.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def run_complete_ingestion_pipeline(pdf_path, persist_directory=None, status_callback=None):
    chunk_dicts = partition_and_chunk_document(pdf_path, status_callback)
    summarised = summarise_chunks(chunk_dicts, status_callback)
    return create_vector_store(summarised, persist_directory)

def rag_query(query: str, persist_directory=None):
    import chromadb
    from chromadb.config import Settings
    
    if persist_directory is None:
        persist_directory = os.path.join(os.getcwd(), "chroma_db") if os.name == 'nt' else "/tmp/chroma_db"
    
    if not os.path.exists(persist_directory):
        raise ValueError("No document index found. Please upload a PDF first.")

    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(allow_reset=True, anonymized_telemetry=False)
    )
    
    db = Chroma(
        client=client,
        embedding_function=get_embedding_model(),
        collection_name=_COLLECTION_NAME
    )
    
    chunks = db.as_retriever(search_kwargs={"k": 3}).invoke(query)
    answer = generate_final_answer(chunks, query)
    return answer, chunks
