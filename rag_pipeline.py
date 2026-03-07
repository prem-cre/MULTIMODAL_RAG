import os
import sys
import json
import base64
from typing import List, Dict, Any

# ==========================================
# Windows PATH configuration for Unstructured
# ==========================================
if sys.platform == "win32":
    # Common Tesseract installation path
    tesseract_path = r"C:\Program Files\Tesseract-OCR"
    if os.path.exists(tesseract_path) and tesseract_path not in os.environ["PATH"]:
         os.environ["PATH"] += os.pathsep + tesseract_path
    
    # Poppler path for winget-installed version
    # Searching for the bin directory dynamically to be robust
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
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

import google.generativeai as genai

load_dotenv()

# Initialize Gemini key securely
def get_api_key():
    try:
        import streamlit as st
        # First check streamlit secrets if deployed
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except ImportError:
        pass
    except FileNotFoundError:
        pass
    
    # Fallback to local environment variable
    return os.getenv("GEMINI_API_KEY")

def configure_genai():
    api_key = get_api_key()
    if api_key:
        genai.configure(api_key=api_key)
        return True
    return False

def partition_document(file_path: str):
    """Extract elements from PDF using unstructured"""
    print(f"📄 Partitioning document: {file_path}")
    
    elements = partition_pdf(
        filename=file_path,  # Path to your PDF file
        strategy="hi_res", # Use the most accurate (but slower) processing method of extraction
        infer_table_structure=True, # Keep tables as structured HTML, not jumbled text
        extract_image_block_types=["Image"], # Grab images found in the PDF
        extract_image_block_to_payload=True # Store images as base64 data you can actually use
    )
    
    print(f"✅ Extracted {len(elements)} elements")
    return elements

def create_chunks_by_title(elements):
    """Create intelligent chunks using title-based strategy"""
    print("🔨 Creating smart chunks...")
    
    chunks = chunk_by_title(
        elements, 
        max_characters=3000, 
        new_after_n_chars=2400, 
        combine_text_under_n_chars=500 
    )
    
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

def separate_content_types(chunk):
    """Analyze what types of content are in a chunk"""
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
        # Changed to Gemini Model
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0, google_api_key=get_api_key())
        
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
            # Gemini expects images format slightly differently than OpenAI, but langchain wrapper handles base64 urls.
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        
        return response.content
        
    except Exception as e:
        print(f"     ❌ AI summary failed: {e}")
        summary = f"{text[:300]}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary

def summarise_chunks(chunks):
    """Process all chunks with AI Summaries"""
    print("🧠 Processing chunks with AI Summaries...")
    
    langchain_documents = []
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        content_types = separate_content_types(chunk)
        
        metadata = {
            "chunk_id": i + 1,
            "original_content": json.dumps({
                "raw_text": content_types['text'],
                "tables_html": content_types['tables'],
                "images_base64": content_types['images']
            })
        }
        
        if 'image' in content_types['types'] or 'table' in content_types['types']:
            enhanced_content = create_ai_enhanced_summary(
                content_types['text'],
                content_types['tables'],
                content_types['images']
            )
            
            doc = Document(
                page_content=enhanced_content,
                metadata=metadata
            )
        else:
            doc = Document(
                page_content=content_types['text'],
                metadata=metadata
            )
            
        langchain_documents.append(doc)
        
    print(f"✅ Processed {len(langchain_documents)} chunks")
    return langchain_documents

def create_vector_store(documents, persist_directory="db_gemini/chroma_db"):
    """Create and persist ChromaDB vector store using Gemini Embeddings"""
    print("🔮 Creating embeddings and storing in ChromaDB...")
        
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=get_api_key())
    
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"✅ Vector store created and saved to {persist_directory}")
    return vectorstore

def generate_final_answer(chunks, query):
    """Generate final answer using multimodal content and Gemini"""
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=get_api_key())
        
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
                images_base64 = original_data.get("images_base64", [])
                
                for image_base64 in images_base64:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    })
        
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        
        return response.content
        
    except Exception as e:
        print(f"❌ Answer generation failed: {e}")
        return "Sorry, I encountered an error while generating the answer."

def run_complete_ingestion_pipeline(pdf_path: str, persist_directory="db_gemini/chroma_db"):
    """Run the complete RAG ingestion pipeline"""
    print("🚀 Starting RAG Ingestion Pipeline")
    
    configure_genai()

    elements = partition_document(pdf_path)
    chunks = create_chunks_by_title(elements)
    summarised_chunks = summarise_chunks(chunks)
    db = create_vector_store(summarised_chunks, persist_directory=persist_directory)
    
    print("🎉 Pipeline completed successfully!")
    return db

def rag_query(query: str, persist_directory="db_gemini/chroma_db"):
    """Perform RAG retrieval and generation"""
    configure_genai()

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=get_api_key())
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    
    retriever = db.as_retriever(search_kwargs={"k": 3})
    chunks = retriever.invoke(query)
    
    final_answer = generate_final_answer(chunks, query)
    return final_answer, chunks
