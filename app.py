import streamlit as st
import os
import tempfile
from rag_pipeline import run_complete_ingestion_pipeline, rag_query, get_api_key

st.set_page_config(page_title="Multimodal RAG with Gemini", page_icon="🤖", layout="wide")

st.title("Multimodal RAG with Gemini 🚀")
st.markdown("Build an intelligent document retrieval system that understands **Text**, **Tables**, and **Images** using LangChain, Chroma, Unstructured, and Google's Gemini models.")

# API Key check
api_key = get_api_key()
if not api_key:
    st.error("Gemini API Key is not set. Please set the GEMINI_API_KEY environment variable or add it to Streamlit Secrets.")
    st.stop()

# Sidebar for controls
with st.sidebar:
    st.header("Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Processing document (this may take a few minutes)..."):
                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    run_complete_ingestion_pipeline(tmp_file_path)
                    st.success("Document processed successfully!")
                except Exception as e:
                    st.error(f"Error processing document: {e}")
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)
    
    st.markdown("---")
    st.header("About")
    st.info(
        "This application uses:\n"
        "- **Unstructured** for parsing PDFs (extracting text, tables, and images)\n"
        "- **Google Gemini** for enhanced text/image summarization and final answer generation\n"
        "- **ChromaDB** for vector storage\n"
        "- **LangChain** for orchestration"
    )

# Main chat area
st.header("Chat with your Document")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about the processed documents..."):
    # Check if vector DB exists
    if not os.path.exists("db_gemini/chroma_db"):
        st.warning("Please process a document first in the sidebar before asking questions.")
    else:
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate bot response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    answer, source_chunks = rag_query(prompt)
                    message_placeholder.markdown(answer)
                    
                    # Add an expander to show sources if needed
                    with st.expander("View Source Information"):
                        st.write(f"Retrieved {len(source_chunks)} chunks for context.")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
