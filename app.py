"""
app.py  –  Streamlit front-end for the Multimodal RAG Pipeline
"""

import os
import json
import tempfile
import streamlit as st

from rag_pipeline import run_complete_ingestion_pipeline, rag_query

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Multimodal RAG — PDF Q&A")
st.caption("Upload a PDF, process it, then ask questions.")

PERSIST_DIR = "db_local/chroma_db"
TOP_K = 3  # number of chunks to retrieve

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Chunks to retrieve (top-k)", min_value=1, max_value=10, value=TOP_K)
    st.markdown("---")
    st.markdown("**Model:** gemini-2.5-flash-lite  \n**Embeddings:** all-MiniLM-L6-v2  \n**Vector DB:** ChromaDB")

# ── Upload ─────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📁 Upload a PDF", type=["pdf"])

if uploaded_file:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.success(f"Loaded: **{uploaded_file.name}**  ({uploaded_file.size / 1024:.1f} KB)")
    with col2:
        process_btn = st.button("⚡ Process Document", type="primary", use_container_width=True)

    # ── Ingestion ──────────────────────────────────────────────────────────────
    if process_btn:
        # Save upload to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.markdown("---")
        st.subheader("📋 Processing Log")

        # We keep a list of log lines so we can re-render them all at once
        log_lines: list[str] = []
        log_placeholder = st.empty()

        def append_log(msg: str):
            """Add a line to the live log and re-render."""
            log_lines.append(msg)
            # Render as a styled card so it feels like a live terminal
            rendered = "\n\n".join(log_lines)
            log_placeholder.markdown(
                f"""
<div style="
    background:#0e1117;
    border:1px solid #2d2d2d;
    border-radius:8px;
    padding:16px 20px;
    font-family:monospace;
    font-size:14px;
    line-height:1.8;
    color:#e0e0e0;
">
{rendered.replace(chr(10), '<br>')}
</div>
""",
                unsafe_allow_html=True,
            )

        progress_bar = st.progress(0, text="Starting…")

        def update_progress(value: float):
            pct = int(value * 100)
            progress_bar.progress(value, text=f"Processing chunks… {pct}%")

        try:
            run_complete_ingestion_pipeline(
                pdf_path=tmp_path,
                persist_directory=PERSIST_DIR,
                status_callback=append_log,
                progress_callback=update_progress,
            )
            progress_bar.progress(1.0, text="Done! ✅")
            st.session_state["processed"] = True

        except Exception as e:
            st.error(f"❌ Pipeline failed: {e}")
        finally:
            os.unlink(tmp_path)

    # ── Query ──────────────────────────────────────────────────────────────────
    if st.session_state.get("processed") or os.path.exists(PERSIST_DIR):
        st.markdown("---")
        st.subheader("💬 Ask a Question")

        query = st.text_input(
            "Your question",
            placeholder="e.g. What are the key findings of this report?",
        )

        if st.button("🔎 Search", type="primary") and query.strip():
            with st.spinner("Retrieving relevant chunks and generating answer…"):
                answer, chunks = rag_query(query, persist_directory=PERSIST_DIR, top_k=top_k)

            # ── Answer ─────────────────────────────────────────────────────────
            st.markdown("### 📝 Answer")
            st.markdown(answer)

            # ── Retrieved Chunks ───────────────────────────────────────────────
            st.markdown("---")
            st.subheader(f"📚 Retrieved {len(chunks)} Source Chunk{'s' if len(chunks) != 1 else ''} (top-{top_k})")

            for idx, chunk in enumerate(chunks, start=1):
                chunk_id = chunk.metadata.get("chunk_id", "?")

                # Parse original content for richer display
                original: dict = {}
                if "original_content" in chunk.metadata:
                    try:
                        original = json.loads(chunk.metadata["original_content"])
                    except Exception:
                        pass

                raw_text    = original.get("raw_text", chunk.page_content)
                tables_html = original.get("tables_html", [])
                images_b64  = original.get("images_base64", [])

                # Build badge row
                badges = []
                badges.append(f"📝 Text ({len(raw_text)} chars)")
                if tables_html:
                    badges.append(f"📊 {len(tables_html)} Table{'s' if len(tables_html) > 1 else ''}")
                if images_b64:
                    badges.append(f"🖼️ {len(images_b64)} Image{'s' if len(images_b64) > 1 else ''}")

                header = f"Chunk #{chunk_id}  ·  " + "  |  ".join(badges)

                with st.expander(f"🗂️ Source {idx} — {header}", expanded=(idx == 1)):
                    # ── Text ───────────────────────────────────────────────────
                    if raw_text.strip():
                        st.markdown("**📝 Text Content**")
                        st.markdown(
                            f"""<div style="
                                background:#1a1a2e;
                                border-left:3px solid #4f8ef7;
                                border-radius:4px;
                                padding:12px 16px;
                                font-size:13px;
                                line-height:1.6;
                                color:#d0d0e0;
                                white-space:pre-wrap;
                            ">{raw_text.strip()}</div>""",
                            unsafe_allow_html=True,
                        )

                    # ── Tables ─────────────────────────────────────────────────
                    if tables_html:
                        st.markdown(f"**📊 Table{'s' if len(tables_html) > 1 else ''}**")
                        for t_idx, tbl in enumerate(tables_html, 1):
                            if len(tables_html) > 1:
                                st.caption(f"Table {t_idx}")
                            st.markdown(
                                f"""<div style="overflow-x:auto;">{tbl}</div>""",
                                unsafe_allow_html=True,
                            )

                    # ── Images ─────────────────────────────────────────────────
                    if images_b64:
                        st.markdown(f"**🖼️ Image{'s' if len(images_b64) > 1 else ''}**")
                        img_cols = st.columns(min(len(images_b64), 3))
                        for i_idx, img_b64 in enumerate(images_b64):
                            with img_cols[i_idx % 3]:
                                try:
                                    import base64
                                    img_bytes = base64.b64decode(img_b64)
                                    st.image(img_bytes, caption=f"Image {i_idx + 1}", use_container_width=True)
                                except Exception:
                                    st.caption(f"⚠️ Could not render image {i_idx + 1}")

                    # ── AI Summary (page_content) ──────────────────────────────
                    if chunk.page_content != raw_text:
                        with st.expander("🤖 AI-generated summary used for retrieval"):
                            st.markdown(chunk.page_content)