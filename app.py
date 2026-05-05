"""
Streamlit App — Document Q&A RAG Assistant
-------------------------------------------
Standalone: calls rag_pipeline.py directly, no FastAPI required.
Deployable to Streamlit Cloud out of the box.

Run locally:   streamlit run app.py
Deploy:        push to GitHub → connect repo on share.streamlit.io
               set OPENAI_API_KEY in Streamlit Cloud secrets
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from rag_pipeline import (
    RAGConfig,
    VectorStoreManager,
    load_document,
    chunk_documents,
    query,
)

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

SUPPORTED_TYPES = ["pdf", "docx", "doc", "csv", "txt", "md"]

st.set_page_config(
    page_title="DocMind — Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.source-card {
    background: #1a1a2e;
    border: 1px solid #2a2a3e;
    border-left: 3px solid #00d4aa;
    border-radius: 4px;
    padding: 12px 16px;
    margin: 8px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #aaa;
}

.score-badge {
    display: inline-block;
    background: #00d4aa22;
    border: 1px solid #00d4aa55;
    color: #00d4aa;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 2px;
    margin-left: 6px;
}

.rerank-badge {
    display: inline-block;
    background: #a78bfa22;
    border: 1px solid #a78bfa55;
    color: #a78bfa;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 2px;
    margin-left: 4px;
}

.not-found-msg {
    border-left: 3px solid #ff6b6b;
    padding-left: 12px;
    color: #ff6b6b;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session state initialisation
# ──────────────────────────────────────────────

@st.cache_resource
def get_vsm() -> VectorStoreManager:
    """
    Initialise VectorStoreManager once per Streamlit session.
    Cached so the FAISS index is not reloaded on every rerun.
    """
    config = RAGConfig()
    vsm = VectorStoreManager(config)
    vsm.load_or_create()
    return vsm


def get_config() -> RAGConfig:
    """Build RAGConfig from current sidebar slider values."""
    return RAGConfig(
        top_k=st.session_state.get("top_k", 5),
        rerank_top_n=st.session_state.get("rerank_top_n", 3),
        score_threshold=st.session_state.get("score_threshold", 0.30),
    )


if "history" not in st.session_state:
    st.session_state.history = []


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📚 DocMind")
    st.markdown("*Document Q&A with citations*")
    st.divider()

    # API key check
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("⚠️ OPENAI_API_KEY not set.\nAdd it to `.env` or Streamlit Cloud secrets.")
        st.stop()

    vsm = get_vsm()
    doc_count = vsm.document_count()
    st.success(f"Ready · {doc_count} document{'s' if doc_count != 1 else ''} indexed")

    st.divider()

    # ── Upload ──
    st.markdown("### Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop files here",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            btn_label = f"Index: {uploaded_file.name}"
            if st.button(btn_label, key=f"btn_{uploaded_file.name}"):
                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    # Write to temp file so loaders can open it by path
                    suffix = Path(uploaded_file.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    try:
                        docs = load_document(tmp_path)
                        file_hash = docs[0].metadata.get("file_hash", "")

                        if vsm.is_indexed(file_hash):
                            st.info(f"'{uploaded_file.name}' is already indexed.")
                        else:
                            # Fix source name (tmp path → original filename)
                            for doc in docs:
                                doc.metadata["source_file"] = uploaded_file.name

                            chunks = chunk_documents(docs, get_config())
                            vsm.add_documents(chunks, uploaded_file.name, file_hash)
                            st.success(f"✓ {len(chunks)} chunks indexed")
                            st.rerun()

                    except Exception as e:
                        st.error(f"Failed: {e}")
                    finally:
                        os.unlink(tmp_path)

    st.divider()

    # ── Retrieval settings ──
    st.markdown("### Retrieval Settings")
    st.session_state["top_k"] = st.slider("Top-K candidates (FAISS)", 1, 10, 5)
    st.session_state["rerank_top_n"] = st.slider("Top-N after reranking", 1, 5, 3)
    st.session_state["score_threshold"] = st.slider("Min similarity score", 0.0, 1.0, 0.30, 0.05)

    filter_file = None
    indexed = vsm.get_indexed_files()
    if len(indexed) > 1:
        st.divider()
        st.markdown("### Filter by document")
        file_names = ["All documents"] + [d["file"] for d in indexed]
        selected = st.selectbox("Search within", file_names, label_visibility="collapsed")
        if selected != "All documents":
            filter_file = selected

    st.divider()

    # ── Indexed documents ──
    st.markdown("### Indexed Documents")
    if indexed:
        for d in indexed:
            icon = "📊" if d.get("file_type") == "csv" else "📄"
            st.markdown(f"{icon} `{d['file']}` — {d['chunks']} chunks")
    else:
        st.caption("No documents indexed yet.")

    if indexed and st.button("🗑️ Clear all documents", type="secondary"):
        import shutil
        index_dir = Path(vsm.config.index_dir)
        meta_path = Path(vsm.config.metadata_path)
        if index_dir.exists():
            shutil.rmtree(index_dir)
        if meta_path.exists():
            meta_path.unlink()
        st.cache_resource.clear()
        st.success("Index cleared.")
        st.rerun()


# ──────────────────────────────────────────────
# Main — Q&A
# ──────────────────────────────────────────────

st.markdown("# Document Q&A Assistant")
st.markdown(
    "Upload documents in the sidebar, then ask questions. "
    "Every answer includes source citations with similarity and rerank scores."
)
st.divider()

question = st.chat_input("Ask a question about your documents...")

if question:
    if vsm.vectorstore is None:
        st.warning("No documents indexed yet. Upload a document first.")
    else:
        with st.spinner("Retrieving · reranking · generating..."):
            result = query(
                question=question,
                vectorstore=vsm.vectorstore,
                config=get_config(),
                filter_file=filter_file,
            )
        st.session_state.history.append(result)

# ── Chat history ──
for entry in reversed(st.session_state.history):
    with st.chat_message("user"):
        st.write(entry.question)

    with st.chat_message("assistant"):
        if not entry.found_in_docs:
            st.markdown(
                f'<div class="not-found-msg">{entry.answer}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(entry.answer)

        if entry.sources:
            reranked_label = " · reranked ✓" if entry.reranked else ""
            with st.expander(
                f"📎 {len(entry.sources)} source chunk(s) cited{reranked_label}",
                expanded=True,
            ):
                for s in entry.sources:
                    cos_html = (
                        f'<span class="score-badge">cosine: {s.similarity_score:.3f}</span>'
                        if s.similarity_score is not None else ""
                    )
                    rerank_html = (
                        f'<span class="rerank-badge">rerank: {s.rerank_score:.2f}</span>'
                        if s.rerank_score is not None else ""
                    )
                    page_info = f" · page {s.page}" if s.page is not None else ""
                    excerpt = s.excerpt + ("..." if len(s.excerpt) >= 300 else "")

                    st.markdown(
                        f"""<div class="source-card">
                        <strong>{s.file}</strong>{page_info} · chunk #{s.chunk_index}
                        {cos_html}{rerank_html}
                        <br><br>{excerpt}
                        </div>""",
                        unsafe_allow_html=True,
                    )

        meta_parts = [f"`{entry.model_used}`"]
        if entry.reranked:
            meta_parts.append("cross-encoder reranked")
        st.caption(" · ".join(meta_parts))
