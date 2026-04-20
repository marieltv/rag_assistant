"""
Streamlit UI — Document Q&A RAG Assistant
-----------------------------------------
Run: streamlit run frontend/app.py
Requires: FastAPI backend running at http://localhost:8000
"""

import streamlit as st
import requests
import json
import os
from pathlib import Path

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
SUPPORTED_TYPES = ["pdf", "docx", "doc", "csv", "txt", "md"]

st.set_page_config(
    page_title="DocMind — Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background: #0f0f0f;
    color: #e8e8e8;
}

.stApp { background: #0f0f0f; }

.source-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
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
    margin-left: 8px;
}

.answer-box {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 20px 24px;
    margin: 12px 0;
    line-height: 1.7;
}

.not-found {
    border-left: 3px solid #ff6b6b;
    background: #1a1212;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# API Helpers
# ──────────────────────────────────────────────

def api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json() if r.ok else None
    except:
        return None

def api_upload(file_bytes, filename):
    r = requests.post(
        f"{API_URL}/upload",
        files={"file": (filename, file_bytes)},
        timeout=60,
    )
    return r.json() if r.ok else {"error": r.text}

def api_query(question, top_k=5, threshold=0.30):
    r = requests.post(
        f"{API_URL}/query",
        json={"question": question, "top_k": top_k, "score_threshold": threshold},
        timeout=30,
    )
    return r.json() if r.ok else {"error": r.text}

def api_list_docs():
    r = requests.get(f"{API_URL}/documents", timeout=5)
    return r.json() if r.ok else {"documents": []}

def api_clear():
    r = requests.delete(f"{API_URL}/documents", timeout=10)
    return r.ok


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📚 DocMind")
    st.markdown("*Document Q&A with citations*")
    st.divider()

    # Health
    health = api_health()
    if health:
        st.success(f"API connected · {health['indexed_documents']} docs indexed")
    else:
        st.error("⚠️ API offline — start the FastAPI backend")

    st.divider()

    # Upload
    st.markdown("### Upload Documents")
    uploaded = st.file_uploader(
        "Drop files here",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        for f in uploaded:
            if st.button(f"Index: {f.name}", key=f"btn_{f.name}"):
                with st.spinner(f"Indexing {f.name}..."):
                    result = api_upload(f.getvalue(), f.name)
                if "error" in result:
                    st.error(result["error"])
                elif result.get("already_indexed"):
                    st.info(result["message"])
                else:
                    st.success(f"✓ {result['chunks_created']} chunks indexed")

    st.divider()

    # Settings
    st.markdown("### Retrieval Settings")
    top_k = st.slider("Top-K chunks", 1, 10, 5)
    threshold = st.slider("Min similarity score", 0.0, 1.0, 0.30, 0.05)

    st.divider()

    # Indexed docs
    st.markdown("### Indexed Documents")
    docs_data = api_list_docs()
    docs = docs_data.get("documents", [])
    if docs:
        for d in docs:
            st.markdown(f"📄 `{d['file']}` — {d['chunks']} chunks")
    else:
        st.caption("No documents indexed yet.")

    if docs and st.button("🗑️ Clear all documents", type="secondary"):
        if api_clear():
            st.success("Index cleared.")
            st.rerun()


# ──────────────────────────────────────────────
# Main — Q&A
# ──────────────────────────────────────────────

st.markdown("# Document Q&A Assistant")
st.markdown("Ask questions about your uploaded documents. Every answer includes source citations and similarity scores.")
st.divider()

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Input
question = st.chat_input("Ask a question about your documents...")

if question:
    with st.spinner("Searching documents and generating answer..."):
        result = api_query(question, top_k=top_k, threshold=threshold)

    if "error" in result:
        st.error(result["error"])
    else:
        st.session_state.history.append(result)

# Display history (newest first)
for entry in reversed(st.session_state.history):
    with st.chat_message("user"):
        st.write(entry["question"])

    with st.chat_message("assistant"):
        found = entry.get("found_in_docs", True)
        css_class = "answer-box" if found else "answer-box not-found"
        st.markdown(f'<div class="{css_class}">{entry["answer"]}</div>', unsafe_allow_html=True)

        sources = entry.get("sources", [])
        if sources:
            with st.expander(f"📎 {len(sources)} source chunk(s) cited", expanded=True):
                for s in sources:
                    score = s.get("similarity_score")
                    score_html = f'<span class="score-badge">score: {score:.3f}</span>' if score else ""
                    page_info = f" · page {s['page']}" if s.get("page") is not None else ""
                    st.markdown(
                        f"""<div class="source-card">
                        <strong>{s['file']}</strong>{page_info} · chunk #{s['chunk_index']}{score_html}
                        <br><br>{s['excerpt']}{"..." if len(s['excerpt']) >= 300 else ""}
                        </div>""",
                        unsafe_allow_html=True,
                    )
        elif found:
            st.caption("No source chunks returned above threshold.")

        st.caption(f"Model: `{entry.get('model_used', 'gpt-4o-mini')}`")
