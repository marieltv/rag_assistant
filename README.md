# Document Q&A RAG Assistant

End-to-end Retrieval-Augmented Generation (RAG) system for document question answering with source citations, similarity scores, and cross-encoder reranking.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                      │
│                                                             │
│  Upload (PDF / DOCX / CSV / TXT)                            │
│       ↓                                                     │
│  Format-aware loader                                        │
│    PDF  → PyPDFLoader (page-aware)                         │
│    DOCX → Docx2txtLoader                                    │
│    CSV  → row-level loader (col: val | col: val)            │
│    TXT  → TextLoader                                        │
│       ↓                                                     │
│  Format-aware chunking                                      │
│    Prose → RecursiveCharacterTextSplitter (800 chars)       │
│    CSV   → row grouping (10 rows per chunk)                 │
│       ↓                                                     │
│  OpenAI Embeddings (text-embedding-3-small)                 │
│       ↓                                                     │
│  FAISS vector store (persisted to disk)                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                         │
│                                                             │
│  User question                                              │
│       ↓                                                     │
│  FAISS similarity search → top_k candidates                 │
│       ↓                                                     │
│  Score threshold filter (cosine ≥ 0.30)                     │
│       ↓                                                     │
│  Optional metadata filter (by source filename)              │
│       ↓                                                     │
│  Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)          │
│    → top rerank_top_n chunks                                │
│       ↓                                                     │
│  Lost-in-the-middle reordering                              │
│    → best evidence at prompt edges                          │
│       ↓                                                     │
│  GPT-4o-mini (stuffed context prompt)                       │
│       ↓                                                     │
│  Answer + citations (cosine score + rerank score)           │
└─────────────────────────────────────────────────────────────┘
```

---

## Stack

| Layer | Technology | Notes |
|---|---|---|
| LLM | GPT-4o-mini | Swappable to GPT-4o or local Ollama |
| Embeddings | text-embedding-3-small | Must match model used at index time |
| Vector store | FAISS | Local, persisted. ChromaDB drop-in available |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | CPU-friendly, ~80MB. Graceful fallback if missing |
| RAG framework | LangChain | |
| Backend API | FastAPI | |
| Frontend | Streamlit | |
| Testing | pytest | 30 unit tests, no API key required |

---

## Project structure

```
rag_assistant/
├── rag_pipeline.py       # Core logic: load → chunk → embed → retrieve → rerank → answer
├── api.py                # FastAPI: /upload, /query, /documents, /health
├── app.py                # Streamlit UI
├── tests/
│   └── test_rag_pipeline.py
├── .env.example          # All variables documented with tradeoff notes
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
copy .env.example .env
# Edit .env and set OPENAI_API_KEY

# 3. Start backend
uvicorn api:app --reload --port 8000

# 4. Start frontend (new terminal)
streamlit run app.py

# 5. Run tests
pytest tests/ -v
```

---

## Key design decisions

### Format-aware chunking
Character-based chunking is the single most common mistake in beginner RAG projects. A 800-character chunk sliced across CSV rows destroys row semantics — the embedding for `"50000\n30000\nauto\nmortgage"` carries no useful signal.

This project applies different strategies per format:
- **PDF/DOCX/TXT**: recursive character splitting with 150-character overlap, which preserves sentence boundaries and prevents answers being cut at chunk edges
- **CSV**: rows are loaded individually as `"col: val | col: val"` strings (column names included for semantic context), then grouped into 10-row chunks before embedding

### Cross-encoder reranking
FAISS retrieves by cosine similarity between embeddings, which measures topical overlap but not answer relevance. A query like "what is the default rate for SMEs?" can surface chunks that mention SMEs and default rates separately, without either chunk containing the answer.

A cross-encoder scores `(question, chunk)` pairs jointly, after seeing both together. This catches false positives that embedding similarity misses. The pipeline fetches `top_k=5` candidates from FAISS, then reranks to `rerank_top_n=3` before passing to the LLM.

Falls back gracefully to FAISS order if `sentence-transformers` is not installed.

### Lost-in-the-middle mitigation
LLMs attend most strongly to context at the beginning and end of their input (Liu et al., 2023). After reranking, chunks are reordered so the highest-scoring evidence appears first and last, with lower-scoring chunks in the middle.

### Metadata filtering
When multiple documents are indexed, queries can be scoped to a single source file via the `filter_file` parameter. This prevents low-relevance chunks from unrelated documents polluting the context.

---

## API reference

### `POST /upload`
```json
{
  "filename": "credit_policy.pdf",
  "chunks_created": 47,
  "already_indexed": false,
  "message": "Successfully indexed 'credit_policy.pdf' into 47 chunks."
}
```

### `POST /query`
```json
// Request
{
  "question": "What is the minimum CET1 ratio?",
  "top_k": 5,
  "score_threshold": 0.30,
  "filter_file": "basel_iii_framework.pdf"
}

// Response
{
  "question": "What is the minimum CET1 ratio?",
  "answer": "Under Basel III, the minimum CET1 ratio is 4.5% of risk-weighted assets.",
  "found_in_docs": true,
  "model_used": "gpt-4o-mini",
  "reranked": true,
  "total_sources": 2,
  "sources": [
    {
      "file": "basel_iii_framework.pdf",
      "page": 12,
      "chunk_index": 34,
      "excerpt": "Common Equity Tier 1 capital must be maintained at a minimum of 4.5 percent...",
      "similarity_score": 0.912,
      "rerank_score": 14.73
    }
  ]
}
```

---

## Known limitations and production roadmap

These are conscious tradeoffs at portfolio scale, not oversights.

**Stuff chain → map_reduce for long documents**
The current pipeline concatenates all retrieved chunks into a single prompt. For large ERP manuals with many relevant sections, this can exceed the context window. Production systems use `map_reduce` or `refine` chain types: each chunk is summarised independently, then the summaries are combined. Trade-off: higher latency and token cost.

**No HyDE (Hypothetical Document Embedding)**
Short user questions embed differently from the document text that answers them. HyDE generates a hypothetical answer first, embeds that, and retrieves against it — improving recall for vague queries. Adds one LLM call per query.

**FAISS has no metadata filtering at the index level**
FAISS does not support server-side filtering (e.g. "only search chunks from PDFs uploaded this week"). The current `filter_file` implementation retrieves globally then filters in Python, which wastes retrieval slots when filtering heavily. ChromaDB supports native metadata filtering and is a drop-in replacement.

**Single-process FAISS**
FAISS is not safe for concurrent writes. Under multi-user load, writes need a lock or the index should be replaced with a client-server vector database (Qdrant, Weaviate, Pinecone).

**No observability**
Production RAG needs retrieval quality metrics: MRR, NDCG@k, answer faithfulness scores (using an LLM evaluator or RAGAS). These are not implemented here.
