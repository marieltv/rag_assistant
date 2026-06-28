# Document Q&A RAG Assistant

End-to-end Retrieval-Augmented Generation (RAG) system for document question answering with **hybrid BM25 + vector search**, source citations, and cross-encoder reranking.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                      │
│                                                             │
│  Upload (PDF / DOCX / CSV / TXT / MD)                       │
│       ↓                                                     │
│  Format-aware loader + text cleaning                        │
│       ↓                                                     │
│  Word-based chunking (300 words, 100 overlap)               │
│    CSV → row grouping (10 rows per chunk)                   │
│       ↓                                                     │
│  Local embeddings (all-mpnet-base-v2)                       │
│       ↓                                                     │
│  FAISS (cosine) + BM25 index (persisted to disk)            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                         │
│                                                             │
│  User question                                              │
│       ↓                                                     │
│  Hybrid search: BM25 (keywords) + FAISS (semantic)          │
│    → merged via Reciprocal Rank Fusion (RRF)                │
│       ↓                                                     │
│  Optional metadata filter (by source filename)              │
│       ↓                                                     │
│  Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)           │
│    → top rerank_top_n chunks                                │
│       ↓                                                     │
│  Lost-in-the-middle reordering → GPT-4o-mini → answer       │
└─────────────────────────────────────────────────────────────┘
```

---

## Stack

| Layer | Technology | Notes |
|---|---|---|
| LLM | GPT-4o-mini | Answer generation (OpenAI API) |
| Embeddings | all-mpnet-base-v2 (local) | Free, runs on CPU; switch to OpenAI via env |
| Keyword search | BM25 (rank-bm25) | Catches exact terms embeddings miss |
| Vector store | FAISS (cosine) | Local, persisted |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Joint question+chunk scoring |
| RAG framework | LangChain | |
| Frontend | Streamlit | Standalone UI (`app.py`) — calls pipeline directly |
| Backend API | FastAPI | Optional production API (`api.py`) |
| Testing | pytest | Unit tests, no API key required |

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

# 3. Start the Streamlit UI (recommended for local use)
streamlit run app.py

# Optional: start FastAPI backend (separate terminal)
uvicorn api:app --reload --port 8000

# 4. Run tests
pytest tests/ -v
```

---

## Uploading documents (Streamlit UI)

1. Open the sidebar **Upload Documents** section.
2. Browse or drag-and-drop supported files (PDF, DOCX, CSV, TXT, MD).
3. Selected files appear in a **Selected files** list with file size.
4. Click **Index all** to embed and store them in the FAISS + BM25 index.


Legacy `.doc` (Word 97–2003) is not supported — save as `.docx` first.

If uploads fail silently in a dev container or behind a proxy, try:

```bash
streamlit run app.py --server.enableXsrfProtection false
```

---

## Key design decisions

### Hybrid BM25 + vector search
Pure embedding search misses exact keyword matches; pure keyword search misses paraphrases. We combine BM25 and FAISS results with Reciprocal Rank Fusion — no OpenSearch server required.

### Local sentence-transformer embeddings
`all-mpnet-base-v2` runs on CPU with no embedding API cost and typically gives better recall on articles and PDFs than small OpenAI embedding models for local document Q&A.

### No pre-filter threshold by default
Similarity scores from FAISS vary widely by content type. The default threshold is **0 (disabled)** — top hybrid results always reach the cross-encoder reranker and LLM. Raise the threshold in the sidebar only if you need strict filtering.

### Word-based chunking (300 / 100)
Matches the reference repo: 300-word chunks with 100-word overlap preserve sentence semantics better than arbitrary character splits for news articles and reports.

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
  "rerank_top_n": 3,
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
