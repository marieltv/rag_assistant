"""
FastAPI Backend — Document Q&A RAG Assistant
--------------------------------------------
Endpoints:
  POST /upload        - Upload and index a document
  POST /query         - Ask a question, get answer + citations
  GET  /documents     - List indexed documents
  DELETE /documents   - Clear the index
  GET  /health        - Health check
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from rag_pipeline import (
    RAGConfig,
    VectorStoreManager,
    load_document,
    chunk_documents,
    query,
    RAGAnswer,
    SourceChunk,
)


# ──────────────────────────────────────────────
# App & Config
# ──────────────────────────────────────────────

app = FastAPI(
    title="Document Q&A RAG Assistant",
    description="Upload documents, ask questions, get cited answers.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

config = RAGConfig()
vsm = VectorStoreManager(config)

# Load existing index on startup
@app.on_event("startup")
def startup():
    vsm.load_or_create()
    logger.info("RAG API started. Indexed files: {}", vsm.document_count())


# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    score_threshold: Optional[float] = None
    filter_file: Optional[str] = None     # restrict retrieval to a single source file


class SourceChunkOut(BaseModel):
    file: str
    page: Optional[int]
    chunk_index: int
    excerpt: str
    similarity_score: Optional[float]     # cosine similarity from FAISS
    rerank_score: Optional[float]         # cross-encoder score (if reranking was applied)


class QueryResponse(BaseModel):
    question: str
    answer: str
    found_in_docs: bool
    model_used: str
    reranked: bool                         # whether cross-encoder reranking was applied
    sources: list[SourceChunkOut]
    total_sources: int


class UploadResponse(BaseModel):
    filename: str
    chunks_created: int
    already_indexed: bool
    message: str


class DocumentListResponse(BaseModel):
    total_documents: int
    documents: list[dict]


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "indexed_documents": vsm.document_count(),
        "index_ready": vsm.vectorstore is not None,
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF, DOCX, CSV, or TXT file and index it."""
    allowed_extensions = {".pdf", ".docx", ".doc", ".csv", ".txt", ".md"}
    ext = Path(file.filename).suffix.lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed_extensions}",
        )

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Load
        docs = load_document(tmp_path)

        # Check deduplication
        file_hash = docs[0].metadata.get("file_hash", "")
        if vsm.is_indexed(file_hash):
            return UploadResponse(
                filename=file.filename,
                chunks_created=0,
                already_indexed=True,
                message=f"'{file.filename}' was already indexed. No changes made.",
            )

        # Fix source file name (tmp path → original name)
        for doc in docs:
            doc.metadata["source_file"] = file.filename

        # Chunk
        chunks = chunk_documents(docs, config)

        # Embed & store
        vsm.add_documents(chunks, file.filename, file_hash)

        return UploadResponse(
            filename=file.filename,
            chunks_created=len(chunks),
            already_indexed=False,
            message=f"Successfully indexed '{file.filename}' into {len(chunks)} chunks.",
        )

    except Exception as e:
        logger.error(f"Upload failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        os.unlink(tmp_path)


@app.post("/query", response_model=QueryResponse)
def ask_question(req: QueryRequest):
    """Ask a question. Returns answer + source citations with similarity scores."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Allow per-request overrides
    local_config = RAGConfig(
        top_k=req.top_k or config.top_k,
        score_threshold=req.score_threshold or config.score_threshold,
        llm_model=config.llm_model,
        embedding_model=config.embedding_model,
    )

    result: RAGAnswer = query(
        req.question,
        vsm.vectorstore,
        local_config,
        filter_file=req.filter_file,
    )

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        found_in_docs=result.found_in_docs,
        model_used=result.model_used,
        reranked=result.reranked,
        sources=[
            SourceChunkOut(
                file=s.file,
                page=s.page,
                chunk_index=s.chunk_index,
                excerpt=s.excerpt,
                similarity_score=s.similarity_score,
                rerank_score=s.rerank_score,
            )
            for s in result.sources
        ],
        total_sources=len(result.sources),
    )


@app.get("/documents", response_model=DocumentListResponse)
def list_documents():
    """List all indexed documents."""
    docs = vsm.get_indexed_files()
    return DocumentListResponse(total_documents=len(docs), documents=docs)


@app.delete("/documents")
def clear_index():
    """Clear the entire FAISS index and registry."""
    index_dir = Path(config.index_dir)
    if index_dir.exists():
        shutil.rmtree(index_dir)
    meta = Path(config.metadata_path)
    if meta.exists():
        meta.unlink()

    vsm.vectorstore = None
    vsm._doc_registry = {}
    index_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Index cleared.")
    return {"message": "Index cleared successfully."}


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
