"""
Core RAG Pipeline
-----------------
Handles: document loading → chunking → embedding → vector storage → retrieval → reranking → answer

Design decisions and known tradeoffs are documented inline.
"""

import os
import csv
import hashlib
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from loguru import logger
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

@dataclass
class RAGConfig:
    # Chunking — prose documents (PDF, DOCX, TXT)
    chunk_size: int = 800
    chunk_overlap: int = 150

    # CSV chunking — rows grouped per chunk (not character-based)
    csv_rows_per_chunk: int = 10

    # Retrieval
    top_k: int = 5                          # candidates fetched from FAISS
    rerank_top_n: int = 3                   # chunks passed to LLM after reranking
    score_threshold: float = 0.30           # min cosine similarity to include chunk

    # Models
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.0
    max_tokens: int = 1024

    # Paths
    index_dir: str = "data/faiss_index"
    metadata_path: str = "data/doc_metadata.json"


# ──────────────────────────────────────────────
# Document Loaders
# ──────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".csv", ".txt", ".md"}


def load_document(file_path: str) -> list[Document]:
    """
    Load a document and return LangChain Document objects.

    Each format uses the most appropriate loader:
    - PDF: page-aware (preserves page numbers in metadata)
    - DOCX: full-text extraction
    - CSV: row-level (see _load_csv — NOT character-chunked)
    - TXT/MD: plain text
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}"
        )

    logger.info(f"Loading {ext} document: {path.name}")

    try:
        if ext == ".pdf":
            docs = PyPDFLoader(file_path).load()
        elif ext in (".docx", ".doc"):
            docs = Docx2txtLoader(file_path).load()
        elif ext == ".csv":
            # CSVs are NOT character-chunked. _load_csv returns one Document
            # per row; chunk_documents() groups them into row-group chunks.
            docs = _load_csv(file_path)
        else:
            docs = TextLoader(file_path, encoding="utf-8").load()

        file_hash = _file_hash(file_path)
        for doc in docs:
            doc.metadata.update({
                "source_file": path.name,
                "file_type": ext.lstrip("."),
                "file_hash": file_hash,
            })

        logger.info(f"Loaded {len(docs)} section(s) from {path.name}")
        return docs

    except Exception as e:
        logger.error(f"Failed to load {path.name}: {e}")
        raise


def _load_csv(file_path: str) -> list[Document]:
    """
    Load a CSV as one Document per row.

    Format: "col1: val1 | col2: val2 | ..."

    Column names are included in the content so the embedding carries
    semantic context. A bare value like "50000" is ambiguous;
    "loan_amount: 50000" embeds correctly.
    Empty values are skipped to keep chunks clean.
    """
    docs = []
    path = Path(file_path)

    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            content = " | ".join(
                f"{k.strip()}: {v.strip()}"
                for k, v in row.items()
                if v and v.strip()
            )
            if content:
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "source_file": path.name,
                        "file_type": "csv",
                        "csv_row": i,
                    },
                ))

    logger.info(f"Loaded {len(docs)} rows from {path.name}")
    return docs


def _file_hash(file_path: str) -> str:
    """SHA-256 hash of file content for deduplication."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


# ──────────────────────────────────────────────
# Chunking — format-aware
# ──────────────────────────────────────────────

def chunk_documents(docs: list[Document], config: RAGConfig) -> list[Document]:
    """
    Chunk documents using format-appropriate strategies.

    - CSV docs: grouped by N rows per chunk
    - All other formats: recursive character splitting with overlap

    Mixed-type batches are split by file_type and processed separately.
    """
    if not docs:
        return []

    csv_docs = [d for d in docs if d.metadata.get("file_type") == "csv"]
    prose_docs = [d for d in docs if d.metadata.get("file_type") != "csv"]

    chunks = []
    if csv_docs:
        chunks.extend(_chunk_csv_rows(csv_docs, config))
    if prose_docs:
        chunks.extend(_chunk_prose(prose_docs, config))

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    logger.info(
        f"Created {len(chunks)} chunks "
        f"({len(csv_docs)} csv rows → {len([c for c in chunks if c.metadata.get('file_type') == 'csv'])} csv chunks, "
        f"{len(prose_docs)} prose docs)"
    )
    return chunks


def _chunk_prose(docs: list[Document], config: RAGConfig) -> list[Document]:
    """Recursive character splitting for prose documents (PDF, DOCX, TXT)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return splitter.split_documents(docs)


def _chunk_csv_rows(docs: list[Document], config: RAGConfig) -> list[Document]:
    """
    Group N CSV row documents into single chunks.

    Rationale: a single CSV row is too sparse to embed meaningfully.
    Grouping rows gives the embedding model enough signal to distinguish
    between e.g. "high-default SME loans" vs "low-risk mortgage rows".
    The default of 10 rows balances context richness against chunk size.
    """
    chunks = []
    n = config.csv_rows_per_chunk

    for i in range(0, len(docs), n):
        group = docs[i:i + n]
        combined_content = "\n".join(d.page_content for d in group)
        meta = group[0].metadata.copy()
        meta["csv_row_start"] = group[0].metadata.get("csv_row", i)
        meta["csv_row_end"] = group[-1].metadata.get("csv_row", i + len(group) - 1)
        chunks.append(Document(page_content=combined_content, metadata=meta))

    return chunks


# ──────────────────────────────────────────────
# Vector Store Manager
# ──────────────────────────────────────────────

class VectorStoreManager:
    """
    Manages FAISS vector store: build, persist, load, update.
    Thread-safe for single-process Streamlit usage.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.index_dir = Path(config.index_dir)
        self.metadata_path = Path(config.metadata_path)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )
        self.vectorstore: Optional[FAISS] = None
        self._doc_registry: dict = self._load_registry()

    def _load_registry(self) -> dict:
        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                return json.load(f)
        return {}

    def _save_registry(self):
        with open(self.metadata_path, "w") as f:
            json.dump(self._doc_registry, f, indent=2)

    def is_indexed(self, file_hash: str) -> bool:
        return file_hash in self._doc_registry

    def load_or_create(self) -> bool:
        index_file = self.index_dir / "index.faiss"
        if index_file.exists():
            logger.info("Loading existing FAISS index...")
            self.vectorstore = FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("FAISS index loaded.")
            return True
        return False

    def add_documents(self, chunks: list[Document], source_file: str, file_hash: str):
        if self.is_indexed(file_hash):
            logger.warning(f"{source_file} already indexed (hash {file_hash}). Skipping.")
            return

        logger.info(f"Embedding {len(chunks)} chunks for {source_file}...")

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vectorstore.add_documents(chunks)

        self.vectorstore.save_local(str(self.index_dir))

        self._doc_registry[file_hash] = {
            "file": source_file,
            "chunks": len(chunks),
            "file_type": chunks[0].metadata.get("file_type", "unknown") if chunks else "unknown",
        }
        self._save_registry()
        logger.info(f"Indexed {source_file} → {len(chunks)} chunks stored.")

    def get_indexed_files(self) -> list[dict]:
        return list(self._doc_registry.values())

    def document_count(self) -> int:
        return len(self._doc_registry)


# ──────────────────────────────────────────────
# Reranker
# ──────────────────────────────────────────────

def rerank(
    question: str,
    candidates: list[tuple[Document, float]],
    top_n: int,
) -> list[tuple[Document, float]]:
    """
    Cross-encoder reranking of FAISS candidates.

    FAISS returns chunks by embedding similarity (cosine distance), which
    measures topical overlap but not answer relevance. A cross-encoder
    scores (question, chunk) pairs jointly, catching cases where a chunk
    is semantically similar but does not actually answer the question.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    (~80MB, CPU-friendly, trained on MS MARCO passage ranking)

    Degrades gracefully to FAISS order if sentence-transformers is not
    installed, so lightweight environments work without modification.
    """
    if not candidates:
        return []

    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(question, doc.page_content) for doc, _ in candidates]
        ce_scores = model.predict(pairs)

        reranked = sorted(
            zip([doc for doc, _ in candidates], ce_scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(doc, float(score)) for doc, score in reranked[:top_n]]

    except ImportError:
        logger.warning(
            "sentence-transformers not installed — skipping cross-encoder reranking. "
            "pip install sentence-transformers to enable."
        )
        return candidates[:top_n]


# ──────────────────────────────────────────────
# Lost-in-the-middle mitigation
# ──────────────────────────────────────────────

def reorder_for_lost_in_middle(docs: list[Document]) -> list[Document]:
    """
    Reorder chunks so the most relevant content appears at the edges of the prompt.

    LLMs attend most strongly to content at the beginning and end of their
    context window. Content in the middle receives significantly less
    attention (Liu et al., 2023 — "Lost in the Middle").

    Strategy: interleave from edges inward.
    Input [best, 2nd, 3rd, 4th, 5th] → Output [best, 3rd, 5th, 4th, 2nd]
    This places the best evidence first and second-best last.
    """
    if len(docs) <= 2:
        return docs

    result = []
    left, right = 0, len(docs) - 1
    turn = "left"

    while left <= right:
        if turn == "left":
            result.append(docs[left])
            left += 1
            turn = "right"
        else:
            result.append(docs[right])
            right -= 1
            turn = "left"

    return result


# ──────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a precise document assistant. Answer ONLY from the provided context.
If the answer is not in the context, say exactly: "I couldn't find this in the uploaded documents."

Context:
{context}

Question: {question}

Rules:
- Be concise and factual. No hedging or filler.
- Cite which document or section the information comes from.
- Do not add information not present in the context.
- If the context contains partial information, answer what you can and state what is missing.

Answer:""",
)


# ──────────────────────────────────────────────
# Answer dataclasses
# ──────────────────────────────────────────────

@dataclass
class SourceChunk:
    file: str
    page: Optional[int]
    chunk_index: int
    excerpt: str
    similarity_score: Optional[float] = None  # cosine similarity from FAISS
    rerank_score: Optional[float] = None       # cross-encoder score (higher = more relevant)


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list[SourceChunk] = field(default_factory=list)
    found_in_docs: bool = True
    model_used: str = ""
    reranked: bool = False                     # whether cross-encoder was applied


# ──────────────────────────────────────────────
# Query — full pipeline
# ──────────────────────────────────────────────

def query(
    question: str,
    vectorstore: Optional[FAISS],
    config: RAGConfig,
    filter_file: Optional[str] = None,
) -> RAGAnswer:
    """
    Full RAG query pipeline:
      1. FAISS similarity search → top_k candidates
      2. Score threshold filter
      3. Optional metadata filter by source filename
      4. Cross-encoder reranking → top rerank_top_n chunks
      5. Lost-in-the-middle reordering
      6. Context assembly → LLM → answer + citations

    Args:
        filter_file: restrict retrieval to chunks from this filename only.
                     Enables targeted queries against a single document when
                     multiple are indexed.
    """
    if vectorstore is None:
        return RAGAnswer(
            question=question,
            answer="No documents have been indexed yet. Please upload documents first.",
            found_in_docs=False,
        )

    # Step 1 — FAISS retrieval
    scored_docs = vectorstore.similarity_search_with_relevance_scores(
        question, k=config.top_k
    )

    # Step 2 — Score threshold
    relevant = [
        (doc, score)
        for doc, score in scored_docs
        if score >= config.score_threshold
    ]

    # Step 3 — Metadata filter
    if filter_file:
        relevant = [
            (doc, score)
            for doc, score in relevant
            if doc.metadata.get("source_file") == filter_file
        ]

    if not relevant:
        return RAGAnswer(
            question=question,
            answer=(
                "I couldn't find relevant information in the uploaded documents "
                f"for this question. All candidates scored below the similarity "
                f"threshold of {config.score_threshold}."
            ),
            found_in_docs=False,
            model_used=config.llm_model,
        )

    # Step 4 — Rerank
    cosine_scores = {id(doc): score for doc, score in relevant}
    reranked = rerank(question, relevant, top_n=config.rerank_top_n)
    reranked_flag = len(relevant) > 1

    # Step 5 — Lost-in-the-middle reordering
    reranked_docs_ordered = reorder_for_lost_in_middle([doc for doc, _ in reranked])
    rerank_score_map = {id(doc): score for doc, score in reranked}

    # Build source citations
    sources = [
        SourceChunk(
            file=doc.metadata.get("source_file", "unknown"),
            page=doc.metadata.get("page", None),
            chunk_index=doc.metadata.get("chunk_index", -1),
            excerpt=doc.page_content[:300].strip(),
            similarity_score=round(float(cosine_scores.get(id(doc), 0.0)), 3),
            rerank_score=round(float(rerank_score_map.get(id(doc), 0.0)), 3),
        )
        for doc in reranked_docs_ordered
    ]

    # Step 6 — LLM call
    context = "\n\n---\n\n".join(doc.page_content for doc in reranked_docs_ordered)

    llm = ChatOpenAI(
        model=config.llm_model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )

    prompt = RAG_PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt)

    return RAGAnswer(
        question=question,
        answer=response.content,
        sources=sources,
        found_in_docs=True,
        model_used=config.llm_model,
        reranked=reranked_flag,
    )
