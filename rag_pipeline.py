"""
Core RAG Pipeline
-----------------
Hybrid retrieval (BM25 + dense vectors) 

Pipeline: load → clean → chunk → embed → index → hybrid retrieve → rerank → answer
"""

import os
import re
import csv
import hashlib
import json
from pathlib import Path
from typing import Optional, Protocol
from dataclasses import dataclass, field
from collections import deque

from loguru import logger
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from rank_bm25 import BM25Okapi


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

SCHEMA_VERSION = 3
DEFAULT_LOCAL_EMBEDDING = "sentence-transformers/all-mpnet-base-v2"
TOKEN_PATTERN = re.compile(r"[\w]+", flags=re.UNICODE)


@dataclass
class RAGConfig:
    # Chunking — word-based for prose (see jamwithai reference: 300 words, 100 overlap)
    chunk_size: int = 300
    chunk_overlap: int = 100

    # CSV chunking — rows grouped per chunk
    csv_rows_per_chunk: int = 10

    # Retrieval
    top_k: int = 12                         # candidates from hybrid search
    rerank_top_n: int = 6                   # chunks passed to LLM after reranking
    score_threshold: float = 0.0            # 0 = disabled; filter only if > 0
    hybrid_fetch_multiplier: int = 2        # fetch top_k * N per channel before RRF
    rrf_k: int = 60                         # reciprocal rank fusion constant

    # Embeddings — local sentence-transformers by default (better semantic recall)
    embedding_backend: str = "local"        # "local" | "openai"
    embedding_model: str = DEFAULT_LOCAL_EMBEDDING

    # LLM (OpenAI for answer generation)
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024

    # Paths
    index_dir: str = "data/faiss_index"
    metadata_path: str = "data/doc_metadata.json"

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Build config from environment variables when present."""
        return cls(
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "300")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "100")),
            top_k=int(os.getenv("RAG_TOP_K", "12")),
            rerank_top_n=int(os.getenv("RAG_RERANK_TOP_N", "6")),
            score_threshold=float(os.getenv("RAG_SCORE_THRESHOLD", "0.0")),
            embedding_backend=os.getenv("RAG_EMBEDDING_BACKEND", "local"),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_LOCAL_EMBEDDING),
            llm_model=os.getenv("RAG_LLM_MODEL", "gpt-4o-mini"),
            index_dir=os.getenv("RAG_INDEX_DIR", "data/faiss_index"),
            metadata_path=os.getenv("RAG_METADATA_PATH", "data/doc_metadata.json"),
        )


# ──────────────────────────────────────────────
# Text utilities (from jamwithai reference)
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Clean extracted text — fix line breaks and OCR-style artifacts."""
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_text_words(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word chunks (reference repo strategy)."""
    text = clean_text(text)
    tokens = text.split()
    if not tokens:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = " ".join(tokens[start:end])
        if chunk:
            chunks.append(chunk)
        if end >= len(tokens):
            break
        start = max(end - overlap, start + 1)

    return chunks


def tokenize_for_search(text: str) -> list[str]:
    """
    Normalize text for lexical retrieval.

    BM25 is very sensitive to casing and punctuation. The previous whitespace
    split made queries like "CET1?" fail to match chunks containing "CET1" and
    made title-cased document terms miss lower-cased query terms.
    """
    return TOKEN_PATTERN.findall(text.lower())


# ──────────────────────────────────────────────
# Document Loaders
# ──────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".csv", ".txt", ".md"}


def load_document(file_path: str) -> list[Document]:
    """Load a document and return LangChain Document objects."""
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
        elif ext == ".docx":
            docs = Docx2txtLoader(file_path).load()
        elif ext == ".csv":
            docs = _load_csv(file_path)
        else:
            docs = TextLoader(file_path, encoding="utf-8").load()

        file_hash = _file_hash(file_path)
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata.update({
                "source_file": path.name,
                "file_type": ext.lstrip("."),
                "file_hash": file_hash,
            })

        if not docs or not any(doc.page_content.strip() for doc in docs):
            raise ValueError(
                f"No content could be extracted from '{path.name}'. "
                "The file may be empty or unreadable."
            )

        logger.info(f"Loaded {len(docs)} section(s) from {path.name}")
        return docs

    except Exception as e:
        logger.error(f"Failed to load {path.name}: {e}")
        raise


def _load_csv(file_path: str) -> list[Document]:
    """Load a CSV as one Document per row."""
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
# Chunking
# ──────────────────────────────────────────────

def chunk_documents(docs: list[Document], config: RAGConfig) -> list[Document]:
    """Chunk documents using format-appropriate strategies."""
    if not docs:
        return []

    csv_docs = [d for d in docs if d.metadata.get("file_type") == "csv"]
    prose_docs = [d for d in docs if d.metadata.get("file_type") != "csv"]

    chunks: list[Document] = []
    if csv_docs:
        chunks.extend(_chunk_csv_rows(csv_docs, config))
    if prose_docs:
        chunks.extend(_chunk_prose(prose_docs, config))

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    logger.info(f"Created {len(chunks)} chunks from {len(docs)} source section(s)")
    return chunks


def _chunk_prose(docs: list[Document], config: RAGConfig) -> list[Document]:
    """Word-based overlapping chunks for prose documents."""
    chunks: list[Document] = []
    for doc in docs:
        for piece in chunk_text_words(doc.page_content, config.chunk_size, config.chunk_overlap):
            meta = doc.metadata.copy()
            chunks.append(Document(page_content=piece, metadata=meta))
    return chunks


def _chunk_csv_rows(docs: list[Document], config: RAGConfig) -> list[Document]:
    """Group N CSV row documents into single chunks."""
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
# Embeddings
# ──────────────────────────────────────────────

class EmbeddingsProtocol(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def build_embeddings(config: RAGConfig) -> EmbeddingsProtocol:
    """Create embedding model — local sentence-transformers or OpenAI."""
    if config.embedding_backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required when RAG_EMBEDDING_BACKEND=openai")
        return OpenAIEmbeddings(
            model=config.embedding_model or "text-embedding-3-small",
            openai_api_key=api_key,
        )

    from langchain_community.embeddings import HuggingFaceEmbeddings
    logger.info(f"Loading local embedding model: {config.embedding_model}")
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ──────────────────────────────────────────────
# Hybrid retrieval helpers
# ──────────────────────────────────────────────

def _doc_key(doc: Document) -> str:
    """Stable key for merging BM25 and vector hits."""
    return (
        f"{doc.metadata.get('source_file', '')}|"
        f"{doc.metadata.get('chunk_index', '')}|"
        f"{doc.page_content[:120]}"
    )


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Document, float]]],
    rrf_k: int = 60,
) -> list[tuple[Document, float]]:
    """
    Merge multiple ranked result lists with Reciprocal Rank Fusion.
    Same approach used in production hybrid search pipelines.
    """
    scores: dict[str, float] = {}
    docs_by_key: dict[str, Document] = {}
    raw_scores: dict[str, list[float]] = {}

    for ranked in ranked_lists:
        for rank, (doc, score) in enumerate(ranked):
            key = _doc_key(doc)
            docs_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            raw_scores.setdefault(key, []).append(score)

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        (docs_by_key[key], max(raw_scores[key]))
        for key, _ in merged
    ]


# ──────────────────────────────────────────────
# Vector Store Manager (FAISS + BM25 hybrid)
# ──────────────────────────────────────────────

class VectorStoreManager:
    """
    Manages FAISS vector store + BM25 index for hybrid retrieval.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.index_dir = Path(config.index_dir)
        self.metadata_path = Path(config.metadata_path)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        self.embeddings = build_embeddings(config)
        self.vectorstore: Optional[FAISS] = None
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: list[Document] = []
        self._registry_payload = self._load_registry()

    @property
    def _doc_registry(self) -> dict:
        return self._registry_payload.setdefault("documents", {})

    @_doc_registry.setter
    def _doc_registry(self, value: dict) -> None:
        self._registry_payload["documents"] = value

    def _load_registry(self) -> dict:
        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                data = json.load(f)
            if isinstance(data, dict) and "documents" in data:
                return data
            # Migrate legacy flat registry
            return {
                "schema_version": 1,
                "embedding_backend": self.config.embedding_backend,
                "embedding_model": self.config.embedding_model,
                "documents": data,
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "embedding_backend": self.config.embedding_backend,
            "embedding_model": self.config.embedding_model,
            "documents": {},
        }

    def _save_registry(self) -> None:
        self._registry_payload.update({
            "schema_version": SCHEMA_VERSION,
            "embedding_backend": self.config.embedding_backend,
            "embedding_model": self.config.embedding_model,
        })
        with open(self.metadata_path, "w") as f:
            json.dump(self._registry_payload, f, indent=2)

    def needs_reindex(self) -> bool:
        """True when stored index was built with a different embedding setup."""
        if not self._doc_registry:
            return False
        stored_backend = self._registry_payload.get("embedding_backend")
        stored_model = self._registry_payload.get("embedding_model")
        stored_version = self._registry_payload.get("schema_version", 1)
        return (
            stored_version != SCHEMA_VERSION
            or stored_backend != self.config.embedding_backend
            or stored_model != self.config.embedding_model
        )

    def is_indexed(self, file_hash: str) -> bool:
        return file_hash in self._doc_registry

    def _sync_bm25_from_docstore(self) -> None:
        if self.vectorstore is None:
            self._bm25_docs = []
            self._bm25 = None
            return
        self._bm25_docs = list(self.vectorstore.docstore._dict.values())  # noqa: SLF001
        tokenized = [tokenize_for_search(doc.page_content) for doc in self._bm25_docs]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def load_or_create(self) -> bool:
        index_file = self.index_dir / "index.faiss"
        if not index_file.exists():
            return False

        if self.needs_reindex():
            logger.warning(
                "Index was built with a different embedding model/version. "
                "Clear the index and re-upload documents."
            )

        logger.info("Loading existing FAISS index...")
        self.vectorstore = FAISS.load_local(
            str(self.index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
            distance_strategy=DistanceStrategy.COSINE,
        )
        self._sync_bm25_from_docstore()
        logger.info(f"FAISS index loaded — {len(self._bm25_docs)} chunks, BM25 ready.")
        return True

    def add_documents(self, chunks: list[Document], source_file: str, file_hash: str) -> None:
        if self.is_indexed(file_hash):
            logger.warning(f"{source_file} already indexed (hash {file_hash}). Skipping.")
            return

        logger.info(f"Embedding {len(chunks)} chunks for {source_file}...")

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(
                chunks,
                self.embeddings,
                distance_strategy=DistanceStrategy.COSINE,
            )
        else:
            self.vectorstore.add_documents(chunks)

        self.vectorstore.save_local(str(self.index_dir))
        self._sync_bm25_from_docstore()

        self._doc_registry[file_hash] = {
            "file": source_file,
            "chunks": len(chunks),
            "file_type": chunks[0].metadata.get("file_type", "unknown") if chunks else "unknown",
        }
        self._save_registry()
        logger.info(f"Indexed {source_file} → {len(chunks)} chunks stored.")

    def hybrid_search(
        self,
        question: str,
        k: int,
        filter_file: Optional[str] = None,
    ) -> list[tuple[Document, float]]:
        """
        Hybrid search: dense vector (semantic) + BM25 (keyword) merged via RRF.
        Mirrors the jamwithai OpenSearch hybrid pattern without requiring OpenSearch.
        """
        if self.vectorstore is None:
            return []

        fetch_k = max(k * self.config.hybrid_fetch_multiplier, k)

        vector_hits = self.vectorstore.similarity_search_with_relevance_scores(
            question, k=fetch_k
        )

        bm25_hits: list[tuple[Document, float]] = []
        if self._bm25 and self._bm25_docs:
            query_tokens = tokenize_for_search(question)
            bm25_scores = self._bm25.get_scores(query_tokens)
            ranked_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True,
            )
            for idx in ranked_indices[:fetch_k]:
                score = float(bm25_scores[idx])
                if score > 0:
                    bm25_hits.append((self._bm25_docs[idx], score))

        if bm25_hits:
            merged = reciprocal_rank_fusion([vector_hits, bm25_hits], rrf_k=self.config.rrf_k)
        else:
            merged = vector_hits

        if filter_file:
            merged = [
                (doc, score)
                for doc, score in merged
                if doc.metadata.get("source_file") == filter_file
            ]

        return merged[:k]

    def get_indexed_files(self) -> list[dict]:
        return list(self._doc_registry.values())

    def document_count(self) -> int:
        return len(self._doc_registry)


# ──────────────────────────────────────────────
# Reranker
# ──────────────────────────────────────────────

_cross_encoder = None


def _get_cross_encoder():
    """Load the cross-encoder once and reuse across queries."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def rerank(
    question: str,
    candidates: list[tuple[Document, float]],
    top_n: int,
) -> tuple[list[tuple[Document, float]], bool]:
    """Cross-encoder reranking of hybrid search candidates."""
    if not candidates:
        return [], False

    try:
        model = _get_cross_encoder()
        pairs = [(question, doc.page_content) for doc, _ in candidates]
        ce_scores = model.predict(pairs)

        reranked = sorted(
            zip([doc for doc, _ in candidates], ce_scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(doc, float(score)) for doc, score in reranked[:top_n]], True

    except Exception as e:
        logger.warning(
            f"Cross-encoder reranking unavailable ({e}) — using hybrid ranking."
        )
        return candidates[:top_n], False


# ──────────────────────────────────────────────
# Lost-in-the-middle mitigation
# ──────────────────────────────────────────────

def reorder_for_lost_in_middle(docs: list[Document]) -> list[Document]:
    """Reorder chunks so the most relevant content appears at prompt edges."""
    if len(docs) <= 2:
        return docs

    result: list[Document] = []
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


def format_context_chunk(doc: Document, source_number: int) -> str:
    """Format one retrieved chunk with stable citation metadata for the LLM."""
    source_file = doc.metadata.get("source_file", "document")
    page = doc.metadata.get("page")
    chunk_index = doc.metadata.get("chunk_index", -1)

    location_parts = [f"file={source_file}", f"chunk={chunk_index}"]
    if page is not None:
        location_parts.append(f"page={page}")

    return (
        f"[Source {source_number}: {'; '.join(location_parts)}]\n"
        f"{doc.page_content.strip()}"
    )


# ──────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a careful document question-answering assistant. Answer using ONLY the provided context.
If the answer is not explicitly supported by the context, say exactly: "I couldn't find this in the uploaded documents."

Context:
{context}

Question: {question}

Instructions:
- First decide whether the context contains direct evidence for the question.
- Answer clearly and directly in complete sentences.
- Cite the source labels you used, e.g. [Source 1], and mention filenames when useful.
- Do not invent facts not present in the context.
- If sources disagree, explain the disagreement and cite both sources.
- If the context is partially relevant, answer only the supported part and state what is missing.

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
    similarity_score: Optional[float] = None
    rerank_score: Optional[float] = None


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list[SourceChunk] = field(default_factory=list)
    found_in_docs: bool = True
    model_used: str = ""
    reranked: bool = False
    hybrid_search: bool = True


# ──────────────────────────────────────────────
# Query — full pipeline
# ──────────────────────────────────────────────

def query(
    question: str,
    vsm: Optional[VectorStoreManager],
    config: RAGConfig,
    filter_file: Optional[str] = None,
) -> RAGAnswer:
    """
    Full RAG query pipeline:
      1. Hybrid search (BM25 + vectors) → top_k candidates
      2. Optional score threshold (disabled when 0)
      3. Cross-encoder reranking → top rerank_top_n chunks
      4. Lost-in-the-middle reordering
      5. LLM answer + citations
    """
    if vsm is None or vsm.vectorstore is None:
        return RAGAnswer(
            question=question,
            answer="No documents have been indexed yet. Please upload documents first.",
            found_in_docs=False,
            hybrid_search=False,
        )

    # Step 1 — Hybrid retrieval (always returns best matches, no hard semantic gate)
    candidates = vsm.hybrid_search(question, k=config.top_k, filter_file=filter_file)

    if not candidates:
        return RAGAnswer(
            question=question,
            answer=(
                "I couldn't find any indexed content matching your question. "
                "Try rephrasing, clearing the document filter, or re-indexing your files."
            ),
            found_in_docs=False,
            model_used=config.llm_model,
        )

    # Step 2 — Optional threshold (0 = skip)
    if config.score_threshold > 0:
        candidates = [(doc, s) for doc, s in candidates if s >= config.score_threshold]
        if not candidates:
            hits = vsm.hybrid_search(question, k=1, filter_file=filter_file)
            best_score = hits[0][1] if hits else 0.0
            return RAGAnswer(
                question=question,
                answer=(
                    f"No chunks passed the similarity threshold ({config.score_threshold}). "
                    f"Best match scored {best_score:.3f}. Set threshold to 0 to disable filtering."
                ),
                found_in_docs=False,
                model_used=config.llm_model,
            )

    # Step 3 — Rerank (semantic relevance on full question+chunk pairs)
    retrieval_scores = {id(doc): score for doc, score in candidates}
    reranked, reranked_flag = rerank(question, candidates, top_n=config.rerank_top_n)

    # Step 4 — Lost-in-the-middle reordering
    reranked_docs_ordered = reorder_for_lost_in_middle([doc for doc, _ in reranked])
    rerank_score_map = {id(doc): score for doc, score in reranked}

    sources = [
        SourceChunk(
            file=doc.metadata.get("source_file", "unknown"),
            page=doc.metadata.get("page"),
            chunk_index=doc.metadata.get("chunk_index", -1),
            excerpt=doc.page_content[:300].strip(),
            similarity_score=round(float(retrieval_scores.get(id(doc), 0.0)), 3),
            rerank_score=round(float(rerank_score_map.get(id(doc), 0.0)), 3),
        )
        for doc in reranked_docs_ordered
    ]

    # Step 5 — LLM
    context = "\n\n---\n\n".join(
        format_context_chunk(doc, i + 1)
        for i, doc in enumerate(reranked_docs_ordered)
    )

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
        hybrid_search=True,
    )
