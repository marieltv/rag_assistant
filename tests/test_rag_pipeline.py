"""
Tests for RAG pipeline core logic.
No API key required — all external calls are mocked.
Run: pytest tests/ -v
"""

import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from rag_pipeline import (
    RAGConfig,
    chunk_documents,
    chunk_text_words,
    clean_text,
    reciprocal_rank_fusion,
    _chunk_csv_rows,
    _chunk_prose,
    _load_csv,
    _file_hash,
    reorder_for_lost_in_middle,
    rerank,
    SourceChunk,
    RAGAnswer,
)


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

def test_default_config():
    cfg = RAGConfig()
    assert cfg.chunk_size == 300
    assert cfg.chunk_overlap == 100
    assert cfg.csv_rows_per_chunk == 10
    assert cfg.top_k == 12
    assert cfg.rerank_top_n == 6
    assert cfg.score_threshold == 0.0
    assert cfg.embedding_backend == "local"
    assert cfg.llm_model == "gpt-4o-mini"


def test_custom_config():
    cfg = RAGConfig(chunk_size=500, top_k=3, score_threshold=0.5, csv_rows_per_chunk=5)
    assert cfg.chunk_size == 500
    assert cfg.top_k == 3
    assert cfg.score_threshold == 0.5
    assert cfg.csv_rows_per_chunk == 5


# ──────────────────────────────────────────────
# Text utilities
# ──────────────────────────────────────────────

def test_clean_text_fixes_hyphenated_line_breaks():
    raw = "artifi-\ncial intelligence"
    assert clean_text(raw) == "artificial intelligence"


def test_chunk_text_words_overlap():
    text = " ".join(f"word{i}" for i in range(50))
    chunks = chunk_text_words(text, chunk_size=20, overlap=5)
    assert len(chunks) >= 2
    assert "word0" in chunks[0]
    assert "word19" in chunks[0]


# ──────────────────────────────────────────────
# CSV loading
# ──────────────────────────────────────────────

def test_load_csv_formats_rows(tmp_path):
    f = tmp_path / "test.csv"
    f.write_text("loan_id,amount,status\n1,50000,active\n2,30000,closed\n")
    docs = _load_csv(str(f))
    assert len(docs) == 2
    assert "loan_id: 1" in docs[0].page_content
    assert "amount: 50000" in docs[0].page_content
    assert "status: active" in docs[0].page_content


def test_load_csv_skips_empty_values(tmp_path):
    f = tmp_path / "sparse.csv"
    f.write_text("a,b,c\n1,,3\n")
    docs = _load_csv(str(f))
    assert len(docs) == 1
    assert "b:" not in docs[0].page_content
    assert "a: 1" in docs[0].page_content
    assert "c: 3" in docs[0].page_content


def test_load_csv_row_metadata(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("x,y\n10,20\n30,40\n50,60\n")
    docs = _load_csv(str(f))
    assert docs[0].metadata["csv_row"] == 0
    assert docs[1].metadata["csv_row"] == 1
    assert docs[2].metadata["csv_row"] == 2


def test_load_csv_empty_file(tmp_path):
    f = tmp_path / "empty.csv"
    f.write_text("col1,col2\n")
    docs = _load_csv(str(f))
    assert docs == []


# ──────────────────────────────────────────────
# CSV chunking
# ──────────────────────────────────────────────

def make_csv_docs(n: int) -> list[Document]:
    return [
        Document(
            page_content=f"loan_id: {i} | amount: {i * 1000}",
            metadata={"file_type": "csv", "source_file": "loans.csv", "csv_row": i},
        )
        for i in range(n)
    ]


def test_csv_chunk_groups_rows():
    cfg = RAGConfig(csv_rows_per_chunk=3)
    docs = make_csv_docs(10)
    chunks = _chunk_csv_rows(docs, cfg)
    assert len(chunks) == 4  # 3+3+3+1


def test_csv_chunk_row_range_metadata():
    cfg = RAGConfig(csv_rows_per_chunk=5)
    docs = make_csv_docs(12)
    chunks = _chunk_csv_rows(docs, cfg)
    assert chunks[0].metadata["csv_row_start"] == 0
    assert chunks[0].metadata["csv_row_end"] == 4
    assert chunks[1].metadata["csv_row_start"] == 5
    assert chunks[1].metadata["csv_row_end"] == 9


def test_csv_chunk_content_contains_all_rows():
    cfg = RAGConfig(csv_rows_per_chunk=3)
    docs = make_csv_docs(3)
    chunks = _chunk_csv_rows(docs, cfg)
    assert len(chunks) == 1
    for i in range(3):
        assert f"loan_id: {i}" in chunks[0].page_content


def test_csv_chunk_exact_multiple():
    cfg = RAGConfig(csv_rows_per_chunk=3)
    docs = make_csv_docs(6)
    chunks = _chunk_csv_rows(docs, cfg)
    assert len(chunks) == 2


# ──────────────────────────────────────────────
# Prose chunking
# ──────────────────────────────────────────────

def make_prose_docs(texts: list[str]) -> list[Document]:
    return [
        Document(page_content=t, metadata={"source_file": "doc.pdf", "file_type": "pdf"})
        for t in texts
    ]


def test_prose_chunk_short_doc():
    cfg = RAGConfig(chunk_size=1000, chunk_overlap=100)
    docs = make_prose_docs(["Short document."])
    chunks = _chunk_prose(docs, cfg)
    assert len(chunks) >= 1


def test_prose_chunk_long_doc():
    cfg = RAGConfig(chunk_size=50, chunk_overlap=10)
    docs = make_prose_docs(["word " * 500])
    chunks = _chunk_prose(docs, cfg)
    assert len(chunks) > 1


def test_prose_chunk_metadata_preserved():
    cfg = RAGConfig()
    docs = make_prose_docs(["Credit risk analysis content."])
    chunks = _chunk_prose(docs, cfg)
    assert chunks[0].metadata["file_type"] == "pdf"


# ──────────────────────────────────────────────
# chunk_documents — format routing
# ──────────────────────────────────────────────

def test_chunk_documents_routes_csv_separately():
    cfg = RAGConfig(chunk_size=50, chunk_overlap=10, csv_rows_per_chunk=2)
    prose = make_prose_docs(["word " * 100])
    csv_docs = make_csv_docs(4)
    chunks = chunk_documents(prose + csv_docs, cfg)
    csv_chunks = [c for c in chunks if c.metadata.get("file_type") == "csv"]
    assert len(csv_chunks) == 2  # 4 rows / 2 per chunk


def test_chunk_documents_assigns_global_index():
    cfg = RAGConfig()
    docs = make_prose_docs(["Short text."])
    chunks = chunk_documents(docs, cfg)
    assert all("chunk_index" in c.metadata for c in chunks)


def test_chunk_documents_empty_input():
    assert chunk_documents([], RAGConfig()) == []


# ──────────────────────────────────────────────
# Hybrid RRF
# ──────────────────────────────────────────────

def test_reciprocal_rank_fusion_merges_lists():
    doc_a = Document(page_content="alpha", metadata={"source_file": "a.txt", "chunk_index": 0})
    doc_b = Document(page_content="beta", metadata={"source_file": "b.txt", "chunk_index": 1})
    doc_c = Document(page_content="gamma", metadata={"source_file": "c.txt", "chunk_index": 2})

    list1 = [(doc_a, 0.9), (doc_b, 0.8)]
    list2 = [(doc_c, 5.0), (doc_a, 4.0)]

    merged = reciprocal_rank_fusion([list1, list2])
    keys = [d.page_content for d, _ in merged]
    assert keys[0] == "alpha"  # appears in both lists → top RRF score


# ──────────────────────────────────────────────
# Lost-in-the-middle reordering
# ──────────────────────────────────────────────

def make_doc(label: str) -> Document:
    return Document(page_content=label, metadata={})


def test_reorder_two_docs_unchanged():
    docs = [make_doc("A"), make_doc("B")]
    result = reorder_for_lost_in_middle(docs)
    assert [d.page_content for d in result] == ["A", "B"]


def test_reorder_single_doc_unchanged():
    result = reorder_for_lost_in_middle([make_doc("A")])
    assert result[0].page_content == "A"


def test_reorder_places_best_first():
    docs = [make_doc(label) for label in ["best", "2nd", "3rd", "4th", "5th"]]
    result = reorder_for_lost_in_middle(docs)
    assert result[0].page_content == "best"


def test_reorder_five_docs_order():
    docs = [make_doc(str(i)) for i in range(5)]
    result = reorder_for_lost_in_middle(docs)
    assert [d.page_content for d in result] == ["0", "4", "1", "3", "2"]


def test_reorder_empty_list():
    assert reorder_for_lost_in_middle([]) == []


# ──────────────────────────────────────────────
# Reranker
# ──────────────────────────────────────────────

def test_rerank_empty_candidates():
    assert rerank("question", [], top_n=3) == ([], False)


def test_rerank_fallback_without_sentence_transformers():
    docs = [(make_doc(f"doc{i}"), float(i) * 0.1) for i in range(5)]
    with patch.dict("sys.modules", {"sentence_transformers": None}):
        result, used_ce = rerank("question", docs, top_n=3)
    assert len(result) == 3
    assert result[0][0].page_content == "doc0"
    assert used_ce is False


def test_rerank_with_mock_cross_encoder():
    docs = [
        (make_doc("irrelevant chunk"), 0.90),
        (make_doc("directly answers the question"), 0.60),
        (make_doc("vaguely related"), 0.75),
    ]
    mock_ce = MagicMock()
    mock_ce.predict.return_value = [0.1, 0.95, 0.3]

    mock_module = MagicMock()
    mock_module.CrossEncoder.return_value = mock_ce

    with patch.dict("sys.modules", {"sentence_transformers": mock_module}):
        result, used_ce = rerank("What is the answer?", docs, top_n=2)

    assert len(result) == 2
    assert result[0][0].page_content == "directly answers the question"
    assert used_ce is True


# ──────────────────────────────────────────────
# File hash
# ──────────────────────────────────────────────

def test_file_hash_deterministic(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("hello world")
    assert _file_hash(str(f)) == _file_hash(str(f))
    assert len(_file_hash(str(f))) == 12


def test_file_hash_different_content(tmp_path):
    f1, f2 = tmp_path / "a.txt", tmp_path / "b.txt"
    f1.write_text("content A")
    f2.write_text("content B")
    assert _file_hash(str(f1)) != _file_hash(str(f2))


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

def test_source_chunk_has_both_scores():
    sc = SourceChunk(
        file="policy.pdf", page=3, chunk_index=7,
        excerpt="Capital adequacy...",
        similarity_score=0.847,
        rerank_score=12.3,
    )
    assert sc.similarity_score == 0.847
    assert sc.rerank_score == 12.3


def test_rag_answer_reranked_flag():
    assert RAGAnswer(question="q", answer="a", reranked=True).reranked is True


def test_rag_answer_defaults():
    ans = RAGAnswer(question="What is PD?", answer="Probability of default.")
    assert ans.sources == []
    assert ans.found_in_docs is True
    assert ans.reranked is False
    assert ans.hybrid_search is True


# ──────────────────────────────────────────────
# VectorStoreManager (mocked — no API key needed)
# ──────────────────────────────────────────────

def test_registry_dedup(tmp_path):
    from rag_pipeline import VectorStoreManager

    cfg = RAGConfig(
        index_dir=str(tmp_path / "idx"),
        metadata_path=str(tmp_path / "meta.json"),
    )
    with patch("rag_pipeline.build_embeddings"), \
         patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        vsm = VectorStoreManager(cfg)
        vsm._doc_registry = {"abc123": {"file": "test.pdf", "chunks": 10}}
        assert vsm.is_indexed("abc123")
        assert not vsm.is_indexed("xyz999")


def test_registry_get_files(tmp_path):
    from rag_pipeline import VectorStoreManager

    cfg = RAGConfig(
        index_dir=str(tmp_path / "idx"),
        metadata_path=str(tmp_path / "meta.json"),
    )
    with patch("rag_pipeline.build_embeddings"), \
         patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        vsm = VectorStoreManager(cfg)
        vsm._doc_registry = {
            "h1": {"file": "a.pdf", "chunks": 5, "file_type": "pdf"},
            "h2": {"file": "b.csv", "chunks": 12, "file_type": "csv"},
        }
        files = vsm.get_indexed_files()
        assert len(files) == 2
        assert any(f["file"] == "a.pdf" for f in files)
        assert any(f["file"] == "b.csv" for f in files)


# ──────────────────────────────────────────────
# Query (mocked — no API key needed)
# ──────────────────────────────────────────────

def _mock_vsm(candidates: list[tuple[Document, float]]) -> MagicMock:
    mock_vsm = MagicMock()
    mock_vsm.vectorstore = MagicMock()
    mock_vsm.hybrid_search.return_value = candidates
    return mock_vsm


def test_query_no_vectorstore():
    from rag_pipeline import query
    result = query("What is credit risk?", vsm=None, config=RAGConfig())
    assert not result.found_in_docs
    assert "No documents" in result.answer


def test_query_below_threshold():
    from rag_pipeline import query

    doc = Document(page_content="text", metadata={"source_file": "a.pdf", "chunk_index": 0})
    mock_vsm = _mock_vsm([(doc, 0.10)])

    result = query("question", vsm=mock_vsm, config=RAGConfig(score_threshold=0.99))
    assert not result.found_in_docs
    assert result.sources == []


def test_query_no_threshold_passes_low_scores():
    """Default threshold 0.0 should not block low-scoring but relevant chunks."""
    from rag_pipeline import query

    doc = Document(
        page_content="Climate change affects global temperatures.",
        metadata={"source_file": "climate.pdf", "chunk_index": 0},
    )
    mock_vsm = _mock_vsm([(doc, 0.05)])
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value.content = "Climate change affects temperatures."

    with patch("rag_pipeline.rerank", return_value=([(doc, 1.0)], True)), \
         patch("rag_pipeline.ChatOpenAI", return_value=mock_llm_instance), \
         patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        result = query("What about climate?", vsm=mock_vsm, config=RAGConfig(score_threshold=0.0))

    assert result.found_in_docs


def test_query_metadata_filter():
    from rag_pipeline import query

    doc_a = Document(page_content="content from A", metadata={"source_file": "a.pdf", "chunk_index": 0})
    doc_b = Document(page_content="content from B", metadata={"source_file": "b.pdf", "chunk_index": 1})

    mock_vsm = _mock_vsm([(doc_a, 0.85)])

    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value.content = "Answer from A only."

    with patch("rag_pipeline.rerank", return_value=([(doc_a, 0.85)], True)), \
         patch("rag_pipeline.ChatOpenAI", return_value=mock_llm_instance), \
         patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        result = query(
            "question",
            vsm=mock_vsm,
            config=RAGConfig(score_threshold=0.0),
            filter_file="a.pdf",
        )

    assert all(s.file == "a.pdf" for s in result.sources)
    mock_vsm.hybrid_search.assert_called_once()


def test_query_returns_both_scores():
    from rag_pipeline import query

    doc = Document(
        page_content="Basel III requires CET1 of 4.5%.",
        metadata={"source_file": "basel.pdf", "page": 2, "chunk_index": 3},
    )
    mock_vsm = _mock_vsm([(doc, 0.87)])

    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value.content = "CET1 minimum is 4.5%."

    with patch("rag_pipeline.rerank", return_value=([(doc, 12.5)], True)), \
         patch("rag_pipeline.ChatOpenAI", return_value=mock_llm_instance), \
         patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        result = query("What is CET1?", vsm=mock_vsm, config=RAGConfig())

    assert result.found_in_docs
    assert result.reranked is True
    assert result.hybrid_search is True
    assert result.sources[0].similarity_score == 0.87
    assert result.sources[0].rerank_score == 12.5


def test_load_document_empty_file_raises(tmp_path):
    from rag_pipeline import load_document

    f = tmp_path / "empty.txt"
    f.write_text("")
    with pytest.raises(ValueError, match="No content could be extracted"):
        load_document(str(f))
