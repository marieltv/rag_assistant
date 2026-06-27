import sys
from pathlib import Path

import pytest

# Add project root to Python path so tests can import rag_pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def reset_cross_encoder_cache():
    """Ensure reranker tests do not share a cached cross-encoder instance."""
    import rag_pipeline
    rag_pipeline._cross_encoder = None
    yield
    rag_pipeline._cross_encoder = None
