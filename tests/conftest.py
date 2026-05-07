import sys
from pathlib import Path

# Add project root to Python path so tests can import rag_pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))
