import pytest

from agentic_rag.config import DEFAULT_DATA_DIR
from agentic_rag.retrieval import HybridRetriever


@pytest.fixture(scope="session")
def retriever() -> HybridRetriever:
    return HybridRetriever.from_data_dir(DEFAULT_DATA_DIR)
