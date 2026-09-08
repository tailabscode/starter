import pytest

from multi_agent_research.config import DEFAULT_DATA_DIR
from multi_agent_research.corpus import Corpus


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    return Corpus.from_data_dir(DEFAULT_DATA_DIR)
