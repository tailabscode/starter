import pytest

from tool_using_agent.tools import ToolError
from tool_using_agent.tools.lookup_fact import run


def test_known_topic_returns_fact():
    result = run({"topic": "python"})
    assert "programming language" in result.lower()


def test_lookup_is_case_insensitive():
    result = run({"topic": "PYTHON"})
    assert "programming language" in result.lower()


def test_unknown_topic_raises_tool_error_with_available_topics():
    with pytest.raises(ToolError, match="No fact found"):
        run({"topic": "cold fusion"})


def test_empty_topic_raises_tool_error():
    with pytest.raises(ToolError):
        run({"topic": "  "})


def test_non_string_topic_raises_tool_error():
    with pytest.raises(ToolError):
        run({"topic": 123})
