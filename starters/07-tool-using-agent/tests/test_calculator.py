import pytest

from tool_using_agent.tools import ToolError
from tool_using_agent.tools.calculator import evaluate, run


def test_basic_addition():
    assert evaluate("2 + 3") == 5


def test_precedence():
    assert evaluate("2 + 3 * 4") == 14


def test_parentheses_override_precedence():
    assert evaluate("(2 + 3) * 4") == 20


def test_decimals():
    assert evaluate("1.5 + 2.5") == 4.0


def test_unary_minus():
    assert evaluate("-5 + 10") == 5


def test_division():
    assert evaluate("10 / 4") == 2.5


def test_division_by_zero_raises_calculator_error():
    with pytest.raises(Exception, match="division by zero"):
        evaluate("1 / 0")


def test_run_happy_path_returns_readable_string():
    result = run({"expression": "6 * 7"})
    assert result == "6 * 7 = 42"


def test_run_division_by_zero_is_tool_error():
    with pytest.raises(ToolError, match="division by zero"):
        run({"expression": "1 / 0"})


def test_run_rejects_unsupported_characters():
    with pytest.raises(ToolError):
        run({"expression": "2 ** 3"})


def test_run_rejects_non_string_expression():
    with pytest.raises(ToolError):
        run({"expression": 42})


def test_run_rejects_empty_expression():
    with pytest.raises(ToolError):
        run({"expression": "   "})


def test_missing_closing_paren_is_rejected():
    with pytest.raises(ToolError):
        run({"expression": "(2 + 3"})
