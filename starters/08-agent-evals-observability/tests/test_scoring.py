from agent_evals.example_agent import ToolCall
from agent_evals.scoring import score_keyword_coverage, score_tool_use


def test_keyword_coverage_all_present():
    assert score_keyword_coverage("The answer is 42, found in 3 days", ["42", "3 days"]) == 1.0


def test_keyword_coverage_partial():
    assert score_keyword_coverage("The answer is 42", ["42", "banana"]) == 0.5


def test_keyword_coverage_case_insensitive():
    assert score_keyword_coverage("REFUNDS WITHIN 30 DAYS", ["30 days"]) == 1.0


def test_keyword_coverage_none_present():
    assert score_keyword_coverage("nothing relevant here", ["banana", "spaceship"]) == 0.0


def test_keyword_coverage_empty_expected_is_vacuous_pass():
    assert score_keyword_coverage("anything at all", []) == 1.0


def test_tool_use_exact_match_with_sane_args():
    calls = [ToolCall(name="calculator", input={"expression": "2 + 2"}, output="4")]
    assert score_tool_use(calls, ["calculator"]) == 1.0


def test_tool_use_wrong_tool_called():
    calls = [ToolCall(name="search_kb", input={"query": "warranty"}, output="...")]
    assert score_tool_use(calls, ["calculator"]) == 0.0


def test_tool_use_no_call_expected_and_none_made():
    assert score_tool_use([], []) == 1.0


def test_tool_use_unexpected_call_when_none_expected():
    calls = [ToolCall(name="calculator", input={"expression": "1+1"}, output="2")]
    assert score_tool_use(calls, []) == 0.0


def test_tool_use_missing_expected_call():
    assert score_tool_use([], ["calculator"]) == 0.0


def test_tool_use_empty_args_is_not_sane():
    calls = [ToolCall(name="calculator", input={}, output="4")]
    assert score_tool_use(calls, ["calculator"]) == 0.0


def test_tool_use_blank_string_arg_is_not_sane():
    calls = [ToolCall(name="search_kb", input={"query": "   "}, output="...")]
    assert score_tool_use(calls, ["search_kb"]) == 0.0


def test_tool_use_multiple_expected_calls_order_independent():
    calls = [
        ToolCall(name="search_kb", input={"query": "warranty"}, output="..."),
        ToolCall(name="calculator", input={"expression": "1+1"}, output="2"),
    ]
    assert score_tool_use(calls, ["calculator", "search_kb"]) == 1.0
