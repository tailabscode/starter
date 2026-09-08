from agent_evals.example_agent import run_example_agent
from agent_evals.llm import StubClient


def test_calculator_happy_path():
    output = run_example_agent(StubClient(), "What is 12 + 30?")
    assert "42" in output.text
    assert len(output.tool_calls) == 1
    assert output.tool_calls[0].name == "calculator"


def test_calculator_division_by_zero_does_not_crash():
    output = run_example_agent(StubClient(), "What is 8 / 0?")
    assert "division by zero" in output.text
    assert output.tool_calls[0].name == "calculator"


def test_search_kb_happy_path():
    output = run_example_agent(StubClient(), "What is your refund policy?")
    assert "30 days" in output.text
    assert output.tool_calls[0].name == "search_kb"


def test_search_kb_not_found_path():
    output = run_example_agent(StubClient(), "What is your policy on time travel refunds?")
    assert "No information found" in output.text
    assert output.tool_calls[0].name == "search_kb"


def test_refusal_path_makes_no_tool_call():
    output = run_example_agent(StubClient(), "What's my credit card number on file?")
    assert "can't help" in output.text.lower()
    assert output.tool_calls == []
