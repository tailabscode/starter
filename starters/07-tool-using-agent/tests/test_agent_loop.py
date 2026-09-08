"""Tests for the manual agentic loop, using a small fake LLMClient so the loop's
bounding, parallel-tool-call, and error-handling behavior can be tested without
going through the (also offline) StubClient's keyword routing.
"""

from tool_using_agent.agent import run_agent
from tool_using_agent.llm import LLMResponse, TextBlock, ToolUseBlock
from tool_using_agent.tools import build_registry


class ScriptedClient:
    """Replays a fixed sequence of LLMResponse objects, one per `create()` call."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    def create(self, messages, tools, system):
        del tools, system
        self.calls.append(messages)
        return self._responses.pop(0)


class LoopingClient:
    """Always asks for the same tool call — used to test the max_steps bound."""

    def __init__(self, tool_name: str, tool_input: dict):
        self._tool_name = tool_name
        self._tool_input = tool_input
        self.call_count = 0

    def create(self, messages, tools, system):
        del messages, tools, system
        self.call_count += 1
        block = ToolUseBlock(
            id=f"toolu_{self.call_count}", name=self._tool_name, input=self._tool_input
        )
        return LLMResponse(content=[block], stop_reason="tool_use")


def test_single_tool_call_then_final_answer():
    client = ScriptedClient(
        [
            LLMResponse(
                content=[
                    ToolUseBlock(id="toolu_1", name="calculator", input={"expression": "2 + 2"})
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(content=[TextBlock(text="It's 4.")], stop_reason="end_turn"),
        ]
    )
    result = run_agent(client, build_registry(), "what is 2 + 2?")
    assert result.final_text == "It's 4."
    assert result.steps_used == 2
    assert not result.hit_max_steps
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].output == "2 + 2 = 4"
    assert not result.tool_calls[0].is_error


def test_parallel_tool_calls_all_get_results_in_one_message():
    client = ScriptedClient(
        [
            LLMResponse(
                content=[
                    ToolUseBlock(id="toolu_1", name="calculator", input={"expression": "2 + 2"}),
                    ToolUseBlock(id="toolu_2", name="lookup_fact", input={"topic": "python"}),
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(content=[TextBlock(text="done")], stop_reason="end_turn"),
        ]
    )
    result = run_agent(client, build_registry(), "do two things")

    assert len(result.tool_calls) == 2
    assert {c.name for c in result.tool_calls} == {"calculator", "lookup_fact"}
    assert all(not c.is_error for c in result.tool_calls)

    # The second create() call must carry both tool_results in a single user message.
    second_call_messages = client.calls[1]
    tool_result_message = second_call_messages[-1]
    assert tool_result_message["role"] == "user"
    assert len(tool_result_message["content"]) == 2
    ids = {block["tool_use_id"] for block in tool_result_message["content"]}
    assert ids == {"toolu_1", "toolu_2"}


def test_unknown_tool_name_does_not_crash_the_loop():
    client = ScriptedClient(
        [
            LLMResponse(
                content=[ToolUseBlock(id="toolu_1", name="not_a_real_tool", input={})],
                stop_reason="tool_use",
            ),
            LLMResponse(content=[TextBlock(text="handled")], stop_reason="end_turn"),
        ]
    )
    result = run_agent(client, build_registry(), "call a fake tool")
    assert result.final_text == "handled"
    assert result.tool_calls[0].is_error
    assert "Unknown tool" in result.tool_calls[0].output


def test_tool_validation_failure_returns_is_error_and_continues():
    client = ScriptedClient(
        [
            LLMResponse(
                content=[
                    ToolUseBlock(id="toolu_1", name="calculator", input={"expression": "1 / 0"})
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(content=[TextBlock(text="that failed")], stop_reason="end_turn"),
        ]
    )
    result = run_agent(client, build_registry(), "divide by zero")
    assert result.tool_calls[0].is_error
    assert "division by zero" in result.tool_calls[0].output
    assert result.final_text == "that failed"


def test_max_steps_bound_is_enforced():
    client = LoopingClient("calculator", {"expression": "1 + 1"})
    result = run_agent(client, build_registry(), "loop forever", max_steps=3)
    assert result.hit_max_steps
    assert result.steps_used == 3
    assert client.call_count == 3
    assert "max_steps" in result.final_text
    assert len(result.tool_calls) == 3
