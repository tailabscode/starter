from tool_using_agent.llm import StubClient, ToolUseBlock


def test_stub_routes_arithmetic_to_calculator():
    client = StubClient()
    response = client.create(
        messages=[{"role": "user", "content": "What is 6 * 7?"}], tools=[], system=""
    )
    assert response.stop_reason == "tool_use"
    block = response.content[0]
    assert isinstance(block, ToolUseBlock)
    assert block.name == "calculator"
    assert "6" in block.input["expression"] and "7" in block.input["expression"]


def test_stub_routes_weather_question_to_http_style_adapter():
    client = StubClient()
    response = client.create(
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=[],
        system="",
    )
    block = response.content[0]
    assert block.name == "http_style_adapter"
    assert block.input["city"] == "Paris"


def test_stub_routes_convert_phrase_to_unit_convert():
    client = StubClient()
    response = client.create(
        messages=[{"role": "user", "content": "Convert 10 km to miles"}], tools=[], system=""
    )
    block = response.content[0]
    assert block.name == "unit_convert"
    assert block.input == {"value": 10.0, "from_unit": "km", "to_unit": "miles"}


def test_stub_falls_back_to_lookup_fact():
    client = StubClient()
    response = client.create(
        messages=[{"role": "user", "content": "Tell me about photosynthesis"}], tools=[], system=""
    )
    block = response.content[0]
    assert block.name == "lookup_fact"
    assert block.input["topic"] == "photosynthesis"


def test_stub_summarizes_a_successful_tool_result_on_the_next_turn():
    client = StubClient()
    messages = [
        {"role": "user", "content": "What is 6 * 7?"},
        {"role": "assistant", "content": []},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "6 * 7 = 42"}],
        },
    ]
    response = client.create(messages=messages, tools=[], system="")
    assert response.stop_reason == "end_turn"
    assert "42" in response.content[0].text


def test_stub_summarizes_a_failed_tool_result_on_the_next_turn():
    client = StubClient()
    messages = [
        {"role": "user", "content": "What is 1 / 0?"},
        {"role": "assistant", "content": []},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "division by zero",
                    "is_error": True,
                }
            ],
        },
    ]
    response = client.create(messages=messages, tools=[], system="")
    assert "couldn't complete" in response.content[0].text
