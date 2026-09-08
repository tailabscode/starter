import json

import pytest

from tool_using_agent.tools import ToolError
from tool_using_agent.tools.http_style_adapter import run


def test_known_city_returns_synthetic_shaped_response():
    result = json.loads(run({"city": "Paris"}))
    assert result["city"] == "Paris"
    assert result["condition"] == "Sunny"
    assert result["synthetic"] is True
    expected_keys = {"city", "condition", "temperature_c", "humidity_pct", "wind_kph", "synthetic"}
    assert set(result) == expected_keys


def test_city_lookup_is_case_insensitive():
    result = json.loads(run({"city": "tOkYo"}))
    assert result["city"] == "Tokyo"


def test_unknown_city_raises_tool_error_listing_known_cities():
    with pytest.raises(ToolError, match="Unknown city"):
        run({"city": "Atlantis"})


def test_empty_city_raises_tool_error():
    with pytest.raises(ToolError):
        run({"city": ""})
