import pytest

from tool_using_agent.tools import ToolError
from tool_using_agent.tools.unit_convert import convert, run


def test_length_km_to_miles():
    result = convert(1.0, "km", "miles")
    assert result == pytest.approx(0.621371, rel=1e-4)


def test_weight_kg_to_lb():
    result = convert(1.0, "kg", "lb")
    assert result == pytest.approx(2.20462, rel=1e-4)


def test_temperature_celsius_to_fahrenheit():
    assert convert(0.0, "celsius", "fahrenheit") == pytest.approx(32.0)
    assert convert(100.0, "c", "f") == pytest.approx(212.0)


def test_temperature_celsius_to_kelvin():
    assert convert(0.0, "celsius", "kelvin") == pytest.approx(273.15)


def test_same_unit_is_identity():
    assert convert(42.0, "m", "m") == 42.0


def test_run_happy_path():
    result = run({"value": 10, "from_unit": "km", "to_unit": "m"})
    assert result == "10 km = 10000.0 m"


def test_unknown_unit_raises_tool_error():
    with pytest.raises(ToolError, match="Unknown unit"):
        run({"value": 1, "from_unit": "smoots", "to_unit": "m"})


def test_cross_category_raises_tool_error():
    with pytest.raises(ToolError, match="different unit categories"):
        run({"value": 1, "from_unit": "kg", "to_unit": "meters"})


def test_non_numeric_value_raises_tool_error():
    with pytest.raises(ToolError):
        run({"value": "ten", "from_unit": "km", "to_unit": "m"})
