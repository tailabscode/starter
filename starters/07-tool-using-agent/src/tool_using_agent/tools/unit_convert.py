"""unit_convert — conversion over a small fixed table of length/weight/temperature units.

Every unit is validated against the table below; unknown units and
cross-category conversions (e.g. length -> weight) fail with a clear
`ToolError` rather than producing a nonsense number.
"""

from .base import ToolError

_LENGTH_TO_METERS = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "km": 1000.0,
    "kilometer": 1000.0,
    "kilometers": 1000.0,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
    "mile": 1609.344,
    "miles": 1609.344,
    "mi": 1609.344,
    "yard": 0.9144,
    "yards": 0.9144,
    "yd": 0.9144,
    "foot": 0.3048,
    "feet": 0.3048,
    "ft": 0.3048,
    "inch": 0.0254,
    "inches": 0.0254,
    "in": 0.0254,
}

_WEIGHT_TO_GRAMS = {
    "g": 1.0,
    "gram": 1.0,
    "grams": 1.0,
    "kg": 1000.0,
    "kilogram": 1000.0,
    "kilograms": 1000.0,
    "lb": 453.592,
    "lbs": 453.592,
    "pound": 453.592,
    "pounds": 453.592,
    "oz": 28.3495,
    "ounce": 28.3495,
    "ounces": 28.3495,
}

_TEMPERATURE_UNITS = {"c", "celsius", "f", "fahrenheit", "k", "kelvin"}


class UnitConvertError(Exception):
    """Raised for an unknown unit or a cross-category conversion."""


def _category_of(unit: str) -> str | None:
    if unit in _TEMPERATURE_UNITS:
        return "temperature"
    if unit in _LENGTH_TO_METERS:
        return "length"
    if unit in _WEIGHT_TO_GRAMS:
        return "weight"
    return None


def _to_celsius(value: float, unit: str) -> float:
    if unit in ("c", "celsius"):
        return value
    if unit in ("f", "fahrenheit"):
        return (value - 32) * 5 / 9
    return value - 273.15  # kelvin


def _from_celsius(value: float, unit: str) -> float:
    if unit in ("c", "celsius"):
        return value
    if unit in ("f", "fahrenheit"):
        return value * 9 / 5 + 32
    return value + 273.15  # kelvin


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert `value` between two units. Raises `UnitConvertError` on any mismatch."""
    f, t = from_unit.strip().lower(), to_unit.strip().lower()
    cf, ct = _category_of(f), _category_of(t)
    if cf is None or ct is None:
        unknown = [u for u, c in ((from_unit, cf), (to_unit, ct)) if c is None]
        raise UnitConvertError(
            f"Unknown unit(s): {', '.join(unknown)}. Supported categories: "
            "length, weight, temperature."
        )
    if cf != ct:
        raise UnitConvertError(
            f"Cannot convert '{from_unit}' ({cf}) to '{to_unit}' ({ct}): different unit categories."
        )
    if cf == "temperature":
        return _from_celsius(_to_celsius(value, f), t)
    table = _LENGTH_TO_METERS if cf == "length" else _WEIGHT_TO_GRAMS
    return value * table[f] / table[t]


SCHEMA = {
    "name": "unit_convert",
    "description": (
        "Convert a numeric value between units of length, weight, or temperature "
        "from a small fixed table (e.g. km, miles, kg, lb, celsius, fahrenheit). "
        "Rejects unknown units and conversions across categories."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "value": {"type": "number", "description": "The numeric value to convert."},
            "from_unit": {"type": "string", "description": "Unit to convert from, e.g. 'km'."},
            "to_unit": {"type": "string", "description": "Unit to convert to, e.g. 'miles'."},
        },
        "required": ["value", "from_unit", "to_unit"],
        "additionalProperties": False,
    },
}


def run(tool_input: dict) -> str:
    value = tool_input.get("value")
    from_unit = tool_input.get("from_unit")
    to_unit = tool_input.get("to_unit")
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ToolError("`value` must be a number.")
    if not isinstance(from_unit, str) or not isinstance(to_unit, str):
        raise ToolError("`from_unit` and `to_unit` must be strings.")
    try:
        result = convert(float(value), from_unit, to_unit)
    except UnitConvertError as e:
        raise ToolError(str(e)) from e
    return f"{value} {from_unit} = {round(result, 6)} {to_unit}"
