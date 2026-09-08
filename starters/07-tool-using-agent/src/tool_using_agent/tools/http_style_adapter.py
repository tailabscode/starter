"""http_style_adapter — adapter pattern for an external/API-style tool, WITHOUT any
real network call.

This is a SYNTHETIC weather adapter: fixed, deterministic data shaped
exactly like a real weather API response would be (city, condition,
temperature, humidity, wind), with the request-validation ->
"call" -> response-shaping -> error-handling pipeline a real HTTP adapter
would have. It never opens a socket. The response payload also carries
`"synthetic": true` so nothing downstream can mistake it for live data.
"""

import json

from .base import ToolError

# SYNTHETIC DATA — not a live weather feed. Fixed per city so results are
# deterministic for tests and the offline demo.
_SYNTHETIC_WEATHER = {
    "london": {"condition": "Cloudy", "temperature_c": 14, "humidity_pct": 78, "wind_kph": 19},
    "paris": {"condition": "Sunny", "temperature_c": 21, "humidity_pct": 55, "wind_kph": 10},
    "tokyo": {"condition": "Rainy", "temperature_c": 24, "humidity_pct": 85, "wind_kph": 14},
    "new york": {"condition": "Clear", "temperature_c": 18, "humidity_pct": 60, "wind_kph": 22},
    "sydney": {"condition": "Windy", "temperature_c": 19, "humidity_pct": 65, "wind_kph": 33},
    "cairo": {"condition": "Sunny", "temperature_c": 34, "humidity_pct": 20, "wind_kph": 12},
    "reykjavik": {"condition": "Snowy", "temperature_c": -2, "humidity_pct": 80, "wind_kph": 28},
    "mumbai": {"condition": "Humid", "temperature_c": 30, "humidity_pct": 88, "wind_kph": 15},
}


def _request(city: str) -> dict:
    """Validate the request. Mirrors what a real adapter would do before calling out."""
    if not isinstance(city, str) or not city.strip():
        raise ToolError("`city` must be a non-empty string.")
    return {"city": city.strip()}


def _call(city: str) -> dict | None:
    """The 'call' step. Looks up fixed synthetic data instead of hitting a network."""
    return _SYNTHETIC_WEATHER.get(city.lower())


def _shape_response(city: str, record: dict) -> dict:
    """Shape the raw record into the same envelope a real API would return."""
    return {
        "city": city.strip().title(),
        "condition": record["condition"],
        "temperature_c": record["temperature_c"],
        "humidity_pct": record["humidity_pct"],
        "wind_kph": record["wind_kph"],
        "synthetic": True,
    }


SCHEMA = {
    "name": "http_style_adapter",
    "description": (
        "Fetch current weather for a known city. SYNTHETIC adapter: returns fixed, "
        "deterministic data shaped like a real weather API response — it never makes "
        "a network call. Demonstrates the adapter pattern (request validation, "
        "response shaping, error handling for an unknown city) for wrapping an "
        "external/API-style service without needing credentials."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name, e.g. 'Paris'."}},
        "required": ["city"],
        "additionalProperties": False,
    },
}


def run(tool_input: dict) -> str:
    city = tool_input.get("city")
    request = _request(city)
    record = _call(request["city"])
    if record is None:
        known = ", ".join(sorted(name.title() for name in _SYNTHETIC_WEATHER))
        raise ToolError(f"Unknown city '{city}'. Known cities: {known}.")
    return json.dumps(_shape_response(request["city"], record))
