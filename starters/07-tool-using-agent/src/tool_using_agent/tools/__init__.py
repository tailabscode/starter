"""The tool registry: every tool this starter's agent can call."""

from . import calculator, http_style_adapter, lookup_fact, unit_convert
from .base import ToolError, ToolSpec

_MODULES = (calculator, unit_convert, lookup_fact, http_style_adapter)


def build_registry() -> dict[str, ToolSpec]:
    """Build the name -> ToolSpec map the agent loop dispatches tool calls against."""
    return {
        module.SCHEMA["name"]: ToolSpec(
            name=module.SCHEMA["name"], schema=module.SCHEMA, run=module.run
        )
        for module in _MODULES
    }


__all__ = ["ToolError", "ToolSpec", "build_registry"]
