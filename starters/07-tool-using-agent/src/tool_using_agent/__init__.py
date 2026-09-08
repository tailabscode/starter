"""tool_using_agent: a bounded, tool-calling agent loop built directly on the Claude API.

Public surface for library use; the CLI (see `cli.py`) is the primary entry point.
"""

from .agent import AgentResult, ToolCallRecord, run_agent
from .tools import ToolError, build_registry

__version__ = "0.1.0"

__all__ = [
    "AgentResult",
    "ToolCallRecord",
    "ToolError",
    "build_registry",
    "run_agent",
    "__version__",
]
