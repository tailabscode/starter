"""memory_agent: an agent with genuinely separate session vs. persistent memory.

Session memory is the current conversation's message list -- ephemeral,
in-process, gone when the process exits. Persistent memory is a SQLite-backed
store of discrete memory records that survives across sessions and processes.
The `demo` command proves the difference concretely with a real two-session
transcript rather than just asserting it.
"""

__version__ = "0.1.0"
