"""Session memory: the current conversation's message list. Ephemeral, in-process.

This class does zero I/O on purpose -- it is the concrete embodiment of "not
persisted." A new `SessionMemory` always starts with an empty conversation,
even for a `--session-id` that was used before: the id is only a label used
to tag which persistent memories a given exchange produced (see
`store.MemoryStore`'s `source_session_id` column), not a key that reloads
prior conversation turns. Anything that needs to survive belongs in the
persistent store instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionMemory:
    session_id: str
    messages: list[dict[str, str]] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
