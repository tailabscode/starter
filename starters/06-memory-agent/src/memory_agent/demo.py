"""The two-session proof: session 1 establishes facts, session 2 (a fresh process,
empty conversation history) correctly answers using only persistent memory.

Each "session" turn genuinely runs `python -m memory_agent chat` in its own
subprocess -- not an in-process shortcut -- so the empty conversation history
in session 2 is not simulated, it is simply true: a brand new process has no
memory of anything except what is on disk in the SQLite file both sessions
share.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass

SESSION_1_ID = "session-1"
SESSION_2_ID = "session-2"

SESSION_1_TURNS = [
    "I prefer Python over JavaScript for backend work.",
    "I live in Berlin. I'm allergic to peanuts.",
]
# Phrased as a question ("Do I prefer... where do I live") on purpose: it must
# NOT itself read as a first-person statement, so the extractor (offline stub
# or real model) doesn't propose a spurious memory from the question itself.
SESSION_2_QUESTION = "Do I prefer Python and where do I live?"


@dataclass
class ChatTurnTranscript:
    session_id: str
    message: str
    stdout: str
    returncode: int


@dataclass
class DemoResult:
    turns: list[ChatTurnTranscript]

    @property
    def session2_reply(self) -> ChatTurnTranscript:
        return self.turns[-1]


def _run_chat_subprocess(
    message: str, session_id: str, db_path: str, offline: bool
) -> ChatTurnTranscript:
    env = dict(os.environ)
    env["MEMORY_AGENT_DB"] = db_path
    args = [sys.executable, "-m", "memory_agent", "chat", message, "--session-id", session_id]
    if offline:
        args.append("--offline")
    result = subprocess.run(args, capture_output=True, text=True, env=env, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"chat subprocess failed (exit {result.returncode}): {result.stderr}")
    return ChatTurnTranscript(
        session_id=session_id, message=message, stdout=result.stdout, returncode=result.returncode
    )


def run_two_session_demo(*, db_path: str, offline: bool = True) -> DemoResult:
    """Run session 1 (multiple turns), then session 2 (one fresh turn). Returns the transcript."""
    if os.path.exists(db_path):
        os.remove(db_path)  # start every demo run from a clean, reproducible slate

    turns: list[ChatTurnTranscript] = []
    for message in SESSION_1_TURNS:
        turns.append(_run_chat_subprocess(message, SESSION_1_ID, db_path, offline))
    turns.append(_run_chat_subprocess(SESSION_2_QUESTION, SESSION_2_ID, db_path, offline))
    return DemoResult(turns=turns)
