"""lookup_fact — a read-only local data tool.

Reads a small JSON knowledge base shipped in `data/knowledge.json` (kept
outside the installed package, next to `src/`, like the rest of this
starter's sample data). Demonstrates the "not found" path a real
retrieval tool needs: an unknown topic is a `ToolError`, not a crash or a
made-up answer.
"""

import json
from pathlib import Path

from .base import ToolError

# starters/07-tool-using-agent/src/tool_using_agent/tools/lookup_fact.py
# -> parents[3] is the starter root, where data/ lives.
_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "knowledge.json"


def _load_knowledge() -> dict[str, str]:
    try:
        with _DATA_PATH.open(encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise ToolError(f"Knowledge file not found at {_DATA_PATH}.") from e
    except json.JSONDecodeError as e:
        raise ToolError(f"Knowledge file at {_DATA_PATH} is not valid JSON.") from e


SCHEMA = {
    "name": "lookup_fact",
    "description": (
        "Look up a short factual summary for a topic from a small local "
        "knowledge base. Read-only. Returns an error if the topic is not "
        "in the knowledge base — never invents an answer."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Topic to look up, e.g. 'photosynthesis'.",
            }
        },
        "required": ["topic"],
        "additionalProperties": False,
    },
}


def run(tool_input: dict) -> str:
    topic = tool_input.get("topic")
    if not isinstance(topic, str) or not topic.strip():
        raise ToolError("`topic` must be a non-empty string.")
    knowledge = _load_knowledge()
    key = topic.strip().lower()
    if key not in knowledge:
        available = ", ".join(sorted(knowledge))
        raise ToolError(f"No fact found for topic '{topic}'. Known topics: {available}.")
    return knowledge[key]
