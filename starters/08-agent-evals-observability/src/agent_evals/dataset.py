"""Load eval cases from a JSONL file (see `data/eval_cases.jsonl`)."""

import json
from dataclasses import dataclass
from pathlib import Path

# src/agent_evals/dataset.py -> parents[2] is the starter root, where data/ lives.
DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "eval_cases.jsonl"


@dataclass(frozen=True)
class EvalCase:
    id: str
    input: str
    category: str
    expected_facts: list[str]
    expected_tool_calls: list[str]
    use_llm_judge: bool


def load_cases(path: Path) -> list[EvalCase]:
    """Parse one `EvalCase` per non-empty line. Raises `ValueError` with the
    line number on malformed JSON or a missing required field."""
    cases = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            try:
                cases.append(
                    EvalCase(
                        id=raw["id"],
                        input=raw["input"],
                        category=raw["category"],
                        expected_facts=raw.get("expected_facts", []),
                        expected_tool_calls=raw.get("expected_tool_calls", []),
                        use_llm_judge=raw.get("use_llm_judge", False),
                    )
                )
            except KeyError as e:
                raise ValueError(f"{path}:{line_no}: missing required field {e}") from e
    return cases
