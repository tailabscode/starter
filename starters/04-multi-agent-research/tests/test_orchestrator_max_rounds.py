"""max_rounds is a hard cap: the run terminates cleanly instead of looping forever."""

from __future__ import annotations

from dataclasses import replace

from multi_agent_research.config import load_config
from multi_agent_research.corpus import Corpus
from multi_agent_research.llm import StubClient
from multi_agent_research.orchestrator import run_research


def test_max_rounds_terminates_cleanly_even_when_still_insufficient(tmp_path) -> None:
    base_config = load_config()
    # 1 subtask against a 5-topic corpus guarantees the critic stays unsatisfied; max_rounds=1
    # means no revision round is even attempted.
    config = replace(base_config, runs_dir=tmp_path, max_subtasks=1, max_rounds=1)
    corpus = Corpus.from_data_dir(config.data_dir)

    state = run_research(
        "How risky is the EV battery supply chain?", llm=StubClient(), corpus=corpus, config=config
    )

    assert state.revision_round == 1
    assert len(state.critiques) == 1
    assert state.critiques[0]["is_sufficient"] is False
    assert state.termination_reason == "max_rounds_reached"
    # Still produces a usable report even though the bound was hit -- never leaves it empty.
    assert state.final_report
