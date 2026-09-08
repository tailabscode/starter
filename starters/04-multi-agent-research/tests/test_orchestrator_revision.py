"""An insufficient critique triggers exactly one more round, and then stops."""

from __future__ import annotations

from dataclasses import replace

from multi_agent_research.config import load_config
from multi_agent_research.corpus import Corpus
from multi_agent_research.llm import StubClient
from multi_agent_research.orchestrator import run_research


def test_insufficient_critique_triggers_exactly_one_more_round(tmp_path) -> None:
    base_config = load_config()
    # The bundled corpus has 5 topic docs; max_subtasks=4 guarantees round 1 misses exactly one,
    # so the critic should flag it and round 2 should close the gap.
    config = replace(base_config, runs_dir=tmp_path, max_subtasks=4, max_rounds=2)
    corpus = Corpus.from_data_dir(config.data_dir)

    state = run_research(
        "How risky is the EV battery supply chain?", llm=StubClient(), corpus=corpus, config=config
    )

    assert state.revision_round == 2
    assert len(state.critiques) == 2
    assert state.critiques[0]["is_sufficient"] is False
    assert state.critiques[0]["gaps"]
    assert state.critiques[-1]["is_sufficient"] is True
    assert state.termination_reason == "sufficient"
    assert state.final_report
