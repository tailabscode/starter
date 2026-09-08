"""The synthesiser: the one expensive step, turning findings into a single cited report.

Researchers run with ``output_config={"effort": "low"}`` -- cheap, parallel, disposable. The
synthesiser is the opposite: one call, adaptive thinking, over everything the team collected.
That asymmetry (many cheap workers, one expensive integrator) is the intentional shape of this
pipeline; see the README.
"""

from __future__ import annotations

import json

from multi_agent_research.llm import LLMClient
from multi_agent_research.schemas import SynthesisReport

_SYSTEM = (
    "You are the synthesiser for a research team. Write the final, well-organized answer to "
    "the original question, drawing only on the findings provided below. Cite sources inline "
    "using their doc_id in square brackets, e.g. [mining]. Never cite a doc_id that does not "
    "appear among the findings' sources -- if the findings don't support a claim, leave it out."
)


def synthesize(question: str, findings: list[dict], llm: LLMClient) -> SynthesisReport:
    """Turn collected findings into one final report, citing only sources actually retrieved.

    Citations the model proposes but that don't trace back to a real finding source are dropped
    rather than trusted -- the same "don't invent a citation" contract this library uses
    elsewhere (see starter 03's README for the fuller version of this argument).
    """
    retrieved_doc_ids = {
        src["doc_id"] for finding in findings for src in finding.get("sources", [])
    }
    user_content = (
        f"Question: {question}\n\n"
        f"Findings (JSON): {json.dumps(findings)}\n\n"
        "Write the final report. Cite only doc_ids that appear in the findings above."
    )
    report = llm.structured(
        system=_SYSTEM, user_content=user_content, schema=SynthesisReport, thinking=True
    )
    verified_citations = [c for c in report.citations if c in retrieved_doc_ids]
    return SynthesisReport(report=report.report, citations=verified_citations)
