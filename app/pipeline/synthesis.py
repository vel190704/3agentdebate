"""Builds and runs the arbiter's natural-language synthesis call (CLAUDE.md
Section 5, item 5). The model's only job here is to explain a score that has
already been computed in code (arbiter.py), never to re-score or re-pick the
winner - see SYNTHESIS_SYSTEM_PROMPT for the enforced boundaries. Runs for
both arbiter_resolved and human_decision_required outcomes (every status
score_decision can return except resolved_via_debate, which never produces a
score to explain): a tie with no explanation is exactly the case a human
reviewer needs this most.
"""

from typing import Dict, Tuple

from ..config import settings
from ..llm.anthropic_client import AnthropicClient

SYNTHESIS_SYSTEM_PROMPT = """You are writing the explanation for a decision that has ALREADY
been resolved by deterministic code - not by you. You are not the arbiter. The score, the
per-model point breakdown, and the winner (or the human-review outcome) are all fixed inputs
that you must treat as ground truth.

Your ONLY job: write a short, clear natural-language synthesis of why the given outcome
happened, using only the score breakdown and debate messages provided to you.

STRICT RULES:
- Do NOT recompute, question, or second-guess the score or the winner. If you privately think
  a different model made the better argument, that is irrelevant - your text must still explain
  and justify the winner exactly as given.
- Do NOT introduce facts, evidence, or reasoning that isn't present in the provided debate
  messages, requirements, or score breakdown. Do not speculate about what either model "really
  meant."
- Do NOT change, soften, or hedge the outcome. If the outcome is HUMAN_DECISION_REQUIRED
  because of a tie, say plainly that it is a tie requiring human review - do not suggest which
  side you'd lean toward.
- Cover, in plain prose (3-5 sentences): (1) what the two positions were, (2) which round-2
  citation(s) were valid or invalid and why, (3) how those citation points produced the final
  score and outcome.
- Output plain text only. No JSON, no markdown headers, no restating these instructions.
"""


def build_synthesis_prompt(
    decision_id: str,
    dimension: str,
    description: str,
    transcript: Dict,
    arbiter_result: Dict,
) -> str:
    round1_lines = "\n".join(
        f"- {m['model']}: value={m['value']!r}, claim={m['claim']!r}, "
        f"cited_requirement={m['cited_requirement']!r}"
        for m in transcript["round_1"]
    )
    round2_lines = "\n".join(
        f"- {m['model']}: value={m['value']!r}, claim={m['claim']!r}, "
        f"cited_requirement={m['cited_requirement']!r}"
        for m in transcript["round_2"]
    )

    score_lines = []
    for model_name, s in arbiter_result["scores"].items():
        score_lines.append(
            f"- {model_name}: total_score={s['total_score']} "
            f"(citation: {s['citation_points']:+d} - {s['citation_note']})"
        )
    score_block = "\n".join(score_lines)

    if arbiter_result["status"] == "arbiter_resolved":
        outcome_line = (
            f"OUTCOME: {arbiter_result['winner']} wins with the highest score. "
            f"Resolved value: {arbiter_result['winning_value']!r}."
        )
    else:
        outcome_line = "OUTCOME: TIE - status is HUMAN_DECISION_REQUIRED. No winner was chosen."

    return f"""DECISION: {decision_id} ({dimension})
DESCRIPTION: {description}

ROUND 1 POSITIONS:
{round1_lines}

ROUND 2 POSITIONS (final):
{round2_lines}

SCORE BREAKDOWN (already computed - do not recompute):
{score_block}

{outcome_line}

Write the synthesis now, following the rules in your system prompt."""


async def run_synthesis(
    decision_id: str,
    dimension: str,
    description: str,
    transcript: Dict,
    arbiter_result: Dict,
) -> Tuple[str, int, int]:
    """One Claude API call, for the already-scored outcome of one disputed
    decision. Not used to decide anything - see the module docstring.
    """
    client = AnthropicClient(model=settings.claude_model, api_key=settings.anthropic_api_key)
    prompt = build_synthesis_prompt(decision_id, dimension, description, transcript, arbiter_result)
    raw_text, prompt_tokens, completion_tokens = await client.complete_json(SYNTHESIS_SYSTEM_PROMPT, prompt)
    text = raw_text.strip()
    if not text:
        raise ValueError(f"Synthesis call for decision '{decision_id}' returned an empty response")
    return text, prompt_tokens, completion_tokens
