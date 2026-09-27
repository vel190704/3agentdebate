import asyncio
import json
from typing import Dict, List, Tuple

from pydantic import ValidationError

from ..config import ProposerConfig, settings
from ..llm.openai_compatible_client import OpenAICompatibleClient
from ..schemas import ProposerResponse

PROPOSER_SYSTEM_PROMPT = """You are one of several independent models proposing solutions for a
software architecture task. You do not see other models' answers. Respond ONLY with JSON
matching this shape:

{
  "task_id": "string",
  "model": "string",
  "decisions": [
    {
      "decision_id": "snake_case_id",
      "dimension": "human-readable label",
      "value": "your chosen option",
      "confidence": 0.0,
      "reasoning": "max ~150 words, plain text",
      "cited_requirements": ["requirement ids from shared context, if any"],
      "assumptions": ["explicit assumptions this decision depends on"]
    }
  ]
}

You MUST provide exactly one decision for each decision_id in the provided list, using that
exact decision_id and dimension. If you believe an additional decision is needed that is not
in the list, you may add it with an extra field "proposer_added": true - but do not invent new
decision_ids for items already in the list.

Some decisions come with an "expected options" list - known named answers other models will
also be choosing from. If one of those options is your answer, put that option's name in
"value" VERBATIM (exact spelling/casing), and put any elaboration (e.g. "using the managed
variant on AWS") in "reasoning" instead - do not fold elaboration into the value string itself.
If none of the expected options fit your recommendation, answer with your own specific choice
as normal; do not just write "other".
"""


async def _call_one_proposer(
    cfg: ProposerConfig, task_id: str, prompt: str
) -> Tuple[ProposerResponse, str, int, int]:
    client = OpenAICompatibleClient(model=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url)

    last_error = None
    for attempt in range(settings.max_json_retries + 1):
        this_prompt = prompt
        if attempt > 0:
            this_prompt += (
                f"\n\nYour previous response was invalid: {last_error}\n"
                "Return ONLY valid JSON matching the required shape."
            )
        raw_text, prompt_tokens, completion_tokens = await client.complete_json(
            PROPOSER_SYSTEM_PROMPT, this_prompt
        )
        try:
            data = json.loads(raw_text)
            parsed = ProposerResponse.model_validate(data)
            return parsed, raw_text, prompt_tokens, completion_tokens
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)

    raise ValueError(
        f"Proposer '{cfg.name}' ({cfg.model}) failed to produce valid JSON after retry: {last_error}"
    )


async def run_proposers(
    task_id: str,
    task_description: str,
    context_text: str,
    requirements: List[Dict],
    framing_decisions: List[Dict],
):
    """Calls every configured proposer in parallel. Fails loudly (raises) if any
    proposer never produces a schema-valid response after its retry - CLAUDE.md
    Section 3 says invalid decisions must never be silently dropped, and the diff
    step needs every proposer's output to be meaningful.
    """
    prompt = _build_prompt(task_id, task_description, context_text, requirements, framing_decisions)

    results = await asyncio.gather(
        *[_call_one_proposer(cfg, task_id, prompt) for cfg in settings.proposers],
        return_exceptions=True,
    )

    failures = [r for r in results if isinstance(r, Exception)]
    if failures:
        raise ValueError("One or more proposers failed: " + "; ".join(str(f) for f in failures))

    return list(zip(settings.proposers, results))


def _format_decision_line(d: Dict) -> str:
    line = f"- {d['decision_id']} ({d['dimension']}): {d['description']}"
    value_options = d.get("value_options") or []
    if value_options:
        line += f" [expected options: {', '.join(value_options)}]"
    return line


def _build_prompt(
    task_id: str,
    task_description: str,
    context_text: str,
    requirements: List[Dict],
    framing_decisions: List[Dict],
) -> str:
    req_lines = "\n".join(f"- {r['req_id']}: {r['text']}" for r in requirements) or "(none provided)"
    decision_lines = "\n".join(_format_decision_line(d) for d in framing_decisions)
    return f"""task_id: {task_id}
model: (fill in with your own model name)

TASK:
{task_description}

SHARED CONTEXT:
{context_text or "(none provided)"}

REQUIREMENTS:
{req_lines}

DECISIONS TO ADDRESS (use these exact decision_ids and dimensions):
{decision_lines}

Propose a value for each decision above."""
