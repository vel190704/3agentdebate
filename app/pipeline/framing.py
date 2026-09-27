import json
from typing import List, Dict, Tuple

from pydantic import ValidationError

from ..config import settings
from ..llm.anthropic_client import AnthropicClient
from ..schemas import FramingResponse

FRAMING_SYSTEM_PROMPT = """You are the task-framing step in a multi-LLM deliberation system.
Your ONLY job is to enumerate the distinct decisions a task requires - not to make any of them.

Given a task description and shared context, output a fixed JSON list of decision points that
proposing models will each independently address using the exact same identifiers.

For each decision point, give:
- decision_id: a stable snake_case identifier (e.g. "database", "message_queue", "auth_method")
- dimension: a short human-readable label (e.g. "Database")
- description: one sentence describing what is being decided, with no opinion on the answer
- value_options: OPTIONAL. If, and only if, this decision has a small set of well-known named
  answers (specific technologies, protocols, or products - e.g. message queue tech, database
  engine, auth protocol), list those canonical names here so every proposing model answers
  using the same vocabulary. If the decision is open-ended (schema design, naming conventions,
  anything without a small closed set of standard named answers), leave this as an empty list -
  do not force an open-ended decision into a fake enum.

STRICT RULES:
- Do NOT propose or hint at a value, option, or recommendation for any decision.
- Do NOT include confidence, reasoning, or "leaning towards X" language.
- Only decide WHAT needs to be decided, never WHICH option is best - value_options is an
  inventory of the known named choices in the space, not a ranked or favored list.
- When you do list value_options, the names must not overlap or prefix-match each other
  (e.g. never list both "SQL" and "NoSQL", or both "Auth" and "OAuth" - those are visually
  containable within each other and would corrupt substring-based matching downstream).
  Pick names that are unambiguous as distinct substrings of one another.
- A model may still answer with something not in value_options if none of the listed options
  fit its recommendation - the list is a known-options hint, not an exhaustive restriction, so
  do not add a catch-all "other" entry.
- Output strictly as JSON matching this shape, and nothing else:
{"task_id": "...", "decisions": [{"decision_id": "...", "dimension": "...", "description": "...", "value_options": []}]}
"""


async def run_framing(
    task_id: str, task_description: str, context_text: str, requirements: List[Dict]
) -> Tuple[FramingResponse, int, int]:
    client = AnthropicClient(model=settings.claude_model, api_key=settings.anthropic_api_key)
    prompt = _build_prompt(task_id, task_description, context_text, requirements)

    last_error = None
    for attempt in range(settings.max_json_retries + 1):
        this_prompt = prompt
        if attempt > 0:
            this_prompt += (
                f"\n\nYour previous response was invalid: {last_error}\n"
                "Return ONLY valid JSON matching the required shape."
            )
        try:
            raw_text, prompt_tokens, completion_tokens = await client.complete_json(FRAMING_SYSTEM_PROMPT, this_prompt)
        except Exception as e:
            # Transport failure - the client's own retry (app/llm/base.py) is
            # already exhausted by the time this raises, and there's nothing
            # framing-specific to retry against (this isn't a JSON-shape
            # problem). Wrap it the same way run_proposers wraps a proposer
            # transport failure, so create_task's single `except ValueError`
            # handles both call sites the same clean way instead of one
            # raising a raw SDK exception uncaught.
            raise ValueError(f"Task-framing call failed: {e}") from e
        try:
            data = json.loads(raw_text)
            return FramingResponse.model_validate(data), prompt_tokens, completion_tokens
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)

    raise ValueError(f"Task-framing step failed to produce valid JSON after retry: {last_error}")


def _build_prompt(task_id: str, task_description: str, context_text: str, requirements: List[Dict]) -> str:
    req_lines = "\n".join(f"- {r['req_id']}: {r['text']}" for r in requirements) or "(none provided)"
    return f"""task_id: {task_id}

TASK:
{task_description}

SHARED CONTEXT:
{context_text or "(none provided)"}

REQUIREMENTS:
{req_lines}

Enumerate the decision points this task requires."""
