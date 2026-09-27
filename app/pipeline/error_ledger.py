"""Phase 4 item 1: LLM-narrated error ledger. Extracts a human-readable log
of notable moments from a debate transcript - never a scoring mechanism,
never keyword/deterministic detection (see models.ErrorLedgerEntry's
docstring and CLAUDE.md's Section 5 history on why a hand-built heuristic
was explicitly ruled out here). This module only narrates what a Claude
call reads in the transcript; it never runs during normal task submission -
see backfill_error_ledger, a separate, on-demand step.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

from pydantic import ValidationError
from sqlalchemy.orm import Session

from .. import models
from ..config import settings
from ..llm.anthropic_client import AnthropicClient
from ..schemas import LedgerEntryItem, LedgerExtractionResponse

logger = logging.getLogger(__name__)

ERROR_LEDGER_SYSTEM_PROMPT = """You are extracting a human-readable log of notable moments from a
two-model debate transcript in a multi-LLM deliberation system. This is NOT a scoring or
arbitration step - a deterministic scorer has already decided the outcome using round-2
citations alone, before you see this. Your only job is to narrate, for a human reviewer,
anything genuinely notable that actually happened in THIS SPECIFIC debate - never to invent,
infer, or speculate about anything not explicitly present in the transcript given to you.

CATEGORY DEFINITIONS - apply these precisely and consistently. Use exactly these category
strings when they apply; only invent a different category name for something that doesn't
fit any of these.

- "position_change": use ONLY when a model's `value` field literally differs between its
  round 1 message and its round 2 message (compare the two `value` strings you are given
  directly, character by character in meaning, not just style). If `value` is identical
  across both rounds, this is NOT a position_change, no matter how much the claim /
  tradeoff_accepted / condition_to_change_position text evolved - use "tradeoff_shift"
  instead (below).

- "tradeoff_shift": use when a model's `value` stayed the SAME between rounds, but its
  claim / tradeoff_accepted / condition_to_change_position text meaningfully changed in a
  way worth noting - most commonly, conceding a specific risk or flaw in round 2 that it
  downplayed, omitted, or didn't mention in round 1, while still keeping the same value.
  Never use this category if `value` actually changed between rounds - use
  "position_change" instead. These two categories are mutually exclusive per model per
  decision: check the `value` field first, and that alone decides which of the two applies.

- "specific_flaw_identified": a specific, technically-grounded critique of the opponent's
  design appearing in round 2 is the NORM for how these two models argue - it is not itself
  notable, and the vast majority of disputed decisions will contain one. A well-argued
  mutual disagreement where both sides substantively engage each other's points is the
  EXPECTED, UNREMARKABLE case. Do NOT log it just because a real technical argument exists
  on both sides - that is normal, not notable, and logging every instance defeats the
  purpose of a notable-moments log.
  Only log "specific_flaw_identified" when AT LEAST ONE of the following holds:
  (a) The challenged model's OWN ROUND 2 response fails to address the specific point at
      all - not just disagrees with it, actually doesn't engage it. IMPORTANT: both
      models' round 2 messages are generated without seeing each other's round 2 text,
      only each other's round 1. So this condition only applies to a critique first raised
      in ROUND 1: check whether the challenged model's round 2 response engages with that
      specific round-1 point. A critique raised for the first time in round 2 cannot
      meaningfully be "unaddressed," since the other side's round 2 was written without
      ever seeing it - do not apply condition (a) to a round-2-only critique.
  (b) The outcome (the arbiter's citation-based resolution, or the tie) would look
      different or surprising to a human reviewer if they didn't know about this specific
      flaw - i.e. the flaw is decisive to understanding why the decision came out the way
      it did, not merely present somewhere in the transcript.
  (c) The flaw directly and visibly caused a position change in the SAME round it was
      raised (in this case also check whether that same-model change is a "position_change"
      or "tradeoff_shift" per the rule above, and prefer logging that single entry with the
      causal link stated in its description, rather than two separate entries for one event).
  If none of (a), (b), (c) hold, do not log a specific_flaw_identified entry even if the
  critique itself is well-written and specific.

- "citation_relevance_flag": a model's citation already flagged "unsupported" by the
  automated citation check below. Only log this if there is something specific worth
  narrating about it beyond just restating the flag (e.g. what the claim actually argued
  instead of engaging the cited requirement).

Beyond these four, you may use a different category name for some other concrete, specific
event a human reviewer would clearly want highlighted - but the bar for inventing a new
category is the same as for specific_flaw_identified: normal debate back-and-forth is not
enough, it must be something that stands out.

STRICT RULES - read carefully, these are the most important part of your job:
- Only report what is literally present in the transcript text given to you. Never infer
  motive, never speculate about what a model "really meant," never invent a flaw or event
  that isn't explicitly stated in the claim/tradeoff_accepted/condition_to_change_position
  text you were given.
- Most debates are unremarkable by the definitions above. Outputting zero entries is the
  CORRECT and EXPECTED response for most decisions, and should be the majority of your
  responses across a batch. Do NOT manufacture an entry just to have something to report.
- If you report a position_change, the "description" field MUST be self-documenting: state
  explicitly, in the text itself, which model changed, the exact original value, the exact
  new value, and that this happened between round 1 and round 2. A reader must be able to
  understand the entry without cross-referencing the transcript.
- "detected_by" is either the model name responsible for the observation (the model who
  raised the flaw, or the model whose position/tradeoff changed), or exactly the string
  "citation_check" if the entry is surfacing an already-computed citation-relevance flag
  rather than something you noticed from the debate narrative itself.
- "challenged_model" is the OTHER model being critiqued, only if this entry is adversarial
  (one model's claim vs. another's design). Set it to null if the entry describes a model's
  own behavior (e.g. its own position_change, tradeoff_shift, or citation_relevance_flag
  about its own citation) with no specific opponent being called out. A model's own citation
  being flagged unsupported is a fact about that model's citation, not an opponent's
  critique of it - it does not by itself make an entry adversarial.
- "resolution" is a short phrase stating what actually happened as a result, only if that is
  knowable from the data given (e.g. "position held through round 2", "model revised its
  value in round 2 and that became the arbiter's winning value", "decision was tied and
  required human review") - never invent a resolution that isn't supported by the given
  final status/winner.
- Output strictly as JSON matching this shape, and nothing else:
  {"entries": [{"category": "...", "detected_by": "...", "challenged_model": "..." or null,
  "description": "...", "resolution": "..."}]}
  An empty entries list, {"entries": []}, is a normal, common, and often correct response.
"""


def _format_round(messages: List[Dict]) -> str:
    return "\n".join(
        f"- {m['model']}: value={m['value']!r}, claim={m['claim']!r}, "
        f"cited_requirement={m['cited_requirement']!r}, "
        f"tradeoff_accepted={m['tradeoff_accepted']!r}, "
        f"condition_to_change_position={m['condition_to_change_position']!r}"
        for m in messages
    )


def build_error_ledger_prompt(
    decision_id: str,
    dimension: str,
    transcript: Dict,
    arbiter_result: Optional[Dict],
) -> str:
    scores = (arbiter_result or {}).get("scores") or {}
    if scores:
        citation_lines = "\n".join(
            f"- {model_name}: citation_points={s.get('citation_points')}, "
            f"citation_relevance={s.get('citation_relevance')}"
            for model_name, s in scores.items()
        )
    else:
        citation_lines = "(no scoring occurred - round 2 converged before arbitration)"

    status = (arbiter_result or {}).get("status", "unknown")
    winner = (arbiter_result or {}).get("winner")
    outcome_line = f"FINAL STATUS: {status}" + (f", winner={winner}" if winner else "")

    return f"""DECISION: {decision_id} ({dimension})

ROUND 1:
{_format_round(transcript["round_1"])}

ROUND 2 (final):
{_format_round(transcript["round_2"])}

AUTOMATED CITATION-RELEVANCE FLAGS (already computed by deterministic code, informational only):
{citation_lines}

{outcome_line}

Extract 0-N notable-moment ledger entries per your instructions. Remember: zero entries is
the expected response unless something genuinely specific and notable is present above."""


async def run_error_ledger_extraction(
    decision_id: str,
    dimension: str,
    transcript: Dict,
    arbiter_result: Optional[Dict],
) -> Tuple[List[LedgerEntryItem], int, int]:
    client = AnthropicClient(model=settings.claude_model, api_key=settings.anthropic_api_key)
    prompt = build_error_ledger_prompt(decision_id, dimension, transcript, arbiter_result)

    last_error = None
    for attempt in range(settings.max_json_retries + 1):
        this_prompt = prompt
        if attempt > 0:
            this_prompt += (
                f"\n\nYour previous response was invalid: {last_error}\n"
                "Return ONLY valid JSON matching the required shape."
            )
        raw_text, prompt_tokens, completion_tokens = await client.complete_json(
            ERROR_LEDGER_SYSTEM_PROMPT, this_prompt
        )
        try:
            data = json.loads(raw_text)
            parsed = LedgerExtractionResponse.model_validate(data)
            return parsed.entries, prompt_tokens, completion_tokens
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)

    raise ValueError(f"Error-ledger extraction for decision '{decision_id}' failed after retry: {last_error}")


async def backfill_error_ledger(task_id: str, db: Session, force: bool = False) -> Dict:
    """Runs the LLM-narrated notable-moments extraction for every disputed
    decision (has a debate transcript) in one task. This is the on-demand,
    separate step referenced in models.ErrorLedgerEntry's docstring - it
    never runs as part of run_pipeline/task submission, since it costs real
    tokens and isn't needed for the pipeline to function.

    Idempotent by default: a decision that already has ledger entries is
    skipped, so re-running the backfill doesn't double-spend or duplicate
    rows. Pass force=True to delete and regenerate entries for decisions
    that already have them.

    Never touches DecisionResult, status, or scoring - purely additive.
    """
    from ..routers.tasks import build_task_detail  # deferred: avoid import cycle at module load

    detail = build_task_detail(task_id, db)
    if detail is None:
        raise ValueError(f"task '{task_id}' not found")

    summary = {
        "decisions_processed": 0,
        "decisions_skipped_existing": 0,
        "decisions_failed": 0,
        "entries_created": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    for d in detail["decisions"]:
        if not d.get("debate"):
            continue

        existing = (
            db.query(models.ErrorLedgerEntry)
            .filter_by(task_id=task_id, decision_id=d["decision_id"])
            .first()
        )
        if existing and not force:
            summary["decisions_skipped_existing"] += 1
            continue
        if existing and force:
            db.query(models.ErrorLedgerEntry).filter_by(
                task_id=task_id, decision_id=d["decision_id"]
            ).delete()

        try:
            entries, prompt_tokens, completion_tokens = await run_error_ledger_extraction(
                decision_id=d["decision_id"],
                dimension=d["dimension"],
                transcript=d["debate"],
                arbiter_result=d.get("arbiter"),
            )
        except Exception as e:
            # Lowest-stakes call in the pipeline - purely narrative, never
            # scores anything. One decision's extraction failing (after
            # transport retries are exhausted) must not stop the backfill
            # from processing every other decision in the task.
            logger.warning(
                "Error-ledger extraction failed for task %s decision '%s': %s", task_id, d["decision_id"], e
            )
            summary["decisions_failed"] += 1
            continue

        summary["decisions_processed"] += 1
        summary["prompt_tokens"] += prompt_tokens
        summary["completion_tokens"] += completion_tokens

        for entry in entries:
            db.add(
                models.ErrorLedgerEntry(
                    task_id=task_id,
                    decision_id=d["decision_id"],
                    category=entry.category,
                    detected_by=entry.detected_by,
                    challenged_model=entry.challenged_model,
                    description=entry.description,
                    resolution=entry.resolution,
                    status="informational",
                )
            )
            summary["entries_created"] += 1
        db.commit()

    return summary
