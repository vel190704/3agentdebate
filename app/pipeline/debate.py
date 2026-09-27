import asyncio
import json
from typing import Dict, List

from pydantic import ValidationError

from ..config import ProposerConfig, settings
from ..llm.openai_compatible_client import OpenAICompatibleClient
from ..schemas import DebateMessage

DEBATE_SYSTEM_PROMPT = """You are one of the models participating in a targeted debate over a
single disputed decision in a multi-LLM deliberation system. Only models who disagreed on this
decision participate - you are not seeing or debating any other decision right now.

Respond ONLY with JSON matching this shape:
{
  "round": 1,
  "model": "your model name",
  "decision_id": "the decision_id you were given",
  "value": "your current position - the exact option you are advocating for this round",
  "claim": "max 100 words - your argument for this value",
  "cited_requirement": "a requirement id from the shared context that supports your claim, or null",
  "tradeoff_accepted": "max 50 words - the downside of your own position that you concede",
  "condition_to_change_position": "max 50 words - what evidence or constraint would change your mind"
}

Do not restate the full task or context back - focus only on your argument for this decision.
"""


def _normalize(value: str) -> str:
    return value.strip().lower()


async def _call_one_debate_turn(cfg: ProposerConfig, prompt: str, round_num: int) -> DebateMessage:
    client = OpenAICompatibleClient(model=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url)

    last_error = None
    for attempt in range(settings.max_json_retries + 1):
        this_prompt = prompt
        if attempt > 0:
            this_prompt += (
                f"\n\nYour previous response was invalid: {last_error}\n"
                "Return ONLY valid JSON matching the required shape."
            )
        raw_text, prompt_tokens, completion_tokens = await client.complete_json(DEBATE_SYSTEM_PROMPT, this_prompt)
        try:
            data = json.loads(raw_text)
            message = DebateMessage.model_validate(data)
            # Identity is tracked by our own config name (cfg.name), never by
            # what the model wrote into "model" - same rule diff.py/proposers.py
            # already follow for the proposal schema's `model` field. Overriding
            # here keeps round1_by_model/round2 lookups and cross-decision keys
            # (values_by_model, details_by_model) all consistent.
            message.model = cfg.name
            message.prompt_tokens = prompt_tokens
            message.completion_tokens = completion_tokens
            return message
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)

    raise ValueError(
        f"Proposer '{cfg.name}' ({cfg.model}) failed to produce a valid round {round_num} "
        f"debate message after retry: {last_error}"
    )


def _format_own_position(detail: Dict) -> str:
    return (
        f"Your original position: {detail['value']}\n"
        f"Your original reasoning: {detail['reasoning']}\n"
        f"Your original assumptions: {', '.join(detail['assumptions']) or '(none stated)'}\n"
        f"Requirements you cited: {', '.join(detail['cited_requirements']) or '(none)'}"
    )


def _format_opponent_positions(cfg_name: str, participant_details: Dict[str, Dict]) -> str:
    lines = []
    for model_name, detail in participant_details.items():
        if model_name == cfg_name:
            continue
        lines.append(
            f"- {model_name} proposed: {detail['value']}\n"
            f"  reasoning: {detail['reasoning']}\n"
            f"  cited requirements: {', '.join(detail['cited_requirements']) or '(none)'}"
        )
    return "\n".join(lines)


def _build_round1_prompt(
    cfg: ProposerConfig,
    decision_id: str,
    dimension: str,
    description: str,
    task_description: str,
    context_text: str,
    requirements: List[Dict],
    participant_details: Dict[str, Dict],
) -> str:
    req_lines = "\n".join(f"- {r['req_id']}: {r['text']}" for r in requirements) or "(none provided)"
    return f"""ROUND: 1
decision_id: {decision_id} ({dimension})
DECISION DESCRIPTION: {description}

TASK:
{task_description}

SHARED CONTEXT:
{context_text or "(none provided)"}

REQUIREMENTS:
{req_lines}

{_format_own_position(participant_details[cfg.name])}

OTHER MODEL(S) PROPOSED FOR THIS SAME DECISION:
{_format_opponent_positions(cfg.name, participant_details)}

State your position for round 1 using the required JSON schema."""


def _build_round2_prompt(
    cfg: ProposerConfig,
    decision_id: str,
    dimension: str,
    description: str,
    task_description: str,
    context_text: str,
    requirements: List[Dict],
    participant_details: Dict[str, Dict],
    round1_by_model: Dict[str, DebateMessage],
) -> str:
    req_lines = "\n".join(f"- {r['req_id']}: {r['text']}" for r in requirements) or "(none provided)"

    own_r1 = round1_by_model[cfg.name]
    opponent_r1_lines = []
    for model_name, msg in round1_by_model.items():
        if model_name == cfg.name:
            continue
        opponent_r1_lines.append(
            f"- {model_name} (round 1): value={msg.value!r}, claim={msg.claim!r}, "
            f"cited_requirement={msg.cited_requirement!r}, "
            f"condition_to_change_position={msg.condition_to_change_position!r}"
        )
    opponent_r1 = "\n".join(opponent_r1_lines)

    return f"""ROUND: 2 (final round - no round 3 after this)
decision_id: {decision_id} ({dimension})
DECISION DESCRIPTION: {description}

TASK:
{task_description}

SHARED CONTEXT:
{context_text or "(none provided)"}

REQUIREMENTS:
{req_lines}

YOUR ROUND 1 MESSAGE:
value={own_r1.value!r}, claim={own_r1.claim!r}, cited_requirement={own_r1.cited_requirement!r},
condition_to_change_position={own_r1.condition_to_change_position!r}

OTHER MODEL(S) ROUND 1 MESSAGES:
{opponent_r1}

You may HOLD your position (set "value" to the exact same value as your round 1 message) or
REVISE it in light of the arguments above (set "value" to your new position and explain why in
"claim"). This is the final round - after this, an arbiter resolves the decision using only
what has been said in rounds 1 and 2. Respond with the required JSON schema for round 2."""


async def run_debate_for_decision(
    decision_id: str,
    dimension: str,
    description: str,
    task_description: str,
    context_text: str,
    requirements: List[Dict],
    participant_details: Dict[str, Dict],
) -> Dict:
    """Runs the fixed 2-round debate (CLAUDE.md Section 5) for one disputed
    decision. Only models present in participant_details (i.e. models that
    actually proposed a value for this decision_id) take part. Returns the
    full transcript - does NOT score or resolve anything; that's the
    arbiter's job, deliberately not implemented yet.

    Never raises: a decision's debate needs BOTH participants to complete
    BOTH rounds to produce anything score_decision can use (round 2's
    prompt is built from round 1's messages), so a call-site failure in
    either round marks the whole decision "failed" (see failed/failed_round
    below) rather than letting one broken call escape and, via the outer
    asyncio.gather in run_debates, cancel every OTHER decision's debate too
    - a real behavior that existed before this fix. Whatever round(s)
    genuinely completed before the failure are preserved, never discarded.
    """
    cfg_by_name = {cfg.name: cfg for cfg in settings.proposers}
    participants = [cfg_by_name[name] for name in participant_details if name in cfg_by_name]

    try:
        round1_messages = await asyncio.gather(
            *[
                _call_one_debate_turn(
                    cfg,
                    _build_round1_prompt(
                        cfg, decision_id, dimension, description, task_description, context_text,
                        requirements, participant_details,
                    ),
                    round_num=1,
                )
                for cfg in participants
            ]
        )
    except Exception as e:
        return {
            "decision_id": decision_id,
            "failed": True,
            "failed_round": 1,
            "error": str(e),
            "round_1": [],
            "round_2": [],
            "position_changes": [],
        }

    round1_by_model = {m.model: m for m in round1_messages}

    try:
        round2_messages = await asyncio.gather(
            *[
                _call_one_debate_turn(
                    cfg,
                    _build_round2_prompt(
                        cfg, decision_id, dimension, description, task_description, context_text,
                        requirements, participant_details, round1_by_model,
                    ),
                    round_num=2,
                )
                for cfg in participants
            ]
        )
    except Exception as e:
        # Round 1 genuinely completed - preserve it - but round 2 never did.
        return {
            "decision_id": decision_id,
            "failed": True,
            "failed_round": 2,
            "error": str(e),
            "round_1": [m.model_dump() for m in round1_messages],
            "round_2": [],
            "position_changes": [],
        }

    position_changes = []
    for m2 in round2_messages:
        m1 = round1_by_model.get(m2.model)
        if m1 is not None and _normalize(m1.value) != _normalize(m2.value):
            position_changes.append({"model": m2.model, "from": m1.value, "to": m2.value})

    return {
        "decision_id": decision_id,
        "failed": False,
        "round_1": [m.model_dump() for m in round1_messages],
        "round_2": [m.model_dump() for m in round2_messages],
        "position_changes": position_changes,
    }


async def run_debates(
    disputed_results: List[Dict],
    framing_decisions: List[Dict],
    task_description: str,
    context_text: str,
    requirements: List[Dict],
) -> Dict[str, Dict]:
    """Runs debate for every disputed decision from the diff, in parallel
    across decisions (each decision's own two rounds still run model-calls in
    parallel within themselves, via run_debate_for_decision).
    """
    if not disputed_results:
        return {}

    description_by_id = {d["decision_id"]: d.get("description", "") for d in framing_decisions}

    # return_exceptions=True: run_debate_for_decision itself never raises (it
    # catches its own failures and returns a failed=True dict instead), but
    # this is the boundary that actually determines whether one decision's
    # problem can cancel a sibling decision's in-flight debate via
    # asyncio.gather's default cancel-on-first-exception behavior. Kept as a
    # defensive second layer - if that per-decision catching ever has a gap,
    # this stops the gap from taking down every other decision too.
    transcripts = await asyncio.gather(
        *[
            run_debate_for_decision(
                decision_id=r["decision_id"],
                dimension=r["dimension"],
                description=description_by_id.get(r["decision_id"], ""),
                task_description=task_description,
                context_text=context_text,
                requirements=requirements,
                participant_details=r["details_by_model"],
            )
            for r in disputed_results
        ],
        return_exceptions=True,
    )

    result: Dict[str, Dict] = {}
    for r, t in zip(disputed_results, transcripts):
        if isinstance(t, Exception):
            result[r["decision_id"]] = {
                "decision_id": r["decision_id"],
                "failed": True,
                "failed_round": None,
                "error": str(t),
                "round_1": [],
                "round_2": [],
                "position_changes": [],
            }
        else:
            result[t["decision_id"]] = t
    return result
