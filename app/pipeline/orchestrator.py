import asyncio
import logging
from typing import Dict, List

from sqlalchemy.orm import Session

from .. import models
from .arbiter import resolve_winning_value, score_decision
from .debate import run_debates
from .diff import diff_decisions
from .framing import run_framing
from .proposers import run_proposers
from .synthesis import run_synthesis

logger = logging.getLogger(__name__)


async def run_pipeline(db: Session, task: models.Task, requirements: List[Dict]) -> List[Dict]:
    framing, framing_prompt_tokens, framing_completion_tokens = await run_framing(
        task_id=task.id,
        task_description=task.description,
        context_text=task.context_text,
        requirements=requirements,
    )
    task.framing_prompt_tokens = framing_prompt_tokens
    task.framing_completion_tokens = framing_completion_tokens

    framing_rows: List[Dict] = []
    for item in framing.decisions:
        db.add(
            models.FramingDecision(
                task_id=task.id,
                decision_id=item.decision_id,
                dimension=item.dimension,
                description=item.description,
                value_options=item.value_options,
            )
        )
        framing_rows.append(
            {
                "decision_id": item.decision_id,
                "dimension": item.dimension,
                "description": item.description,
                "value_options": item.value_options,
            }
        )
    db.flush()

    proposer_results = await run_proposers(
        task_id=task.id,
        task_description=task.description,
        context_text=task.context_text,
        requirements=requirements,
        framing_decisions=framing_rows,
    )

    for cfg, (parsed, _raw_text, prompt_tokens, completion_tokens) in proposer_results:
        response_row = models.ModelResponse(
            task_id=task.id,
            model_name=cfg.name,
            raw_response=parsed.model_dump(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        db.add(response_row)
        db.flush()
        for item in parsed.decisions:
            db.add(
                models.ProposedDecision(
                    model_response_id=response_row.id,
                    task_id=task.id,
                    model_name=cfg.name,
                    decision_id=item.decision_id,
                    dimension=item.dimension,
                    value=item.value,
                    confidence=item.confidence,
                    reasoning=item.reasoning,
                    cited_requirements=item.cited_requirements,
                    assumptions=item.assumptions,
                    proposer_added=item.proposer_added,
                )
            )

    requirement_ids = {r["req_id"] for r in requirements}
    diff_results = diff_decisions(framing_rows, proposer_results, requirement_ids)

    # Targeted debate (CLAUDE.md Section 5) for DISPUTED decisions only - never
    # for consensus/auto_accepted/unopposed.
    disputed_results = [r for r in diff_results if r["status"] == "disputed"]
    transcripts_by_decision = await run_debates(
        disputed_results=disputed_results,
        framing_decisions=framing_rows,
        task_description=task.description,
        context_text=task.context_text,
        requirements=requirements,
    )

    value_options_by_id = {d["decision_id"]: d.get("value_options") or [] for d in framing_rows}
    description_by_id = {d["decision_id"]: d.get("description", "") for d in framing_rows}

    debated_results = []
    for r in diff_results:
        transcript = transcripts_by_decision.get(r["decision_id"])
        if transcript is None:
            continue
        r["debate"] = transcript

        if transcript.get("failed"):
            # A transport failure survived transport retries for this one
            # decision's debate (see debate.py). It's isolated - every other
            # decision's debate and the rest of the pipeline still complete
            # normally - but this decision has no complete round-2 data for
            # score_decision to work with, so it gets its own status instead
            # of crashing the task or silently reporting a fake result.
            # values_by_model/coupled_with/agreement_score from the initial
            # diff are untouched and still shown - that part of the pipeline
            # succeeded fine.
            #
            # The failure/failed_round/error facts only exist in-memory on
            # `transcript` right now - nothing about a debate_failed decision
            # is stored in DebateTurn rows (there may be none at all, if
            # round 1 itself never completed), so without persisting this
            # somewhere, a board/API read *after* this request finishes
            # would show the badge but nothing about why or where it failed.
            # arbiter_result is otherwise unused for this status (there was
            # no score), so it's reused here rather than adding a new column.
            r["status"] = "debate_failed"
            r["winning_value"] = None
            r["arbiter"] = {
                "failed": True,
                "failed_round": transcript.get("failed_round"),
                "error": transcript.get("error"),
            }
            continue

        # Deterministic scoring (CLAUDE.md Section 5, corrected rubric) - pure
        # code, no LLM call. Replaces "disputed" with either a scored winner
        # or human_decision_required; never an arbitrary pick.
        arbiter_result = score_decision(
            decision_id=r["decision_id"],
            requirements=requirements,
            requirement_ids=requirement_ids,
            proposal_details_by_model=r["details_by_model"],
            transcript=transcript,
            value_options=value_options_by_id.get(r["decision_id"], []),
        )
        r["arbiter"] = arbiter_result
        r["status"] = arbiter_result["status"]
        r["winning_value"] = resolve_winning_value(
            arbiter_result, value_options_by_id.get(r["decision_id"], [])
        )
        debated_results.append(r)

    # One Claude synthesis call per disputed decision that was actually
    # scored, run in parallel across decisions - for BOTH arbiter_resolved
    # and human_decision_required. A tied score with no explanation is
    # exactly the case a human reviewer needs this most, so it is never
    # skipped for that outcome. resolved_via_debate decisions are excluded:
    # synthesis explains a score (see synthesis.py's module docstring), and
    # convergence never produced one - same reason a plain consensus from
    # the initial diff never gets a synthesis call either. debate_failed
    # decisions never reach debated_results at all (see the `continue`
    # above), so they're already excluded here too - there's no score to
    # explain when the debate itself never completed.
    synthesis_eligible = [r for r in debated_results if r["status"] != "resolved_via_debate"]
    if synthesis_eligible:
        # return_exceptions=True: synthesis is purely explanatory prose over
        # an already-final, already-correct status/winning_value/arbiter -
        # a synthesis failure has nothing to do with whether the decision
        # itself resolved correctly, so it must never take the rest of the
        # task down with it. Degrades to a null synthesis field, logged, not
        # raised.
        synthesis_results = await asyncio.gather(
            *[
                run_synthesis(
                    decision_id=r["decision_id"],
                    dimension=r["dimension"],
                    description=description_by_id.get(r["decision_id"], ""),
                    transcript=r["debate"],
                    arbiter_result=r["arbiter"],
                )
                for r in synthesis_eligible
            ],
            return_exceptions=True,
        )
        for r, result in zip(synthesis_eligible, synthesis_results):
            if isinstance(result, Exception):
                logger.warning(
                    "Synthesis failed for task %s decision '%s': %s", task.id, r["decision_id"], result
                )
                r["synthesis"] = None
                r["synthesis_prompt_tokens"] = 0
                r["synthesis_completion_tokens"] = 0
            else:
                text, prompt_tokens, completion_tokens = result
                r["synthesis"] = text
                r["synthesis_prompt_tokens"] = prompt_tokens
                r["synthesis_completion_tokens"] = completion_tokens

    for r in diff_results:
        db.add(
            models.DecisionResult(
                task_id=task.id,
                decision_id=r["decision_id"],
                dimension=r["dimension"],
                status=r["status"],
                winning_value=r.get("winning_value"),
                agreement_score=r.get("agreement_score"),
                values_by_model=r.get("values_by_model", {}),
                coupled_with=r.get("coupled_with", []),
                arbiter_result=r.get("arbiter"),
                synthesis=r.get("synthesis"),
                synthesis_prompt_tokens=r.get("synthesis_prompt_tokens", 0),
                synthesis_completion_tokens=r.get("synthesis_completion_tokens", 0),
            )
        )

        transcript = transcripts_by_decision.get(r["decision_id"])
        if transcript is not None:
            for round_key, round_num in (("round_1", 1), ("round_2", 2)):
                for msg in transcript[round_key]:
                    db.add(
                        models.DebateTurn(
                            task_id=task.id,
                            decision_id=r["decision_id"],
                            round=round_num,
                            model_name=msg["model"],
                            value=msg["value"],
                            claim=msg["claim"],
                            cited_requirement=msg["cited_requirement"],
                            tradeoff_accepted=msg["tradeoff_accepted"],
                            condition_to_change_position=msg["condition_to_change_position"],
                            prompt_tokens=msg.get("prompt_tokens", 0),
                            completion_tokens=msg.get("completion_tokens", 0),
                        )
                    )

    db.commit()

    return diff_results
