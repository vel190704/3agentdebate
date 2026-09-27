import json
import os
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from .. import models
from ..board_display import decision_badge
from ..db import get_db
from ..pipeline.evidence_verification import ESTIMATED_COST_RANGE, estimate_cost, verify_claim
from ..token_summary import compute_token_summary
from .tasks import build_task_detail, create_task

router = APIRouter(prefix="/board", tags=["board"])

_TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
templates = Jinja2Templates(directory=_TEMPLATES_DIR)


@router.get("")
def list_tasks(request: Request, db: Session = Depends(get_db)):
    tasks = db.query(models.Task).order_by(models.Task.created_at.desc()).all()
    return templates.TemplateResponse(request, "board_list.html", {"tasks": tasks})


@router.post("/submit")
async def submit_task(
    project_name: str = Form(...),
    task_description: str = Form(...),
    context_text: str = Form(""),
    requirements_text: str = Form(""),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
):
    """Form-friendly wrapper around the JSON /tasks/ endpoint: takes
    newline-separated requirements from a textarea instead of a JSON array,
    runs the same synchronous pipeline (this is the request that blocks for
    real model calls - the "before running" state is just this page load
    taking ~30-60s), then redirects into the board for the new task.
    """
    requirement_lines = [line.strip() for line in requirements_text.splitlines() if line.strip()]
    result = await create_task(
        project_name=project_name,
        task_description=task_description,
        context_text=context_text,
        requirements=json.dumps(requirement_lines),
        file=file,
        db=db,
    )
    return RedirectResponse(url=f"/board/{result['task_id']}", status_code=303)


@router.post("/{task_id}/verify-claim")
async def verify_claim_endpoint(
    task_id: str,
    decision_id: str = Form(...),
    source_field: str = Form(...),
    claim_text: str = Form(...),
    db: Session = Depends(get_db),
):
    """One explicit human click, per claim, per field - never automatic,
    never batched (see EvidenceVerification's docstring). Idempotent in the
    same sense as the ledger's per-decision skip: if source_field already
    has a stored result, don't spend again, just redirect back to the
    already-rendered cached result.
    """
    existing = (
        db.query(models.EvidenceVerification)
        .filter_by(task_id=task_id, decision_id=decision_id, source_field=source_field)
        .first()
    )
    if existing is None:
        task = db.query(models.Task).filter_by(id=task_id).first()
        if task is None:
            raise HTTPException(404, "task not found")
        decision = db.query(models.DecisionResult).filter_by(task_id=task_id, decision_id=decision_id).first()
        dimension = decision.dimension if decision else decision_id
        context = f"Task: {task.description}\nDecision being discussed: {dimension}"

        verdict, input_tokens, output_tokens, web_search_requests = await verify_claim(claim_text, context)
        cost_usd = estimate_cost(input_tokens, output_tokens, web_search_requests)

        db.add(
            models.EvidenceVerification(
                task_id=task_id,
                decision_id=decision_id,
                source_field=source_field,
                claim_text=claim_text,
                verdict=verdict.verdict,
                caveats=verdict.caveats,
                rationale=verdict.rationale,
                sources=verdict.sources,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                web_search_requests=web_search_requests,
                cost_usd=cost_usd,
            )
        )
        db.commit()

    return RedirectResponse(url=f"/board/{task_id}#decision-{decision_id}", status_code=303)


@router.get("/{task_id}")
def board_detail(task_id: str, request: Request, db: Session = Depends(get_db)):
    detail = build_task_detail(task_id, db)
    if detail is None:
        raise HTTPException(404, "task not found")

    for decision in detail["decisions"]:
        decision["badge"] = decision_badge(decision)
        decision["coupled_count"] = len(decision.get("coupled_with") or [])
        # values_by_model is frozen at the initial-proposal stage (Section 4
        # diff); a decision that went to debate may have moved since then
        # (revisions logged in debate.position_changes), so once a debate
        # transcript exists, round 2 is the current position, not round 1.
        # A debate_failed decision has a debate dict (failed=True) but an
        # empty round_2 - fall back to the original proposal values rather
        # than showing an empty "values on the table" line.
        debate = decision.get("debate")
        round2_values = {m["model"]: m["value"] for m in debate["round_2"]} if debate else {}
        decision["current_values_by_model"] = round2_values or (decision.get("values_by_model") or {})

    token_summary = compute_token_summary(task_id, db)

    return templates.TemplateResponse(
        request,
        "board_detail.html",
        {
            "task": detail,
            "tokens": token_summary,
            "evidence_by_field": detail.get("evidence_by_field", {}),
            "evidence_cost_range": ESTIMATED_COST_RANGE,
        },
    )
