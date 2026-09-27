import json
from typing import Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from .. import models
from ..db import get_db
from ..pdf_extract import extract_text_from_pdf
from ..pipeline.orchestrator import run_pipeline

router = APIRouter(prefix="/tasks", tags=["tasks"])


def build_task_detail(task_id: str, db: Session) -> Optional[Dict]:
    """Assembles the full per-decision board payload for one task - shared by
    the JSON API (get_task below) and the server-rendered board (routers/board.py)
    so there's exactly one place that knows how to join decisions, proposed
    values, and debate turns back together.
    """
    task = db.query(models.Task).filter_by(id=task_id).first()
    if task is None:
        return None

    decisions = db.query(models.DecisionResult).filter_by(task_id=task_id).all()

    proposed = db.query(models.ProposedDecision).filter_by(task_id=task_id).all()
    details_by_decision: dict = {}
    for p in proposed:
        details_by_decision.setdefault(p.decision_id, {})[p.model_name] = {
            "value": p.value,
            "confidence": p.confidence,
            "reasoning": p.reasoning,
            "cited_requirements": p.cited_requirements,
            "assumptions": p.assumptions,
            "proposer_added": p.proposer_added,
        }

    debate_turns = db.query(models.DebateTurn).filter_by(task_id=task_id).all()
    debate_by_decision: dict = {}
    for t in debate_turns:
        transcript = debate_by_decision.setdefault(
            t.decision_id, {"decision_id": t.decision_id, "round_1": [], "round_2": []}
        )
        transcript[f"round_{t.round}"].append(
            {
                "round": t.round,
                "model": t.model_name,
                "decision_id": t.decision_id,
                "value": t.value,
                "claim": t.claim,
                "cited_requirement": t.cited_requirement,
                "tradeoff_accepted": t.tradeoff_accepted,
                "condition_to_change_position": t.condition_to_change_position,
            }
        )
    for decision_id, transcript in debate_by_decision.items():
        round1_by_model = {m["model"]: m for m in transcript["round_1"]}
        position_changes = []
        for m2 in transcript["round_2"]:
            m1 = round1_by_model.get(m2["model"])
            if m1 is not None and m1["value"].strip().lower() != m2["value"].strip().lower():
                position_changes.append({"model": m2["model"], "from": m1["value"], "to": m2["value"]})
        transcript["position_changes"] = position_changes

    # A debate_failed decision may have zero DebateTurn rows at all (if
    # round 1 itself never completed - see debate.py), so the loop above
    # never even creates a debate_by_decision entry for it. Force one here,
    # and attach the failed/failed_round/error facts persisted onto
    # arbiter_result (see orchestrator.py) - without this, the board/API
    # would show the debate_failed badge but nothing about why or where.
    for d in decisions:
        if d.status != "debate_failed":
            continue
        transcript = debate_by_decision.setdefault(
            d.decision_id, {"decision_id": d.decision_id, "round_1": [], "round_2": [], "position_changes": []}
        )
        failure_info = d.arbiter_result or {}
        transcript["failed"] = True
        transcript["failed_round"] = failure_info.get("failed_round")
        transcript["error"] = failure_info.get("error")

    ledger_entries = db.query(models.ErrorLedgerEntry).filter_by(task_id=task_id).all()
    ledger_by_decision: dict = {}
    for e in ledger_entries:
        ledger_by_decision.setdefault(e.decision_id, []).append(
            {
                "category": e.category,
                "detected_by": e.detected_by,
                "challenged_model": e.challenged_model,
                "description": e.description,
                "resolution": e.resolution,
                "status": e.status,
            }
        )

    # Keyed by (decision_id, source_field) - one human-triggered verification
    # per field location, looked up so the board shows the cached result
    # instead of a "verify" button once a field has already been checked
    # (never re-verified automatically - see EvidenceVerification's docstring
    # on why this mechanism has no batch/backfill path at all).
    # String key "{decision_id}:{source_field}", not a tuple - Jinja can only
    # do dict lookups it can build from string concatenation in the template.
    evidence_rows = db.query(models.EvidenceVerification).filter_by(task_id=task_id).all()
    evidence_by_field: dict = {}
    for ev in evidence_rows:
        evidence_by_field[f"{ev.decision_id}:{ev.source_field}"] = {
            "verdict": ev.verdict,
            "caveats": ev.caveats,
            "rationale": ev.rationale,
            "sources": ev.sources,
            "cost_usd": ev.cost_usd,
        }

    return {
        "task_id": task.id,
        "project_id": task.project_id,
        "description": task.description,
        "context_text": task.context_text,
        "created_at": task.created_at,
        "decisions": [
            {
                "decision_id": d.decision_id,
                "dimension": d.dimension,
                "status": d.status,
                "winning_value": d.winning_value,
                "agreement_score": d.agreement_score,
                "values_by_model": d.values_by_model,
                "details_by_model": details_by_decision.get(d.decision_id, {}),
                "coupled_with": d.coupled_with,
                "debate": debate_by_decision.get(d.decision_id),
                "arbiter": d.arbiter_result,
                "synthesis": d.synthesis,
                "error_ledger": ledger_by_decision.get(d.decision_id, []),
            }
            for d in decisions
        ],
        "evidence_by_field": evidence_by_field,
    }


@router.post("/")
async def create_task(
    project_name: str = Form(...),
    task_description: str = Form(...),
    context_text: str = Form(""),
    requirements: str = Form("[]"),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
):
    """Submits shared context + a task, runs the full Phase 1 pipeline
    (task-framing -> parallel proposers -> deterministic diff) synchronously,
    and returns the per-decision board as JSON.
    """
    try:
        requirement_texts = json.loads(requirements)
        if not isinstance(requirement_texts, list):
            raise ValueError
    except ValueError:
        raise HTTPException(400, "requirements must be a JSON array of strings")

    pdf_text = ""
    # A browser's <input type="file"> left unselected still submits a real
    # (non-None) UploadFile in the multipart body - empty filename, zero
    # bytes - so `file is not None` alone is not "a file was attached."
    # Confirmed via a real 500: pypdf.PdfReader rejects a zero-byte stream
    # with EmptyFileError, uncaught, from the board's HTML form (no curl/API
    # test this project has run ever included a file field at all, so this
    # path was never exercised until a real browser hit it).
    if file is not None and file.filename:
        content = await file.read()
        if content:
            pdf_text = extract_text_from_pdf(content)

    full_context = "\n\n".join(t for t in [context_text, pdf_text] if t)

    project = db.query(models.Project).filter_by(name=project_name).first()
    if project is None:
        project = models.Project(name=project_name)
        db.add(project)
        db.flush()

    task = models.Task(project_id=project.id, description=task_description, context_text=full_context)
    db.add(task)
    db.flush()

    requirements_rows = []
    for i, text in enumerate(requirement_texts, start=1):
        req_id = f"req_{i}"
        db.add(models.Requirement(task_id=task.id, req_id=req_id, text=text))
        requirements_rows.append({"req_id": req_id, "text": text})
    db.flush()

    try:
        diff_results = await run_pipeline(db, task, requirements_rows)
    except ValueError as e:
        db.rollback()
        raise HTTPException(502, str(e))

    return {
        "task_id": task.id,
        "project": project.name,
        "decisions": diff_results,
    }


@router.get("/{task_id}")
def get_task(task_id: str, db: Session = Depends(get_db)):
    detail = build_task_detail(task_id, db)
    if detail is None:
        raise HTTPException(404, "task not found")
    return detail
