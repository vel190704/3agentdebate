from typing import Dict

from sqlalchemy import func
from sqlalchemy.orm import Session

from . import models


def compute_token_summary(task_id: str, db: Session) -> Dict:
    """Sums real, persisted token counts per pipeline phase for one task -
    no estimation, just adding up what framing.py/proposers.py/debate.py/
    synthesis.py already recorded on Task/ModelResponse/DebateTurn/
    DecisionResult. Phase 3's cost counter reads this directly.
    """
    task = db.query(models.Task).filter_by(id=task_id).first()
    framing_prompt = task.framing_prompt_tokens if task else 0
    framing_completion = task.framing_completion_tokens if task else 0

    proposer_prompt, proposer_completion = db.query(
        func.coalesce(func.sum(models.ModelResponse.prompt_tokens), 0),
        func.coalesce(func.sum(models.ModelResponse.completion_tokens), 0),
    ).filter(models.ModelResponse.task_id == task_id).one()

    debate_prompt, debate_completion = db.query(
        func.coalesce(func.sum(models.DebateTurn.prompt_tokens), 0),
        func.coalesce(func.sum(models.DebateTurn.completion_tokens), 0),
    ).filter(models.DebateTurn.task_id == task_id).one()

    synthesis_prompt, synthesis_completion = db.query(
        func.coalesce(func.sum(models.DecisionResult.synthesis_prompt_tokens), 0),
        func.coalesce(func.sum(models.DecisionResult.synthesis_completion_tokens), 0),
    ).filter(models.DecisionResult.task_id == task_id).one()

    phases = {
        "framing": {"prompt": framing_prompt, "completion": framing_completion},
        "proposers": {"prompt": proposer_prompt, "completion": proposer_completion},
        "debate": {"prompt": debate_prompt, "completion": debate_completion},
        "arbitration_synthesis": {"prompt": synthesis_prompt, "completion": synthesis_completion},
    }
    for phase in phases.values():
        phase["total"] = phase["prompt"] + phase["completion"]

    return {"phases": phases, "total": sum(p["total"] for p in phases.values())}
