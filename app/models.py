import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from .db import Base


def gen_id() -> str:
    return str(uuid.uuid4())


class Project(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=gen_id)
    name = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    tasks = relationship("Task", back_populates="project")


class Task(Base):
    __tablename__ = "tasks"

    id = Column(String, primary_key=True, default=gen_id)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    description = Column(Text, nullable=False)
    context_text = Column(Text, default="")
    # Tokens for the single Claude task-framing call (Section 6 Phase 3 needs
    # this counted alongside proposer/debate/synthesis tokens for the
    # per-task cost display - there's exactly one framing call per task, so
    # it lives on Task rather than a separate per-call table.
    framing_prompt_tokens = Column(Integer, default=0)
    framing_completion_tokens = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="tasks")
    requirements = relationship("Requirement", back_populates="task")
    framing_decisions = relationship("FramingDecision", back_populates="task")
    model_responses = relationship("ModelResponse", back_populates="task")
    decision_results = relationship("DecisionResult", back_populates="task")


class Requirement(Base):
    __tablename__ = "requirements"

    id = Column(String, primary_key=True, default=gen_id)
    task_id = Column(String, ForeignKey("tasks.id"), nullable=False)
    req_id = Column(String, nullable=False)  # e.g. "req_1", stable within a task
    text = Column(Text, nullable=False)

    task = relationship("Task", back_populates="requirements")


class FramingDecision(Base):
    """The fixed decision_id/dimension vocabulary Claude generates before either
    proposer runs, so both proposers answer using identical decision_ids and the
    diff step can actually compare like-for-like (CLAUDE.md Section 4 resolution #3).
    Never carries a value, confidence, or reasoning - naming a decision is not
    proposing an answer to it.
    """

    __tablename__ = "framing_decisions"

    id = Column(String, primary_key=True, default=gen_id)
    task_id = Column(String, ForeignKey("tasks.id"), nullable=False)
    decision_id = Column(String, nullable=False)
    dimension = Column(String, nullable=False)
    description = Column(Text, default="")
    value_options = Column(JSON, default=list)

    task = relationship("Task", back_populates="framing_decisions")


class ModelResponse(Base):
    __tablename__ = "model_responses"

    id = Column(String, primary_key=True, default=gen_id)
    task_id = Column(String, ForeignKey("tasks.id"), nullable=False)
    model_name = Column(String, nullable=False)
    raw_response = Column(JSON, nullable=False)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    retry_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    task = relationship("Task", back_populates="model_responses")
    decisions = relationship("ProposedDecision", back_populates="model_response")


class ProposedDecision(Base):
    __tablename__ = "proposed_decisions"

    id = Column(String, primary_key=True, default=gen_id)
    model_response_id = Column(String, ForeignKey("model_responses.id"), nullable=False)
    task_id = Column(String, ForeignKey("tasks.id"), nullable=False)
    model_name = Column(String, nullable=False)
    decision_id = Column(String, nullable=False)
    dimension = Column(String, nullable=False)
    value = Column(Text, nullable=False)
    confidence = Column(Float, default=0.0)
    reasoning = Column(Text, default="")
    cited_requirements = Column(JSON, default=list)
    assumptions = Column(JSON, default=list)
    proposer_added = Column(Boolean, default=False)

    model_response = relationship("ModelResponse", back_populates="decisions")


class DebateTurn(Base):
    """One model's message in one round of the Section 5 targeted debate for
    one disputed decision. Resolution (arbiter scoring) is not stored here -
    that lands in DecisionResult once the arbiter step exists.
    """

    __tablename__ = "debate_turns"

    id = Column(String, primary_key=True, default=gen_id)
    task_id = Column(String, ForeignKey("tasks.id"), nullable=False)
    decision_id = Column(String, nullable=False)
    round = Column(Integer, nullable=False)
    model_name = Column(String, nullable=False)
    value = Column(Text, nullable=False)
    claim = Column(Text, default="")
    cited_requirement = Column(String, nullable=True)
    tradeoff_accepted = Column(Text, default="")
    condition_to_change_position = Column(Text, default="")
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class DecisionResult(Base):
    """Output of the deterministic diff (CLAUDE.md Section 4), possibly
    replaced by the debate/arbiter outcome (Section 5) when the diff status
    was disputed. status is one of: consensus | auto_accepted | unopposed
    (never resolved by debate) | resolved_via_debate (round 2 converged, so
    the arbiter's scoring rubric never ran - see arbiter.score_decision) |
    arbiter_resolved (rubric ran, produced a winner via a valid round-2
    requirement citation) | human_decision_required (round 2 still
    disagreed and no citation-backed winner resulted, a genuine tie) |
    debate_failed (a model call for this decision's debate failed a
    transport-level retry - timeout, connection error, 429, 5xx, or a
    non-retryable error like a 402 insufficient-balance response - and never
    produced a complete round-2 transcript; see pipeline.debate and
    pipeline.orchestrator's failure-handling, Phase 4 item 2). This is
    distinct from human_decision_required: a tie means the pipeline worked
    correctly and a human needs to make a call; debate_failed means a
    technical failure happened and needs operational attention, not a
    product decision. winning_value is always null for this status, and
    values_by_model/coupled_with/agreement_score still reflect the initial
    diff, which completed fine before the debate failure - only the debate
    itself, and everything downstream of it (scoring, synthesis), is
    missing, with whatever partial round-1/round-2 transcript data existed
    before the failure preserved rather than discarded.

    There used to be a fifth status, resolved_on_technicality, for a winner
    produced entirely by a keyword-overlap contradiction penalty rather than
    a citation difference. CLAUDE.md Section 5 removed that penalty (and the
    status and resolution_basis field along with it) after a 12-task
    validation batch found it had a 0% genuine-catch rate.
    """

    __tablename__ = "decision_results"

    id = Column(String, primary_key=True, default=gen_id)
    task_id = Column(String, ForeignKey("tasks.id"), nullable=False)
    decision_id = Column(String, nullable=False)
    dimension = Column(String, nullable=False)
    status = Column(String, nullable=False)
    winning_value = Column(Text, nullable=True)
    agreement_score = Column(Float, nullable=True)
    values_by_model = Column(JSON, default=dict)
    coupled_with = Column(JSON, default=list)
    arbiter_result = Column(JSON, nullable=True)
    synthesis = Column(Text, nullable=True)
    synthesis_prompt_tokens = Column(Integer, default=0)
    synthesis_completion_tokens = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    task = relationship("Task", back_populates="decision_results")


class ErrorLedgerEntry(Base):
    """Phase 4 item 1: a human-readable narrative log of notable moments in
    a debate transcript - never a scoring or arbitration mechanism, and
    never populated by keyword/deterministic detection. Populated entirely
    by one LLM call per disputed decision (see pipeline.error_ledger),
    explicitly instructed to report only what's literally present in the
    transcript and to output nothing for an unremarkable debate.

    status is always "informational" - a constant, not a state machine -
    so a row here can never be mistaken for an arbiter verdict. This is a
    log for a human to read, not a decision record.

    detected_by is either a model name (the model whose behavior grounds
    the entry - the one who raised a flaw, or the one whose position
    changed) or the literal string "citation_check" when the entry surfaces
    an already-computed citation_relevance flag (arbiter.py) rather than
    something noticed purely from the transcript narrative.

    challenged_model is nullable: only set when the entry is adversarial
    (one model's claim specifically critiqued in relation to another's
    design); null when the entry describes a model's own behavior (e.g. a
    position change) with no specific opponent being called out.
    """

    __tablename__ = "error_ledger_entries"

    id = Column(String, primary_key=True, default=gen_id)
    task_id = Column(String, ForeignKey("tasks.id"), nullable=False)
    decision_id = Column(String, nullable=False)
    category = Column(String, nullable=False)
    detected_by = Column(String, nullable=False)
    challenged_model = Column(String, nullable=True)
    description = Column(Text, nullable=False)
    resolution = Column(Text, nullable=False)
    status = Column(String, nullable=False, default="informational")
    created_at = Column(DateTime, default=datetime.utcnow)


class EvidenceVerification(Base):
    """Phase 4 item 3: one human-triggered, web_search-backed verdict on one
    specific piece of claim text (app/pipeline/evidence_verification.py).

    Unlike the error ledger, this NEVER runs as a batch backfill - the
    cost/variance profile (measured at $0.07-$0.20+ per call, non-
    deterministic even for the identical claim run twice) argues against
    ever running it unattended. Every row here exists because a human
    clicked "Verify this claim" on one specific field's exact text, after
    seeing the cost range up front - never automatic, never batched.

    source_field identifies WHERE in the decision's data this claim text
    came from (e.g. "proposal_reasoning:gpt", "debate_round2_claim:deepseek",
    "proposal_assumption:deepseek:0") - stable enough to look up "has this
    exact field already been verified" without re-storing the full claim
    text as a lookup key. claim_text is stored anyway, verbatim, so the
    record is self-contained even if the source data later changes.

    Non-scoring, exactly like citation_relevance and the error ledger: this
    never touches DecisionResult.status/winning_value/arbiter_result.
    """

    __tablename__ = "evidence_verifications"

    id = Column(String, primary_key=True, default=gen_id)
    task_id = Column(String, ForeignKey("tasks.id"), nullable=False)
    decision_id = Column(String, nullable=False)
    source_field = Column(String, nullable=False)
    claim_text = Column(Text, nullable=False)
    verdict = Column(String, nullable=False)
    caveats = Column(Text, default="")
    rationale = Column(Text, nullable=False)
    sources = Column(JSON, default=list)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    web_search_requests = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class HumanDecision(Base):
    """A human's own recorded answer for a decision the system didn't settle
    with confidence (human_decision_required - a genuine tie - or
    debate_failed - no system answer at all because a model call broke).
    This exists purely so the project can later ask "does the citation-only
    rubric's near-misses correlate with what a human actually picks" - it is
    NEVER read by the pipeline and NEVER changes DecisionResult.status or
    winning_value. The system's output and the human's recorded choice are
    kept fully separate so the two can be compared later; overwriting one
    with the other would destroy the thing this table exists to preserve.

    One row per (task_id, decision_id) - a re-submission overwrites in place
    (same idempotent-by-natural-key posture as EvidenceVerification, except
    EvidenceVerification refuses a second write entirely while this one is
    expected to be revised, so version/updated_at exist to show it changed).

    chosen_model is derived, not entered directly: the endpoint compares
    chosen_value against the decision's current per-model values (the same
    round-2-aware values the board itself shows as "current") and records
    which model it matches, or null if the human's wording matches neither
    verbatim - asking the human to separately self-report which model they
    agree with would be redundant with chosen_value and could disagree with
    it.
    """

    __tablename__ = "human_decisions"

    id = Column(String, primary_key=True, default=gen_id)
    task_id = Column(String, ForeignKey("tasks.id"), nullable=False)
    decision_id = Column(String, nullable=False)
    chosen_value = Column(Text, nullable=False)
    chosen_model = Column(String, nullable=True)
    rationale = Column(Text, default="")
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
