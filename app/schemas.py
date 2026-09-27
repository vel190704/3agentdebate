from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class FramingDecisionItem(BaseModel):
    decision_id: str
    dimension: str
    description: str = ""
    # Canonical answer set for decisions with a small, well-known set of named
    # options (e.g. message queue tech). Left empty for open-ended decisions.
    # Lets the diff step match "Amazon MSK (Kafka)" to "Kafka" without an LLM
    # call - see diff.py's containment fallback, scoped to this list only.
    value_options: List[str] = Field(default_factory=list)


class FramingResponse(BaseModel):
    """Output of the Claude task-framing step. Deliberately has no value,
    confidence, or reasoning fields - Claude is naming decisions here, not
    proposing answers to them (CLAUDE.md Section 2 / resolution #3).
    """

    task_id: str
    decisions: List[FramingDecisionItem]


class ProposedDecisionItem(BaseModel):
    """Matches the structured decision schema in CLAUDE.md Section 3."""

    decision_id: str
    dimension: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    cited_requirements: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    proposer_added: bool = False

    @field_validator("reasoning")
    @classmethod
    def _soft_trim_reasoning(cls, v: str) -> str:
        # "max ~150 words" is a soft prompt guideline, not a hard contract -
        # trim instead of rejecting a model's otherwise-valid response over it.
        words = v.split()
        if len(words) > 170:
            v = " ".join(words[:170])
        return v


class ProposerResponse(BaseModel):
    task_id: str
    model: str
    decisions: List[ProposedDecisionItem]


class DebateMessage(BaseModel):
    """CLAUDE.md Section 5 targeted-debate message. `value` is not in the
    literal schema shown there, but is required to detect a round-2 position
    change structurally instead of parsing free text out of `claim` - see
    the note flagged alongside this change.
    """

    round: Literal[1, 2]
    model: str
    decision_id: str
    value: str
    claim: str
    cited_requirement: Optional[str] = None
    tradeoff_accepted: str = ""
    condition_to_change_position: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @field_validator("claim")
    @classmethod
    def _soft_trim_claim(cls, v: str) -> str:
        words = v.split()
        return " ".join(words[:120]) if len(words) > 120 else v

    @field_validator("tradeoff_accepted", "condition_to_change_position")
    @classmethod
    def _soft_trim_short(cls, v: str) -> str:
        words = v.split()
        return " ".join(words[:60]) if len(words) > 60 else v


class LedgerEntryItem(BaseModel):
    """One candidate error-ledger entry extracted by the LLM narration call
    (app/pipeline/error_ledger.py). An empty list from the model is the
    normal, expected response for an unremarkable debate - nothing here
    forces a minimum count.
    """

    category: str
    detected_by: str
    challenged_model: Optional[str] = None
    description: str
    resolution: str


class EvidenceVerdict(BaseModel):
    """One claim's web-search-backed verdict (Phase 4 item 3,
    app/pipeline/evidence_verification.py). "unverifiable" is expected to be
    common, not a rare fallback - see that module's docstring. caveats is
    deliberately a separate field from verdict: a claim can be "supported"
    (the core mechanism is correct) while still carrying a caveats entry
    (the claim's absolute/strong wording has a documented exception) -
    collapsing the two would hide exactly the "overconfident-but-true"
    signal this feature exists to surface.
    """

    verdict: Literal["supported", "contradicted", "unverifiable"]
    caveats: str = ""
    rationale: str
    sources: List[str] = Field(default_factory=list)


class LedgerExtractionResponse(BaseModel):
    entries: List[LedgerEntryItem] = Field(default_factory=list)
