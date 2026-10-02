from typing import Dict

# Statuses where the system never produced a confident answer a human can
# just defer to - a genuine tie (human_decision_required) or no answer at
# all because a model call broke (debate_failed). This is where recording a
# human's own decision (HumanDecision) actually adds information; offering
# it under a status the system already resolved confidently would blur the
# signal this feature exists to capture (see HumanDecision's docstring).
HUMAN_INPUT_ELIGIBLE_STATUSES = {"human_decision_required", "debate_failed"}


def current_values_by_model(decision: Dict) -> Dict[str, str]:
    """The per-model values a human comparing their own answer should be
    compared against - the most recent values that actually exist, in order:
    round 2 (once a full debate transcript exists) -> round 1 (a
    debate_failed decision that broke between round 1 and round 2 still has
    a real, possibly-revised round 1 position - round 1 restates a model's
    position in its own words rather than echoing the original proposal
    verbatim, so it can genuinely differ from values_by_model) -> the
    original pre-debate proposal (no debate at all, or debate_failed before
    even round 1 completed).

    Falling straight from round 2 to the original proposal (skipping a
    round 1 that did complete) would silently show a stale value for a
    failed_round=2 decision whenever round 1's wording moved - this chain
    exists so that gap can't happen.
    """
    debate = decision.get("debate") or {}
    round2_values = {m["model"]: m["value"] for m in debate.get("round_2") or []}
    if round2_values:
        return round2_values
    round1_values = {m["model"]: m["value"] for m in debate.get("round_1") or []}
    if round1_values:
        return round1_values
    return decision.get("values_by_model") or {}


def decision_badge(decision: Dict) -> Dict:
    """Maps a decision's status onto the badge the board shows. Pure
    presentation logic - reads fields the pipeline already computed, decides
    nothing new. See CLAUDE.md Section 6 Phase 3 spec for the exact
    distinctions required: consensus/auto_accepted and resolved_via_debate
    both read as "agreed" but must look distinguishable; unopposed reads
    differently from either; debate_failed (Phase 4 item 2) must read as an
    operational failure, not as human_decision_required's routine tie.

    arbiter_resolved used to branch on resolution_basis (citation vs.
    contradiction vs. mixed), since a contradiction-driven win could flip on
    word choice while both sides were equally compliant. That distinction
    and the resolved_on_technicality status are gone because the mechanism
    behind them is gone: CLAUDE.md Section 5 removed the keyword-overlap
    contradiction penalty entirely after a 12-task validation batch found a
    0% genuine-catch rate. Scoring is citation-only now, so every
    arbiter_resolved decision is a clean citation-backed win - there is
    nothing left to downgrade.
    """
    status = decision["status"]

    if status in ("consensus", "auto_accepted"):
        return {
            "css_class": "badge-agreed",
            "label": "Agreed",
            "icon": "✓",
            "subtitle": "All models independently aligned",
        }

    if status == "resolved_via_debate":
        return {
            "css_class": "badge-agreed-debate",
            "label": "Agreed",
            "icon": "✓",
            "subtitle": "Models converged after debate",
        }

    if status == "arbiter_resolved":
        return {
            "css_class": "badge-resolved-confident",
            "label": "Resolved",
            "icon": "✓",
            "subtitle": "Decided by a valid requirement citation",
        }

    if status == "human_decision_required":
        return {
            "css_class": "badge-action",
            "label": "Needs your decision",
            "icon": "⚑",
            "subtitle": "Models tied or no valid citation broke the tie",
        }

    if status == "unopposed":
        return {
            "css_class": "badge-unopposed",
            "label": "Unopposed",
            "icon": "–",
            "subtitle": "Only one model addressed this",
        }

    if status == "debate_failed":
        # Deliberately NOT badge-action: human_decision_required means the
        # pipeline worked and a human needs to make a product call this is
        # a technical failure needing operational attention instead, and
        # conflating the two would hide an infrastructure problem behind
        # what looks like a routine decision waiting on someone's judgment.
        return {
            "css_class": "badge-error",
            "label": "Debate failed",
            "icon": "⚠",
            "subtitle": "A model call failed and could not be recovered - see drill-down for details",
        }

    # Defensive fallback - status vocabulary is closed per models.py's
    # DecisionResult docstring, but never silently mislabel an outcome
    # the board doesn't recognize as if it were settled.
    return {
        "css_class": "badge-action",
        "label": status,
        "icon": "?",
        "subtitle": "",
    }
