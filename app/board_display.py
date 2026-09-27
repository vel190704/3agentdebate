from typing import Dict


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
