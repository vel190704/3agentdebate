"""Server-rendered, zero-JS SVG debate-flow diagram for one decision.

Purely a rendering layer on top of data build_task_detail already computes
(values_by_model, debate.round_1/round_2/position_changes, status) - no new
scoring or status logic lives here, and nothing here is read back by the
pipeline. The one piece of pipeline logic borrowed is diff.py's
values_converged() (the same check the arbiter uses to decide debate
convergence); it is reused here ONLY to decide how far a track visually
bends toward the other track's endpoint. That reuse is one-directional -
this module never writes back into scoring or status.

Layout: two horizontal tracks (one per proposer) running left to right
through five stages - Framing (shared) -> Proposal -> Round 1 -> Round 2 ->
Outcome (shared). A track that never revised its position (no entry in
debate.position_changes) stays a straight horizontal line. A track that DID
revise bends during the Round 1 -> Round 2 segment, toward the other
track's endpoint, by an amount set by _closeness() - not by whether the
values now match some third scoring rule.
"""
from html import escape
from typing import Dict, List, Optional

from markupsafe import Markup

from .pipeline.diff import values_converged

# Mirrors board_display.decision_badge's css_class -> palette mapping and
# _base.html's :root color variables exactly, so this diagram's outcome dot
# can never visually disagree with the board's own status badge.
_OUTCOME_COLORS = {
    "consensus": "#1a7f37",
    "auto_accepted": "#1a7f37",
    "resolved_via_debate": "#0f6d76",
    "arbiter_resolved": "#1a56db",
    "human_decision_required": "#a3181f",
    "debate_failed": "#92620a",
}
_DEFAULT_OUTCOME_COLOR = "#57606a"

# Two fixed, distinguishable colors for model tracks - deliberately outside
# the status palette above so a track's color is never mistaken for a status
# signal (that meaning belongs to the outcome dot alone).
_TRACK_COLORS = ["#6f42c1", "#c2255c"]

_WIDTH = 560
_HEIGHT = 150
_X = {"framing": 40, "proposal": 175, "round1": 310, "round2": 445, "outcome": 520}
_Y_TOP = 38
_Y_BOTTOM = 112
_Y_MID = 75
_NODE_R = 5
_STAGE_LABELS = [("framing", "Framing"), ("proposal", "Proposal"), ("round1", "Round 1"), ("round2", "Round 2"), ("outcome", "Outcome")]


def _norm(value: str) -> str:
    return value.strip().lower()


def _closeness(value_a: str, value_b: str, value_options: List[str]) -> float:
    """0.0-1.0: how close two final (round 2) values are, for bend
    positioning ONLY - never fed back into scoring or status.

    1.0 - the pair converges per diff.py's own values_converged() check,
    the exact same test the arbiter uses to call a debate resolved.
    0.5 - not a canonicalize-level match, but one value's normalized text
    is contained in the other's (e.g. "choreography" inside "Pub/sub with
    event sourcing and choreography") - a looser textual overlap that still
    reads as "closer" than two unrelated answers.
    0.15 - a fixed, small nudge: the track moved (it has a position_changes
    entry) but landed somewhere textually unrelated to the other track, so
    the bend should be barely visible rather than implying an agreement
    diff.py itself would not recognize.
    """
    if values_converged([value_a, value_b], value_options):
        return 1.0
    na, nb = _norm(value_a), _norm(value_b)
    if na and nb and (na in nb or nb in na):
        return 0.5
    return 0.15


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def build_debate_diagram(decision: Dict, value_options: Optional[List[str]] = None) -> Optional[Markup]:
    """Returns a standalone <svg>...</svg> fragment for one decision's debate
    flow, or None if the decision never went to debate (consensus /
    auto_accepted / unopposed decisions have no round_1/round_2 to show -
    this diagram is specifically about the debate, not a substitute for the
    proposal-only cards those statuses already render).
    """
    debate = decision.get("debate")
    if not debate or not debate.get("round_1"):
        return None

    round1_by_model = {t["model"]: t["value"] for t in debate["round_1"]}
    round2_by_model = {t["model"]: t["value"] for t in debate.get("round_2") or []}
    models = list(debate["round_1"][0:2])
    model_names = [t["model"] for t in debate["round_1"]][:2]
    if len(model_names) < 2:
        # MVP is always exactly 2 proposers (CLAUDE.md Section 2/8); a
        # single-model debate track shouldn't happen, but degrade to "no
        # diagram" rather than guessing at a second track.
        return None

    initial_by_model = decision.get("values_by_model") or {}
    value_options = value_options or []

    has_round2 = len(round2_by_model) == 2
    closeness = None
    if has_round2:
        closeness = _closeness(round2_by_model[model_names[0]], round2_by_model[model_names[1]], value_options)

    changed_models = {pc["model"] for pc in debate.get("position_changes") or []}

    ys = [_Y_TOP, _Y_BOTTOM]
    other_ys = [_Y_BOTTOM, _Y_TOP]

    outcome_color = _OUTCOME_COLORS.get(decision.get("status"), _DEFAULT_OUTCOME_COLOR)

    svg_parts = [
        f'<svg viewBox="0 0 {_WIDTH} {_HEIGHT}" width="100%" style="max-width: {_WIDTH}px" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Debate flow diagram">'
    ]

    # Stage headers, shared across both tracks.
    for key, label in _STAGE_LABELS:
        svg_parts.append(
            f'<text x="{_X[key]}" y="16" font-size="10" fill="#6b7280" '
            f'text-anchor="middle" font-family="inherit">{escape(label)}</text>'
        )

    for i, model in enumerate(model_names):
        own_y = ys[i]
        other_y = other_ys[i]
        color = _TRACK_COLORS[i % len(_TRACK_COLORS)]

        round1_y = own_y
        if has_round2 and model in changed_models:
            round2_y = _lerp(own_y, other_y, closeness)
        else:
            round2_y = own_y

        # Straight segments up to round 1 (a track never bends before a
        # position change could have happened), then either a straight
        # continuation (no change) or a smooth curve into round 2 (change).
        path_d = (
            f"M {_X['framing']} {_Y_MID} "
            f"L {_X['proposal']} {own_y} "
            f"L {_X['round1']} {round1_y} "
        )
        if round2_y == round1_y:
            path_d += f"L {_X['round2']} {round2_y} "
        else:
            mid_x = (_X["round1"] + _X["round2"]) / 2
            path_d += f"Q {mid_x} {round1_y} {_X['round2']} {round2_y} "
        # Final connector into the single shared outcome node.
        path_d += f"L {_X['outcome']} {_Y_MID}"

        svg_parts.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2" opacity="0.85" />')

        # Nodes: proposal, round1, round2 (if present).
        svg_parts.append(f'<circle cx="{_X["proposal"]}" cy="{own_y}" r="{_NODE_R}" fill="{color}" />')
        svg_parts.append(f'<circle cx="{_X["round1"]}" cy="{round1_y}" r="{_NODE_R}" fill="{color}" />')
        if has_round2:
            svg_parts.append(f'<circle cx="{_X["round2"]}" cy="{round2_y}" r="{_NODE_R}" fill="{color}" />')

        label_y = own_y - 12 if i == 0 else own_y + 20
        svg_parts.append(
            f'<text x="{_X["framing"]}" y="{label_y}" font-size="11" font-weight="600" '
            f'fill="{color}" font-family="inherit">{escape(model)}</text>'
        )
        initial_value = initial_by_model.get(model, "")
        if initial_value:
            svg_parts.append(
                f'<text x="{_X["proposal"]}" y="{own_y - 10}" font-size="9" fill="#6b7280" '
                f'text-anchor="middle" font-family="inherit">{escape(_truncate(initial_value))}</text>'
            )

    # Shared framing and outcome nodes.
    svg_parts.append(f'<circle cx="{_X["framing"]}" cy="{_Y_MID}" r="{_NODE_R}" fill="#6b7280" />')
    svg_parts.append(f'<circle cx="{_X["outcome"]}" cy="{_Y_MID}" r="7" fill="{outcome_color}" stroke="white" stroke-width="1.5" />')
    winning = decision.get("winning_value")
    outcome_label = _truncate(winning) if winning else decision.get("status", "")
    svg_parts.append(
        f'<text x="{_X["outcome"]}" y="{_Y_MID + 24}" font-size="10" font-weight="600" '
        f'fill="{outcome_color}" text-anchor="middle" font-family="inherit">{escape(outcome_label)}</text>'
    )

    svg_parts.append("</svg>")
    return Markup("".join(svg_parts))


def _truncate(text: str, limit: int = 22) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"
