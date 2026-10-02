"""Server-rendered, zero-JS SVG debate-flow diagram for one decision.

Purely a rendering layer on top of data build_task_detail already computes
(values_by_model, value_options, debate.round_1/round_2/position_changes,
status) - no new scoring or status logic lives here, and nothing here is
read back by the pipeline. The one piece of pipeline logic borrowed is
diff.py's values_converged() (the same check the arbiter uses to decide
debate convergence); it is reused here ONLY to decide how far a track
visually bends toward the other track's endpoint. That reuse is
one-directional - this module never writes back into scoring or status.

Layout: two horizontal tracks (one per proposer) running left to right
through five stages - Framing (shared) -> Proposal -> Round 1 -> Round 2 ->
Outcome (shared). A track that never revised its position (no entry in
debate.position_changes) stays a straight horizontal line. A track that DID
revise bends during the Round 1 -> Round 2 segment, toward the other
track's endpoint, by an amount set by _closeness() - not by whether the
values now match some third scoring rule.

Track identity (which line is which model) is carried by TWO independent
channels - color AND dash pattern - deliberately not color alone. A
grid-search over the free hue bands left by the six existing status colors
(see the color-separation check run before this was wired in) found a best
achievable worst-case RGB separation of ~87, versus >100 for most
same-palette comparisons - the status palette already spans nearly the
whole hue wheel, so no track-color choice can be as cleanly separated from
every status color as the status colors are from each other. The dash
pattern exists so identifying which track is which never depends on that
imperfect color margin.
"""
from typing import Dict, List, Optional
from html import escape

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
_FAILURE_COLOR = "#92620a"  # same amber as badge-error / debate_failed, used for the break marker too.
_MUTED = "#6b7280"

# Two track styles, each identified by color AND dash pattern (see module
# docstring on why one channel alone isn't trusted). Chosen from the two
# hue bands (~75-105 and ~245-330) not already occupied by a status color;
# solid vs dashed is the structural fallback since even the best available
# hues can't clear the same separation margin the status colors have from
# each other.
_TRACK_STYLES = [
    {"color": "#66a824", "dash": None},
    {"color": "#6a2fbc", "dash": "7,4"},
]

_WIDTH = 620
_HEIGHT = 150
_X = {"framing": 50, "proposal": 190, "round1": 330, "round2": 460, "outcome": 560}
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

    1.0 - the pair converges per diff.py's own values_converged() check
    (which does the value_options canonicalize/containment mapping), the
    exact same test the arbiter uses to call a debate resolved.
    0.5 - not a canonicalize-level match, but one value's normalized text
    is a plain substring of the other's (e.g. "choreography" inside "Pub/sub
    with event sourcing and choreography") - a looser textual overlap that
    still reads as "closer" than two unrelated answers. This check is
    independent of value_options; it's a fallback for when there's no
    curated option list to canonicalize against.
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


def _truncate(text: str, limit: int = 22) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _shared_prefix_len(a: str, b: str) -> int:
    """Case-insensitive, since the failure mode this guards against is
    visual sameness (a viewer reading two truncated labels as identical),
    not literal string equality - "centralized..." vs "Centralized..."
    should count as sharing the whole common phrase, not diverge at
    character 0 just because of a capitalization difference.
    """
    n = min(len(a), len(b))
    for i in range(n):
        if a[i].lower() != b[i].lower():
            return i
    return n


def _truncate_pair(value_a: str, value_b: str, limit: int = 22, pad: int = 8, max_limit: int = 40) -> "tuple[str, str]":
    """Truncates two proposal-label strings for side-by-side display,
    extending both past their shared prefix when needed so a viewer sees
    where they actually diverge instead of two identical-looking cutoffs.

    Normal case (the two values diverge before the truncation point would
    even matter, e.g. "microservices" vs. "Event-driven microservices"):
    unchanged fixed-limit truncation, independently per string.

    Long-shared-prefix case (e.g. "centralized state machine service" vs.
    "Centralized state machine library enforced in Order service..."):
    the default limit would cut inside the shared portion, so both labels
    would read as the same phrase. Instead, extend the visible length past
    the first point of divergence (plus a small pad so the differing word
    isn't cut off at its very first letter), capped at max_limit so one
    long straggler can't blow out the diagram's layout.
    """
    a, b = value_a.strip(), value_b.strip()
    divergence = _shared_prefix_len(a, b)
    if divergence <= limit - 2:
        return _truncate(a, limit), _truncate(b, limit)
    extended = min(divergence + pad, max_limit)
    return _truncate(a, extended), _truncate(b, extended)


def _dash_attr(dash: Optional[str]) -> str:
    return f' stroke-dasharray="{dash}"' if dash else ""


def _break_marker(x: float, y: float) -> str:
    """Two short parallel diagonal ticks across the line - the conventional
    'break in the axis' glyph - marking where a debate round's data stops
    because the call failed, not because the model held its position. A
    color/status change alone (amber outcome dot) wouldn't read as
    unambiguously as an actual gap in the line itself.
    """
    out = []
    for dx in (-5, 4):
        out.append(
            f'<line x1="{x + dx - 4}" y1="{y - 7}" x2="{x + dx + 4}" y2="{y + 7}" '
            f'stroke="{_FAILURE_COLOR}" stroke-width="2.2" stroke-linecap="round" />'
        )
    return "".join(out)


def _failure_outcome_marker(x: float, y: float) -> str:
    """Amber warning triangle instead of the normal circular outcome dot -
    a shape change, not just a color change, so debate_failed's diagram
    ending is distinct even if the amber/red palette were ever confused.
    """
    return (
        f'<polygon points="{x},{y - 10} {x - 9},{y + 7} {x + 9},{y + 7}" '
        f'fill="{_FAILURE_COLOR}" stroke="white" stroke-width="1.3" stroke-linejoin="round" />'
        f'<text x="{x}" y="{y + 4.5}" font-size="10" font-weight="700" fill="white" '
        f'text-anchor="middle" font-family="inherit">!</text>'
    )


def build_debate_diagram(decision: Dict) -> Optional[Markup]:
    """Returns a standalone <svg>...</svg> fragment for one decision's debate
    flow, or None if the decision never went to debate (consensus /
    auto_accepted / unopposed decisions have no round_1/round_2 to show -
    this diagram is specifically about the debate, not a substitute for the
    proposal-only cards those statuses already render).

    Reads decision["value_options"] (set by build_task_detail from the
    FramingDecision row) for the convergence check inside _closeness();
    defaults to [] for any caller that hasn't attached it, which degrades
    to plain string comparison - never worse than before that field existed.
    """
    debate = decision.get("debate")
    if not debate:
        return None

    failed = bool(debate.get("failed"))
    failed_round = debate.get("failed_round") if failed else None

    initial_by_model = decision.get("values_by_model") or {}
    value_options = decision.get("value_options") or []

    if failed and failed_round == 1:
        # Round 1 itself never completed - debate.py returns an empty
        # round_1 in this case (see its docstring), so the only model
        # identities we have are from the pre-debate proposals.
        model_names = list(initial_by_model.keys())[:2]
        round1_by_model: Dict[str, str] = {}
        round2_by_model: Dict[str, str] = {}
    else:
        if not debate.get("round_1"):
            return None
        model_names = [t["model"] for t in debate["round_1"]][:2]
        round1_by_model = {t["model"]: t["value"] for t in debate["round_1"]}
        round2_by_model = {t["model"]: t["value"] for t in debate.get("round_2") or []}

    if len(model_names) < 2:
        # MVP is always exactly 2 proposers (CLAUDE.md Section 2/8); a
        # single-model debate track shouldn't happen, but degrade to "no
        # diagram" rather than guessing at a second track.
        return None

    has_round1 = len(round1_by_model) == 2
    has_round2 = len(round2_by_model) == 2
    closeness = None
    if has_round2:
        closeness = _closeness(round2_by_model[model_names[0]], round2_by_model[model_names[1]], value_options)

    changed_models = {pc["model"] for pc in debate.get("position_changes") or []}

    # Computed once as a pair (not independently per model) so a long shared
    # prefix between the two proposals extends both labels to their actual
    # point of divergence, rather than each being truncated in isolation
    # and happening to look identical.
    proposal_labels: Dict[str, str] = {}
    initial_a = initial_by_model.get(model_names[0], "")
    initial_b = initial_by_model.get(model_names[1], "")
    if initial_a and initial_b:
        label_a, label_b = _truncate_pair(initial_a, initial_b)
        proposal_labels[model_names[0]] = label_a
        proposal_labels[model_names[1]] = label_b
    else:
        if initial_a:
            proposal_labels[model_names[0]] = _truncate(initial_a)
        if initial_b:
            proposal_labels[model_names[1]] = _truncate(initial_b)

    ys = [_Y_TOP, _Y_BOTTOM]
    other_ys = [_Y_BOTTOM, _Y_TOP]

    outcome_color = _OUTCOME_COLORS.get(decision.get("status"), _DEFAULT_OUTCOME_COLOR)

    svg_parts = [
        f'<svg viewBox="0 0 {_WIDTH} {_HEIGHT}" width="100%" style="max-width: {_WIDTH}px" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Debate flow diagram">'
    ]

    for key, label in _STAGE_LABELS:
        svg_parts.append(
            f'<text x="{_X[key]}" y="16" font-size="10" fill="{_MUTED}" '
            f'text-anchor="middle" font-family="inherit">{escape(label)}</text>'
        )

    # x-position where a failed round's data stops - partway between the
    # last completed stage and the one that never returned, so the break
    # marker sits visibly before the stage it prevented rather than on top
    # of a node that doesn't exist.
    if failed and failed_round == 1:
        cutoff_x = (_X["proposal"] + _X["round1"]) / 2
    elif failed and failed_round == 2:
        cutoff_x = (_X["round1"] + _X["round2"]) / 2
    else:
        cutoff_x = None

    for i, model in enumerate(model_names):
        own_y = ys[i]
        other_y = other_ys[i]
        style = _TRACK_STYLES[i % len(_TRACK_STYLES)]
        color = style["color"]
        dash = _dash_attr(style["dash"])

        if failed:
            # No bending logic for a failed debate - bending is derived from
            # round-2 data we don't have, so the track is flat up to
            # whatever it actually completed, then breaks.
            path_d = f"M {_X['framing']} {_Y_MID} L {_X['proposal']} {own_y} "
            if has_round1:
                path_d += f"L {_X['round1']} {own_y} "
            path_d += f"L {cutoff_x} {own_y}"
            svg_parts.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2" opacity="0.85"{dash} />')

            svg_parts.append(f'<circle cx="{_X["proposal"]}" cy="{own_y}" r="{_NODE_R}" fill="{color}" />')
            if has_round1:
                svg_parts.append(f'<circle cx="{_X["round1"]}" cy="{own_y}" r="{_NODE_R}" fill="{color}" />')
            svg_parts.append(_break_marker(cutoff_x, own_y))

            # Dotted amber connector from the break to the shared outcome
            # marker - deliberately not the model's own color, since
            # nothing past this point was actually authored by either model.
            svg_parts.append(
                f'<path d="M {cutoff_x} {own_y} L {_X["outcome"]} {_Y_MID}" fill="none" '
                f'stroke="{_FAILURE_COLOR}" stroke-width="1.5" stroke-dasharray="2,3" opacity="0.7" />'
            )
        else:
            round1_y = own_y
            if has_round2 and model in changed_models:
                round2_y = _lerp(own_y, other_y, closeness)
            else:
                round2_y = own_y

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
            path_d += f"L {_X['outcome']} {_Y_MID}"

            svg_parts.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2" opacity="0.85"{dash} />')

            svg_parts.append(f'<circle cx="{_X["proposal"]}" cy="{own_y}" r="{_NODE_R}" fill="{color}" />')
            svg_parts.append(f'<circle cx="{_X["round1"]}" cy="{round1_y}" r="{_NODE_R}" fill="{color}" />')
            if has_round2:
                svg_parts.append(f'<circle cx="{_X["round2"]}" cy="{round2_y}" r="{_NODE_R}" fill="{color}" />')

        label_y = own_y - 12 if i == 0 else own_y + 20
        svg_parts.append(
            f'<text x="{_X["framing"]}" y="{label_y}" font-size="11" font-weight="600" '
            f'fill="{color}" font-family="inherit">{escape(model)}</text>'
        )
        proposal_label = proposal_labels.get(model, "")
        if proposal_label:
            svg_parts.append(
                f'<text x="{_X["proposal"]}" y="{own_y - 10}" font-size="9" fill="{_MUTED}" '
                f'text-anchor="middle" font-family="inherit">{escape(proposal_label)}</text>'
            )

    svg_parts.append(f'<circle cx="{_X["framing"]}" cy="{_Y_MID}" r="{_NODE_R}" fill="{_MUTED}" />')

    if failed:
        svg_parts.append(_failure_outcome_marker(_X["outcome"], _Y_MID))
        outcome_label = f"Failed (round {failed_round})"
    else:
        svg_parts.append(f'<circle cx="{_X["outcome"]}" cy="{_Y_MID}" r="7" fill="{outcome_color}" stroke="white" stroke-width="1.5" />')
        # Shorter truncate limit than other labels - this text is centered
        # on a node near the right edge of the viewBox, with no room to
        # overflow past it the way a mid-diagram label does. Applies to the
        # status fallback too (e.g. "human_decision_required" is itself 24
        # characters).
        outcome_label = _truncate(decision.get("winning_value") or decision.get("status", ""), limit=16)

    svg_parts.append(
        f'<text x="{_X["outcome"]}" y="{_Y_MID + 24}" font-size="10" font-weight="600" '
        f'fill="{outcome_color if not failed else _FAILURE_COLOR}" text-anchor="middle" '
        f'font-family="inherit">{escape(_truncate(outcome_label, limit=20))}</text>'
    )

    svg_parts.append("</svg>")
    return Markup("".join(svg_parts))
