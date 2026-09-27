from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple


def _normalize(value: str) -> str:
    return value.strip().lower()


def _canonicalize(raw_value: str, value_options: List[str]) -> Tuple[str, bool]:
    """Maps a proposer's raw value onto a decision's canonical value_options,
    if one exists. Returns (comparison_value, matched).

    matched=True means the raw value was recognized as one of the canonical
    options (exactly, or as the sole option contained in the raw text - e.g.
    "Amazon MSK (Kafka)" contains "kafka" and nothing else from the list).
    matched=False means either there's no canonical set for this decision (in
    which case comparison_value is just the plain normalized raw value, and
    behavior is identical to before this fix), or the raw value didn't map
    onto exactly one canonical option (zero matches, or an ambiguous multiple
    match) - in both of those cases we fall back to the plain normalized raw
    value too, so the worst case is unchanged from the original exact-match
    behavior, never worse.

    The containment check is scoped to this one decision's own value_options
    list, never compared across two arbitrary proposer strings - that's what
    keeps it safe from false-merging real opposites like "SQL" vs "NoSQL"
    (which the framing prompt is also told never to list side by side).
    """
    normalized_raw = _normalize(raw_value)
    if not value_options:
        return normalized_raw, False

    normalized_options = [_normalize(o) for o in value_options]
    if normalized_raw in normalized_options:
        return normalized_raw, True

    contained = [opt for opt in normalized_options if opt and opt in normalized_raw]
    if len(contained) == 1:
        return contained[0], True

    return normalized_raw, False


def canonicalize_display_value(raw_value: str, value_options: List[str]) -> str:
    """Public helper for other pipeline steps (the arbiter) that need to show a
    resolved value the same canonical, display-cased way the diff does - e.g.
    turning a winning debate round-2 "kafka" back into "Kafka".
    """
    canonical, matched = _canonicalize(raw_value, value_options)
    if not matched:
        return raw_value
    lookup = {_normalize(o): o for o in value_options}
    return lookup.get(canonical, raw_value)


def values_converged(values: List[str], value_options: List[str]) -> bool:
    """True if every value canonicalizes to the same comparison value - the
    exact same test diff_decisions uses to call a decision CONSENSUS at the
    initial-proposal stage (see the len(set(comparison_values)) == 1 check
    above). The arbiter uses this on round-2 debate values so a decision the
    models revised into agreement is recognized as resolved, not scored as
    if it were still disputed.
    """
    if not values:
        return True
    comparison_values = {_canonicalize(v, value_options)[0] for v in values}
    return len(comparison_values) == 1


def _item_detail(item: Any, value_options: List[str]) -> Dict:
    _, matched = _canonicalize(item.value, value_options)
    # proposer_added already means "this model answered outside what framing
    # expected" for decision_ids the model invented; a value that misses every
    # canonical option for a decision framing DID enumerate is the same kind
    # of gap at the value level, so it reuses the same flag rather than adding
    # a second one - lets the UI treat both as "outside expected options."
    fell_outside_options = bool(value_options) and not matched
    return {
        "value": item.value,
        "confidence": item.confidence,
        "reasoning": item.reasoning,
        "cited_requirements": item.cited_requirements,
        "assumptions": item.assumptions,
        "proposer_added": item.proposer_added or fell_outside_options,
    }


def diff_decisions(
    framing_decisions: List[Dict],
    proposer_results: List[Tuple[Any, Tuple]],
    requirement_ids: set,
) -> List[Dict]:
    """Pure, deterministic diff per CLAUDE.md Section 4 - no LLM call.

    proposer_results: list of (ProposerConfig, (ProposerResponse, raw_text, prompt_tokens, completion_tokens))
    """
    by_decision: Dict[str, List[Tuple[str, Any]]] = defaultdict(list)
    dimension_by_id: Dict[str, str] = {}
    value_options_by_id: Dict[str, List[str]] = {
        d["decision_id"]: d.get("value_options") or [] for d in framing_decisions
    }

    for cfg, (parsed, _raw, _pt, _ct) in proposer_results:
        for item in parsed.decisions:
            by_decision[item.decision_id].append((cfg.name, item))
            dimension_by_id.setdefault(item.decision_id, item.dimension)

    results: List[Dict] = []
    for decision_id, entries in by_decision.items():
        dimension = dimension_by_id[decision_id]
        value_options = value_options_by_id.get(decision_id, [])
        # Original-cased canonical option, keyed by its normalized form, so a
        # resolved winning_value can be displayed as "Kafka" rather than "kafka".
        canonical_lookup = {_normalize(o): o for o in value_options}

        values_by_model = {model_name: item.value for model_name, item in entries}
        # Full reasoning/confidence/cited_requirements/assumptions per model, for
        # drill-down and for verifying proposers cite real requirement ids rather
        # than hallucinating them - attached to every branch below, not just consensus.
        details_by_model = {model_name: _item_detail(item, value_options) for model_name, item in entries}

        if len(entries) == 1:
            _, item = entries[0]
            results.append(
                {
                    "decision_id": decision_id,
                    "dimension": dimension,
                    "status": "unopposed",
                    "winning_value": item.value,
                    "agreement_score": None,
                    "values_by_model": values_by_model,
                    "details_by_model": details_by_model,
                }
            )
            continue

        # Compare on the canonicalized value when this decision has a known
        # value_options set (so "Kafka" and "Amazon MSK (Kafka)" compare as
        # equal); otherwise this is unchanged plain-normalized comparison.
        comparison_values = [_canonicalize(item.value, value_options)[0] for _, item in entries]

        if len(set(comparison_values)) == 1:
            # CONSENSUS: all comparison values match.
            avg_confidence = sum(item.confidence for _, item in entries) / len(entries)
            union_requirements = sorted({r for _, item in entries for r in item.cited_requirements})
            winning_value = canonical_lookup.get(comparison_values[0], entries[0][1].value)
            results.append(
                {
                    "decision_id": decision_id,
                    "dimension": dimension,
                    "status": "consensus",
                    "winning_value": winning_value,
                    "agreement_score": 1.0,
                    "values_by_model": values_by_model,
                    "details_by_model": details_by_model,
                    "confidence": avg_confidence,
                    "cited_requirements": union_requirements,
                }
            )
            continue

        # Disagreement. agreement_score = majority count / total opinions.
        # A true plurality tie can never reach the 0.8 auto-accept bucket (two
        # values each >=0.8*total would sum to >1.0*total), so it always falls
        # through to DISPUTED without needing separate tie-handling - this is
        # also how "no majority -> DISPUTED" (resolution #2) falls out naturally.
        counts = Counter(comparison_values)
        top_value, top_count = counts.most_common(1)[0]
        total = len(entries)
        agreement_score = top_count / total

        if agreement_score >= 0.8:
            status = "auto_accepted"
            winning_value = canonical_lookup.get(top_value, top_value)
        else:
            # Covers both [0.5, 0.8) and <0.5 - both DISPUTED per Section 4.
            status = "disputed"
            winning_value = None

        results.append(
            {
                "decision_id": decision_id,
                "dimension": dimension,
                "status": status,
                "winning_value": winning_value,
                "agreement_score": agreement_score,
                "values_by_model": values_by_model,
                "details_by_model": details_by_model,
            }
        )

    _flag_coupled_disputes(results, proposer_results)
    return results


def _flag_coupled_disputes(results: List[Dict], proposer_results: List[Tuple[Any, Tuple]]) -> None:
    """CLAUDE.md Section 4 coupling heuristic: two disputed decisions are flagged
    together (not auto-merged) if they share a cited requirement, or one's
    reasoning/assumptions text mentions the other's decision_id or dimension.
    Simple keyword match, not an LLM call.
    """
    for r in results:
        r.setdefault("coupled_with", [])

    disputed = [r for r in results if r["status"] == "disputed"]
    if len(disputed) < 2:
        return

    requirements_by_decision: Dict[str, set] = defaultdict(set)
    text_by_decision: Dict[str, str] = defaultdict(str)

    for _cfg, (parsed, _raw, _pt, _ct) in proposer_results:
        for item in parsed.decisions:
            requirements_by_decision[item.decision_id].update(item.cited_requirements)
            text_by_decision[item.decision_id] += " " + item.reasoning + " " + " ".join(item.assumptions)

    for k in text_by_decision:
        text_by_decision[k] = text_by_decision[k].lower()

    for i, a in enumerate(disputed):
        for b in disputed[i + 1 :]:
            shared_requirements = requirements_by_decision[a["decision_id"]] & requirements_by_decision[b["decision_id"]]
            a_mentions_b = (
                b["decision_id"] in text_by_decision[a["decision_id"]]
                or b["dimension"].lower() in text_by_decision[a["decision_id"]]
            )
            b_mentions_a = (
                a["decision_id"] in text_by_decision[b["decision_id"]]
                or a["dimension"].lower() in text_by_decision[b["decision_id"]]
            )
            if shared_requirements or a_mentions_b or b_mentions_a:
                a["coupled_with"].append(b["decision_id"])
                b["coupled_with"].append(a["decision_id"])
