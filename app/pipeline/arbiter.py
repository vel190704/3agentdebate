import re
from typing import Dict, List, Optional

from .diff import canonicalize_display_value, values_converged

# Plain topic-overlap stopwords - no negation-detection logic here, unlike
# the removed contradiction check. This check never needs to know whether
# text affirms or negates anything, only whether two specific, paired texts
# (a model's own claim and the requirement IT named in the same message)
# share a topic. That's a structurally weaker, safer question than the one
# the removed check asked - see score_decision's docstring.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "for", "with", "and", "or", "but",
    "this", "that", "these", "those", "we", "our", "it", "its", "as", "by",
    "from", "will", "shall", "should", "would", "may", "might", "if",
    "then", "so", "do", "does", "did", "has", "have", "had", "must",
    "no", "not", "never", "without", "cannot", "can't", "nor",
    "any", "all", "use", "using", "used", "into", "onto", "than",
}


def _stem(word: str) -> str:
    """Minimal suffix-stripping (Porter-lite, not a real stemmer) so the
    citation-relevance check below can match "passwordless"/"password" and
    "processed"/"process" as the same topic, instead of missing them purely
    on word form. Order matters: each branch returns immediately, so a more
    specific suffix (e.g. "-less") is tried before a more general one (the
    plain "-s" branch would otherwise fire first, since "passwordless" also
    ends in "s"). Length guards avoid stripping down to noise (e.g. "as" is
    never touched), and the "ss" exclusion on the "-s" branch stops "class"/
    "process" from being corrupted to "clas"/"proces". This is deliberately
    crude - real over-stemming (e.g. "caching" -> "cach", not "cache") is an
    accepted tradeoff rather than a full NLP dependency for four suffixes.
    """
    if len(word) > 6 and word.endswith("less"):
        return word[:-4]
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _significant_keywords(text: str) -> set:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {_stem(w) for w in words if w not in _STOPWORDS and len(w) >= 3}


def _citation_relevance(claim_text: str, requirement_text: str) -> str:
    """Weak topical-overlap check between a round-2 claim and the
    requirement it cites - NOT a violation/compliance judgment. Only asks
    whether the claim shares any significant vocabulary with the
    requirement it names, as a cheap sanity check against citing a
    real-but-irrelevant requirement id.

    "unsupported" is a weaker signal than it sounds: a well-reasoned
    citation that paraphrases the requirement instead of quoting it can
    land here too, so this is surfaced as a flag for a human to weigh, not
    used to change the score - a shakier keyword signal must not move a
    decision's winner, which is exactly the mistake the removed
    contradiction check made.
    """
    claim_kw = _significant_keywords(claim_text)
    req_kw = _significant_keywords(requirement_text)
    if claim_kw & req_kw:
        return "supported"
    return "unsupported"


def _score_round2_citation(round2_msg: Dict, requirement_ids: set, requirement_text_by_id: Dict[str, str]) -> Dict:
    cited = round2_msg.get("cited_requirement")
    if cited and cited in requirement_ids:
        relevance = _citation_relevance(round2_msg.get("claim", ""), requirement_text_by_id.get(cited, ""))
        return {
            "points": 1,
            "note": f"round 2 cited_requirement '{cited}' matches a real requirement id (+1)",
            "citation_relevance": relevance,
        }
    if cited:
        return {
            "points": 0,
            "note": f"round 2 cited_requirement '{cited}' does not match any known requirement id (0)",
            "citation_relevance": None,
        }
    return {"points": 0, "note": "no requirement cited in round 2 (0)", "citation_relevance": None}


def score_decision(
    decision_id: str,
    requirements: List[Dict],
    requirement_ids: set,
    proposal_details_by_model: Dict[str, Dict],
    transcript: Dict,
    value_options: Optional[List[str]] = None,
) -> Dict:
    """Pure, deterministic, citation-only scoring of a disputed decision's
    round-2 positions. No LLM call.

    This used to also apply a -1 keyword-overlap contradiction penalty
    (CLAUDE.md Section 5's third rubric item). That mechanism was removed
    entirely, not tuned further: a 12-task, 29-instance fresh-domain
    validation batch found a 0% genuine-catch rate, split across three
    failure modes a keyword check cannot avoid by construction - a model
    stating compliance in the requirement's own vocabulary and being
    penalized for it, a domain's central noun (e.g. "tenant") acting as an
    unavoidable trigger since it's the grammatical subject of the
    prohibition rather than the prohibited act, and misattribution, where
    one model's critique of the OPPONENT's design gets scored as its own
    contradiction. See CLAUDE.md Section 5 for the full rationale. Citation
    checking is unaffected - it never produced a false positive across four
    real runs (the requirement id either exists or it doesn't).

    Returns per-model score breakdown plus the resolution: either a single
    winner (highest total_score) or human_decision_required on a tie -
    unless round 2 converged, in which case scoring never runs at all.
    """
    round2_values = [m["value"] for m in transcript["round_2"]]
    if values_converged(round2_values, value_options or []):
        # The disputing models revised their way into agreement in round 2.
        # Citation scoring exists to break a tie between positions that
        # still disagree - running it on values that already match isn't a
        # scoring problem to get right, it's a category error (there's
        # nothing left to arbitrate). No scores are computed at all.
        return {
            "decision_id": decision_id,
            "scores": {},
            "status": "resolved_via_debate",
            "winner": None,
            "winning_value": round2_values[0] if round2_values else None,
        }

    requirement_text_by_id = {r["req_id"]: r["text"] for r in requirements}
    round2_by_model = {m["model"]: m for m in transcript["round_2"]}

    scores: Dict[str, Dict] = {}
    for model_name, round2_msg in round2_by_model.items():
        citation = _score_round2_citation(round2_msg, requirement_ids, requirement_text_by_id)
        scores[model_name] = {
            "citation_points": citation["points"],
            "citation_note": citation["note"],
            "citation_relevance": citation["citation_relevance"],
            "total_score": citation["points"],
            "round2_value": round2_msg.get("value"),
        }

    if not scores:
        return {
            "decision_id": decision_id,
            "scores": scores,
            "status": "human_decision_required",
            "winner": None,
            "winning_value": None,
        }

    max_score = max(s["total_score"] for s in scores.values())
    top_models = [m for m, s in scores.items() if s["total_score"] == max_score]

    if len(top_models) == 1:
        winner = top_models[0]
        return {
            "decision_id": decision_id,
            "scores": scores,
            "status": "arbiter_resolved",
            "winner": winner,
            "winning_value": scores[winner]["round2_value"],
        }

    return {
        "decision_id": decision_id,
        "scores": scores,
        "status": "human_decision_required",
        "winner": None,
        "winning_value": None,
    }


def resolve_winning_value(arbiter_result: Dict, value_options: List[str]) -> Optional[str]:
    """Applies the same canonical display-casing the diff uses (e.g. "kafka"
    round-2 value -> "Kafka") to the arbiter's winning value, if any.
    """
    if arbiter_result["winning_value"] is None:
        return None
    return canonicalize_display_value(arbiter_result["winning_value"], value_options)
