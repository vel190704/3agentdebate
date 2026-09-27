"""Phase 4 item 3: evidence verification, first pass. Checks a single
checkable factual claim (NOT a judgment/preference claim - see the manual
bucketing done before this module was written) against real sources via the
Anthropic API's web_search tool, and produces a non-scoring, informational
verdict for a human to review.

Deliberately narrow, matching CLAUDE.md Section 5's own lesson from the
removed contradiction check and citation_relevance's cautious rollout:
- Never changes any decision's status, winner, or score. Verdicts are
  stored (once this passes review) purely as annotation.
- Only runs on claims a human/process has already classified as "checkable"
  (a specific, falsifiable statement about a named technology) - never on
  judgment/preference claims ("X is simpler," "Y is better suited"), which
  would require the same semantic/value adjudication that broke the
  contradiction check, just relocated to a search-grounded mechanism
  instead of a keyword one.
- Runs as a separate, on-demand step (same pattern as error_ledger's
  backfill), never during task submission - one web_search-backed call per
  claim costs real tokens and time.
- "unverifiable" must be a common, expected outcome, not a rare fallback -
  a mechanism that always comes back confidently supported or contradicted
  isn't actually distinguishing decisive evidence from thin evidence, the
  same shape of red flag as resolved_on_technicality firing on everything.
"""

import json
from typing import List, Tuple

from anthropic import AsyncAnthropic
from pydantic import ValidationError

from ..config import settings
from ..llm.base import call_with_transport_retry
from ..schemas import EvidenceVerdict

EVIDENCE_VERIFICATION_SYSTEM_PROMPT = """You are fact-checking ONE specific technical claim using
web search. This is NOT a scoring or arbitration step - your verdict is stored as an
informational, non-scoring annotation for a human to review. It never changes any decision's
outcome, winner, or score.

Use the web_search tool to check the claim against real, current sources, then respond with
ONLY a JSON object matching this shape - nothing else, no markdown fences, no prose outside it:

{
  "verdict": "supported" | "contradicted" | "unverifiable",
  "caveats": "any exception, precondition, or nuance to the claim's specific wording that your
              search surfaced - empty string if genuinely none",
  "rationale": "1-3 sentences explaining the verdict, referencing what you actually found",
  "sources": ["https://...", "https://..."]
}

STRICT RULES:
- "unverifiable" is a normal, common, and often correct answer. Use it whenever your search
  does not turn up clear, decisive evidence either way. Do NOT force a supported/contradicted
  verdict when the evidence is genuinely thin, mixed, or you could not find authoritative
  sources on the specific point - a confident guess dressed up as a verdict is worse than
  admitting the search didn't settle it.
- Pay close attention to absolute or strong wording in the claim itself (e.g. "cannot,"
  "always," "never," "correctly handles," "guarantees," "ensures"). If your search finds a
  real, specific, documented exception or precondition to that absolute framing, report it in
  "caveats" - even if the claim's underlying mechanism is fundamentally correct and the overall
  verdict is "supported". A caveat does not automatically flip the verdict: use "contradicted"
  when the claim's core factual assertion is wrong, and "supported" with a populated "caveats"
  field when the mechanism is correct but the strong/absolute wording overstates it. Do not
  leave "caveats" empty just because the overall verdict is positive - actively check whether
  the specific wording holds up, not just the general idea.
- Only cite a URL you actually saw in your own search results for this call. Never invent or
  recall a URL from general knowledge.
- Output ONLY the JSON object.
"""

# The current web_search tool version as of this project's timeline. Bumping
# this to a newer dated version if the SDK adds one is a one-line change.
_WEB_SEARCH_TOOL = {"type": "web_search_20260318", "name": "web_search", "max_uses": 4}

# Empirically measured against a real call (see project history): a single
# verified claim's response, including the model's own tool-orchestration
# text, ran to ~2100 output tokens and was cut off at 1500. 4096 leaves
# comfortable headroom for the final JSON on top of search/tool-use content.
_MAX_TOKENS = 4096

# Sonnet 5 promotional rate confirmed via CSV reconciliation (see project
# history) - exported so callers (the board's pre-click estimate, cost
# logging) use the same numbers instead of duplicating them.
INPUT_TOKEN_RATE = 2.0 / 1_000_000
OUTPUT_TOKEN_RATE = 10.0 / 1_000_000
SEARCH_RATE = 0.01

# Measured across 6 real repeat/comparison calls on this project (see
# project history): per-claim total cost ranged $0.07-$0.20+ depending on
# how many searches the model decided it needed (2-4, non-deterministic
# even for the identical claim run back to back). Shown to a human BEFORE
# they trigger a real call, since a point estimate would understate the
# true spread by roughly 2x in either direction.
ESTIMATED_COST_RANGE = (0.07, 0.20)


def estimate_cost(input_tokens: int, output_tokens: int, web_search_requests: int) -> float:
    return (
        input_tokens * INPUT_TOKEN_RATE
        + output_tokens * OUTPUT_TOKEN_RATE
        + web_search_requests * SEARCH_RATE
    )


def _extract_final_text(content_blocks) -> str:
    """The response interleaves thinking/server_tool_use/tool_result blocks
    with the model's own text - only the text blocks matter for our JSON
    answer, and in practice there is exactly one, at the end.
    """
    text_parts = [b.text for b in content_blocks if getattr(b, "type", None) == "text"]
    return "".join(text_parts).strip()


def _extract_json_object(text: str) -> str:
    """Despite an explicit "output ONLY the JSON object" instruction, this
    model sometimes wraps the answer in a markdown code fence or prefixes it
    with a short lead-in sentence ("I have enough evidence...") - observed
    empirically, not hypothetical. Rather than parse fence syntax
    specifically, just slice from the first '{' to the last '}': it handles
    both cases (and plain unwrapped JSON) uniformly. If no braces are found
    at all, return the text unchanged so json.loads raises its own clear
    error instead of this silently returning an empty string.
    """
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace < first_brace:
        return text
    return text[first_brace : last_brace + 1]


async def verify_claim(claim_text: str, context: str = "") -> Tuple[EvidenceVerdict, int, int, int]:
    """Verifies one checkable factual claim via a single web_search-backed
    Anthropic API call. `context` is optional short framing (e.g. which
    decision/task this claim came from) to help the model search precisely -
    it is not itself claim text and is never quoted back as if it were.

    Returns (verdict, input_tokens, output_tokens, web_search_requests) - the
    search count is a real, separately-billed line item ($0.01/search on top
    of token cost) and was previously dropped on the floor here, making this
    mechanism's true per-call cost impossible to reconcile. Pure plumbing:
    the count was always on response.usage.server_tool_use.web_search_requests,
    just never read.

    Raises ValueError if the model's response never resolves to valid JSON
    matching EvidenceVerdict (no retry loop here: unlike a JSON-shape retry
    on a cheap call, retrying a multi-search verification call on a parse
    failure would double real search cost for a rare failure mode - if this
    turns out to matter in practice, it can be added later).
    """
    client = AsyncAnthropic(api_key=settings.anthropic_api_key, timeout=settings.llm_call_timeout_seconds * 3)

    prompt = f'CLAIM TO VERIFY: "{claim_text}"'
    if context:
        prompt += f"\n\nCONTEXT (for search precision only, not part of the claim itself): {context}"
    prompt += "\n\nVerify this claim now per your instructions."

    async def _call():
        return await client.messages.create(
            model=settings.claude_model,
            max_tokens=_MAX_TOKENS,
            system=EVIDENCE_VERIFICATION_SYSTEM_PROMPT,
            tools=[_WEB_SEARCH_TOOL],
            messages=[{"role": "user", "content": prompt}],
        )

    response = await call_with_transport_retry(_call)

    if response.stop_reason == "max_tokens":
        raise ValueError(f"Evidence verification for claim {claim_text!r} was cut off at max_tokens before finishing")

    raw_text = _extract_final_text(response.content)
    json_text = _extract_json_object(raw_text)
    try:
        data = json.loads(json_text)
        verdict = EvidenceVerdict.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Evidence verification for claim {claim_text!r} produced invalid JSON: {e}\nRaw: {raw_text[:500]}")

    usage = response.usage
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    server_tool_use = getattr(usage, "server_tool_use", None)
    web_search_requests = getattr(server_tool_use, "web_search_requests", 0) or 0
    return verdict, input_tokens, output_tokens, web_search_requests
