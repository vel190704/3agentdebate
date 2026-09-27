# CLAUDE.md — Multi-LLM Consensus Platform (MVP Build)

You are building the MVP of a multi-LLM deliberation platform. Read this whole file before writing any code. It converts the design plan into concrete, buildable specs. Where the plan said "NEEDS DESIGN," this file makes the decision — implement what's specified here, and flag in your own words anywhere you think a spec below is wrong or underspecified before you build around it.

Work in phases (Section 6). Do not try to build everything in one pass. After each phase, stop and show me the result before continuing to the next.

---

## 1. What this system does (one paragraph)

A user gives a project context (requirements, constraints, docs) and a task ("design the database architecture"). Two or three LLMs independently propose solutions as structured, schema-constrained decisions (not free text). The system diffs those decisions algorithmically — no LLM needed for that step. Where models agree, the decision auto-resolves. Where they disagree, a short, format-constrained debate happens between only the disagreeing models, and a dedicated arbiter (never one of the proposing models) resolves it or flags it for a human. The user sees a per-decision board, not a chat transcript.

## 2. Key design decision: role separation (do not deviate from this)

**Claude is the arbiter/synthesizer. Claude never proposes a competing solution in the same task it's arbitrating.** This sidesteps the single biggest flaw identified in the design review: a model can't fairly judge a dispute it's a party to.

Concretely for MVP:
- **Proposing models:** two non-Claude models (e.g. GPT-4-class + DeepSeek-class via their APIs). Configurable via `.env` / config file — don't hardcode specific model names deep in the logic.
- **Arbiter:** Claude (via this session or the Anthropic API). Claude never generates an initial proposal for a task it's arbitrating. Claude's job is: run the diff, decide if a dispute needs debate, moderate the debate, apply the deterministic scoring rubric (Section 4), and write the final synthesis.
- If a later phase adds Claude as a third proposer, that requires a different reassessment mechanism (anonymized candidates, or a fourth model as arbiter) — don't build that yet, just don't design yourself into a corner that makes it impossible later.

## 3. Structured decision schema (item 4 from the plan — build this first)

Every proposing model must return JSON matching this shape for a given task — no free-form prose as the primary output (prose can accompany it as `reasoning`, but the fields below are what the system operates on):

```json
{
  "task_id": "string",
  "model": "string",
  "decisions": [
    {
      "decision_id": "snake_case_id, e.g. 'database', 'message_queue', 'auth_method'",
      "dimension": "human-readable label, e.g. 'Database'",
      "value": "the chosen option, e.g. 'PostgreSQL'",
      "confidence": 0.0,
      "reasoning": "max ~150 words, plain text",
      "cited_requirements": ["requirement or constraint IDs from shared context, if any"],
      "assumptions": ["explicit assumptions this decision depends on"]
    }
  ]
}
```

Enforce this with a JSON schema validator on every model response. If a model returns invalid JSON or missing required fields, retry once with an error message appended to the prompt, then fail loudly (don't silently drop the decision).

## 4. Diff + consensus scoring (item 6 — deterministic, no LLM call)

For each `decision_id` that appears in more than one model's response:

1. Normalize `value` strings (lowercase, strip whitespace) before comparing.
2. If all normalized values match → **CONSENSUS**. Store the decision with averaged confidence and the union of cited requirements.
3. If values differ → compute:
   ```
   agreement_score = (count of models agreeing with the majority value) / (total models with an opinion on this decision_id)
   ```
4. Thresholds:
   - `agreement_score >= 0.8` → auto-accept majority value, log the dissent for visibility but don't debate.
   - `0.5 <= agreement_score < 0.8` → **DISPUTED**, goes to targeted debate (Section 5).
   - `agreement_score < 0.5` → **DISPUTED**, same as above (with MVP's 2-model setup this collapses to "any disagreement between two models = debate," which is fine and expected at this scale).
5. A `decision_id` present in only one model's output is **UNOPPOSED**, not automatically consensus — surface it clearly in the UI as "only one model addressed this" rather than implying agreement.

**Coupled-decision heuristic (item 7, MVP-simple version):** when two disputed `decision_id`s share at least one `cited_requirements` entry, or one decision's `reasoning`/`assumptions` text mentions another disputed `decision_id` or its `dimension` label (simple keyword match, not an LLM call), flag them together as "possibly coupled — resolve together" in the output. Don't try to auto-merge them; just surface the flag. Full semantic coupling detection is explicitly out of scope for MVP.

## 5. Targeted debate protocol (max 2 rounds, fixed schema)

Only models that disagree on a specific `decision_id` participate. Message schema per turn:

```json
{
  "round": 1,
  "model": "string",
  "decision_id": "string",
  "claim": "max 100 words",
  "cited_requirement": "requirement id or null",
  "tradeoff_accepted": "max 50 words",
  "condition_to_change_position": "max 50 words"
}
```

- Round 1: each disputing model states its claim in this schema.
- Round 2: each disputing model may revise its `value` (logged as a position change, with the round-1 claim kept for audit) or hold. No round 3 — after round 2, hand off to Claude as arbiter.
- Claude (arbiter) receives both rounds' structured messages, applies the deterministic scoring rubric below, and either resolves the decision or marks it `HUMAN_DECISION_REQUIRED`.

**Before scoring runs at all:** if every disputing model's round-2 `value` converges (same value after normalization/canonicalization — the same comparison the Section 4 diff uses for CONSENSUS), the decision is resolved via debate convergence, not scored. Citation checking exists to break a tie between positions that still disagree; running it on values that already match is a category error, not a scoring problem to get right. This produces its own status (`resolved_via_debate`), distinct from both the initial-diff CONSENSUS and an arbiter-scored win, so the UI can show that this one took real back-and-forth.

**Deterministic scoring rubric (for Claude's resolution — this constrains the arbiter, it's not just Claude's free judgment):**
- +1 point per requirement citation that's actually present in the shared context (verify the ID exists — don't trust the model's claim uncritically).
- +1 point if the position didn't change between round 1 and round 2 unprompted by new information (mild penalty against contradictions logged elsewhere, not scored here).
- ~~−1 point for any `assumption` that contradicts a stated constraint in the shared context (this is a hard check, not a judgment call — flag it as a detected error, not a debate point).~~

  **REMOVED.** Implemented as a keyword-overlap check: a model's assumption/claim/tradeoff/condition text was flagged if it shared a significant word with a negative requirement's negated clause. A 12-task, 29-instance fresh-domain validation batch (see project history — the batch covered caching, auth, API versioning, IoT ingestion, notifications, multi-tenancy, rate limiting, job processing, analytics, and schema migration, each with its own negative-constraint wording) found a **0% genuine-catch rate**. Every flagged case fell into one of three failure modes a keyword check cannot avoid by construction, not by insufficient tuning:
  1. **Affirmation misread as violation** — a model explicitly states it satisfies the constraint ("this directly satisfies req_2," "ensuring events are only sent after consent is granted") and gets penalized for using the requirement's own vocabulary to describe compliance.
  2. **Domain-noun-as-trigger** — the requirement's grammatical subject (e.g. "no *tenant* may query...", "a user must never receive a duplicate *notification*...") is also the domain's central noun, so it fires on nearly every sentence in that domain regardless of relevance. A hardcoded exclusion list fixed this for one prior batch's vocabulary ("service", "accept") but cannot generalize — every new domain mints its own trigger word, so this is whack-a-mole, not a fixable bug.
  3. **Opponent-critique misattribution** — model B argues model A's design risks a violation ("a fixed visibility timeout... directly violates req_2") as part of a substantive technical critique of A's proposal, and the penalty attaches to B, not A, since the check has no sense of who is being described.

  Two of these (1 and 3) require distinguishing affirmation from negation and self from opponent — reasoning a keyword-overlap check cannot do by construction. This is exactly the gap Phase 4's evidence-verification item is meant to eventually close with something that can actually reason about violation vs. compliance; until then, this rubric stays citation-only. A smaller, correct rubric beats a broader one that fabricates confidence.
- Highest score wins. Tie → `HUMAN_DECISION_REQUIRED`.

Claude should show its scoring inline in the output (which points were awarded and why), not just declare a winner — this is the traceability requirement from the plan.

## 6. Build phases — do these in order, stop after each for review

**Phase 1 — Core pipeline, no debate yet**
- Project/task/context data model (Postgres — use FastAPI + SQLAlchemy unless you have a strong reason not to; this matches the user's existing stack familiarity).
- Endpoint: submit shared context (text + PDF upload, simple text extraction, no vector DB yet) + a task.
- Call both proposing models in parallel with the schema from Section 3, validate responses.
- Run the diff (Section 4) and store per-decision results.
- Return a JSON response showing consensus/disputed/unopposed decisions. No UI yet — test via API calls.
- **Acceptance check:** submit a real small architecture question, confirm both models return valid structured decisions, confirm the diff correctly classifies at least one agreement and (if you pick a naturally contentious question) one disagreement.

**Phase 2 — Debate + arbitration**
- Implement the debate protocol (Section 5) for `DISPUTED` decisions only.
- Implement Claude-as-arbiter with the scoring rubric — make the rubric's point calculations actual code, not a prompt asking Claude to "score it," except for the final natural-language synthesis.
- **Acceptance check:** force a disagreement, confirm debate runs exactly 2 rounds max, confirm the arbiter's resolution shows the rubric scoring inline, confirm a tied score correctly produces `HUMAN_DECISION_REQUIRED` instead of an arbitrary pick.

**Phase 3 — Minimal UI**
- Per-decision board: each `decision_id` as a row/card showing status (✓ consensus / ⚠ debating / ⚠ human decision required / — unopposed), the winning value, and a drill-down to see the full model reasoning and debate transcript.
- No live streaming of raw model text by default — show "thinking" state, then populate the board when structured output lands. Drill-down can show the full reasoning text.
- Cost/token counter per task, visible to the user before and after running.

**Phase 4 — Deferred, do not build yet**
- Error ledger, external evidence verification (web search / code execution / benchmark lookups), full 5-model support, vector-DB retrieval, multiple debate rounds beyond 2, model failure/timeout fallback logic beyond a basic try/retry.

## 7. Explicit non-goals for this build

- Don't let Claude propose a competing solution in the same task it arbitrates (Section 2).
- Don't let any LLM decide "is this agreement or disagreement" — that's the deterministic diff (Section 4), full stop.
- Don't build a chat-transcript-first UI — the decision board is primary, transcripts are drill-down evidence.
- Don't add more than 2 proposing models until Phase 1–3 are validated against real tasks.
- Don't build evidence verification (web search, code execution) yet — that's Phase 4+, and the MVP's "contradicts a stated constraint" check (Section 5 rubric) only checks against the shared context that was already provided, nothing external.

## 8. Config / secrets

- API keys for each proposing model + Claude via `.env`, never hardcoded, never logged.
- Make the proposing-model list configurable (a simple list of provider/model-name pairs), so swapping which two models propose doesn't require code changes.

## 9. Before you start

Confirm your understanding of Section 2 (role separation) and Section 4 (diff thresholds) back to me in your own words before scaffolding the repo — these are the two decisions everything else depends on, and I'd rather catch a misunderstanding now than after Phase 1 is built.
