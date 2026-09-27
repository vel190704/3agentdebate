"""Regression test: board form submission with an unselected file input.

Bug (found via a real user submission through the board's HTML form, not
the JSON API): a browser's <input type="file"> left unselected still
submits a real (non-None) UploadFile inside the multipart/form-data body -
empty filename, zero-byte content - it does not omit the field. The
original create_task() code checked only `if file is not None:`, which is
true for this case, so it called extract_text_from_pdf(b"") and pypdf
raised EmptyFileError, uncaught, producing a raw 500 Internal Server Error
with no board-visible message.

This is the same *shape* of bug as the very first issue in this project
(Phase 1's "requirements must be a JSON array of strings" error): a
defensive-parsing gap the board's real HTML form surfaces that direct
JSON-API/curl testing never exercised, because no test in this project's
history ever included a file field in a request at all. Fixed in
app/routers/tasks.py by additionally requiring a non-empty filename and
non-empty content before attempting to parse a PDF.

The task content below (REAL_TASK_DESCRIPTION/REAL_CONTEXT_TEXT/
REAL_REQUIREMENTS) is the actual real-world submission that triggered the
500, not a synthetic minimal repro - a vulnerability-scanner server-
reachability design review, kept verbatim as evidence.

Run directly (no pytest harness set up in this project):
    .venv/bin/python tests/test_regression_empty_file_upload.py

Uses mocked LLM clients (no real API cost) - this test is about the file-
handling code path, not about validating real model output. See the board
for the real, live pipeline run against this same content.
"""

import asyncio
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from starlette.datastructures import UploadFile

from app.db import SessionLocal
from app.routers.tasks import create_task


async def _fake_framing(self, system_prompt, user_prompt):
    return (
        json.dumps(
            {
                "task_id": "x",
                "decisions": [
                    {"decision_id": "d1", "dimension": "D1", "description": "d", "value_options": []}
                ],
            }
        ),
        10,
        10,
    )


async def _fake_proposer(self, system_prompt, user_prompt):
    return (
        json.dumps(
            {
                "task_id": "x",
                "model": self.model,
                "decisions": [
                    {
                        "decision_id": "d1",
                        "dimension": "D1",
                        "value": "same",
                        "confidence": 0.9,
                        "reasoning": "r",
                        "cited_requirements": [],
                        "assumptions": [],
                    }
                ],
            }
        ),
        10,
        10,
    )



# The actual real-world input that triggered the original 500: a
# multi-paragraph task description, a long shared-context block with
# history/constraints, and 6 requirements each a full sentence on its own
# line - submitted through the board's HTML form with the optional file
# input left unselected. Kept verbatim as evidence, same as this project's
# other real-input regression artifacts.
REAL_TASK_DESCRIPTION = """Review the server-reachability check in an internal Python vulnerability
scanner service (run-once, scheduled every 15 minutes via Task Scheduler/
cron) and recommend whether the current design is sound enough to present
and roll out, or whether it needs further hardening before production use.

The scanner watches IP-named folders written by a separate remediation
pipeline. If a vulnerability sits unfixed past a 12-hour threshold, it
emails the team with the VIT number(s), the server's IP address, and
whether the server itself might be unreachable - using deliberately
hedged language ("server might be down") rather than a confident claim,
since the check cannot use credentials."""

REAL_CONTEXT_TEXT = """Constraints: no SSH/WinRM credentials available for the check (simplicity
requirement); must not add new CLI flags or restructure the existing
scanner.py; must never suppress an alert based on reachability, only
annotate it (fail-safe-open philosophy used throughout this codebase).

History: we initially tried ping (ICMP) alone - too many false negatives,
since the company's own firewall silently drops ICMP to genuinely healthy
servers. We then tried a raw TCP connect alone - too many false
positives, since a firewall/NAT device would complete a TCP handshake on
behalf of a server that was actually down. We also prototyped an
application-layer check (WS-Man/Test-WSMan) which caught the TCP false
positives but introduced its own failure mode: some genuinely-down
servers returned a fabricated/stub WS-Man response (OS version reported
as literal 0.0.0), which we now detect and treat as UNKNOWN rather than
UP. We explored going deeper (SMB2 negotiate, RPC Endpoint Mapper bind)
for even stronger evidence, but paused that because hand-crafted
SMB/RPC packets risk being flagged by IDS/IPS as reconnaissance activity
from our own scanning host.

Current decision: dropped ping entirely, using TCP-connect only on the
WinRM port (5985), with hedged "might be up / might be down" language in
the alert email rather than an assertive claim.

Open question: whether the alert should also include a resolved server
name (hostname), separate from the IP address - currently only IP exists
in the data model; pending confirmation from a teammate on whether this
is actually required."""

REAL_REQUIREMENTS = [
    "Assess whether TCP-connect-only reachability (no ping, no WS-Man/SMB/RPC) is defensible for a production alerting system, or whether it's a regression worth flagging",
    "Identify any remaining false-positive or false-negative scenarios this design doesn't account for",
    "Evaluate whether the hedged \"might be up/down\" wording is the right level of certainty given the underlying check, or too weak/too strong",
    "Recommend whether pursuing WS-Man-only (no SMB/RPC) as a middle ground is worth the added complexity given the IDS/IPS risk",
    "Advise on the hostname vs. IP-only question - is a resolved server name meaningfully useful for on-call triage, or is IP address sufficient",
    "Flag any other design or security concern in this reachability approach we haven't already surfaced",
]


async def main() -> None:
    db = SessionLocal()

    # Exactly what a browser sends for an unselected <input type="file">:
    # the field is present, filename is empty, content is zero bytes.
    empty_file = UploadFile(filename="", file=io.BytesIO(b""))

    with patch("app.llm.anthropic_client.AnthropicClient.complete_json", _fake_framing), patch(
        "app.llm.openai_compatible_client.OpenAICompatibleClient.complete_json", _fake_proposer
    ):
        result = await create_task(
            project_name="Epicure Vulnerability Scanner — Reachability & Alerting Design",
            task_description=REAL_TASK_DESCRIPTION,
            context_text=REAL_CONTEXT_TEXT,
            requirements=json.dumps(REAL_REQUIREMENTS),
            file=empty_file,
            db=db,
        )

    ok = result["task_id"] is not None and len(result["decisions"]) == 1
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] empty file input no longer crashes create_task - task_id={result['task_id']}, decisions={len(result['decisions'])}")
    assert ok


if __name__ == "__main__":
    asyncio.run(main())
