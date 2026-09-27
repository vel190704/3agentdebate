import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ProposerConfig:
    name: str
    provider: str
    model: str
    api_key: str
    base_url: str


@dataclass(frozen=True)
class Settings:
    database_url: str
    anthropic_api_key: str
    claude_model: str
    proposers: List[ProposerConfig] = field(default_factory=list)
    max_json_retries: int = 1
    # Per-try timeout for every LLM call (framing/proposer/debate/synthesis/
    # ledger extraction), and how many additional tries a transport failure
    # (timeout/connection error/429/5xx) gets before giving up - separate
    # from max_json_retries above, which retries a parseable-but-invalid
    # response, not a response that never arrived. See app/llm/base.py.
    llm_call_timeout_seconds: float = 45.0
    llm_transport_max_retries: int = 2


def _load_proposers() -> List[ProposerConfig]:
    """Reads PROPOSER_1_*, PROPOSER_2_*, ... until a numbered slot has no MODEL set.

    This is the mechanism that keeps the proposer list swappable via .env
    instead of hardcoded model names in the pipeline logic (CLAUDE.md Section 8).
    """
    proposers = []
    i = 1
    while True:
        prefix = f"PROPOSER_{i}_"
        model = os.getenv(prefix + "MODEL")
        if not model:
            break
        proposers.append(
            ProposerConfig(
                name=os.getenv(prefix + "NAME", f"proposer_{i}"),
                provider=os.getenv(prefix + "PROVIDER", "openai"),
                model=model,
                api_key=os.getenv(prefix + "API_KEY", ""),
                base_url=os.getenv(prefix + "BASE_URL", "https://api.openai.com/v1"),
            )
        )
        i += 1

    if len(proposers) < 2:
        raise RuntimeError(
            "At least two PROPOSER_N_MODEL entries must be configured (see .env.example)."
        )
    return proposers


def get_settings() -> Settings:
    return Settings(
        database_url=os.getenv("DATABASE_URL", "postgresql://localhost/fiveagent"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        claude_model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5"),
        proposers=_load_proposers(),
        llm_call_timeout_seconds=float(os.getenv("LLM_CALL_TIMEOUT_SECONDS", "45")),
        llm_transport_max_retries=int(os.getenv("LLM_TRANSPORT_MAX_RETRIES", "2")),
    )


settings = get_settings()
