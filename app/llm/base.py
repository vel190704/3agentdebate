import asyncio
from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Tuple, TypeVar

import anthropic
import openai

from ..config import settings


class LLMClient(ABC):
    @abstractmethod
    async def complete_json(self, system_prompt: str, user_prompt: str) -> Tuple[str, int, int]:
        """Returns (raw_text_response, prompt_tokens, completion_tokens)."""
        raise NotImplementedError


# Transient, retry-worthy failures on either provider SDK: the request never
# got a real answer, so trying again might work. Deliberately narrow -
# everything else (401 bad key, 400 malformed request, 402 insufficient
# balance, 404, ...) fails immediately, since retrying a non-transient error
# just delays the same inevitable failure. This is a separate concern from
# each pipeline step's own JSON-validation retry (max_json_retries): that
# retries a parseable-but-wrong response; this retries a response that never
# arrived at all.
RETRYABLE_EXCEPTIONS = (
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)

_BACKOFF_SECONDS = [1, 2]

T = TypeVar("T")


async def call_with_transport_retry(fn: Callable[[], Awaitable[T]]) -> T:
    """Calls fn(), retrying up to settings.llm_transport_max_retries additional
    times (3 total attempts by default) on RETRYABLE_EXCEPTIONS only, with a
    fixed 1s-then-2s backoff. Any other exception - including a 402
    insufficient-balance error, which surfaces as a plain APIStatusError on
    both SDKs, not a named subclass - propagates immediately on first
    occurrence.
    """
    last_error: Exception = RuntimeError("call_with_transport_retry: no attempt was made")
    max_attempts = settings.llm_transport_max_retries + 1
    for attempt in range(max_attempts):
        try:
            return await fn()
        except RETRYABLE_EXCEPTIONS as e:
            last_error = e
            if attempt < max_attempts - 1:
                delay = _BACKOFF_SECONDS[min(attempt, len(_BACKOFF_SECONDS) - 1)]
                await asyncio.sleep(delay)
    raise last_error
