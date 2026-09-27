from typing import Tuple

from openai import AsyncOpenAI

from ..config import settings
from .base import LLMClient, call_with_transport_retry


class OpenAICompatibleClient(LLMClient):
    """Works for any OpenAI-compatible chat completions API (OpenAI, DeepSeek, etc)
    by swapping base_url/model - this is what lets the proposer roster in
    CLAUDE.md Section 8 stay config-driven instead of hardcoded per provider.
    """

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=settings.llm_call_timeout_seconds)

    async def complete_json(self, system_prompt: str, user_prompt: str) -> Tuple[str, int, int]:
        async def _call():
            return await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )

        response = await call_with_transport_retry(_call)
        text = response.choices[0].message.content or ""
        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        return text, prompt_tokens, completion_tokens
