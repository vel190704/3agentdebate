from typing import Tuple

from anthropic import AsyncAnthropic

from ..config import settings
from .base import LLMClient, call_with_transport_retry


class AnthropicClient(LLMClient):
    """Used only for the task-framing step (Phase 1) and, in Phase 2, arbitration.
    Never used to generate a competing proposal - see CLAUDE.md Section 2.
    """

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.client = AsyncAnthropic(api_key=api_key, timeout=settings.llm_call_timeout_seconds)

    async def complete_json(self, system_prompt: str, user_prompt: str) -> Tuple[str, int, int]:
        async def _call():
            return await self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

        response = await call_with_transport_retry(_call)
        text = "".join(block.text for block in response.content if block.type == "text")
        usage = response.usage
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
        return text, input_tokens, output_tokens
