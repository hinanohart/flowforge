"""HuggingFace Inference API client for the bootstrap (S0/S1) phases.

Used only when the local Qwen backend cannot be initialised (e.g., on
CPU-only hosts during testing). Production runs (S3) must use the local
backend; the router will raise if HF API is asked to mutate during S3.
"""

from __future__ import annotations

import logging
import os
from typing import Any


log = logging.getLogger(__name__)


class HfApiClient:
    """Lazy client that calls the HF Inference API for short JSON completions.

    Reads the token from the `HF_API_TOKEN` env var; the token is never logged
    or persisted by this class.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        timeout_s: float = 30.0,
        max_new_tokens: int = 512,
    ):
        self.model_id = model_id
        self.timeout_s = timeout_s
        self.max_new_tokens = max_new_tokens
        self._client: Any = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from huggingface_hub import InferenceClient
        except ImportError as e:
            raise RuntimeError(
                "huggingface_hub not installed; run `pip install huggingface_hub`"
            ) from e
        token = os.environ.get("HF_API_TOKEN") or os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_API_TOKEN env var not set; refusing to call HF API anonymously")
        self._client = InferenceClient(model=self.model_id, token=token, timeout=self.timeout_s)

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """Send a chat-style request expecting JSON. Returns the parsed dict.

        Raises ValueError on malformed output. The router catches and falls
        back to a random mutation.
        """
        self._ensure_client()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            text = (
                self._client.chat_completion(
                    messages=messages, max_tokens=self.max_new_tokens, temperature=0.7
                )
                .choices[0]
                .message.content
            )
        except (RuntimeError, ValueError, OSError, ConnectionError) as e:
            raise ValueError(f"HF API error: {e}") from e
        return parse_first_json_block(text)
