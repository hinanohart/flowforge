"""Local Qwen-Coder mutation backend for headless 24 GB GPU.

S3 (main evolve loop) requires this backend; HF API is forbidden during S3.
The recipe below loads the model once and reuses it across generations.

The exact Qwen variant is **chosen at bootstrap time** by querying the HF Hub
for the most-downloaded Coder model and writing
`.flowforge/qwen_candidates.json`. The orchestrator passes a concrete model_id
to this class; we never hard-code the version.
"""

from __future__ import annotations

import logging
from typing import Any

from flowforge.mutate._json_extract import parse_first_json_block


log = logging.getLogger(__name__)


class LocalQwenClient:
    """Lazy-loaded local Qwen-Coder for JSON completions.

    On non-GPU hosts (`torch.cuda.is_available() == False`) the constructor
    raises RuntimeError so the orchestrator can fall back gracefully in S0/S1
    only (S3 will refuse to run).
    """

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 512,
        require_gpu: bool = True,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.require_gpu = require_gpu
        self._tokenizer: Any = None
        self._model: Any = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "transformers + torch required; install via `pip install flowforge[llm]`"
            ) from e
        if self.require_gpu and not torch.cuda.is_available():
            raise RuntimeError("local Qwen requires CUDA GPU; none detected")
        log.info("loading local Qwen model: %s", self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        kwargs: dict[str, Any] = {"trust_remote_code": False}
        if torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        self._ensure_model()
        prompt = self._tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        out = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        text = self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )
        return parse_first_json_block(text)
