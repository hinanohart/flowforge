"""Permissively pull the first JSON object out of an LLM completion."""

from __future__ import annotations

import json
from typing import Any


def parse_first_json_block(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("empty completion")
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in completion")
    depth = 0
    end = -1
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        raise ValueError("unbalanced braces in completion")
    return json.loads(text[start:end])
