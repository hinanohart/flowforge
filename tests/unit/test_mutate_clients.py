"""Tests for HF / local-Qwen client helpers."""

import pytest

from flowforge.mutate.hf_api_client import HfApiClient, _parse_first_json_block
from flowforge.mutate.local_qwen import LocalQwenClient


def test_parse_first_json_block_basic():
    text = 'garbage prefix {"a":1, "b": [1,2,3]} trailing'
    parsed = _parse_first_json_block(text)
    assert parsed == {"a": 1, "b": [1, 2, 3]}


def test_parse_first_json_block_nested():
    text = '{"outer": {"inner": 7}}'
    parsed = _parse_first_json_block(text)
    assert parsed["outer"]["inner"] == 7


def test_parse_first_json_block_no_braces():
    with pytest.raises(ValueError):
        _parse_first_json_block("no json here")


def test_parse_first_json_block_unbalanced():
    with pytest.raises(ValueError):
        _parse_first_json_block("{unbalanced")


def test_hf_client_requires_token(monkeypatch):
    monkeypatch.delenv("HF_API_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    client = HfApiClient(model_id="some/model")
    with pytest.raises(RuntimeError):
        client._ensure_client()


def test_local_qwen_constructor_does_not_load():
    # ensure_model is lazy; bare constructor must not import torch
    c = LocalQwenClient(model_id="Qwen/Qwen2.5-Coder-7B-Instruct")
    assert c._model is None
