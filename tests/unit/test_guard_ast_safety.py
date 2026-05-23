"""Tests for AST safety validator."""

from flowforge.guard import ast_safety


def test_self_test_passes():
    ok, failures = ast_safety.self_test()
    assert ok, failures


def test_valid_schedule_passes():
    ok, reasons = ast_safety.validate(
        "def schedule(t):\n    return 1.0 - t\n", allowed_def_names=("schedule",)
    )
    assert ok, reasons


def test_forbidden_import_caught():
    ok, reasons = ast_safety.validate(
        "import os\ndef schedule(t):\n    return 0.0\n", allowed_def_names=("schedule",)
    )
    assert not ok
    assert any("os" in r for r in reasons)


def test_eval_call_caught():
    ok, reasons = ast_safety.validate(
        "def schedule(t):\n    return eval('1.0')\n", allowed_def_names=("schedule",)
    )
    assert not ok
    assert any("eval" in r for r in reasons)


def test_open_call_caught():
    ok, reasons = ast_safety.validate(
        "def f(t):\n    open('/etc/passwd')\n    return 0\n", allowed_def_names=("f",)
    )
    assert not ok
    assert any("open" in r for r in reasons)


def test_disallowed_def_name():
    ok, reasons = ast_safety.validate(
        "def evil(t):\n    return 0\n", allowed_def_names=("schedule",)
    )
    assert not ok
    assert any("evil" in r for r in reasons)


def test_no_functions_rejected():
    ok, reasons = ast_safety.validate("x = 1\n", allowed_def_names=("schedule",))
    assert not ok
    assert any("no functions" in r for r in reasons)


def test_syntax_error_rejected():
    ok, reasons = ast_safety.validate(
        "def schedule(t)\n    return 0\n", allowed_def_names=("schedule",)
    )
    assert not ok
    assert any("SyntaxError" in r for r in reasons)


def test_while_true_rejected():
    src = "def schedule(t):\n    while True:\n        return 0\n"
    ok, reasons = ast_safety.validate(src, allowed_def_names=("schedule",))
    assert not ok
    assert any("while True" in r for r in reasons)


def test_global_rejected():
    src = "x=1\ndef schedule(t):\n    global x\n    return x\n"
    ok, reasons = ast_safety.validate(src, allowed_def_names=("schedule",))
    assert not ok
    assert any("global" in r for r in reasons)


def test_subprocess_attr_rejected():
    src = "import math\ndef schedule(t):\n    return math.cos(t)\nimport subprocess  # noqa\n"
    # subprocess is also rejected as an import (not in allow-list)
    ok, _ = ast_safety.validate(src, allowed_def_names=("schedule",))
    assert not ok


def test_math_import_accepted():
    src = "import math\ndef schedule(t):\n    return math.cos(t)\n"
    ok, reasons = ast_safety.validate(src, allowed_def_names=("schedule",))
    assert ok, reasons
