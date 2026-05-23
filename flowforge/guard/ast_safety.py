"""AST-level safety validator for LLM-generated mutation code.

**v0.1.0 scope**: Mutations operate on fixed templates with coefficient-only
changes; arbitrary code execution is therefore not part of the runtime path.
This module provides a static validator that v0.2 (full function mutation) will
require. It deliberately does NOT call exec / compile on the validated source —
v0.1.0 instantiates concrete template functions via direct Python imports
(see `flowforge.evolve.templates`).

Rules enforced (architecture § 4 / F3):
- only a whitelist of imports allowed (math, typing)
- no file I/O (open, os, pathlib, shutil)
- no network (socket, urllib, requests, http)
- no subprocess / dynamic exec / compile / eval / __import__
- no global mutation (no `global`, `nonlocal`)
- only `def` of named templates allowed; body length ≤ MAX_BODY_STMTS

The validator returns (ok: bool, reasons: list[str]).
"""

from __future__ import annotations

import ast

ALLOWED_IMPORT_MODULES = frozenset({"math", "typing"})
FORBIDDEN_CALLS = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
    }
)
FORBIDDEN_ATTR_PREFIXES = (
    "os.",
    "sys.",
    "subprocess.",
    "socket.",
    "urllib.",
    "shutil.",
    "pathlib.",
    "requests.",
    "http.",
    "ctypes.",
    "importlib.",
)
MAX_BODY_STMTS = 50


def validate(
    source: str, allowed_def_names: tuple[str, ...] | None = None
) -> tuple[bool, list[str]]:
    """Statically validate `source` against safety rules.

    Args:
        source: Python source code.
        allowed_def_names: if provided, restricts top-level `def`s to these names
            (e.g. ("schedule", "shape_reward")).

    Returns:
        (ok, reasons) — reasons is empty iff ok.
    """
    reasons: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, [f"SyntaxError: {e}"]

    def _attr_chain(node: ast.AST) -> str:
        parts: list[str] = []
        cur: ast.AST = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))

    def_count = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] not in ALLOWED_IMPORT_MODULES:
                        reasons.append(f"forbidden import: {alias.name}")
            else:
                mod = node.module or ""
                if mod.split(".")[0] not in ALLOWED_IMPORT_MODULES:
                    reasons.append(f"forbidden from-import: {mod}")
        elif isinstance(node, ast.FunctionDef):
            def_count += 1
            if allowed_def_names is not None and node.name not in allowed_def_names:
                reasons.append(
                    f"forbidden top-level def: {node.name} (allowed: {allowed_def_names})"
                )
            if sum(1 for _ in ast.walk(node)) > 400:
                reasons.append(f"def {node.name} too large (AST nodes > 400)")
            if len(node.body) > MAX_BODY_STMTS:
                reasons.append(
                    f"def {node.name} body too long ({len(node.body)} > {MAX_BODY_STMTS})"
                )
        elif isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign)):
            continue  # constants / type aliases ok
        else:
            reasons.append(f"forbidden top-level node: {type(node).__name__}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                reasons.append(f"forbidden call: {node.func.id}()")
            if isinstance(node.func, ast.Attribute):
                chain = _attr_chain(node.func)
                if any(chain.startswith(p) for p in FORBIDDEN_ATTR_PREFIXES):
                    reasons.append(f"forbidden attr call: {chain}()")
        elif isinstance(node, ast.Global):
            reasons.append("`global` not allowed")
        elif isinstance(node, ast.Nonlocal):
            reasons.append("`nonlocal` not allowed")
        elif isinstance(node, ast.Try) and any(
            isinstance(h.type, ast.Name) and h.type.id == "BaseException" for h in node.handlers
        ):
            reasons.append("`except BaseException` not allowed (would mask sandbox aborts)")
        elif (
            isinstance(node, ast.While)
            and isinstance(node.test, ast.Constant)
            and node.test.value is True
        ):
            reasons.append("`while True:` not allowed (use bounded loops)")

    if def_count == 0:
        reasons.append("source defines no functions")

    return (len(reasons) == 0), reasons


def self_test() -> tuple[bool, list[str]]:
    """Verifies the validator itself; called by bootstrap.sh."""
    failures: list[str] = []

    ok_src = "def schedule(t):\n    return 1.0 - t\n"
    ok, _ = validate(ok_src, allowed_def_names=("schedule",))
    if not ok:
        failures.append("valid source rejected")

    bad_src = "import os\ndef schedule(t):\n    return os.getpid()\n"
    ok, reasons = validate(bad_src, allowed_def_names=("schedule",))
    if ok or not any("os" in r for r in reasons):
        failures.append("forbidden import not caught")

    eval_src = "def schedule(t):\n    return 0.0\nx = sorted([1])\n"
    # `sorted` is fine — this is a smoke test that benign code passes
    ok, _ = validate(eval_src, allowed_def_names=("schedule",))
    if not ok:
        failures.append("benign code rejected")

    forbidden_call_src = "def schedule(t):\n    return float(t)\nimport math\nv = math.sin(1.0)\n"
    ok, _ = validate(forbidden_call_src, allowed_def_names=("schedule",))
    if not ok:
        failures.append("math import wrongly rejected")

    return (len(failures) == 0), failures
