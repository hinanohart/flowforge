#!/usr/bin/env bash
# Sample wrapper for cron-driven resume on bare Linux hosts.
# See docs/cron_resume.md for the full setup.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$PROJECT_ROOT/.flowforge/cron.log"
mkdir -p "$(dirname "$LOG")"

# If the venv is missing the user has not installed yet; refuse rather
# than touch any state.
VENV="$PROJECT_ROOT/.venv"
if [ ! -x "$VENV/bin/flowforge" ]; then
    echo "$(date -Iseconds) cron_resume: .venv/bin/flowforge missing — run pip install first" >> "$LOG"
    exit 0
fi

# If a HITL flag exists, do nothing.
if [ -f "$PROJECT_ROOT/.flowforge/HITL_REQUIRED" ]; then
    echo "$(date -Iseconds) cron_resume: HITL_REQUIRED set — skipping" >> "$LOG"
    exit 0
fi

cd "$PROJECT_ROOT"
"$VENV/bin/flowforge" auto --session-bound --max-steps 4 >> "$LOG" 2>&1
