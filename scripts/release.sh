#!/usr/bin/env bash
# Sample release helper. Run under the user's authority — this script does NOT
# execute automatically from the orchestrator (S7_release only writes a marker
# file). The user reviews the script, then runs it manually.
#
# Usage:
#   ./scripts/release.sh v0.1.0a1
#
# Prerequisites:
#   - gh CLI is logged in with `repo` scope
#   - working tree is clean and on `main`
#   - tests pass (.venv/bin/pytest tests/)
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <tag>     # e.g. v0.1.0a1" >&2
    exit 1
fi
TAG="$1"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 1. Sanity: state-machine reached the release_ready marker.
if [ ! -f ".flowforge/release_ready" ]; then
    echo "release_ready marker absent. Run \`flowforge auto\` until S7_release completes." >&2
    exit 2
fi

# 2. Sanity: tests pass.
.venv/bin/python -m pytest tests/ -q

# 3. Create the GitHub repo if missing (idempotent).
if ! gh repo view hinanohart/flowforge >/dev/null 2>&1; then
    gh repo create hinanohart/flowforge --public --source=. --remote=origin --push
else
    if ! git remote get-url origin >/dev/null 2>&1; then
        git remote add origin "https://github.com/hinanohart/flowforge.git"
    fi
    git push -u origin HEAD
fi

# 4. Tag.
git tag -a "$TAG" -m "FlowForge $TAG"
git push origin "$TAG"

# 5. Branch protection (admin bypass enabled so the solo author can patch).
gh api -X PUT "repos/hinanohart/flowforge/branches/main/protection" \
    --input - <<EOF
{
  "required_status_checks": null,
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
EOF

# 6. Create release notes.
gh release create "$TAG" --title "FlowForge $TAG" --notes-file - <<'NOTES'
Phase-0 release. See README for honest claims.

What's in v0.1.0a1:
- 8-state machine (S0..S7) with state.json persistence and 42-day hard cap
- Coefficient-only function-template evolution (polynomial / piecewise / cosine + potential / dense / sparse)
- Optuna and random-grid baselines
- LIBERO eval harness with bootstrap CI 95% (StubEnv fallback on CPU-only)
- Local-Qwen / HF API mutation router with sandboxing
- AST-safety validator (deny-list)

What's **not** in v0.1.0a1:
- Real LIBERO training has not been run by this release
- Pretrained checkpoints
- openpi pi0.5 fine-tuning weights (download separately per docs)
NOTES

echo "Release $TAG complete."
