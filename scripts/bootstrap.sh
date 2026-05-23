#!/usr/bin/env bash
# FlowForge bootstrap: openpi + pi0.5 checkpoint + Qwen latest + sandbox check.
# Exits 0 on success; on any blocking failure writes .flowforge/HITL_REQUIRED with reason.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FF_DIR="$PROJECT_ROOT/.flowforge"
LOG="$FF_DIR/bootstrap.log"
mkdir -p "$FF_DIR"
: > "$LOG"

log() { printf "[%s] %s\n" "$(date -Iseconds)" "$*" | tee -a "$LOG"; }

hitl() {
    local reason="$1"
    log "HITL_REQUIRED: $reason"
    printf "%s\n%s\n" "$(date -Iseconds)" "$reason" > "$FF_DIR/HITL_REQUIRED"
    exit 0   # graceful — orchestrator will detect HITL flag
}

log "FlowForge bootstrap start (project_root=$PROJECT_ROOT)"

# ---------- Step 1: GPU check ----------
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_LINE=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    log "GPU detected: $GPU_LINE"
    echo "$GPU_LINE" > "$FF_DIR/gpu_info.txt"
else
    log "WARN: nvidia-smi not found — CPU-only mode. Skeleton tests OK, training requires GPU."
    echo "cpu_only" > "$FF_DIR/gpu_info.txt"
    # not a hard fail; skeleton + unit tests work CPU-only
fi

# ---------- Step 2: Python + base deps ----------
log "Python $(python3 --version 2>&1)"
if ! python3 -c "import flowforge" 2>/dev/null; then
    log "Installing flowforge in editable mode"
    python3 -m pip install -e "$PROJECT_ROOT" >> "$LOG" 2>&1 || hitl "pip install -e . failed (see $LOG)"
fi

# ---------- Step 3: ShinkaEvolve ----------
if ! python3 -c "import shinka" 2>/dev/null; then
    log "Installing shinka-evolve"
    python3 -m pip install shinka-evolve >> "$LOG" 2>&1 || log "WARN: shinka-evolve install failed; evolve gen will be skipped"
fi

# ---------- Step 4: Optuna ----------
python3 -m pip install "optuna>=3.6" >> "$LOG" 2>&1 || log "WARN: optuna install failed"

# ---------- Step 5: huggingface_hub + list latest Qwen-Coder ----------
python3 -m pip install "huggingface_hub>=0.24" >> "$LOG" 2>&1 || hitl "huggingface_hub install failed"

python3 - <<'PY' >> "$LOG" 2>&1 || log "WARN: Qwen listing failed; will fallback to API mutation in S0 only"
import json, pathlib
from huggingface_hub import HfApi
api = HfApi()
out_path = pathlib.Path(".flowforge/qwen_candidates.json")
out_path.parent.mkdir(exist_ok=True)
try:
    models = api.list_models(author="Qwen", search="Coder", limit=50)
    cands = []
    for m in models:
        mid = m.id if hasattr(m, "id") else m.modelId
        cands.append({"id": mid, "downloads": getattr(m, "downloads", 0)})
    cands.sort(key=lambda x: x["downloads"], reverse=True)
    out_path.write_text(json.dumps(cands[:20], indent=2))
    print(f"Qwen-Coder candidates written: {len(cands)} models")
except Exception as e:
    print(f"Qwen listing exception: {e}")
PY

# ---------- Step 6: openpi (pi0.5 + LIBERO) — optional, large download ----------
OPENPI_DIR="$PROJECT_ROOT/external/openpi"
if [ ! -d "$OPENPI_DIR/.git" ]; then
    log "Cloning openpi (shallow)"
    mkdir -p "$PROJECT_ROOT/external"
    git clone --depth 1 https://github.com/Physical-Intelligence/openpi.git "$OPENPI_DIR" >> "$LOG" 2>&1 || log "WARN: openpi clone failed; policy/openpi_adapter will be stubbed"
fi

# pi0.5 LIBERO checkpoint is in gs:// bucket; we do NOT auto-download here (multi-GB, requires gsutil).
# Document the path; orchestrator will check on S0 entry.
cat > "$FF_DIR/openpi_paths.json" <<EOF
{
  "openpi_root": "$OPENPI_DIR",
  "pi05_libero_checkpoint": "gs://openpi-assets/checkpoints/pi05_libero",
  "note": "Download manually with: gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_libero ./external/openpi_assets/"
}
EOF

# ---------- Step 7: LIBERO simulator check ----------
if python3 -c "import libero" 2>/dev/null; then
    log "LIBERO installed"
else
    log "WARN: LIBERO not installed. To enable real eval: pip install libero (and follow libero-project.github.io)"
fi

# ---------- Step 8: Sandbox check ----------
python3 - <<'PY' >> "$LOG" 2>&1
from flowforge.guard import ast_safety
ok, reasons = ast_safety.self_test()
print(f"AST safety self-test: ok={ok}, reasons={reasons}")
if not ok:
    raise SystemExit(2)
PY
SANDBOX_RC=$?
if [ "$SANDBOX_RC" -ne 0 ]; then
    hitl "AST safety self-test failed; refusing to proceed"
fi

# ---------- Step 9: WSL2 vmIdleTimeout warning ----------
if grep -qi microsoft /proc/version 2>/dev/null; then
    log "WSL2 detected. vmIdleTimeout default is 60000 ms (60 s). For long unattended runs,"
    log "  set [wsl2] vmIdleTimeout=21600000 in %UserProfile%\\.wslconfig (6 h)"
    log "  or use systemd unit / cron from docs/cron_resume.md"
fi

# ---------- Step 10: Write done marker ----------
date -Iseconds > "$FF_DIR/bootstrap_done"
log "bootstrap complete"
