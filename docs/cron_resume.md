# Cron-driven `flowforge auto` resume (bare Linux only)

`flowforge auto --session-bound` is foreground-only by design (see README §
Honest claims — WSL2 `vmIdleTimeout` defaults to 60 s, so background `nohup`
processes die quickly). On a bare Linux box with no idle-out, you can drive
the state machine to completion across many invocations via cron.

## Sample cron entry

```cron
# Re-enter the state machine every 6 hours; the orchestrator persists
# .flowforge/state.json after each transition so the next invocation
# resumes from the last completed state.
0 */6 * * *  cd $HOME/projects/flowforge && \
             ./.venv/bin/flowforge auto --session-bound --max-steps 4 \
             >> $HOME/projects/flowforge/.flowforge/cron.log 2>&1
```

`--max-steps 4` keeps each invocation short (one state per cron tick is
plenty for S0/S5/S6/S7 and one generation per tick for S3). The orchestrator
exits cleanly on `done`, `hitl_required`, or when `--max-steps` is reached.

## Sample systemd unit (Linux ≥ 232)

```ini
# ~/.config/systemd/user/flowforge.service
[Unit]
Description=FlowForge state-machine resume

[Service]
Type=oneshot
WorkingDirectory=%h/projects/flowforge
ExecStart=%h/projects/flowforge/.venv/bin/flowforge auto --session-bound --max-steps 4

# ~/.config/systemd/user/flowforge.timer
[Unit]
Description=Resume FlowForge every 6 h

[Timer]
OnBootSec=10min
OnUnitActiveSec=6h
Persistent=true

[Install]
WantedBy=timers.target
```

Enable with:

```bash
systemctl --user enable --now flowforge.timer
```

## WSL2 fallback

If you must run on WSL2, raise `vmIdleTimeout` in `%UserProfile%\.wslconfig`:

```ini
[wsl2]
vmIdleTimeout=21600000   # 6 hours in milliseconds
```

After saving, `wsl --shutdown` once, then re-enter the distro. `flowforge
auto --session-bound` will then survive idle windows up to 6 h.

## Watching the HITL flag

If the orchestrator emits `.flowforge/HITL_REQUIRED`, the cron / systemd
loop will keep no-op'ing until you remove the file. Inspect the message:

```bash
cat ~/projects/flowforge/.flowforge/HITL_REQUIRED
```

and fix the upstream issue before deleting the flag.
