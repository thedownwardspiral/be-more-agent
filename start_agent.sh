#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

# Bring .venv up to date with uv.lock first, so a `git pull` that adds a
# dependency doesn't leave the agent on a stale environment. The desktop
# session's PATH may lack ~/.local/bin, where the uv installer puts uv.
# Offline, or if the sync fails, start on the existing .venv anyway.
UV="$(command -v uv || true)"
if [ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ]; then
    UV="$HOME/.local/bin/uv"
fi
if [ -n "$UV" ]; then
    # --inexact keeps extras such as the dev group's pytest
    "$UV" sync --frozen --inexact --quiet || echo "start_agent.sh: uv sync failed; using the existing .venv" >&2
fi

exec .venv/bin/python agent.py
