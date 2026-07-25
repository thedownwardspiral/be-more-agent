---
name: feedback_venv_path
description: The be-more-agent project uses uv with a .venv/ virtualenv (migrated from .bmo/ on 2026-07-25)
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 5c7d1ceb-30c6-4bf5-8969-324dd801288e
  modified: 2026-07-25T15:50:08.482Z
---

Use `uv` for all Python environment and package work in be-more-agent. The
virtualenv is `.venv/`, defined by `pyproject.toml` + `uv.lock`.

**Why:** The user originally corrected `venv` → `.bmo`, then on 2026-07-25 asked
to migrate to uv, replacing `.bmo/` with `.venv/` and `requirements.txt` with
`pyproject.toml`.

**How to apply:** Run commands with `uv run <cmd>` (e.g. `uv run agent.py`), or
use `.venv/bin/python` directly. Add or change dependencies by editing
`pyproject.toml` and running `uv sync` — never `pip install` into the venv.
Note `uv` lives at `~/.local/bin/uv`, which may not be on a non-login PATH.
`openwakeword` is pinned `<0.5` because 0.5+ needs tflite-runtime, which has no
CPython 3.13/aarch64 wheel.
