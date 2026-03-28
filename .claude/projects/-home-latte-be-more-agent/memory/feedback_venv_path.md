---
name: venv-path
description: The project's Python virtual environment is at .bmo/, not venv/
type: feedback
---

Use `.bmo` as the virtualenv directory, not `venv`.

**Why:** The user corrected this — the project uses `.bmo/bin/activate` for its Python environment.

**How to apply:** Any time you need to activate the venv or install packages, use `source /home/latte/be-more-agent/.bmo/bin/activate`.
