# AGENTS.md

Agent instructions for this repository live in [CLAUDE.md](CLAUDE.md): read it before making changes.

The rules that matter most:

- Pure logic goes in `bmo/` with a test; `agent.py` holds the GUI, audio and thread code.
- Check changes with `uv run --group dev pytest` and `uvx ruff check agent.py bmo tests`.
- `.env`, `memory.json` and `agent.log*` hold the API key and conversation transcripts: they stay out of git.
