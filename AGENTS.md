# AGENTS.md

Agent instructions for this repository live in [CLAUDE.md](CLAUDE.md). Read it before making changes; it is kept current and this file is not.

Rules to know before touching the repo:

- It is a single-file app: everything is in `agent.py` (one `BotGUI` class). Keep it that way unless asked otherwise.
- There are no tests. Check changes with `uvx ruff check agent.py` and `python3 -m py_compile agent.py`.
- Never commit `.env`, `memory.json` or `agent.log*`: they hold the API key and conversation transcripts.
