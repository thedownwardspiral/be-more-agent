# AGENTS.md

This file provides guidance to AI coding agents working with this repository.

## Project Overview

Be More Agent is a single-file Python application (`agent.py`) that turns a Raspberry Pi into a conversational AI agent. It uses wake word detection, local speech-to-text (whisper.cpp), the Anthropic API (Claude) for LLM intelligence, and local text-to-speech (Piper TTS).

## Quick Start

```bash
# Setup (installs system deps, whisper.cpp, Piper TTS, .bmo venv)
chmod +x setup.sh && ./setup.sh

# Configure API key
cp example.env .env
# Edit .env and set ANTHROPIC_API_KEY

# Start persistent Piper TTS server (systemd)
sudo cp piper-tts.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now piper-tts

# Run (auto-targets DSI display via DISPLAY=:0)
source .bmo/bin/activate && python agent.py
```

There are no tests or linting configured.

## Key Files

| File | Purpose |
|------|---------|
| `agent.py` | Entire application — single-file ~920-line tkinter GUI + LLM agent |
| `piper_server.py` | Persistent Piper TTS HTTP server (keeps model loaded in memory) |
| `piper-tts.service` | systemd unit file for auto-starting the TTS server |
| `config.json` | User-facing settings (model name, voice, camera, system prompt) |
| `example.env` | Template for `.env` file (API keys) |
| `setup.sh` | One-shot installer — system deps, whisper.cpp, Piper TTS, `.bmo` venv |
| `requirements.txt` | Python dependencies |

## Architecture

### LLM Stack

```
agent.py  →  Anthropic Python SDK  →  Anthropic API  →  Claude (Sonnet/Opus/Haiku)
```

- **agent.py** uses the `anthropic` Python package to call the Anthropic Messages API
- **API key** is loaded from `.env` via `python-dotenv` (the `ANTHROPIC_API_KEY` env var)
- **Model** defaults to `claude-sonnet-4-20250514`, configurable via `ANTHROPIC_MODEL` env var or `text_model` in `config.json`
- **Vision** is supported natively — images are sent as base64-encoded content blocks

### Application Structure (agent.py)

The entire app is one class, `BotGUI`, with these sections marked by comment banners:

1. **Configuration & Constants** — `config.json` + `.env` loading, `DEFAULT_CONFIG`, Anthropic client setup, system prompt, `BotStates` enum
2. **GUI Class** — tkinter fullscreen app (800x480, targets DSI display via `DISPLAY=:0`), PNG face animations from `faces/[state]/`
3. **Action Router** (`execute_action_and_get_result`) — JSON action parsing, three tools: `get_time`, `search_web`, `capture_image`
4. **Core Logic** (`safe_main_execution`) — Main loop: wake word/PTT → record → transcribe → chat → speak
5. **Chat & Respond** (`chat_and_respond`) — Streaming Anthropic API calls, action mode vs chat mode detection, vision via base64 images. TTS sentence buffer requires minimum length (`TTS_MIN_SENTENCE_LENGTH`, 80 chars) before flushing to avoid choppy short-fragment playback

### Threading Model

- Main loop runs in a daemon thread off tkinter's main thread
- TTS has its own worker thread with a queue
- Thinking sounds loop in short-lived daemon threads
- All state coordination uses `threading.Event` objects

### Vision Handling

Vision uses Claude's native multimodal support. Images are base64-encoded and sent in Anthropic's content block format:
```python
{"role": "user", "content": [
    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
    {"type": "text", "text": "What do you see?"}
]}
```

## Configuration Details

### .env (API keys — gitignored)

Copy `example.env` to `.env` and set your key:

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key |
| `ANTHROPIC_MODEL` | No | Override model (default: `claude-sonnet-4-20250514`) |

### config.json (agent settings)

| Key | Default | Description |
|-----|---------|-------------|
| `text_model` | `"claude-sonnet-4-20250514"` | Anthropic model name (overridden by `ANTHROPIC_MODEL` env var) |
| `voice_model` | `"piper/en_GB-semaine-medium.onnx"` | Piper TTS voice model path |
| `whisper_model` | `"./whisper.cpp/models/ggml-base.en.bin"` | Whisper.cpp model file path |
| `whisper_threads` | `2` | CPU threads for whisper transcription |
| `audio_energy_threshold` | `0.002` | RMS energy below which audio is skipped without transcription |
| `chat_memory` | `true` | Persist conversation history to `memory.json` |
| `camera_rotation` | `0` | Rotate camera captures (0/90/180/270) |
| `system_prompt_extras` | `""` | Appended to the base system prompt |

## External Dependencies (not in repo)

| Tool | Location | Installed By |
|------|----------|-------------|
| whisper.cpp | `./whisper.cpp/build/bin/whisper-cli` (model/threads configurable via `config.json`) | `setup.sh` |
| Piper TTS | `./piper/piper` (via `piper_server.py` HTTP service on port 5111) | `setup.sh` |
| OpenWakeWord | `wakeword.onnx` | `setup.sh` |

## Display

The GUI targets the Raspberry Pi's DSI touchscreen (800x480). `agent.py` sets `os.environ.setdefault("DISPLAY", ":0")` before importing tkinter, so it works from SSH or systemd without extra config. Override by setting `DISPLAY` before launching.

## Common Tasks

### Changing the Claude model
Set `ANTHROPIC_MODEL` in `.env` or update `text_model` in `config.json`. Available models include `claude-sonnet-4-20250514`, `claude-opus-4-20250514`, `claude-haiku-4-5-20251001`.
