# AGENTS.md

This file provides guidance to AI coding agents working with this repository.

## Project Overview

Be More Agent is a single-file Python application (`agent.py`) that turns a Raspberry Pi into a conversational AI agent. It uses wake word detection, local speech-to-text (whisper.cpp), the Anthropic API (Claude) for LLM intelligence, and local text-to-speech (Piper TTS).

## Quick Start

```bash
# Setup (installs system deps, .bmo venv with piper-tts + anthropic SDK,
# downloads a Piper voice into piper/, builds whisper.cpp)
chmod +x setup.sh && ./setup.sh

# Configure API key
cp example.env .env
# Edit .env and set ANTHROPIC_API_KEY

# Run (auto-targets DSI display via DISPLAY=:0)
source .bmo/bin/activate && python agent.py
```

There are no tests or linting configured.

## Key Files

| File | Purpose |
|------|---------|
| `agent.py` | Entire application — single-file tkinter GUI + LLM agent (Piper voice loaded in-process via `PiperVoice.load()`) |
| `config.json` | User-facing settings (model name, voice, camera, system prompt) |
| `example.env` | Template for `.env` file (API keys) |
| `setup.sh` | One-shot installer — system deps, `.bmo` venv (incl. `piper-tts`), downloads a Piper voice, builds whisper.cpp, installs desktop autostart |
| `requirements.txt` | Python dependencies (includes `piper-tts`) |
| `start_agent.sh` | Launch script — activates `.bmo` venv and execs `agent.py` |
| `be-more-agent.desktop` | Autostart template; `setup.sh` substitutes the repo path and copies it to `~/.config/autostart/` |

## Architecture

### LLM Stack

```
agent.py  →  Anthropic Python SDK  →  Anthropic API  →  Claude (Sonnet/Opus/Haiku)
```

- **agent.py** uses the `anthropic` Python package to call the Anthropic Messages API
- **API key** is loaded from `.env` via `python-dotenv` (the `ANTHROPIC_API_KEY` env var)
- **Model** defaults to `claude-haiku-4-5`, configurable via `ANTHROPIC_MODEL` env var or `text_model` in `config.json`
- **Tools** use Anthropic native tool use (the `tools=` parameter with `TOOLS` schemas), not prompt-engineered JSON
- **Vision** is supported natively — the `capture_image` tool returns the photo as a base64 image block inside the `tool_result`, so Claude sees it with full conversation history

### Application Structure (agent.py)

The entire app is one class, `BotGUI`, with these sections marked by comment banners:

1. **Configuration & Constants** — `config.json` + `.env` loading, `DEFAULT_CONFIG`, Anthropic client setup, system prompt, `BotStates` enum
2. **GUI Class** — tkinter fullscreen app (800x480, targets DSI display via `DISPLAY=:0`), PNG face animations from `faces/[state]/`
3. **Tool Execution** (`execute_tool`) — runs tools Claude requests via native tool use: `get_time`, `search_web`, `capture_image` (returns the photo as an image block in the `tool_result`)
4. **Core Logic** (`safe_main_execution`) — Main loop: wake word/PTT → record → transcribe → chat → speak
5. **Chat & Respond** (`chat_and_respond`) — Streams responses with `tools=TOOLS`; on `stop_reason == "tool_use"` it executes the requested tools, appends `tool_result` blocks, and loops (up to `MAX_TOOL_ROUNDS`). TTS sentence buffer requires minimum length (`TTS_MIN_SENTENCE_LENGTH`, 80 chars) before flushing to avoid choppy short-fragment playback

### Threading Model

- Main loop runs in a daemon thread off tkinter's main thread
- TTS has its own worker thread with a queue
- Thinking sounds loop in short-lived daemon threads
- All state coordination uses `threading.Event` objects

### Vision Handling

Vision uses Claude's native multimodal support via the `capture_image` tool. When Claude requests a photo, the camera capture is base64-encoded and returned as an image block inside the `tool_result`, preserving the full conversation history:
```python
{"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "...", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}}
    ]}
]}
```

## Configuration Details

### .env (API keys — gitignored)

Copy `example.env` to `.env` and set your key:

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key |
| `ANTHROPIC_MODEL` | No | Override model (default: `claude-haiku-4-5`) |

### config.json (agent settings)

| Key | Default | Description |
|-----|---------|-------------|
| `text_model` | `"claude-haiku-4-5"` | Anthropic model name (overridden by `ANTHROPIC_MODEL` env var) |
| `voice_model` | `"piper/en_US-bmo-medium.onnx"` | Piper TTS voice model path (sibling `.onnx.json` must be next to it; both written by `python -m piper.download_voices`) |
| `whisper_model` | `"./whisper.cpp/models/ggml-base.en.bin"` | Whisper.cpp model file path |
| `whisper_threads` | `2` | CPU threads for whisper transcription |
| `audio_energy_threshold` | `0.002` | RMS energy below which audio is skipped without transcription |
| `chat_memory` | `true` | Persist conversation history to `memory.json` |
| `camera_rotation` | `0` | Rotate camera captures (0/90/180/270) |
| `system_prompt_extras` | `""` | Appended to the base system prompt |
| `input_device` | `null` | Microphone selection — device index, name substring, or `null` for system default |
| `input_sample_rate` | `null` | Preferred input sample rate; verified with `sd.check_input_settings()`, falls back through 48000/44100/32000/16000 |

## External Dependencies (not in repo)

| Tool | Location | Installed By |
|------|----------|-------------|
| whisper.cpp | `./whisper.cpp/build/bin/whisper-cli` (model/threads configurable via `config.json`) | `setup.sh` |
| Piper TTS | [`piper-tts`](https://github.com/OHF-Voice/piper1-gpl) Python package; voice files in `piper/` | `setup.sh` (`pip install` + `python -m piper.download_voices --data-dir piper <voice>`) |
| OpenWakeWord | `wakeword.onnx` | `setup.sh` |

## Display

The GUI targets the Raspberry Pi's DSI touchscreen (800x480). `agent.py` sets `os.environ.setdefault("DISPLAY", ":0")` before importing tkinter, so it works from SSH or systemd without extra config. Override by setting `DISPLAY` before launching.

`setup.sh` installs a desktop autostart entry (`~/.config/autostart/be-more-agent.desktop`) that runs `start_agent.sh` when the desktop session starts. Delete that file to disable autostart.

## Common Tasks

### Changing the Claude model
Set `ANTHROPIC_MODEL` in `.env` or update `text_model` in `config.json`. Available models include `claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-opus-4-8`.
