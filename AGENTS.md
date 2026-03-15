# AGENTS.md

This file provides guidance to AI coding agents working with this repository.

## Project Overview

Be More Agent is a single-file Python application (`agent.py`) that turns a Raspberry Pi into a local, offline conversational AI agent. It uses wake word detection, speech-to-text, a local LLM via llama.cpp + llama-swap, and text-to-speech — all running on-device.

## Quick Start

```bash
# Setup (builds llama.cpp, downloads models, installs deps, enables llama-swap service)
chmod +x setup.sh && ./setup.sh

# Run (auto-targets DSI display via DISPLAY=:0)
source venv/bin/activate && python agent.py
```

There are no tests or linting configured.

## Key Files

| File | Purpose |
|------|---------|
| `agent.py` | Entire application — single-file ~920-line tkinter GUI + LLM agent |
| `config.json` | User-facing settings (model name, voice, camera, system prompt) |
| `config.yaml` | llama-swap config — defines how to launch llama-server per model |
| `llama-swap.service` | systemd unit file for the llama-swap proxy service |
| `setup.sh` | One-shot installer — system deps, llama.cpp build, model download, service setup |
| `requirements.txt` | Python dependencies |

## Architecture

### LLM Stack

```
agent.py  →  OpenAI Python client  →  llama-swap (:8080)  →  llama-server  →  Qwen3.5-4B GGUF
```

- **agent.py** uses the `openai` Python package with `base_url="http://localhost:8080/v1"`
- **llama-swap** is a Go binary running as a systemd service on port 8080. It proxies OpenAI-compatible requests and auto-starts/stops `llama-server` based on the requested model name.
- **llama-server** (from llama.cpp) runs the actual inference. Built from source at `./llama.cpp/build/bin/llama-server`.
- **Qwen3.5-4B** handles both text and vision via a single model (`models/Qwen3.5-4B-Q4_K_M.gguf`) plus a multimodal projector (`models/mmproj-F16.gguf`).

### Application Structure (agent.py)

The entire app is one class, `BotGUI`, with these sections marked by comment banners:

1. **Configuration & Constants** — `config.json` loading, `DEFAULT_CONFIG`, LLM client setup, system prompt, `BotStates` enum
2. **GUI Class** — tkinter fullscreen app (800x480, targets DSI display via `DISPLAY=:0`), PNG face animations from `faces/[state]/`
3. **Action Router** (`execute_action_and_get_result`) — JSON action parsing, three tools: `get_time`, `search_web`, `capture_image`
4. **Core Logic** (`safe_main_execution`) — Main loop: wake word/PTT → record → transcribe → chat → speak
5. **Chat & Respond** (`chat_and_respond`) — Streaming LLM calls, action mode vs chat mode detection, vision via base64 images

### Threading Model

- Main loop runs in a daemon thread off tkinter's main thread
- TTS has its own worker thread with a queue
- Thinking sounds loop in short-lived daemon threads
- All state coordination uses `threading.Event` objects

### Vision Handling

Vision uses the same Qwen3.5-4B model. Images are base64-encoded and sent in OpenAI chat completions format:
```python
{"role": "user", "content": [
    {"type": "text", "text": "What do you see?"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
]}
```

## Configuration Details

### config.json (agent settings)

| Key | Default | Description |
|-----|---------|-------------|
| `text_model` | `"qwen3.5-4b"` | Model name — must match an ID in `config.yaml` |
| `voice_model` | `"piper/en_GB-semaine-medium.onnx"` | Piper TTS voice model path |
| `chat_memory` | `true` | Persist conversation history to `memory.json` |
| `camera_rotation` | `0` | Rotate camera captures (0/90/180/270) |
| `system_prompt_extras` | `""` | Appended to the base system prompt |
| `llm_base_url` | `"http://localhost:8080/v1"` | OpenAI-compatible API endpoint |

### config.yaml (llama-swap)

Defines model → llama-server command mapping. The `${PORT}` macro is auto-assigned by llama-swap. `ttl` controls how long (seconds) a model stays loaded after the last request.

## External Dependencies (not in repo)

| Tool | Location | Installed By |
|------|----------|-------------|
| llama.cpp | `./llama.cpp/build/bin/llama-server` | `setup.sh` (built from source) |
| llama-swap | `./llama-swap` | `setup.sh` (downloaded binary) |
| Qwen3.5-4B model | `models/Qwen3.5-4B-Q4_K_M.gguf` | `setup.sh` (from HuggingFace) |
| Vision projector | `models/mmproj-F16.gguf` | `setup.sh` (from HuggingFace) |
| whisper.cpp | `./whisper.cpp/build/bin/whisper-cli` | Manual / `setup.sh` |
| Piper TTS | `./piper/piper` | `setup.sh` |
| OpenWakeWord | `wakeword.onnx` | `setup.sh` |

## Display

The GUI targets the Raspberry Pi's DSI touchscreen (800x480). `agent.py` sets `os.environ.setdefault("DISPLAY", ":0")` before importing tkinter, so it works from SSH or systemd without extra config. Override by setting `DISPLAY` before launching.

## Common Tasks

### Changing the LLM model
1. Download a new GGUF to `models/`
2. Add a new entry in `config.yaml` with the llama-server command
3. Update `text_model` in `config.json` to match the new model ID
4. Restart llama-swap: `sudo systemctl restart llama-swap`

### Checking llama-swap status
```bash
sudo systemctl status llama-swap
journalctl -u llama-swap -f
curl http://localhost:8080/v1/models
```
