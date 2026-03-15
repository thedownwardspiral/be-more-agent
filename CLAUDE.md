# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Be More Agent is a single-file Python application (`agent.py`) that turns a Raspberry Pi into a local, offline conversational AI agent. It uses wake word detection, speech-to-text, a local LLM, and text-to-speech — all running on-device with no cloud APIs.

## Running the Project

```bash
# First-time setup (installs system deps, Piper TTS, Python venv, llama.cpp, llama-swap, models)
chmod +x setup.sh
./setup.sh

# Ensure llama-swap service is running
sudo systemctl status llama-swap

# Run the agent (auto-targets DSI display via DISPLAY=:0)
source venv/bin/activate
python agent.py
```

There are no tests or linting configured in this project.

## Architecture

Everything lives in `agent.py` — a single ~920-line script with one main class:

**`BotGUI`** — The entire application. It's a tkinter GUI that also manages all background threads. Key sections (marked with comment banners in the file):

1. **Configuration & Constants** (top of file) — Loads `config.json`, defines `BotStates` enum, LLM settings, and the system prompt that instructs the LLM to output JSON for tool actions vs plain text for chat. Creates an OpenAI-compatible client (`llm_client`) pointing at llama-swap.

2. **GUI Class** — Fullscreen tkinter app (800x480, designed for the Pi's DSI touchscreen). The script sets `DISPLAY=:0` via `os.environ.setdefault` before importing tkinter, so it renders to the DSI display even when launched via SSH or systemd. Loads PNG animation sequences from `faces/[state]/` directories. Face state changes based on bot state (idle, listening, thinking, speaking, error, capturing, warmup).

3. **Action Router** (`execute_action_and_get_result`) — Parses JSON actions from LLM output. Three tools: `get_time`, `search_web` (DuckDuckGo), `capture_image` (Pi camera via `rpicam-still`). Includes alias mapping (e.g., "google" → "search_web").

4. **Core Logic** (`safe_main_execution`) — Main loop: detect wake word or PTT → record audio → transcribe → chat → speak. Two recording modes: adaptive silence detection and push-to-talk (Enter key).

5. **Chat & Respond** (`chat_and_respond`) — Streams LLM response via OpenAI-compatible API (llama-swap → llama.cpp). During streaming, detects if output is JSON (action mode) or plain text (chat mode). For actions, executes the tool then sends results back to LLM for summarization. Vision requests send base64-encoded images in OpenAI message format.

**Key threading model:** The main loop runs in a daemon thread off the tkinter main thread. TTS has its own worker thread with a queue. Thinking sounds loop in short-lived threads. All state coordination uses `threading.Event` objects.

## LLM Stack

The LLM stack has three layers:

1. **llama-swap** — A Go binary running as a systemd service on port 8080. It exposes an OpenAI-compatible API and acts as a smart proxy. When a request arrives, it starts/stops the appropriate llama-server process based on the model name. Configured via `config.yaml`.

2. **llama-server** (from llama.cpp) — The actual inference server. Built from source at `./llama.cpp/build/bin/llama-server`. Launched and managed by llama-swap with automatic port assignment via the `${PORT}` macro.

3. **Qwen3.5-4B** — A single model that handles both text and vision. Uses Q4_K_M quantization (`models/Qwen3.5-4B-Q4_K_M.gguf`, 2.6 GB) with a separate multimodal projector (`models/mmproj-F16.gguf`, 642 MB) for image understanding.

The Python code communicates with this stack using the `openai` Python package, with `base_url` pointed at `http://localhost:8080/v1`.

## External Tool Chain (not in repo, installed by setup.sh)

- **llama.cpp** — Built from source at `./llama.cpp/build/bin/llama-server`
- **llama-swap** — Binary at `./llama-swap`, runs as systemd service, configured via `config.yaml`
- **Qwen3.5-4B** — GGUF model files in `models/` (downloaded from `unsloth/Qwen3.5-4B-GGUF`)
- **whisper.cpp** — Speech-to-text at `./whisper.cpp/build/bin/whisper-cli`
- **Piper TTS** — Text-to-speech at `./piper/piper`
- **OpenWakeWord** — Wake word detection from `wakeword.onnx`

## Configuration

`config.json` controls: text model name, voice model path, chat memory toggle, camera rotation, LLM base URL, and a system prompt extension. The script merges user config over `DEFAULT_CONFIG` defaults.

`config.yaml` is the llama-swap configuration that defines how to launch llama-server for each model, including GGUF paths, mmproj for vision, context size, and TTL.

## Display

**A desktop environment (X11/Wayland) is required.** The agent uses tkinter for its GUI, which needs a running display server. A headless/server-only Pi OS installation will fail with `couldn't connect to display ":0"`. If using Pi OS Lite, install `lightdm` and configure desktop autologin via `raspi-config`.

The target display is the Raspberry Pi's DSI touchscreen interface (800x480). The script sets `os.environ.setdefault("DISPLAY", ":0")` at the top of `agent.py` before any tkinter imports, ensuring the GUI renders to the DSI screen regardless of launch context (local terminal, SSH, or systemd service). Users can override by setting `DISPLAY` before running.

## Audio Handling

The agent auto-detects microphone/speaker sample rates and resamples on the fly (using scipy) to match hardware capabilities. This is critical for Pi hardware compatibility — Piper outputs at 22050 Hz but many Pi audio devices only support 48000 Hz.
