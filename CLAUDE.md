# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Be More Agent is a single-file Python application (`agent.py`) that turns a Raspberry Pi into a conversational AI agent. It uses wake word detection, local speech-to-text (whisper.cpp), the Anthropic API (Claude) for LLM intelligence, and local text-to-speech (Piper TTS). An internet connection is required for the Anthropic API.

## Running the Project

```bash
# First-time setup (installs system deps, Piper TTS, .bmo venv, whisper.cpp, anthropic SDK)
chmod +x setup.sh
./setup.sh

# Configure your Anthropic API key
cp example.env .env
# Edit .env and add your ANTHROPIC_API_KEY

# Install and start the persistent Piper TTS server
sudo cp piper-tts.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now piper-tts

# Run the agent (auto-targets DSI display via DISPLAY=:0)
source .bmo/bin/activate
python agent.py
```

There are no tests or linting configured in this project.

## Architecture

Everything lives in `agent.py` — a single ~920-line script with one main class:

**`BotGUI`** — The entire application. It's a tkinter GUI that also manages all background threads. Key sections (marked with comment banners in the file):

1. **Configuration & Constants** (top of file) — Loads `config.json` and `.env`, defines `BotStates` enum, LLM settings, and the system prompt that instructs the LLM to output JSON for tool actions vs plain text for chat. Creates an Anthropic client (`llm_client`) using the `ANTHROPIC_API_KEY` from environment.

2. **GUI Class** — Fullscreen tkinter app (800x480, designed for the Pi's DSI touchscreen). The script sets `DISPLAY=:0` via `os.environ.setdefault` before importing tkinter, so it renders to the DSI display even when launched via SSH or systemd. Loads PNG animation sequences from `faces/[state]/` directories. Face state changes based on bot state (idle, listening, thinking, speaking, error, capturing, warmup).

3. **Action Router** (`execute_action_and_get_result`) — Parses JSON actions from LLM output. Three tools: `get_time`, `search_web` (DuckDuckGo), `capture_image` (Pi camera via `rpicam-still`). Includes alias mapping (e.g., "google" → "search_web").

4. **Core Logic** (`safe_main_execution`) — Main loop: detect wake word or PTT → record audio → transcribe → chat → speak. Two recording modes: adaptive silence detection and push-to-talk (Enter key).

5. **Chat & Respond** (`chat_and_respond`) — Streams LLM response via the Anthropic Messages API. During streaming, detects if output is JSON (action mode) or plain text (chat mode). For actions, executes the tool then sends results back to Claude for summarization. Vision requests send base64-encoded images in Anthropic's native image format. Streamed text is buffered and flushed to TTS only when a sentence-ending punctuation mark is reached **and** the buffer meets a minimum length (`TTS_MIN_SENTENCE_LENGTH`, default 80 chars), preventing choppy playback of short fragments.

**Key threading model:** The main loop runs in a daemon thread off the tkinter main thread. TTS has its own worker thread with a queue. Thinking sounds loop in short-lived threads. All state coordination uses `threading.Event` objects.

## LLM Integration

The agent uses the **Anthropic API** (Claude) for all LLM tasks. The `anthropic` Python SDK communicates directly with Claude's Messages API — no local inference server is needed.

- **API Key**: Read from `ANTHROPIC_API_KEY` in the `.env` file (loaded via `python-dotenv`)
- **Model**: Defaults to `claude-sonnet-4-20250514`. Override via `ANTHROPIC_MODEL` env var or `text_model` in `config.json`
- **Streaming**: Uses `client.messages.stream()` for real-time token delivery during conversation
- **Vision**: Images are sent as base64-encoded content blocks in Anthropic's native format

## External Tool Chain (not in repo, installed by setup.sh)

- **whisper.cpp** — Speech-to-text at `./whisper.cpp/build/bin/whisper-cli`. Model and thread count configurable via `config.json` (`whisper_model`, `whisper_threads`). Defaults to `ggml-base.en.bin` with 2 threads to balance speed and CPU headroom.
- **Piper TTS** — Text-to-speech binary at `./piper/piper`, kept running as a persistent process by `piper_server.py`
- **OpenWakeWord** — Wake word detection from `wakeword.onnx`

## Piper TTS Server

`piper_server.py` runs Piper as a persistent HTTP service on `127.0.0.1:5111` to avoid reloading the ONNX voice model on every utterance. It maintains a single long-lived Piper subprocess, accepts POST requests with `{"text": "..."}`, and returns raw int16 audio at 22050 Hz. The agent tries the server first and falls back to spawning Piper directly if unavailable. A systemd unit file (`piper-tts.service`) is provided for auto-start.

## Configuration

`.env` contains sensitive configuration (API keys). Copy `example.env` to `.env` and fill in your `ANTHROPIC_API_KEY`. The `.env` file is gitignored.

`config.json` controls: Claude model name, voice model path, whisper model path, whisper thread count, audio energy threshold, chat memory toggle, camera rotation, and a system prompt extension. The script merges user config over `DEFAULT_CONFIG` defaults.

## Display

**A desktop environment (X11/Wayland) is required.** The agent uses tkinter for its GUI, which needs a running display server. A headless/server-only Pi OS installation will fail with `couldn't connect to display ":0"`. If using Pi OS Lite, install `lightdm` and configure desktop autologin via `raspi-config`.

The target display is the Raspberry Pi's DSI touchscreen interface (800x480). The script sets `os.environ.setdefault("DISPLAY", ":0")` at the top of `agent.py` before any tkinter imports, ensuring the GUI renders to the DSI screen regardless of launch context (local terminal, SSH, or systemd service). Users can override by setting `DISPLAY` before running.

## Audio Handling

The agent auto-detects microphone/speaker sample rates and resamples on the fly (using scipy) to match hardware capabilities. This is critical for Pi hardware compatibility — Piper outputs at 22050 Hz but many Pi audio devices only support 48000 Hz.

An audio energy gate (`_check_audio_energy`) checks RMS energy of recorded audio before invoking whisper.cpp. If the audio is below the configurable threshold (`audio_energy_threshold` in `config.json`, default `0.002`), transcription is skipped entirely, avoiding unnecessary CPU spikes on blank/silent recordings.
