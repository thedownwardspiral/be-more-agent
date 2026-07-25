# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Be More Agent is a single-file Python application (`agent.py`) that turns a Raspberry Pi into a conversational AI agent. It uses wake word detection, local speech-to-text (whisper.cpp), the Anthropic API (Claude) for LLM intelligence, and local text-to-speech (Piper TTS). An internet connection is required for the Anthropic API.

## Running the Project

```bash
# First-time setup (installs uv + system deps, .venv with piper-tts + anthropic SDK,
# downloads a Piper voice into piper/, builds whisper.cpp)
chmod +x setup.sh
./setup.sh

# Configure your Anthropic API key
cp example.env .env
# Edit .env and add your ANTHROPIC_API_KEY

# Run the agent (auto-targets DSI display via DISPLAY=:0)
uv run agent.py
```

There are no tests or linting configured in this project.

## Architecture

Everything lives in `agent.py` — a single ~920-line script with one main class:

**`BotGUI`** — The entire application. It's a tkinter GUI that also manages all background threads. Key sections (marked with comment banners in the file):

1. **Configuration & Constants** (top of file) — Loads `config.json` and `.env`, defines `BotStates` enum, LLM settings, the system prompt, and the `TOOLS` schemas for Anthropic native tool use. Creates an Anthropic client (`llm_client`) using the `ANTHROPIC_API_KEY` from environment. Also defines `resolve_input_device()` / `choose_input_samplerate()` which pick a microphone (via the `input_device` config key) and a verified sample rate.

2. **GUI Class** — Fullscreen tkinter app (800x480, designed for the Pi's DSI touchscreen). The script sets `DISPLAY=:0` via `os.environ.setdefault` before importing tkinter, so it renders to the DSI display even when launched via SSH or systemd. Loads PNG animation sequences from `faces/[state]/` directories. Face state changes based on bot state (idle, listening, thinking, speaking, error, capturing, warmup).

3. **Tool Execution** (`execute_tool`) — Runs tools requested by Claude via native tool use. Three tools: `get_time`, `search_web` (DuckDuckGo), `capture_image` (Pi camera via `rpicam-still`). `capture_image` returns the photo as a base64 image block inside the `tool_result`, so Claude sees it with the full conversation history.

4. **Core Logic** (`safe_main_execution`) — Main loop: detect wake word or PTT → record audio → transcribe → chat → speak. Two recording modes: adaptive silence detection and push-to-talk (Enter key). The wake-word listener (`_listen_loop`) uses cheap nearest-neighbor resampling, skips inference on near-silence, and retries once with fallback stream settings (blocksize 1024, high latency) before degrading to PTT.

5. **Chat & Respond** (`chat_and_respond`) — Streams the response via the Anthropic Messages API with `tools=TOOLS`. When the stream ends with `stop_reason == "tool_use"`, it executes the requested tools, appends the assistant content and `tool_result` blocks to the messages, and loops (bounded by `MAX_TOOL_ROUNDS`) so Claude can use the results in context. Streamed text is buffered and flushed to TTS only when a sentence-ending punctuation mark is reached **and** the buffer meets a minimum length (`TTS_MIN_SENTENCE_LENGTH`, default 80 chars), preventing choppy playback of short fragments.

**Key threading model:** The main loop runs in a daemon thread off the tkinter main thread. TTS has its own worker thread with a queue. Thinking sounds loop in short-lived threads. All state coordination uses `threading.Event` objects.

## LLM Integration

The agent uses the **Anthropic API** (Claude) for all LLM tasks. The `anthropic` Python SDK communicates directly with Claude's Messages API — no local inference server is needed.

- **API Key**: Read from `ANTHROPIC_API_KEY` in the `.env` file (loaded via `python-dotenv`)
- **Model**: Defaults to `claude-haiku-4-5`. Override via `ANTHROPIC_MODEL` env var or `text_model` in `config.json`
- **Streaming**: Uses `client.messages.stream()` for real-time token delivery during conversation
- **Tool use**: Native Anthropic tool use (`tools=` parameter) — no prompt-engineered JSON protocol or output sniffing
- **Vision**: The `capture_image` tool returns the camera photo as a base64 image block inside the `tool_result`, so vision requests keep the full conversation history

## External Tool Chain (not in repo, installed by setup.sh)

- **whisper.cpp** — Speech-to-text at `./whisper.cpp/build/bin/whisper-cli`. Model and thread count configurable via `config.json` (`whisper_model`, `whisper_threads`). Defaults to `ggml-base.en.bin` with 3 threads. Benchmarked on a Pi 5: whisper pads every input to a 30s window, so transcription latency is fixed regardless of utterance length. `base.en` is the accuracy/speed sweet spot — `tiny.en` misrecognizes short questions outright, and `small.en` costs ~2x for no content gain. Threads scale 6.7s/3.5s/2.5s/2.5s at 1/2/3/4; the 4th thread buys ~40ms while starving the tkinter and audio-callback threads, so 3 is the cap.
- **Piper TTS** — Text-to-speech via the [`piper-tts`](https://github.com/OHF-Voice/piper1-gpl) Python package (installed into `.venv` from `pyproject.toml`). Voice `.onnx` + `.onnx.json` files live in `piper/`, downloaded by `python -m piper.download_voices --data-dir piper <voice-name>` during setup. The model is loaded once at agent startup via `PiperVoice.load()` and stays resident in-process — no subprocess, no separate server.
- **OpenWakeWord** — Wake word detection from `wakeword.onnx`

## Piper TTS

`PiperVoice.load()` is called once in `BotGUI.__init__` and the voice instance is reused for every utterance. `BotGUI.speak()` calls `self.piper_voice.synthesize(text)`, which yields streaming int16 PCM chunks; each chunk's `sample_rate` drives the playback `RawOutputStream` (resampled with `scipy.signal.resample` if the audio device doesn't natively support the voice's rate). Playback writes in 2048-sample sub-chunks and checks `self.interrupted` per sub-chunk so spacebar interrupts cut cleanly.

## Configuration

`.env` contains sensitive configuration (API keys). Copy `example.env` to `.env` and fill in your `ANTHROPIC_API_KEY`. The `.env` file is gitignored.

`config.json` controls: Claude model name, voice model path, whisper model path, whisper thread count, audio energy threshold, chat memory toggle, camera rotation, a system prompt extension, microphone selection (`input_device` — index, name substring, or `null` for default), and a preferred input sample rate (`input_sample_rate`). The script merges user config over `DEFAULT_CONFIG` defaults.

## Display

**A desktop environment (X11/Wayland) is required.** The agent uses tkinter for its GUI, which needs a running display server. A headless/server-only Pi OS installation will fail with `couldn't connect to display ":0"`. If using Pi OS Lite, install `lightdm` and configure desktop autologin via `raspi-config`.

The target display is the Raspberry Pi's DSI touchscreen interface (800x480). The script sets `os.environ.setdefault("DISPLAY", ":0")` at the top of `agent.py` before any tkinter imports, ensuring the GUI renders to the DSI screen regardless of launch context (local terminal, SSH, or systemd service). Users can override by setting `DISPLAY` before running.

`setup.sh` installs a desktop autostart entry: it substitutes the repo path into `be-more-agent.desktop` and copies it to `~/.config/autostart/`, which launches `start_agent.sh` (execs `agent.py` via `.venv`) when the desktop session starts. Remove that file to disable autostart.

## Audio Handling

The agent auto-detects microphone/speaker sample rates and resamples on the fly (using scipy) to match hardware capabilities. This is critical for Pi hardware compatibility — Piper voices typically emit at 22050 Hz (read per-chunk from `chunk.sample_rate`) but many Pi audio devices only support 48000 Hz.

An audio energy gate (`_check_audio_energy`) checks RMS energy of recorded audio before invoking whisper.cpp. If the audio is below the configurable threshold (`audio_energy_threshold` in `config.json`, default `0.002`), transcription is skipped entirely, avoiding unnecessary CPU spikes on blank/silent recordings.
