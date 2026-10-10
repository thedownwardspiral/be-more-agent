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
cp example.env .env && chmod 600 .env
# Edit .env and add your ANTHROPIC_API_KEY

# Run the agent (auto-targets DSI display via DISPLAY=:0)
uv run agent.py
```

There are no tests. Lint with `uvx ruff check agent.py` (config in `pyproject.toml`: line length 120, rules E4/E7/E9/F, E402 ignored for `agent.py`; the remaining findings are E701 one-liners) and syntax-check with `python3 -m py_compile agent.py`.

## Architecture

Everything lives in `agent.py` — a single ~1,600-line script with one main class:

**`BotGUI`** — The entire application. It's a tkinter GUI that also manages all background threads. Key sections (marked with comment banners in the file):

1. **Configuration & Constants** (top of file) — Before the GUI/audio imports, the preamble sets `DISPLAY`, computes `BASE_DIR = Path(__file__).resolve().parent` and `os.chdir`s there (so every relative path works from any launch directory), and sets up logging. That ordering is deliberate and is why ruff's E402 is ignored for `agent.py`. Then: `.env` is loaded from `BASE_DIR`; `load_config()` merges `config.json` over `DEFAULT_CONFIG` via `validate_config()` (see Configuration); `LLM_KWARGS` holds the optional `temperature` / `output_config={"effort": ...}` parameters shared by the chat stream and the warm-up call; `restrict_private_files()` chmods `.env`, `memory.json` and `agent.log` to 0600 (called from `__main__`). Also defines the `BotStates` enum, the system prompt, the `TOOLS` schemas (and `TOOLS_WITHOUT_CAMERA`), `normalize_words()` (lowercase, strip punctuation) with `CAMERA_TRIGGER_RE` and `MEMORY_WIPE_RE` matched against its output, `normalize_for_tts()` (spells out `$`, `%`, `°C/°F/°`, `&`, `+`, `=`, digit`/`digit, `#`digit, keeps apostrophes, then strips emoji/non-speech characters), the whisper transcript regexes, and `resolve_input_device()` / `choose_input_samplerate()`, which pick a microphone (via the `input_device` config key) and a verified sample rate. The Anthropic client (`llm_client`) is created with `timeout=60.0, max_retries=2`.

2. **GUI Class** — Fullscreen tkinter app (800x480, designed for the Pi's DSI touchscreen). The script sets `DISPLAY=:0` via `os.environ.setdefault` before importing tkinter, so it renders to the DSI display even when launched via SSH or systemd. Loads PNG animation sequences from `faces/[state]/` directories. Face state changes based on bot state (idle, listening, thinking, speaking, error, capturing, warmup).

   **UI queue:** worker threads never touch Tk. They call `self._ui(fn, *args)`, which puts a callable on `self.ui_queue`; `_process_ui_queue()` drains it on the Tk thread every 30 ms (FIFO). `set_state()`, `append_to_text()` and `_stream_to_text()` all go through it. `set_state()` updates `self.current_state` synchronously on the calling thread (the worker reads it right after) and queues only the drawing. The only direct `master.after` calls are `update_animation()` and `_process_ui_queue()` rescheduling themselves on the Tk thread, plus the extra `quit` queued by the signal handler (which Python runs on the main thread). The open `BOT: ` line is tracked by `_bot_line_open` (main-loop thread only) and closed by `_end_bot_line()`.

3. **Tool Execution** (`execute_tool`) — Runs tools requested by Claude via native tool use. Three tools: `get_time`, `search_web` (DuckDuckGo), `capture_image` (Pi camera via `rpicam-still`). `capture_image` returns the photo as a base64 image block inside the `tool_result`, so Claude sees it with the full conversation history. The camera tool is offered only on turns whose user text matches `CAMERA_TRIGGER_RE` (built from `camera_trigger_words`); the tool list is chosen once per turn, and a call to a tool not offered that turn gets an `is_error` `tool_result`. Any exception from `execute_tool` is also returned to Claude as an `is_error` `tool_result` instead of aborting the turn (search and camera failures that are caught inside `execute_tool` return plain explanatory strings). `rpicam-still` has a 20 s timeout.

4. **Core Logic** (`safe_main_execution`) — Main loop: detect wake word or PTT → record audio → transcribe → chat → speak. Two recording modes: adaptive silence detection and push-to-talk (Enter key). The wake-word listener (`_listen_loop`) uses cheap nearest-neighbor resampling, skips inference on near-silence, and retries once with fallback stream settings (blocksize 1024, high latency; still reading and resampling full 80 ms frames) before degrading to PTT.

   Recordings (`input.wav`) and camera photos live in a per-run `tempfile.mkdtemp()` directory (`self.tmp_dir`): the recording is deleted right after transcription, the photo after the turn (through the UI queue, after the overlay that uses it), and the directory at exit. Audio is clipped to ±1.0 before int16 conversion. `whisper-cli` runs with a 90 s timeout and a return-code check; every segment line is joined, and `_is_speech()` drops non-speech annotations and known silence hallucinations. If the warm-up call fails, the agent shows the ERROR face ("Can't reach Claude"), plays an error sound, skips the greeting, leaves `llm_ready` false, and still starts the main loop. SIGTERM/SIGINT run `safe_exit()` (save memory, stop audio, remove the temp dir, quit Tk). Memory is saved after every completed turn by `save_chat_history()`: under `memory_lock`, write `memory.json.tmp` with mode 0600, fsync, then `os.replace`. It keeps the system entry plus the last 10 conversation messages; `load_chat_history()` validates the file's shape and starts fresh if it is malformed. With `chat_memory: false`, memory.json is never read or written. The memory wipe fires only when the whole normalized utterance is "forget everything" or "reset (your) memory" (optional leading/trailing "please").

5. **Chat & Respond** (`chat_and_respond`) — Streams the response via the Anthropic Messages API with the turn's tool list and `**LLM_KWARGS`. History comes from `_history_messages()` (last `max_history_messages`, starting at a user message). When the stream ends with `stop_reason == "tool_use"`, it executes the requested tools, appends the assistant content and `tool_result` blocks to the messages, and loops (bounded by `MAX_TOOL_ROUNDS`) so Claude can use the results in context. Streamed text is buffered and flushed to TTS only when a sentence-ending punctuation mark is reached **and** the buffer meets a minimum length (`TTS_MIN_SENTENCE_LENGTH`, default 80 chars), preventing choppy playback of short fragments.

**Key threading model:** The main loop runs in a daemon thread off the tkinter main thread. TTS has its own worker thread with a queue. At most one thinking-sound thread runs at a time (`_start_thinking_sound`), and it never plays while TTS is speaking or has speech queued. Worker threads reach the GUI only through `self.ui_queue` (see the UI queue above). State coordination uses `threading.Event` objects; `memory_lock` serializes memory saves (main loop vs. `safe_exit` on the Tk thread).

## LLM Integration

The agent uses the **Anthropic API** (Claude) for all LLM tasks. The `anthropic` Python SDK communicates directly with Claude's Messages API — no local inference server is needed.

- **API Key**: Read from `ANTHROPIC_API_KEY` in `BASE_DIR/.env` (loaded via `python-dotenv`)
- **Model**: Defaults to `claude-haiku-4-5`. Override via `ANTHROPIC_MODEL` env var or `text_model` in `config.json`
- **Streaming**: Uses `client.messages.stream()` for real-time token delivery during conversation
- **Optional parameters**: `temperature` (default 0.7) and `effort` (default null → `output_config={"effort": ...}` when set) go into `LLM_KWARGS`; a null value omits the parameter. Claude 5.x models reject a non-default temperature and `claude-haiku-4-5` rejects `effort`, so switching to Haiku 5.5 means `"text_model": "claude-haiku-5-5", "temperature": null, "effort": "low"`. The warm-up call uses the same kwargs (`max_tokens=32`), so a mismatch fails at startup
- **Tool use**: Native Anthropic tool use (`tools=` parameter) — no prompt-engineered JSON protocol or output sniffing
- **Vision**: The `capture_image` tool returns the camera photo as a base64 image block inside the `tool_result`, so vision requests keep the full conversation history

## External Tool Chain (not in repo, installed by setup.sh)

- **whisper.cpp** — Speech-to-text at `./whisper.cpp/build/bin/whisper-cli`. Model and thread count configurable via `config.json` (`whisper_model`, `whisper_threads`). Defaults to `ggml-base.en.bin` with 3 threads. Benchmarked on a Pi 5: whisper pads every input to a 30s window, so transcription latency is fixed regardless of utterance length. `base.en` is the accuracy/speed sweet spot — `tiny.en` misrecognizes short questions outright, and `small.en` costs ~2x for no content gain. Threads scale 6.7s/3.5s/2.5s/2.5s at 1/2/3/4; the 4th thread buys ~40ms while starving the tkinter and audio-callback threads, so 3 is the cap.
- **Piper TTS** — Text-to-speech via the [`piper-tts`](https://github.com/OHF-Voice/piper1-gpl) Python package (installed into `.venv` from `pyproject.toml`). Voice `.onnx` + `.onnx.json` files live in `piper/`, downloaded by `python -m piper.download_voices --data-dir piper <voice-name>` during setup. The model is loaded once at agent startup via `PiperVoice.load()` and stays resident in-process — no subprocess, no separate server.
- **OpenWakeWord** — Wake word detection from `wakeword.onnx`

## Piper TTS

`PiperVoice.load()` is called once in `BotGUI.__init__` and the voice instance is reused for every utterance. `BotGUI.speak()` calls `self.piper_voice.synthesize(text)`, which yields streaming int16 PCM chunks; each chunk's `sample_rate` drives the playback `RawOutputStream` (resampled with `scipy.signal.resample` if the audio device doesn't natively support the voice's rate). Playback writes in 2048-sample sub-chunks and checks `self.interrupted` per sub-chunk so spacebar interrupts cut cleanly.

## Configuration

`.env` contains sensitive configuration (API keys). Copy `example.env` to `.env` and fill in your `ANTHROPIC_API_KEY`. It is loaded from `BASE_DIR`, is gitignored, and is kept at mode 0600 by the agent and by `setup.sh`.

`config.json` controls: Claude model name, voice model path, whisper model path, whisper thread count, audio energy threshold, chat memory toggle, camera rotation, a system prompt extension, microphone selection (`input_device` — index, name substring, or `null` for default), a preferred input sample rate (`input_sample_rate`), how many recent chat messages are resent to Claude each turn (`max_history_messages`, default 20), the recording `silence_threshold`, `wake_word_threshold` / `wake_word_consecutive`, `camera_trigger_words` (`[]` = camera never offered), `temperature` (0.0–1.0 or null) and `effort` ("low"/"medium"/"high" or null). The script merges user config over `DEFAULT_CONFIG` defaults after `validate_config()`: unknown keys are warned about and dropped; each value is type-checked against its default's type (or `NULLABLE_CONFIG_TYPES`) and range-checked (`CONFIG_RANGES`), and an invalid value is warned about and replaced by the default. All of these print `[CONFIG]` warnings. If `config.json` is not valid JSON (or not an object), the whole file is ignored with a loud error banner.

## Logging

All output goes through the `logging` module: logger `"bmo"` at INFO, root at WARNING (so library warnings are kept but httpx/anthropic INFO chatter is not). Handlers: stdout and `agent.log` in `BASE_DIR` via a `RotatingFileHandler` subclass (1 MB, 3 backups, always mode 0600 because it holds transcripts). Use `log.info/warning/error/exception`, not `print`. When something fails (API errors, whisper, camera, a crashed main loop), the details and tracebacks are in `agent.log`.

## Linting

`uvx ruff check agent.py` (config in `pyproject.toml`). There are still no tests.

## Display

**A desktop environment (X11/Wayland) is required.** The agent uses tkinter for its GUI, which needs a running display server. A headless/server-only Pi OS installation will fail with `couldn't connect to display ":0"`. If using Pi OS Lite, install `lightdm` and configure desktop autologin via `raspi-config`.

The target display is the Raspberry Pi's DSI touchscreen interface (800x480). The script sets `os.environ.setdefault("DISPLAY", ":0")` at the top of `agent.py` before any tkinter imports, ensuring the GUI renders to the DSI screen regardless of launch context (local terminal, SSH, or systemd service). Users can override by setting `DISPLAY` before running.

`setup.sh` installs a desktop autostart entry: it substitutes the repo path into `be-more-agent.desktop` and copies it to `~/.config/autostart/`, which launches `start_agent.sh` (execs `agent.py` via `.venv`) when the desktop session starts. Remove that file to disable autostart.

## Audio Handling

The agent auto-detects microphone/speaker sample rates and resamples on the fly (using scipy) to match hardware capabilities. This is critical for Pi hardware compatibility — Piper voices typically emit at 22050 Hz (read per-chunk from `chunk.sample_rate`) but many Pi audio devices only support 48000 Hz.

An audio energy gate (`_check_audio_energy`) checks RMS energy of recorded audio before invoking whisper.cpp. If the audio is below the configurable threshold (`audio_energy_threshold` in `config.json`, default `0.002`), transcription is skipped entirely, avoiding unnecessary CPU spikes on blank/silent recordings.
