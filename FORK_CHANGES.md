# Fork Changes

Key differences between this fork and the
[original repository](https://github.com/brenpoly/be-more-agent).

## LLM Backend: Ollama to Anthropic API

The largest change in this fork is replacing the local Ollama inference
stack with the cloud-based Anthropic API (Claude).

### What changed

| Area              | Original (Ollama)                   | Fork (Anthropic API)              |
| ----------------- | ----------------------------------- | --------------------------------- |
| LLM library       | `ollama` Python package             | `anthropic` Python SDK            |
| Text model        | `gemma3:1b` (local)                 | `claude-sonnet-4-6` (cloud)       |
| Vision model      | `moondream` (local, separate model) | Claude native vision (same model) |
| Streaming         | `ollama.chat(stream=True)`          | `client.messages.stream()`        |
| Summarization     | `ollama.chat(stream=False)`         | `client.messages.create()`        |
| API key           | None required                       | `ANTHROPIC_API_KEY` in `.env`     |
| Internet required | No (fully local)                    | Yes (API calls)                   |

### Code-level details

- **Import**: `import ollama` replaced with `import anthropic` and
  `from dotenv import load_dotenv`.
- **Client**: A module-level `llm_client = anthropic.Anthropic()` client
  is created, reading `ANTHROPIC_API_KEY` from the environment.
- **Model config**: The `vision_model` config key was removed. A single
  `text_model` is used for both text and vision. The model can be
  overridden via the `ANTHROPIC_MODEL` environment variable.
- **LLM settings**: `OLLAMA_OPTIONS` dict (keep_alive, num_thread,
  temperature, top_k, top_p) replaced with standalone `LLM_TEMPERATURE`
  and `LLM_TOP_P` constants.
- **Chat messages**: Vision messages changed from Ollama's
  `{"images": [path]}` format to Anthropic's base64 content blocks with
  `{"type": "image", "source": {"type": "base64", ...}}`.
- **Message building**: A new `_build_messages()` helper constructs the
  messages list and handles the history/vision split. The original
  inlined this logic in `chat_and_respond()`.
- **System prompt**: Passed via the `system=` parameter on API calls
  instead of as a message role.
- **Warm-up**: `ollama.generate(keep_alive=-1)` (pre-load model into
  memory) replaced with a lightweight `llm_client.messages.create()`
  call to verify API connectivity.
- **Shutdown**: The `ollama.generate(keep_alive=0)` call (unload model)
  was removed since there is no local model to unload.
- **Sentence buffer flush**: After streaming completes, any remaining
  text in `sentence_buffer` is flushed to TTS. The original did not
  flush leftover text.
- **TTS init**: `self.tts_active` is cleared before starting the TTS
  worker thread. The original set it, which could cause `wait_for_tts()`
  to block on the first call.

## Display: DSI Touchscreen Support

- `os.environ.setdefault("DISPLAY", ":0")` added at the top of
  `agent.py` before any tkinter imports, allowing the GUI to render on
  the Pi's DSI display when launched via SSH or systemd.
- The original had no `DISPLAY` handling; it required a local terminal
  session.

## TTS: Persistent Piper Server

The original spawned a new Piper subprocess for every utterance,
reloading the voice model each time.

### New files

- **`piper_server.py`** -- A persistent HTTP server on
  `127.0.0.1:5111` that keeps a single Piper subprocess alive. Accepts
  `POST /tts` with `{"text": "..."}` and returns raw int16 audio at
  22050 Hz. Includes a `GET /` health check.
- **`piper-tts.service`** -- A systemd unit file to run the server as a
  system service with auto-restart.

### Agent-side changes

- **`_speak_via_server()`** -- New method that sends text to the Piper
  HTTP server and receives raw audio.
- **`_speak_via_subprocess()`** -- Extracted from the original `speak()`
  as a fallback when the server is unavailable.
- **`speak()`** rewritten to try the server first, fall back to
  subprocess, then resample and play audio in interruptible chunks.
  The original streamed directly from Piper's stdout and resampled
  per-chunk; the fork resamples the entire buffer once before playback.

## Whisper: Configurable Model and Threads

The original hardcoded the whisper model (`ggml-base.en.bin`) and thread
count (`-t 4`).

### New config keys

| Key               | Default                                 | Description                        |
| ----------------- | --------------------------------------- | ---------------------------------- |
| `whisper_model`   | `./whisper.cpp/models/ggml-base.en.bin` | Path to the whisper.cpp model file |
| `whisper_threads` | `2`                                     | CPU threads for transcription      |

### Why

- **Threads reduced from 4 to 2**: Leaves CPU headroom for the GUI,
  audio pipeline, and TTS during transcription.
- **Model is configurable**: Users can swap between `tiny`, `base`,
  `small`, or quantized variants without editing code.

## Audio Energy Gate

A new `_check_audio_energy()` method computes the RMS energy of recorded
audio before invoking whisper.cpp. If the energy is below the
configurable threshold, transcription is skipped entirely.

| Key                      | Default | Description                                       |
| ------------------------ | ------- | ------------------------------------------------- |
| `audio_energy_threshold` | `0.002` | RMS level below which audio is considered silence |

This avoids a full CPU spike on blank or silent recordings, which were
observed to account for roughly one-third of interactions during
testing.

## Environment and Dependencies

### New files

| File                | Purpose                                      |
| ------------------- | -------------------------------------------- |
| `example.env`       | Template for `.env` with `ANTHROPIC_API_KEY` |
| `.env`              | Gitignored file holding the actual API key   |
| `CLAUDE.md`         | Project context for Claude Code              |
| `AGENTS.md`         | Project context for AI coding agents         |
| `piper_server.py`   | Persistent Piper TTS HTTP server             |
| `piper-tts.service` | systemd unit for the TTS server              |

### `requirements.txt`

| Original            | Fork            |
| ------------------- | --------------- |
| `ollama`            | `anthropic`     |
| `duckduckgo-search` | `ddgs`          |
| --                  | `python-dotenv` |

The `ollama` package was removed and `anthropic` + `python-dotenv` were
added. The DuckDuckGo search package changed from `duckduckgo-search` to
`ddgs`.

### `setup.sh`

- **Virtual environment**: Renamed from `venv` to `.bmo`.
- **Desktop environment check**: Added a pre-flight check for a display
  manager (`lightdm`, `gdm3`, or `sddm`) with a warning if none is
  found.
- **BLAS package**: Falls back to `libopenblas-dev` if `libatlas-base-dev`
  is not available (compatibility with newer Pi OS releases).
- **Whisper.cpp**: Added a build step (step 6/7) that clones, compiles,
  and downloads the `small.en-q8_0` model. The original did not include
  whisper.cpp setup.
- **Ollama model pull removed**: The `ollama pull gemma3:1b` and
  `ollama pull moondream` steps were removed.

### `config.json`

| Key                      | Original                          | Fork                                    |
| ------------------------ | --------------------------------- | --------------------------------------- |
| `text_model`             | `gemma3:1b`                       | `claude-sonnet-4-6`                     |
| `vision_model`           | `moondream`                       | Removed (Claude handles vision)         |
| `voice_model`            | `piper/en_GB-semaine-medium.onnx` | `piper/en_US-bmo_voice.onnx`            |
| `whisper_model`          | --                                | `./whisper.cpp/models/ggml-base.en.bin` |
| `whisper_threads`        | --                                | `2`                                     |
| `audio_energy_threshold` | --                                | `0.002`                                 |
| `system_prompt`          | Present                           | Present (unchanged structure)           |

## Warning Filter

The `duckduckgo_search` warning filter module name was updated to match
the new `ddgs` package:

```python
# Original
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")

# Fork
warnings.filterwarnings("ignore", category=RuntimeWarning, module="ddgs")
```
