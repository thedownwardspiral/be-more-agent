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
| Text model        | `gemma3:1b` (local)                 | `claude-haiku-4-5` (cloud)        |
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

## TTS: In-Process piper1-gpl

The original spawned a new Piper subprocess for every utterance,
reloading the voice model each time. This fork uses the
[`piper-tts`](https://github.com/OHF-Voice/piper1-gpl) Python package
and loads the voice once at startup, keeping it resident in-process for
the lifetime of the agent. No subprocess, no separate server.

### Dependencies and setup

- `pyproject.toml` adds `piper-tts`.
- `setup.sh` no longer downloads the prebuilt `piper_linux_aarch64`
  binary; instead it runs
  `python -m piper.download_voices --data-dir piper en_US-bmo-medium`
  inside the `.venv` to fetch the voice `.onnx` + `.onnx.json` into
  `piper/`.

### Agent-side changes

- **Import** of `from piper import PiperVoice` added at module load.
- **`BotGUI.__init__`** loads the voice once via
  `PiperVoice.load(voice_model)` and stores it on `self.piper_voice`.
  Failure logs but does not crash the agent — TTS just degrades to
  silent.
- **`speak()`** rewritten to iterate `self.piper_voice.synthesize(text)`
  and stream each int16 PCM chunk into `sd.RawOutputStream`. The
  per-chunk `sample_rate` drives playback (and the `scipy.signal.resample`
  fallback when the audio device doesn't natively support it),
  replacing the hardcoded `PIPER_RATE = 22050` constant.
- **`_speak_via_server`, `_speak_via_subprocess`, and
  `self.current_audio_process`** removed entirely — there is no
  subprocess or HTTP server to talk to.

### Removed files

The earlier persistent-server iteration of this fork has been retired:
`piper_server.py` and `piper-tts.service` are gone.

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

### Dependency manifest (`requirements.txt` → `pyproject.toml`)

| Original            | Fork            |
| ------------------- | --------------- |
| `ollama`            | `anthropic`     |
| `duckduckgo-search` | `ddgs`          |
| --                  | `python-dotenv` |
| --                  | `piper-tts`     |

The `ollama` package was removed and `anthropic`, `python-dotenv`, and
`piper-tts` were added. The DuckDuckGo search package changed from
`duckduckgo-search` to `ddgs`. Piper is now consumed as a Python library
instead of a prebuilt binary.

### `setup.sh`

- **Virtual environment**: `venv` + `pip install -r requirements.txt`
  replaced with [uv](https://docs.astral.sh/uv/) — `setup.sh` installs `uv`
  if missing and runs `uv sync`, which creates `.venv/` from
  `pyproject.toml` and the locked versions in `uv.lock`.
- **Desktop environment check**: Added a pre-flight check for a display
  manager (`lightdm`, `gdm3`, or `sddm`) with a warning if none is
  found.
- **BLAS package**: Falls back to `libopenblas-dev` if `libatlas-base-dev`
  is not available (compatibility with newer Pi OS releases).
- **Whisper.cpp**: Added a build step (step 6/7) that clones, compiles,
  and downloads the `base.en` model. The original did not include
  whisper.cpp setup.
- **Ollama model pull removed**: The `ollama pull gemma3:1b` and
  `ollama pull moondream` steps were removed.

### `config.json`

| Key                      | Original                          | Fork                                    |
| ------------------------ | --------------------------------- | --------------------------------------- |
| `text_model`             | `gemma3:1b`                       | `claude-haiku-4-5`                      |
| `vision_model`           | `moondream`                       | Removed (Claude handles vision)         |
| `voice_model`            | `piper/en_GB-semaine-medium.onnx` | `piper/en_US-bmo-medium.onnx`           |
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

## Native Anthropic Tool Use

The original (and earlier versions of this fork) used a prompt-engineered
protocol: the system prompt instructed the model to emit JSON like
`{"action": "search_web", "value": "..."}`, and the streaming loop sniffed
output for `'{"'` to detect "action mode", then regex-extracted the JSON
and routed it through an alias map.

This fork now uses the Anthropic API's native tool use:

- **`TOOLS`** schemas (`get_time`, `search_web`, `capture_image`) are passed
  via the `tools=` parameter on every streaming call.
- **`execute_tool()`** replaces `execute_action_and_get_result()`. The alias
  map, `CHAT_FALLBACK`, `INVALID_ACTION`, and canned fallback speech lines
  are gone — Claude phrases tool results (including empty searches and
  errors) naturally.
- **`chat_and_respond()`** streams until `stop_reason == "tool_use"`, runs
  the requested tools, appends the assistant content and `tool_result`
  blocks, and loops (bounded by `MAX_TOOL_ROUNDS`). The separate
  "summarize this result" second API call is gone — the follow-up response
  has full conversation context.
- **`extract_json_from_text()`** and the `'{"' in content` stream sniffing
  were removed.

## Vision Keeps Conversation History

Previously a photo request restarted the conversation: the image was sent
as a fresh single-message exchange with no history. Now `capture_image` is
a tool, and the photo is returned as a base64 image block inside the
`tool_result`, so Claude sees the image *and* the full conversation.

The user message is now also persisted to session memory (previously only
assistant replies were saved, so "memory" contained half the conversation).

## Audio Robustness (back-ported from upstream)

Upstream kept improving its audio stack after this fork diverged; these
changes were ported back:

- **`input_device` config key** + `resolve_input_device()` — select a
  microphone by device index or name substring instead of always using the
  system default.
- **`input_sample_rate` config key** + `choose_input_samplerate()` —
  candidate rates are verified with `sd.check_input_settings()` instead of
  trusting the device's reported default.
- **Nearest-neighbor resampling** in the wake-word loop — replaces
  per-chunk `scipy.signal.resample` (FFT-based), which overloaded the Pi
  CPU and caused buffer overflows.
- **Silence gate before wake-word inference** — `oww_model.predict()` is
  skipped when peak amplitude is ≤ 200, saving idle CPU.
- **Stream retry** — if the wake-word stream fails, it is retried once with
  fallback settings (blocksize 1024, high latency) before degrading to
  push-to-talk. Persistent buffer overflows also trigger the fallback.
- **`sd.stop()` + 0.2 s pause** before opening recording streams — hardware
  contention freezes the Pi 5.
- **`[AUDIO ERROR]` logging** in both recording paths (failures were
  previously silent and looked identical to "heard nothing").
- **`safe_exit()` reentrancy guard** — it is reachable from the Exit
  button, Escape, and `atexit`; an `exiting` flag makes it run once, and
  the `sys.exit(0)` call (which re-triggered atexit from inside a Tk
  callback) was removed.

## Desktop Autostart

`start_agent.sh` (execs `agent.py` via `.venv`) and a
`be-more-agent.desktop` template were added, mirroring upstream's launch
story. `setup.sh` substitutes the repo path into the template and installs
it to `~/.config/autostart/`, replacing the systemd unit that was retired
with the Piper server. Remove `~/.config/autostart/be-more-agent.desktop`
to disable.

## config.json Cleanup

The unused `system_prompt` key (which still described the old JSON action
protocol; the code reads `system_prompt_extras`) was removed, and
`input_device` / `input_sample_rate` were added with `null` defaults.
