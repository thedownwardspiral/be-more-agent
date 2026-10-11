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
| Text model        | `gemma3:1b` (local)                 | `claude-haiku-5-5` (cloud)        |
| Vision model      | `moondream` (local, separate model) | Claude native vision (same model) |
| Streaming         | `ollama.chat(stream=True)`          | `client.messages.stream()`        |
| Summarization     | `ollama.chat(stream=False)`         | `client.messages.create()`        |
| API key           | None required                       | `ANTHROPIC_API_KEY` in `.env`     |
| Internet required | No (fully local)                    | Yes (API calls)                   |

### Code-level details

- **Import**: `import ollama` replaced with `import anthropic` and
  `from dotenv import load_dotenv`.
- **Client**: A module-level `llm_client = anthropic.Anthropic(timeout=60.0,
  max_retries=2)` client is created, reading `ANTHROPIC_API_KEY` from the
  environment.
- **Model config**: The `vision_model` config key was removed. A single
  `text_model` is used for both text and vision. The model can be
  overridden via the `ANTHROPIC_MODEL` environment variable.
- **LLM settings**: The `OLLAMA_OPTIONS` dict (keep_alive, num_thread,
  temperature, top_k, top_p) was removed. Optional request parameters now
  come from config (`temperature`, `effort`) via `LLM_KWARGS`; see
  [Review Remediation](#review-remediation-robustness-safety-quality).
- **Chat messages**: Vision messages changed from Ollama's
  `{"images": [path]}` format to Anthropic's base64 content blocks with
  `{"type": "image", "source": {"type": "base64", ...}}`.
- **Message building**: A `_history_messages()` helper returns the chat
  history as Anthropic messages (skipping legacy `system` entries, capped
  at `max_history_messages`, and starting at a user message);
  `chat_and_respond()` appends the new user message. The original inlined
  this logic in `chat_and_respond()`.
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
| `whisper_threads` | `3`                                     | CPU threads for transcription      |

### Why

- **Threads reduced from 4 to 3** (briefly 2): Benchmarked on a Pi 5, the
  4th thread saves ~40 ms while starving the GUI and audio threads; 2
  threads cost about a second per utterance.
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
| `AGENTS.md`         | Short pointer to `CLAUDE.md` for other agents |

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
- **Whisper.cpp**: Added a build step (now step 6/10) that clones,
  compiles, and downloads the `base.en` model. The original did not
  include whisper.cpp setup.
- **Ollama model pull removed**: The `ollama pull gemma3:1b` and
  `ollama pull moondream` steps were removed.

### `config.json`

| Key                      | Original                          | Fork                                    |
| ------------------------ | --------------------------------- | --------------------------------------- |
| `text_model`             | `gemma3:1b`                       | `claude-haiku-5-5`                      |
| `vision_model`           | `moondream`                       | Removed (Claude handles vision)         |
| `voice_model`            | `piper/en_GB-semaine-medium.onnx` | `piper/en_US-bmo-medium.onnx`           |
| `whisper_model`          | --                                | `./whisper.cpp/models/ggml-base.en.bin` |
| `whisper_threads`        | --                                | `3`                                     |
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
  blocks, and loops. After `MAX_TOOL_ROUNDS` rounds of tool use, a final
  call is made with `tool_choice` `"none"` so Claude must answer in text,
  with a spoken fallback if it still produces nothing. The separate
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

## Recording and Listening Fixes

- **Adaptive noise floor** (db44f6e): room noise often sat above the fixed
  silence threshold, so every recording ran the full 30 s.
  `record_voice_adaptive()` now tracks the ambient floor (20th percentile
  of chunk volumes), compares a 250 ms smoothed volume against
  `max(silence_threshold, min(floor * 2.2, 0.015))`, stops 1.5 s after
  speech ends, and allows up to 5 s of leading silence. New config key
  `silence_threshold` (default `0.006`).
- **No self-hearing** (db44f6e): the LLM-error and memory-wipe paths now
  wait for TTS to finish before returning, so the mic never reopens while
  the bot is still talking.
- **Thinking sounds never overlap speech** (db44f6e): clips don't start
  while speech is queued or playing, and an in-flight clip stops itself
  when speech starts, from its own thread (stopping another thread's
  `sd.play` crashes PortAudio).
- **stdin-EOF idle loop** (b29a910): `_listen_loop` polled stdin for a
  terminal "press Enter" trigger, but an EOF stdin (desktop autostart,
  systemd) is permanently readable, so the agent woke, recorded and
  answered on every pass. stdin is now polled only when it is a tty, and
  EOF stops the polling.
- **`_is_speech()` hallucination filter** (b29a910): whisper annotates
  non-speech (`[BLANK_AUDIO]`, `(wind blowing)`, `*coughs*`) instead of
  returning nothing. Transcripts that are only such markers, or exactly a
  known silence hallucination ("you", "thanks for watching", ...), are
  dropped. A regex now strips only the timestamp prefix, so markers are
  parsed intact.
- **Wake-word tuning keys** (b29a910): `wake_word_threshold` (0.5) and
  `wake_word_consecutive` (1), with a `[WAKE]` log line on detection.

## uv and `.venv`

`python3 -m venv .bmo` + `pip install -r requirements.txt` was replaced
with uv (ad1c061): `pyproject.toml` is the single manifest, `uv.lock` pins
versions, and `uv sync` builds `.venv/`. `requirements.txt` was removed.
`start_agent.sh` execs `.venv/bin/python` directly. Two load-bearing
constraints live in `pyproject.toml`: `openwakeword<0.5` (0.5+ needs
`tflite-runtime`, which has no CPython 3.13/aarch64 wheel) and
`python-preference = "only-system"` (tkinter comes from the apt
`python3-tk`, which only equips the system interpreter). Claude Code
memory files swept into that commit were untracked and ignored (dba4db5).

## Setup and Agent-Loop Review Fixes (PR #2)

- **whisper.cpp setup hardening**: the stale `whisper.cpp` submodule
  gitlink was dropped; `setup.sh` runs under `set -euo pipefail`, clones
  when `whisper.cpp/CMakeLists.txt` is missing, checks out a pinned commit
  (`WHISPER_CPP_COMMIT`) on every run, not only after a fresh clone, and
  fails if `whisper-cli` was not built.
- **Whisper segment joining**: every timestamped segment line is joined;
  previously only the last line was kept, truncating longer utterances.
- **Interrupt flag**: cleared after playback finishes, so a spacebar
  interrupt no longer swallows the next wake word.
- **Fallback stream frames**: the fallback wake-word stream (blocksize
  1024, high latency) still reads and resamples full 80 ms frames.
- **History cap**: only the last `max_history_messages` (default 20) chat
  messages are sent to Claude, and session memory is trimmed in place.
- **Single thinking-sound thread**: at most one loop thread runs; two
  threads calling `sd.play()` crashed PortAudio.
- **Wake-word model download**: the old raw/main URL 404s; the fallback
  now downloads the openWakeWord v0.5.1 release asset and installs it
  only if its SHA-256 matches.
- **Cleanup**: `config.yaml` and `llama-swap.service` (an unused
  llama-swap/llama.cpp setup from before the Anthropic migration) were
  removed; `.gitignore` covers `.env.*` (except `example.env`) and
  `.claude/settings.local.json`; the README credits upstream.

The bundled `sounds/*.wav` clips were deleted earlier (07c0230); the
`sounds/` folders are created empty by `setup.sh` for your own clips.

## Review Remediation: Robustness, Safety, Quality

### Robustness

- **Launch from anywhere**: `BASE_DIR = Path(__file__).resolve().parent`
  and `os.chdir(BASE_DIR)` run at startup; `.env` is loaded from
  `BASE_DIR`.
- **Timeouts**: Anthropic client `timeout=60.0, max_retries=2`;
  `whisper-cli` 90 s with a return-code check; `rpicam-still` 20 s.
- **Tool failures**: an exception from a tool becomes an `is_error`
  `tool_result` that Claude can explain, instead of aborting the turn.
- **Warm-up failure**: shows the ERROR face ("Can't reach Claude"), plays
  an error sound and skips the greeting; the main loop still starts.
- **Memory**: saved after every turn, atomically (`memory.json.tmp` at
  mode 0600, fsync, `os.replace`) under a lock; loading validates the
  file's shape and starts fresh if it is malformed. `chat_memory: false`
  now really disables reading and writing `memory.json`. The wipe
  command must be the whole utterance ("forget everything" / "reset
  (your) memory", optional "please"), so "don't forget everything" is
  safe.
- **Shutdown**: SIGTERM/SIGINT run `safe_exit()`.
- **Private files**: `restrict_private_files()` chmods `.env`,
  `memory.json` and `agent.log` to 0600 at startup; `setup.sh` does the
  same (plus `agent.log.*` backups).
- **Temp files**: `input.wav` and camera photos live in a per-run
  `tempfile.mkdtemp()` directory, deleted after use and at exit.
- **Audio**: samples are clipped to ±1.0 before int16 conversion, so loud
  input no longer wraps around.

### Safety and configuration

- **Camera gating**: `capture_image` is offered only on turns whose text
  contains a `camera_trigger_words` entry (whole words/phrases, case and
  punctuation ignored; `[]` = never). The tool list is fixed for the
  whole turn, and calls to tools not offered get an `is_error` result.
  The system prompt says so.
- **TTS normalization** (`normalize_for_tts`): keeps apostrophes and
  speaks `$`, `%`, `°C`/`°F`/`°`, `&`, `+`, `=`, digit/digit and `#`digit
  instead of dropping them.
- **Config validation** (`validate_config`): an unreadable or non-object
  `config.json` is ignored with a loud error banner; unknown keys are
  warned about and dropped; each value is type- and range-checked and
  falls back to its default with a `[CONFIG]` warning.
- **New keys**: `temperature` (default 0.7; null omits it) and `effort`
  (default null; when set, sent as `output_config={"effort": ...}`),
  shared by chat and warm-up via `LLM_KWARGS` (warm-up `max_tokens` is
  32).
- **Default model**: now Claude Haiku 5.5 (`claude-haiku-5-5`) with
  `temperature` null and `effort` "low".

### Code quality

- **Logging**: `print` replaced by the `logging` module — logger `bmo` at
  INFO, root at WARNING, stdout plus a rotating `agent.log` (1 MB × 3
  backups, always mode 0600). `.gitignore` ignores `agent.log*` and
  `memory.json.tmp`.
- **Thread-safe GUI**: worker threads queue UI callables on
  `self.ui_queue`; `_process_ui_queue()` runs them on the Tk thread every
  30 ms. `set_state()` updates `current_state` synchronously; the
  `BOT:` line is tracked by `_bot_line_open`.
- **Cleanup**: bare `except:` → `except Exception`; unused imports and
  `current_volume` removed; the animation fallback for states without
  images now reuses the idle frames.
- **Linting**: ruff config in `pyproject.toml` (line length 120, py311,
  E4/E7/E9/F, E402 ignored for `agent.py`). Run `uvx ruff check agent.py`.

## Speech-to-Text: Parakeet and Silero VAD

whisper.cpp pads every input to a 30 s window, so each turn spent a fixed
2.6–3.7 s transcribing, and the RMS silence detector waited 1.5 s after the
last word before it even started. Both are replaced; whisper.cpp stays as a
fallback.

| Stage           | Before                                  | After                                                  |
| --------------- | --------------------------------------- | ------------------------------------------------------ |
| Speech-to-text  | `whisper-cli` subprocess, base.en       | NVIDIA Parakeet-TDT 0.6B v2 int8, in-process (sherpa-onnx) |
| STT time (Pi 5) | 2.6–3.0 s per question, any length      | 0.22–0.56 s for 1.4–4.1 s questions                    |
| End of speech   | RMS noise floor + fixed 1.5 s tail      | Silero VAD, `vad_min_silence` 0.9 s + `vad_resume_window` 0.6 s (1.6 s measured) |
| Audio handoff   | `input.wav` in the temp dir             | 16 kHz float32 array (WAV only for whisper)            |

- **Dependencies**: `sherpa-onnx==1.13.8` and `sherpa-onnx-core==1.13.8`
  (Apache-2.0). The core package ships the `libonnxruntime.so` the
  extension links against; it is listed explicitly because some
  sherpa-onnx wheels (armv7l) omit it from their metadata and uv locks from
  those, leaving aarch64 with an unimportable module. It coexists with the
  `onnxruntime` wheel used by Piper and openWakeWord in one process.
- **setup.sh step 5/10**: downloads the Parakeet archive (~460 MB) and
  `silero_vad.onnx`, verifies pinned SHA-256 sums, unpacks through a
  scratch directory into `models/parakeet-tdt-0.6b-v2-int8/`, and skips
  files already present. whisper.cpp is now step 6/10, labelled as the
  fallback engine.
- **New config keys**: `stt_engine` (`"parakeet"` | `"whisper"`, default
  parakeet), `stt_threads` (1–4, default 2; 3 is no faster on a Pi 5),
  `vad_min_silence` (0.2–3.0 s, default 0.9), `vad_resume_window` (0.0–2.0 s,
  default 0.6), `vad_threshold` (0.1–0.9,
  default 0.5). `silence_threshold` now only applies to the RMS fallback.
- **Loading**: Parakeet loads once in `__init__` (~3 s, ~1 GB RSS) and
  decodes on the main-loop thread. If sherpa-onnx or a model file is
  missing, or loading fails, the run falls back to whisper with an ERROR
  naming setup.sh step 5 and "Parakeet unavailable: using whisper" in the
  idle status. A missing VAD model falls back to the RMS detector with a
  single warning. `sherpa_onnx` is imported in a `try`, so a stale `.venv`
  still starts.
- **VAD recording**: the input callback only queues 50 ms blocks; the main
  loop resamples each to 16 kHz for the VAD, stops once a segment ends (or
  at 30 s, or after 5 s with no speech, which now returns "Heard nothing."
  without running STT), and cuts the utterance from the whole recording
  around the segment with 0.3 s / 0.2 s of padding.
- **Shorter pre-record wait**: the 0.5 s sleep before both recorders is
  now 0.2 s. Nothing plays between the trigger and the recording and the
  wake-word stream is already closed; the device settle after `sd.stop()`
  (0.2 s) is unchanged.
- **Hallucination filter**: non-speech annotations are still dropped for
  both engines, but `WHISPER_HALLUCINATIONS` ("thank you", "okay", "bye",
  ...) now applies only to whisper, so those replies reach Claude when
  Parakeet hears them.
- **Review fixes**: `vad_resume_window` bridges mid-sentence pauses; fillers
  ("um", "hmm") dropped for both engines; 0.5 s grace on the 5 s no-speech cutoff.
- **Logging**: each transcription logs `[STT] <engine> <seconds>`, and
  the VAD logs the segment bounds and where the recording stopped.
- **Attribution**: Parakeet-TDT 0.6B v2 is CC-BY-4.0 (NVIDIA); credited in
  the README's Acknowledgments. The model is downloaded by setup.sh, not
  committed.

## Tool trace in memory

- **`tool_calls` records**: each saved reply in memory.json keeps one compact
  record per tool executed that turn: a synthetic id (`toolu_mem_<8 hex>`),
  the name, the input (strings capped at 200 chars), the result text (capped
  at 300 chars) and an `error` flag. Photos are never stored; a camera call
  is recorded as "photo captured (image not retained)".
- **Replay**: `_history_messages()` sends such a reply to Claude as the calls
  that produced it: `tool_use` blocks, the matching `tool_result` blocks
  (`is_error` for a failed call), then the reply text. Claude sees a real
  tool call behind an earlier answer instead of a note in its own words.
  Verified against Claude Haiku 5.5 (no thinking blocks are replayed).
- **Migration**: memory files from the previous build (a `tools` list of
  labels) still load; the labels are dropped with one info log line.
- **Note and filter removed**: the first version rendered the trace as a
  bracketed note line inside Claude's earlier replies, with a prompt rule
  and a stream filter to strip copies. Claude copied the note in the first
  live session anyway, since it was imitating its own past text; the
  records keep that evidence out of assistant text, so the note, the prompt
  sentences about it and the filter are gone.
- **Give-up fallback**: when the tool-round cap is hit with no reply, the
  spoken "I couldn't finish that one." is saved with the reply too.

## Pipelined TTS

- **Two stages**: a synth thread turns `tts_queue` text into PCM chunks on a
  bounded `audio_queue` (16) while a playback thread plays the previous
  sentence, so the next sentence is ready when this one ends. The ~1 s gap
  after a short opening sentence is gone (measured: 0 ms between sentences,
  first audio ~0.5 s after the text is queued).
- **One stream per reply**: the playback thread opens one `RawOutputStream`
  on the first chunk and closes it when nothing more is queued or being
  synthesized, instead of opening and closing one per sentence.
- **Draining**: `stop()` does not drain on the Pi's PulseAudio-backed ALSA
  device, so the end of a reply is flushed with a buffer's worth of silence
  before `stop()`; this replaces the 0.5 s sleep after each sentence.
- **Interrupts**: the spacebar handler only sets `interrupted` and empties
  `tts_queue`; the playback thread aborts its own stream within one 2048-frame
  write and empties `audio_queue`.
- **`wait_for_tts()`** now returns only once queued speech has been heard and
  the stream closed (both queues joined, then `playback_idle`); the main
  loop's per-turn error handler calls it too.
- **Idle check**: the playback close condition and `_tts_busy()` use
  `tts_queue.unfinished_tasks == 0`, which covers both queued text and the
  item being synthesized.

## Interrupts cancel blocking calls

The spacebar used to take effect only at the next `interrupted` check, so an
interrupt during a slow request, search or photo left the bot on "Finishing..."
(wake word and PTT refused) until the call returned: up to 2 x 60 s for a
stalled API request, ~7 s for a search whose backends timed out (seen live),
up to 20 s for `rpicam-still`. Each span now checks every `INTERRUPT_POLL`
(0.1 s) and is cancelled:

- **API request, before the response starts**: `_open_stream()` opens the
  stream on a daemon helper thread; an interrupt abandons it (a stream that
  opens later is closed by the helper). Closing the client does not stop a
  request still waiting for headers (tested), so a thread is needed here.
- **API stream, while reading**: the open stream is kept in `active_stream`;
  `handle_speaking_interrupt` shuts its socket down and closes it from the Tk
  thread, so a blocked read wakes immediately and raises `httpx.ReadError`,
  which the stream block treats as the interrupt instead of an error. The
  shutdown matters: on Linux, closing the fd alone does not wake a thread
  blocked on it, so a stalled stream would wait out the 60 s read timeout.
  `_open_stream()` doesn't send at all if `interrupted` is already set (the
  helper re-checks just before sending), and an abandoned helper's request
  error is logged at debug.
- **search_web**: both DDG searches run on an abandoned-on-interrupt daemon
  thread (`_run_interruptible`), with a 20 s overall cap and an explicit 5 s
  per-backend ddgs timeout.
- **capture_image**: `rpicam-still` runs under `Popen` and is terminated
  (then killed after 1 s) on interrupt or after 20 s; a partial photo is
  deleted.

An interrupted tool returns `("interrupted", True)`, which is never sent to
Claude; the turn ends through the usual path (no memory save, TTS drained,
"Ready").
