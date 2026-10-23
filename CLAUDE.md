# CLAUDE.md

Be More Agent turns a Raspberry Pi 5 into a voice assistant: OpenWakeWord wake word → Silero VAD endpointing → Parakeet speech-to-text (sherpa-onnx; whisper.cpp fallback) → Claude via the Anthropic API → Piper TTS, with a fullscreen tkinter face on the 800x480 DSI display.

It is derived from [brenpoly/be-more-agent](https://github.com/brenpoly/be-more-agent) (MIT). The fork link was detached on purpose, so the README, `FORK_CHANGES.md`, `LICENSE`, the `agent.py` header and `pyproject.toml` carry the attribution: keep it when editing them.

## Commands

```bash
./setup.sh                          # apt deps, uv, .venv, models (all checksum-verified), whisper.cpp, autostart entry
cp example.env .env && chmod 600 .env   # then set ANTHROPIC_API_KEY
uv run agent.py                     # run (renders on DISPLAY=:0)
uv run --group dev pytest           # tests
uvx ruff check agent.py bmo tests   # lint (CI pins the ruff version)
```

The Pi usually has an agent already running from the main checkout (autostart). A second instance contends for the mic and speaker, so verify changes with the tests rather than by launching another copy.

## Layout

- `agent.py`: the `BotGUI` class: GUI, recording, STT, the chat/tool loop and the TTS threads, plus the config-derived module constants. Its preamble sets `DISPLAY`, `chdir`s to the repo and configures logging before the GUI/audio imports; that order is why ruff's E402 is ignored for it.
- `bmo/`: pure helpers with no import-time side effects: `config` (defaults, validation), `text` (sentence splitting, TTS normalization, trigger and wipe regexes, transcript filters), `memory` (tool records, memory.json validation and upgrade, history rendering), `stt` (resampling, chunk split points), `tools` (tool schemas, `get_time` and search output), `llm` (status text for API errors, stop-reason fallbacks). New logic that needs no device or thread belongs here, with a test.
- `tests/`: pytest for `bmo/`, plus `test_turn.py`, which drives `chat_and_respond` with a fake Claude client.
- `setup.sh`: every download goes through `download_verified` (pinned URL + SHA-256). Change the hash together with the URL.

## Threading rules

These are the invariants the code comments assume; breaking one shows up as a hang, a PortAudio crash, or the bot answering itself.

- **Threads:** Tk main thread; the main loop (daemon); `tts-synth` and `tts-playback`; at most one thinking-sound thread; short-lived helpers for opening the API stream and for web search.
- **Tk only on the Tk thread.** Workers call `self._ui(fn, ...)`, drained by `_process_ui_queue`. `set_state` updates `current_state` synchronously and queues only the drawing.
- **Playback belongs to the thread that started it.** `sd.stop()` on another thread's `sd.play()` crashes PortAudio, so the recorders join the thinking-sound thread first. The TTS `RawOutputStream` is opened, aborted and closed only by `tts-playback`.
- **The mic stays closed while TTS is busy.** Every path that ends a turn calls `wait_for_tts()` before the next recording.
- **Interrupts:** the spacebar sets `interrupted`; blocking work polls it every `INTERRUPT_POLL`. `_cancel_active_stream` only `shutdown()`s the response socket: a cross-thread `close()` races the reader's fd. Every turn path ends with `interrupted.clear()`.

## Claude API

- Defaults: `claude-haiku-5-5`, `temperature` null (Claude 5.x rejects non-default values), `effort` `"low"`. The warm-up call sends the same `LLM_KWARGS`, so a model/parameter mismatch fails at startup and `describe_api_error` names the cause on the face.
- Tool loop: up to `MAX_TOOL_ROUNDS`, then one round with `tool_choice: none` (`tools=` must stay, since history holds tool blocks). Within a turn the assistant content is appended unchanged; no thinking blocks are stored across turns.
- The camera tool is offered only when `CAMERA_TRIGGER_RE` matches the user's own words, decided once per turn. The default triggers are phrases ("look at", "what do you see"), so a search turn ("look up the news") never offers it.
- memory.json keeps `max_history_messages` entries, each reply with its `tool_calls` records. Web text and photos are never stored (`SEARCH_NOT_RETAINED`, `PHOTO_NOT_RETAINED`); `upgrade_memory` scrubs files from older builds.
- `anthropic` is pinned below 1.0: `_cancel_active_stream` reads the 0.x SDK's `response.extensions["network_stream"]`. Upgrade across the major version deliberately.

## Privacy

`.env`, `memory.json` and `agent.log*` hold the API key and transcripts. The agent and `setup.sh` keep them at mode 0600 and git ignores them; keep it that way. `log_transcripts` defaults to false, so `agent.log` records activity, not words.

The measurements behind the STT, VAD and TTS constants (thread counts, chunk lengths, timeouts) live in the comments beside each constant and in `FORK_CHANGES.md`.
