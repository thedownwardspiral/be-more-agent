# Be More Agent 🤖
**A Customizable AI Agent for Raspberry Pi**

[![Watch the Demo](https://img.youtube.com/vi/l5ggH-YhuAw/maxresdefault.jpg)](https://youtu.be/l5ggH-YhuAw)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue) ![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-red) ![License](https://img.shields.io/badge/License-MIT-green)

This project turns a Raspberry Pi into a fully functional, conversational AI agent. It listens for a wake word, processes speech, "thinks" using the **Anthropic API** (Claude), and speaks back with a low-latency neural voice—all while displaying reactive face animations.

**It is designed as a blank canvas:** You can easily swap the face images and sound effects to create your own character!

## ✨ Features

* **Powered by Claude**: Uses the **Anthropic API** for intelligent conversation and vision understanding. Supports any Claude model (Sonnet, Opus, Haiku).
* **On-Device Speech Pipeline**: **Whisper.cpp** for speech-to-text and **Piper TTS** for text-to-speech run locally on the Pi. Whisper model and thread count are configurable for performance tuning.
* **Open Source Wake Word**: Wakes up to your custom model using **OpenWakeWord** (Offline & Free). No access keys required.
* **Hardware-Aware Audio**: Automatically detects your microphone's sample rate and resamples audio on the fly to prevent ALSA errors. Includes an audio energy gate that skips transcription on silent/blank recordings to save CPU.
* **Smart Web Search**: Uses DuckDuckGo to find real-time news and information when Claude doesn't know the answer.
* **Camera Vision, On Request**: Claude can take a photo with the Pi camera and see it, but the camera is only offered on turns where you say a trigger word such as "look", "see" or "picture" (configurable via `camera_trigger_words`).
* **Reactive Faces**: The GUI updates the character's face based on its state (Listening, Thinking, Speaking, Idle).
* **Fast Text-to-Speech**: Uses **Piper TTS** ([piper1-gpl](https://github.com/OHF-Voice/piper1-gpl)) as an in-process Python library, so the voice model stays loaded in memory — no reload delay between utterances and no separate server to manage.

## 🛠️ Hardware Requirements

* **Raspberry Pi 5** (Recommended) or Pi 4 (4GB RAM minimum)
* USB Microphone & Speaker
* LCD Screen (official 7" DSI touchscreen recommended — 800x480 native resolution)
* Raspberry Pi Camera Module

## 🖥️ Display Requirements

**A desktop environment (X11/Wayland) must be running on the Pi.** The agent uses tkinter for its GUI, which requires a display server. A headless/server-only Raspberry Pi OS installation will not work — you will get a `couldn't connect to display ":0"` error.

**Recommended:** Use Raspberry Pi OS with Desktop and configure it for desktop autologin:
```bash
sudo raspi-config
# Navigate to: System Options → Boot / Auto Login → Desktop Autologin
```

If you installed Raspberry Pi OS Lite (no desktop), you'll need to install a display manager first:
```bash
sudo apt install -y lightdm
sudo raspi-config
# Then select Desktop Autologin as above, and reboot
```

---

## 📂 Project Structure

```text
be-more-agent/
├── agent.py                   # The main brain script
├── setup.sh                   # Auto-installer script
├── start_agent.sh             # Launch script (runs agent.py via .venv)
├── be-more-agent.desktop      # Autostart template (installed to ~/.config/autostart/)
├── .env                       # API keys (gitignored — copy from example.env)
├── example.env                # Template for .env
├── config.json                # User settings (Model, Prompt, Hardware)
├── wakeword.onnx              # OpenWakeWord model (The "Ear")
├── memory.json                # Conversation history
├── agent.log                  # Runtime + conversation log (rotated, mode 600)
├── pyproject.toml             # Python dependencies (managed by uv)
├── uv.lock                    # Locked dependency versions
├── whisper.cpp/               # Speech-to-Text engine
├── piper/                     # Piper TTS voice models (.onnx + .onnx.json)
├── sounds/                    # Sound effects folder
│   ├── greeting_sounds/       # Startup .wav files
│   ├── thinking_sounds/       # Looping .wav files
│   ├── ack_sounds/            # "I heard you" .wav files
│   └── error_sounds/          # Error/Confusion .wav files
└── faces/                     # Face images folder
    ├── idle/                  # .png sequence for idle state
    ├── listening/             # .png sequence for listening
    ├── thinking/              # .png sequence for thinking
    ├── speaking/              # .png sequence for speaking
    ├── error/                 # .png sequence for errors
    └── warmup/                # .png sequence for startup
```

---

## 🚀 Installation

### 1. Prerequisites
Ensure your Raspberry Pi OS is up to date and has a desktop environment (see [Display Requirements](#️-display-requirements) above).
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install git -y
```

### 2. Clone & Setup
```bash
git clone https://github.com/thedownwardspiral/be-more-agent.git
cd be-more-agent
chmod +x setup.sh
./setup.sh
```
*The setup script will install system libraries (and `uv` itself, if missing), create necessary folders, set up the `.venv` Python virtual environment via `uv sync` (including the `piper-tts` and `anthropic` SDKs and the rest of the locked dependency set from `pyproject.toml` / `uv.lock`), download a Piper voice via `python -m piper.download_voices`, build whisper.cpp, and install a desktop autostart entry so the agent launches with the desktop session (remove `~/.config/autostart/be-more-agent.desktop` to disable).*

### 3. Configure Your API Key
```bash
cp example.env .env
chmod 600 .env
```
Edit `.env` and set your `ANTHROPIC_API_KEY`. Get a key at [console.anthropic.com](https://console.anthropic.com/settings/keys).

### 4. Configure the Wake Word
The setup script downloads a default wake word ("Hey Jarvis"). To use your own:
1. Train a model at [OpenWakeWord](https://github.com/dscripka/openWakeWord).
2. Place the `.onnx` file in the root folder.
3. Rename it to `wakeword.onnx`.

### 5. Run the Agent
```bash
uv run agent.py
```

<sub>(or the classic form: `source .venv/bin/activate` then `python agent.py`)</sub>

The GUI is designed for the Raspberry Pi's DSI display interface (800x480). The script automatically sets `DISPLAY=:0` so it renders to the DSI screen even when launched via SSH or a systemd service. If you need to target a different display, set the `DISPLAY` environment variable before launching.

You don't have to launch from the repo directory: `agent.py` changes into its own folder at startup, so `config.json`, `.env`, `faces/`, `sounds/` and the rest are found wherever you start it from (e.g. `/path/to/be-more-agent/.venv/bin/python /path/to/be-more-agent/agent.py`).

---

## 📂 Configuration

### `.env` — API Keys (Required)

Copy `example.env` to `.env` and configure:

```bash
# Anthropic API key (required)
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Model override (optional, defaults to claude-haiku-5-5)
# ANTHROPIC_MODEL=claude-haiku-5-5
```

The `.env` file is gitignored and will never be committed. Keep it private with `chmod 600 .env` (the agent and `setup.sh` also tighten it to mode 600 if it is looser).

### `config.json` — Agent Settings

You can modify the hardware behavior and personality in `config.json`:

```json
{
    "text_model": "claude-haiku-5-5",
    "voice_model": "piper/en_US-bmo-medium.onnx",
    "whisper_model": "./whisper.cpp/models/ggml-base.en.bin",
    "whisper_threads": 3,
    "audio_energy_threshold": 0.002,
    "chat_memory": true,
    "log_transcripts": true,
    "camera_rotation": 0,
    "system_prompt_extras": "You are a helpful robot assistant. Keep responses short and cute.",
    "input_device": null,
    "input_sample_rate": null,
    "temperature": null,
    "effort": "low"
}
```

Every key is optional; anything you leave out uses its default. Values are checked at startup: a wrong type or out-of-range value, or an unknown (misspelled) key, is reported with a `[CONFIG]` warning and replaced by the default.

| Key | Description |
|-----|-------------|
| `text_model` | Anthropic model name (default: `claude-haiku-5-5`; overridden by `ANTHROPIC_MODEL` env var if set) |
| `voice_model` | Path to Piper TTS voice `.onnx` file (with sibling `.onnx.json`, as written by `python -m piper.download_voices`) |
| `whisper_model` | Path to whisper.cpp model file (default: `ggml-base.en.bin`) |
| `whisper_threads` | Number of CPU threads for whisper transcription (default: `3`) |
| `audio_energy_threshold` | RMS energy below which audio is skipped without transcription (default: `0.002`) |
| `chat_memory` | Persist the conversation to `memory.json` so it survives restarts (default: `true`). `false` never reads or writes the file; history then lasts only for the current run |
| `log_transcripts` | Write what you said (`Heard: ...`) and what the bot said (`[PIPER SPEAKING] ...`, web search queries) to `agent.log` (default: `true`). `false` still logs those lines, but as `<redacted, N chars>` |
| `camera_rotation` | Rotate camera image (0, 90, 180, 270) |
| `system_prompt_extras` | Extra personality instructions appended to the system prompt |
| `input_device` | Microphone to use — a device index, a name substring (e.g. `"USB"`), or `null` for the system default |
| `input_sample_rate` | Preferred mic sample rate; the agent verifies it and falls back through 48000/44100/32000/16000 |
| `max_history_messages` | Most recent chat messages resent to Claude each turn (default: `20`, minimum `2`) |
| `silence_threshold` | Minimum volume that counts as speech while recording; the agent also adapts to the room's noise floor (default: `0.006`) |
| `wake_word_threshold` | Wake-word score needed to trigger, above 0 and at most 1 (default: `0.5`) |
| `wake_word_consecutive` | Consecutive 80 ms frames that must clear `wake_word_threshold` (default: `1`) |
| `camera_trigger_words` | Words/phrases that offer the camera tool to Claude on that turn (whole words; case and punctuation ignored). Default includes "look", "see", "picture", "photo", "camera", "what is this". `[]` disables the camera entirely |
| `temperature` | Sampling temperature, `0.0`–`1.0` (default: `null`, which leaves it out so the model's default applies) |
| `effort` | Thinking effort: `"low"`, `"medium"` or `"high"` (default: `"low"`). `null` means not sent |

If you change `text_model` (or set `ANTHROPIC_MODEL` in `.env`, which overrides it), the `temperature`/`effort` keys must suit the model you choose; a mismatch fails at startup with "Can't reach Claude" (see Troubleshooting).

### Privacy & files

* The conversation can end up in two places, each with its own switch:
  * `memory.json` holds the recent conversation (the last 10 messages) so it survives restarts. Controlled by `chat_memory`; with `"chat_memory": false` it is never read or written.
  * `agent.log` records each turn's transcript (`Heard: ...`, `[PIPER SPEAKING] ...`, web search queries) alongside the runtime log. It is rotated at 1 MB with 3 backups (`agent.log.1`–`.3`). Controlled by `log_transcripts`; with `"log_transcripts": false` those lines are still written but logged as `<redacted, N chars>`, so you can see activity without content. Turning `chat_memory` off does **not** stop these log lines; set both to `false` to keep no transcript on disk.
* `memory.json`, `agent.log` and its backups, plus `.env`, are kept at mode 600 (owner only) by the agent at startup and by `setup.sh`.
* Voice recordings and camera photos are written to a private temporary directory and deleted as soon as they have been used (and the directory is removed at exit; one left behind by a crash or power loss is removed on the next start).
* Say "forget everything" or "reset memory" (a short lead-in like "okay" or the wake word is fine) to wipe the stored conversation. This clears the bot's memory and `memory.json` (deleting the file if `chat_memory` is off), but does **not** purge transcripts already written to `agent.log`; delete `agent.log*` yourself for that. The whole utterance must be the command, so a sentence that merely contains the phrase does nothing.

---

## 🎨 Customizing Your Character

This software is a generic framework. You can give it a new personality by replacing the assets:

1.  **Faces:** The script looks for PNG sequences in `faces/[state]/`. It will loop through all images found in the folder.
2.  **Sounds:** Put multiple `.wav` files in the `sounds/[category]/` folders. The robot will pick one at random each time (e.g., different "thinking" hums or "error" buzzes).

---

## ⚠️ Troubleshooting

* **`couldn't connect to display ":0"` error:** The Pi must be running a desktop environment (X11 or Wayland). If you installed Raspberry Pi OS Lite (server/headless), you need to install `lightdm` and configure desktop autologin via `sudo raspi-config` — see [Display Requirements](#️-display-requirements). If running over SSH, ensure the desktop is already active on the Pi. You can override the target display with `DISPLAY=:1 python agent.py` if needed.
* **"AuthenticationError" or API key errors:** Ensure your `.env` file exists and contains a valid `ANTHROPIC_API_KEY`. Copy from `example.env` if needed.
* **`[CONFIG]` warnings at startup:** Config problems are reported in the console and `agent.log` when the agent starts. A bad value or unknown key is skipped (its default is used); if `config.json` is not valid JSON, the **whole file is ignored** with a loud error banner and every setting is at its default, so fix the syntax and restart.
* **Startup shows "Can't reach Claude: check API key / network":** The warm-up request failed. Check `ANTHROPIC_API_KEY` in `.env`, the network connection, and that the `temperature`/`effort` keys suit `text_model`. The exact API error is in `agent.log`.
* **"No search library found":** If web search fails, ensure you are in the `.venv` virtual environment and `ddgs` is installed (`uv sync`).
* **Shutdown Errors:** When you exit the script (Ctrl+C), you might see `Expression 'alsa_snd_pcm_mmap_begin' failed`. **This is normal.** It just means the audio stream was cut off mid-sample. It does not affect the functionality.
* **Audio Glitches:** If the voice sounds fast or slow, the script attempts to auto-detect sample rates. Ensure your `config.json` points to a valid `.onnx` voice model in the `piper/` folder (with its sibling `.onnx.json` next to it). To switch voices, run `uv run python -m piper.download_voices --data-dir piper <voice-name>`.
* **Choppy/Fragmented Speech:** If the TTS output sounds broken into tiny fragments, the `TTS_MIN_SENTENCE_LENGTH` constant in `agent.py` (default: 80 characters) controls how much text is buffered before being sent to Piper. Increase it for longer, smoother utterances. The first sentence of each reply uses the lower `TTS_FIRST_SENTENCE_MIN_LENGTH` (default: 20 characters) so the bot starts speaking sooner; raise it if a short opening sentence sounds clipped from the rest, or lower it for even faster first audio.

## 🙏 Acknowledgments
This project is based on [be-more-agent](https://github.com/brenpoly/be-more-agent) by brenpoly, released under the MIT License. It has since diverged substantially; see [FORK_CHANGES.md](FORK_CHANGES.md) for the list of changes.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## ⚖️ Legal Disclaimer
**"BMO"** and **"Adventure Time"** are trademarks of **Cartoon Network** (Warner Bros. Discovery).

This project is a **fan creation** built for educational and hobbyist purposes only. It is **not** affiliated with, endorsed by, or connected to Cartoon Network or the official Adventure Time brand in any way. The software provided here is a generic agent framework; users are responsible for the assets they load into it.
