# Be More Agent 🤖
**A Customizable, Offline-First AI Agent for Raspberry Pi**

[![Watch the Demo](https://img.youtube.com/vi/l5ggH-YhuAw/maxresdefault.jpg)](https://youtu.be/l5ggH-YhuAw)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-red) ![License](https://img.shields.io/badge/License-MIT-green)

This project turns a Raspberry Pi into a fully functional, conversational AI agent. Unlike cloud-based assistants, this agent runs **100% locally** on your device. It listens for a wake word, processes speech, "thinks" using a local Large Language Model (LLM), and speaks back with a low-latency neural voice—all while displaying reactive face animations.

**It is designed as a blank canvas:** You can easily swap the face images and sound effects to create your own character!

## ✨ Features

* **100% Local Intelligence**: Powered by **llama.cpp** + **llama-swap** (LLM) and **Whisper.cpp** (Speech-to-Text). No API fees, no cloud data usage.
* **Unified Vision & Text**: Uses **Qwen3.5-4B** for both conversation and image understanding — one model does it all.
* **Open Source Wake Word**: Wakes up to your custom model using **OpenWakeWord** (Offline & Free). No access keys required.
* **Hardware-Aware Audio**: Automatically detects your microphone's sample rate and resamples audio on the fly to prevent ALSA errors.
* **Smart Web Search**: Uses DuckDuckGo to find real-time news and information when the LLM doesn't know the answer.
* **Reactive Faces**: The GUI updates the character's face based on its state (Listening, Thinking, Speaking, Idle).
* **Fast Text-to-Speech**: Uses **Piper TTS** for low-latency, high-quality voice generation on the Pi.

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
├── config.json                # User settings (Model, Prompt, Hardware)
├── config.yaml                # llama-swap configuration
├── llama-swap.service         # systemd service for llama-swap
├── wakeword.onnx              # OpenWakeWord model (The "Ear")
├── memory.json                # Conversation history
├── requirements.txt           # Python dependencies
├── llama.cpp/                 # LLM inference engine (built from source)
├── models/                    # GGUF model files
│   ├── Qwen3.5-4B-Q4_K_M.gguf  # Text + Vision model
│   └── mmproj-F16.gguf          # Vision projector
├── whisper.cpp/               # Speech-to-Text engine
├── piper/                     # Piper TTS engine & voice models
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
git clone https://github.com/brenpoly/be-more-agent.git
cd be-more-agent
chmod +x setup.sh
./setup.sh
```
*The setup script will install system libraries, create necessary folders, download Piper TTS, build llama.cpp from source, download the Qwen3.5-4B model files, install llama-swap, enable it as a systemd service, and set up the Python virtual environment.*

### 3. Configure the Wake Word
The setup script downloads a default wake word ("Hey Jarvis"). To use your own:
1. Train a model at [OpenWakeWord](https://github.com/dscripka/openWakeWord).
2. Place the `.onnx` file in the root folder.
3. Rename it to `wakeword.onnx`.

### 4. Run the Agent
```bash
source venv/bin/activate
python agent.py
```

The GUI is designed for the Raspberry Pi's DSI display interface (800x480). The script automatically sets `DISPLAY=:0` so it renders to the DSI screen even when launched via SSH or a systemd service. If you need to target a different display, set the `DISPLAY` environment variable before launching.

---

## 📂 Configuration

### `config.json` — Agent Settings

You can modify the hardware behavior and personality in `config.json`:

```json
{
    "text_model": "qwen3.5-4b",
    "voice_model": "piper/en_GB-semaine-medium.onnx",
    "chat_memory": true,
    "camera_rotation": 0,
    "system_prompt_extras": "You are a helpful robot assistant. Keep responses short and cute."
}
```

| Key | Description |
|-----|-------------|
| `text_model` | Model name (must match a model ID in `config.yaml`) |
| `voice_model` | Path to Piper TTS voice `.onnx` file |
| `chat_memory` | Enable/disable persistent chat history |
| `camera_rotation` | Rotate camera image (0, 90, 180, 270) |
| `system_prompt_extras` | Extra personality instructions appended to the system prompt |
| `llm_base_url` | Override the LLM API endpoint (default: `http://localhost:8080/v1`) |

### `config.yaml` — llama-swap / LLM Server Settings

Controls how llama-swap launches llama-server for each model:

```yaml
models:
  "qwen3.5-4b":
    cmd: |
      ./llama.cpp/build/bin/llama-server \
        --port ${PORT} \
        -m models/Qwen3.5-4B-Q4_K_M.gguf \
        --mmproj models/mmproj-F16.gguf \
        --ctx-size 8192 \
        --chat-template-kwargs '{"enable_thinking":false}'
    ttl: 3600
```

See the [llama-swap documentation](https://github.com/mostlygeek/llama-swap) for advanced configuration options.

---

## 🎨 Customizing Your Character

This software is a generic framework. You can give it a new personality by replacing the assets:

1.  **Faces:** The script looks for PNG sequences in `faces/[state]/`. It will loop through all images found in the folder.
2.  **Sounds:** Put multiple `.wav` files in the `sounds/[category]/` folders. The robot will pick one at random each time (e.g., different "thinking" hums or "error" buzzes).

---

## ⚠️ Troubleshooting

* **`couldn't connect to display ":0"` error:** The Pi must be running a desktop environment (X11 or Wayland). If you installed Raspberry Pi OS Lite (server/headless), you need to install `lightdm` and configure desktop autologin via `sudo raspi-config` — see [Display Requirements](#️-display-requirements). If running over SSH, ensure the desktop is already active on the Pi. You can override the target display with `DISPLAY=:1 python agent.py` if needed.
* **"No search library found":** If web search fails, ensure you are in the virtual environment and `duckduckgo-search` is installed via pip.
* **llama-swap not running:** Check the service status with `sudo systemctl status llama-swap`. View logs with `journalctl -u llama-swap -f`.
* **Shutdown Errors:** When you exit the script (Ctrl+C), you might see `Expression 'alsa_snd_pcm_mmap_begin' failed`. **This is normal.** It just means the audio stream was cut off mid-sample. It does not affect the functionality.
* **Audio Glitches:** If the voice sounds fast or slow, the script attempts to auto-detect sample rates. Ensure your `config.json` points to a valid `.onnx` voice model in the `piper/` folder.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## ⚖️ Legal Disclaimer
**"BMO"** and **"Adventure Time"** are trademarks of **Cartoon Network** (Warner Bros. Discovery).

This project is a **fan creation** built for educational and hobbyist purposes only. It is **not** affiliated with, endorsed by, or connected to Cartoon Network or the official Adventure Time brand in any way. The software provided here is a generic agent framework; users are responsible for the assets they load into it.
