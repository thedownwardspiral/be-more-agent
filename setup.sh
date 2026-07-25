#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🤖 Pi Local Assistant Setup Script${NC}"

# 0. Check for desktop environment (required for tkinter GUI)
if ! command -v lightdm &>/dev/null && ! command -v gdm3 &>/dev/null && ! command -v sddm &>/dev/null; then
    echo -e "${RED}⚠️  No display manager (lightdm/gdm3/sddm) detected.${NC}"
    echo -e "${RED}   The agent requires a desktop environment (X11/Wayland) for its GUI.${NC}"
    echo -e "${RED}   If you're running Raspberry Pi OS Lite, install one with:${NC}"
    echo -e "${RED}     sudo apt install -y lightdm${NC}"
    echo -e "${RED}   Then run: sudo raspi-config → System Options → Boot → Desktop Autologin${NC}"
    echo -e "${YELLOW}   Continuing setup, but the agent won't run without a display server.${NC}"
    echo ""
fi

# 1. Install System Dependencies (The "Hidden" Requirements)
echo -e "${YELLOW}[1/8] Installing System Tools (apt)...${NC}"
sudo apt update
BLAS_PKG="libopenblas-dev"
if apt-cache policy libatlas-base-dev 2>/dev/null | grep -q "Candidate:" && \
   ! apt-cache policy libatlas-base-dev 2>/dev/null | grep -q "Candidate: (none)"; then
    BLAS_PKG="libatlas-base-dev"
fi
sudo apt install -y python3-tk libasound2-dev libportaudio2 "$BLAS_PKG" cmake build-essential espeak-ng git

# 2. Create Folders
echo -e "${YELLOW}[2/8] Creating Folders...${NC}"
mkdir -p piper
mkdir -p sounds/greeting_sounds
mkdir -p sounds/thinking_sounds
mkdir -p sounds/ack_sounds
mkdir -p sounds/error_sounds
mkdir -p faces/idle
mkdir -p faces/listening
mkdir -p faces/thinking
mkdir -p faces/speaking
mkdir -p faces/error
mkdir -p faces/warmup

# 3. Install Python Libraries (Piper TTS now ships as a Python wheel)
echo -e "${YELLOW}[3/8] Installing Python Libraries (uv)...${NC}"
if ! command -v uv &>/dev/null; then
    echo -e "${YELLOW}Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer drops uv in ~/.local/bin, which may not be on PATH yet.
    export PATH="$HOME/.local/bin:$PATH"
fi
# Creates .venv/ and installs the locked dependency set from uv.lock.
uv sync

# 4. Download Voice Model via piper1-gpl's downloader
echo -e "${YELLOW}[4/8] Downloading Voice Model...${NC}"
uv run python -m piper.download_voices --data-dir piper en_US-bmo-medium

# 5. Build whisper.cpp and download model
echo -e "${YELLOW}[5/8] Building whisper.cpp...${NC}"
if [ ! -d "whisper.cpp" ]; then
    git clone https://github.com/ggerganov/whisper.cpp.git
fi
cd whisper.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
if [ ! -f "models/ggml-base.en.bin" ]; then
    bash models/download-ggml-model.sh base.en
fi
cd ..

# 6. OpenWakeWord Model
echo -e "${YELLOW}[6/8] Checking Wake Word Model...${NC}"
if [ ! -f "wakeword.onnx" ]; then
    echo -e "${YELLOW}Downloading default 'Hey Jarvis' wake word...${NC}"
    curl -L -o wakeword.onnx https://github.com/dscripka/openWakeWord/raw/main/openwakeword/resources/models/hey_jarvis_v0.1.onnx
fi

# 7. Launch script
echo -e "${YELLOW}[7/8] Making launch script executable...${NC}"
chmod +x start_agent.sh

# 8. Desktop autostart entry (starts the agent with the desktop session)
echo -e "${YELLOW}[8/8] Installing desktop autostart entry...${NC}"
mkdir -p "$HOME/.config/autostart"
sed "s|__BASE_DIR__|$(pwd)|g" be-more-agent.desktop > "$HOME/.config/autostart/be-more-agent.desktop"

echo -e "${GREEN}✨ Setup Complete! Run 'uv run agent.py' (or 'source .venv/bin/activate' then 'python agent.py')${NC}"
echo -e "${GREEN}   The agent will also autostart with the desktop session (remove ~/.config/autostart/be-more-agent.desktop to disable).${NC}"
