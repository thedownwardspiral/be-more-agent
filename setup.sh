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
echo -e "${YELLOW}[1/7] Installing System Tools (apt)...${NC}"
sudo apt update
BLAS_PKG="libopenblas-dev"
if apt-cache policy libatlas-base-dev 2>/dev/null | grep -q "Candidate:" && \
   ! apt-cache policy libatlas-base-dev 2>/dev/null | grep -q "Candidate: (none)"; then
    BLAS_PKG="libatlas-base-dev"
fi
sudo apt install -y python3-tk libasound2-dev libportaudio2 "$BLAS_PKG" cmake build-essential espeak-ng git

# 2. Create Folders
echo -e "${YELLOW}[2/7] Creating Folders...${NC}"
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
echo -e "${YELLOW}[3/7] Installing Python Libraries...${NC}"
if [ ! -d ".bmo" ]; then
    python3 -m venv .bmo
fi
source .bmo/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Download Voice Model via piper1-gpl's downloader
echo -e "${YELLOW}[4/7] Downloading Voice Model...${NC}"
python -m piper.download_voices --data-dir piper en_US-bmo-medium

# 6. Build whisper.cpp and download model
echo -e "${YELLOW}[6/7] Building whisper.cpp...${NC}"
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

# 7. OpenWakeWord Model
if [ ! -f "wakeword.onnx" ]; then
    echo -e "${YELLOW}Downloading default 'Hey Jarvis' wake word...${NC}"
    curl -L -o wakeword.onnx https://github.com/dscripka/openWakeWord/raw/main/openwakeword/resources/models/hey_jarvis_v0.1.onnx
fi

echo -e "${GREEN}✨ Setup Complete! Run 'source .bmo/bin/activate' then 'python agent.py'${NC}"
