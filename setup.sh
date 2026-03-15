#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🤖 Pi Local Assistant Setup Script${NC}"

# 1. Install System Dependencies (The "Hidden" Requirements)
echo -e "${YELLOW}[1/6] Installing System Tools (apt)...${NC}"
sudo apt update
sudo apt install -y python3-tk libasound2-dev libportaudio2 libatlas-base-dev cmake build-essential espeak-ng git

# 2. Create Folders
echo -e "${YELLOW}[2/6] Creating Folders...${NC}"
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

# 3. Download Piper (Architecture Check)
echo -e "${YELLOW}[3/6] Setting up Piper TTS...${NC}"
ARCH=$(uname -m)
if [ "$ARCH" == "aarch64" ]; then
    # FIXED: Using the specific 2023.11.14-2 release known to work on Pi
    wget -O piper.tar.gz https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz
    tar -xvf piper.tar.gz -C piper --strip-components=1
    rm piper.tar.gz
else
    echo -e "${RED}⚠️  Not on Raspberry Pi (aarch64). Skipping Piper download.${NC}"
fi

# 4. Download Voice Model
echo -e "${YELLOW}[4/6] Downloading Voice Model...${NC}"
cd piper
wget -nc -O en_GB-semaine-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/semaine/medium/en_GB-semaine-medium.onnx
wget -nc -O en_GB-semaine-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/semaine/medium/en_GB-semaine-medium.onnx.json
cd ..

# 5. Install Python Libraries
echo -e "${YELLOW}[5/6] Installing Python Libraries...${NC}"
# Check if venv exists, if not create it
if [ ! -d ".bmo" ]; then
    python3 -m venv .bmo
fi
source .bmo/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 6. Build llama.cpp
echo -e "${YELLOW}[6/9] Building llama.cpp...${NC}"
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git
fi
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
cd ..

# 7. Download GGUF models
echo -e "${YELLOW}[7/9] Downloading Qwen3.5-4B model files...${NC}"
mkdir -p models
pip install huggingface-hub
if [ ! -f "models/Qwen3.5-4B-Q4_K_M.gguf" ]; then
    huggingface-cli download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf --local-dir models
fi
if [ ! -f "models/mmproj-F16.gguf" ]; then
    huggingface-cli download unsloth/Qwen3.5-4B-GGUF mmproj-F16.gguf --local-dir models
fi

# 8. Install llama-swap
echo -e "${YELLOW}[8/9] Installing llama-swap...${NC}"
if [ ! -f "llama-swap" ]; then
    ARCH=$(uname -m)
    if [ "$ARCH" == "aarch64" ]; then
        SWAP_URL=$(curl -s https://api.github.com/repos/mostlygeek/llama-swap/releases/latest \
            | grep "browser_download_url.*linux_arm64" \
            | head -1 | cut -d '"' -f 4)
    else
        SWAP_URL=$(curl -s https://api.github.com/repos/mostlygeek/llama-swap/releases/latest \
            | grep "browser_download_url.*linux_amd64" \
            | head -1 | cut -d '"' -f 4)
    fi
    if [ -n "$SWAP_URL" ]; then
        curl -L -o llama-swap.tar.gz "$SWAP_URL"
        tar -xvf llama-swap.tar.gz llama-swap
        rm llama-swap.tar.gz
        chmod +x llama-swap
    else
        echo -e "${RED}❌ Could not find llama-swap release. Install manually from https://github.com/mostlygeek/llama-swap/releases${NC}"
    fi
fi

# 9. Install llama-swap systemd service
echo -e "${YELLOW}[9/9] Setting up llama-swap service...${NC}"
sudo cp llama-swap.service /etc/systemd/system/llama-swap.service
sudo systemctl daemon-reload
sudo systemctl enable llama-swap
sudo systemctl start llama-swap

# 7. OpenWakeWord Model (Added this back so the user has a default)
if [ ! -f "wakeword.onnx" ]; then
    echo -e "${YELLOW}Downloading default 'Hey Jarvis' wake word...${NC}"
    curl -L -o wakeword.onnx https://github.com/dscripka/openWakeWord/raw/main/openwakeword/resources/models/hey_jarvis_v0.1.onnx
fi

echo -e "${GREEN}✨ Setup Complete! Run 'source .bmo/bin/activate' then 'python agent.py'${NC}"
