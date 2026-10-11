#!/bin/bash
set -euo pipefail

# whisper.cpp commit to build. Bump this to upgrade whisper.cpp.
WHISPER_CPP_REPO="https://github.com/ggerganov/whisper.cpp.git"
WHISPER_CPP_COMMIT="30c5194c9691e4e9a98b3dea9f19727397d3f46e"

# Fallback wake-word model. openWakeWord moved its models out of the repo into release assets.
WAKEWORD_URL="https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.onnx"
WAKEWORD_SHA256="94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb"

# Speech-to-text: NVIDIA Parakeet-TDT 0.6B v2 (int8, CC-BY-4.0), packaged for
# sherpa-onnx, plus the Silero VAD model that ends recordings. Both go in models/.
PARAKEET_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2"
PARAKEET_SHA256="157c157bc51155e03e37d2466522a3a737dd9c72bb25f36eb18912964161e1ad"
PARAKEET_ARCHIVE_DIR="sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"  # top-level dir inside the archive
PARAKEET_DIR="models/parakeet-tdt-0.6b-v2-int8"                    # where agent.py looks for it
SILERO_VAD_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
SILERO_VAD_SHA256="9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6"

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
echo -e "${YELLOW}[1/10] Installing System Tools (apt)...${NC}"
sudo apt update
BLAS_PKG="libopenblas-dev"
# Capture the policy first: piping straight into `grep -q` can SIGPIPE
# apt-cache, which pipefail would report as a failed check.
ATLAS_POLICY="$(apt-cache policy libatlas-base-dev 2>/dev/null || true)"
if grep -q "Candidate:" <<<"$ATLAS_POLICY" && \
   ! grep -q "Candidate: (none)" <<<"$ATLAS_POLICY"; then
    BLAS_PKG="libatlas-base-dev"
fi
sudo apt install -y python3-tk libasound2-dev libportaudio2 "$BLAS_PKG" cmake build-essential espeak-ng git

# 2. Create Folders
echo -e "${YELLOW}[2/10] Creating Folders...${NC}"
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
echo -e "${YELLOW}[3/10] Installing Python Libraries (uv)...${NC}"
if ! command -v uv &>/dev/null; then
    echo -e "${YELLOW}Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer drops uv in ~/.local/bin, which may not be on PATH yet.
    export PATH="$HOME/.local/bin:$PATH"
fi
# Creates .venv/ and installs the locked dependency set from uv.lock.
uv sync

# 4. Download Voice Model via piper1-gpl's downloader
echo -e "${YELLOW}[4/10] Downloading Voice Model...${NC}"
uv run python -m piper.download_voices --data-dir piper en_US-bmo-medium

# 5. Speech-to-text and voice activity models (Parakeet + Silero VAD)
echo -e "${YELLOW}[5/10] Downloading speech-to-text models...${NC}"
mkdir -p models
if [ -f "$PARAKEET_DIR/encoder.int8.onnx" ] && [ -f "$PARAKEET_DIR/decoder.int8.onnx" ] && \
   [ -f "$PARAKEET_DIR/joiner.int8.onnx" ] && [ -f "$PARAKEET_DIR/tokens.txt" ]; then
    echo -e "${GREEN}Parakeet model already present.${NC}"
else
    echo -e "${YELLOW}Downloading Parakeet-TDT 0.6B v2 int8 (~460 MB, ~650 MB unpacked)...${NC}"
    if ! curl -fL -o models/parakeet.tar.bz2.tmp "$PARAKEET_URL"; then
        rm -f models/parakeet.tar.bz2.tmp
        echo -e "${RED}❌ Failed to download the Parakeet model from $PARAKEET_URL${NC}"
        exit 1
    fi
    if ! echo "$PARAKEET_SHA256  models/parakeet.tar.bz2.tmp" | sha256sum -c --quiet; then
        rm -f models/parakeet.tar.bz2.tmp
        echo -e "${RED}❌ Parakeet model checksum mismatch; refusing to install it.${NC}"
        exit 1
    fi
    echo -e "${YELLOW}Unpacking (takes a few minutes on a Pi)...${NC}"
    # Unpack into a scratch dir, then move into place, so an interrupted run
    # never leaves a half-extracted model at the path agent.py checks
    rm -rf models/.parakeet-extract
    mkdir -p models/.parakeet-extract
    tar -xjf models/parakeet.tar.bz2.tmp -C models/.parakeet-extract
    rm -rf "$PARAKEET_DIR"
    mv "models/.parakeet-extract/$PARAKEET_ARCHIVE_DIR" "$PARAKEET_DIR"
    rm -rf models/.parakeet-extract models/parakeet.tar.bz2.tmp "$PARAKEET_DIR/test_wavs"
fi
if [ ! -f "models/silero_vad.onnx" ]; then
    if ! curl -fL -o models/silero_vad.onnx.tmp "$SILERO_VAD_URL"; then
        rm -f models/silero_vad.onnx.tmp
        echo -e "${RED}❌ Failed to download the Silero VAD model from $SILERO_VAD_URL${NC}"
        exit 1
    fi
    if ! echo "$SILERO_VAD_SHA256  models/silero_vad.onnx.tmp" | sha256sum -c --quiet; then
        rm -f models/silero_vad.onnx.tmp
        echo -e "${RED}❌ Silero VAD model checksum mismatch; refusing to install it.${NC}"
        exit 1
    fi
    mv models/silero_vad.onnx.tmp models/silero_vad.onnx
fi

# 6. Build whisper.cpp and download model (fallback engine, used when stt_engine is "whisper")
echo -e "${YELLOW}[6/10] Building whisper.cpp (fallback engine, used when stt_engine is \"whisper\")...${NC}"
# Check for CMakeLists.txt rather than the directory: an empty or partial
# whisper.cpp/ (e.g. left behind by an old submodule pointer or an interrupted
# clone) must not skip the clone.
if [ ! -f "whisper.cpp/CMakeLists.txt" ]; then
    if [ -d "whisper.cpp" ]; then
        echo -e "${YELLOW}whisper.cpp/ is empty or incomplete; re-cloning...${NC}"
        rm -rf whisper.cpp
    fi
    git clone "$WHISPER_CPP_REPO" whisper.cpp
fi
# Apply the pin on every run, not just after a fresh clone, so an existing
# checkout (or a bumped WHISPER_CPP_COMMIT) builds the pinned revision.
# Require whisper.cpp/.git: without it, git would walk up and act on this repo.
if [ -e "whisper.cpp/.git" ] && git -C whisper.cpp rev-parse --git-dir >/dev/null 2>&1; then
    if [ "$(git -C whisper.cpp rev-parse HEAD)" != "$WHISPER_CPP_COMMIT" ]; then
        echo -e "${YELLOW}Checking out pinned whisper.cpp commit ${WHISPER_CPP_COMMIT:0:12}...${NC}"
        # Fetching a bare SHA needs a reasonably recent git; fall back to a full fetch.
        git -C whisper.cpp fetch --quiet origin "$WHISPER_CPP_COMMIT" \
            || git -C whisper.cpp fetch --quiet origin
        git -C whisper.cpp -c advice.detachedHead=false checkout --quiet "$WHISPER_CPP_COMMIT"
    fi
else
    echo -e "${YELLOW}⚠️  whisper.cpp/ is not a git checkout; cannot apply pin ${WHISPER_CPP_COMMIT:0:12}. Building it as-is.${NC}"
fi
(
    cd whisper.cpp
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release -j"$(nproc)"
    if [ ! -f "models/ggml-base.en.bin" ]; then
        bash models/download-ggml-model.sh base.en
    fi
)
if [ ! -x "whisper.cpp/build/bin/whisper-cli" ]; then
    echo -e "${RED}❌ whisper.cpp build failed: whisper.cpp/build/bin/whisper-cli not found.${NC}"
    exit 1
fi

# 7. OpenWakeWord Model
echo -e "${YELLOW}[7/10] Checking Wake Word Model...${NC}"
if [ ! -f "wakeword.onnx" ]; then
    echo -e "${YELLOW}Downloading default 'Hey Jarvis' wake word...${NC}"
    if ! curl -fL -o wakeword.onnx.tmp "$WAKEWORD_URL"; then
        rm -f wakeword.onnx.tmp
        echo -e "${RED}❌ Failed to download wake word model from $WAKEWORD_URL${NC}"
        exit 1
    fi
    if ! echo "$WAKEWORD_SHA256  wakeword.onnx.tmp" | sha256sum -c --quiet; then
        rm -f wakeword.onnx.tmp
        echo -e "${RED}❌ Wake word model checksum mismatch; refusing to install it.${NC}"
        exit 1
    fi
    mv wakeword.onnx.tmp wakeword.onnx
fi

# 8. Launch script
echo -e "${YELLOW}[8/10] Making launch script executable...${NC}"
chmod +x start_agent.sh

# 9. Private files: the API key, chat transcripts and the conversation log
# are owner read/write only. The agent enforces this at startup too.
echo -e "${YELLOW}[9/10] Restricting permissions on private files...${NC}"
for f in .env memory.json agent.log agent.log.*; do
    # An unmatched glob stays literal, so test -f skips it under set -e
    if [ -f "$f" ]; then
        chmod 600 "$f"
    fi
done

# 10. Desktop autostart entry (starts the agent with the desktop session)
echo -e "${YELLOW}[10/10] Installing desktop autostart entry...${NC}"
mkdir -p "$HOME/.config/autostart"
sed "s|__BASE_DIR__|$(pwd)|g" be-more-agent.desktop > "$HOME/.config/autostart/be-more-agent.desktop"

if [ ! -f .env ]; then
    echo -e "${YELLOW}Next: create your API key file and keep it private:${NC}"
    echo -e "${YELLOW}  cp example.env .env && chmod 600 .env   # then set ANTHROPIC_API_KEY in .env${NC}"
fi
echo -e "${GREEN}✨ Setup Complete! Run 'uv run agent.py' (or 'source .venv/bin/activate' then 'python agent.py')${NC}"
echo -e "${GREEN}   The agent will also autostart with the desktop session (remove ~/.config/autostart/be-more-agent.desktop to disable).${NC}"
