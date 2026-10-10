# =========================================================================
#  Be More Agent 🤖
#  A Local, Offline-First AI Agent for Raspberry Pi
#
#  Copyright (c) 2026 brenpoly
#  Licensed under the MIT License
#  Source: https://github.com/brenpoly/be-more-agent
#
#  DISCLAIMER:
#  This software is provided "as is", without warranty of any kind.
#  This project is a generic framework and includes no copyrighted assets.
# =========================================================================

import os
os.environ.setdefault("DISPLAY", ":0")

from pathlib import Path

# Every relative path in this file (config.json, .env, memory.json, faces/,
# sounds/, piper/, wakeword.onnx, and the whisper.cpp paths in config.json) is
# relative to the repo root. Anchor the process there so the agent works no
# matter which directory it was launched from (SSH, systemd, autostart).
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

import logging
import logging.handlers
import sys

class _PrivateRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """RotatingFileHandler whose log file is always mode 600.

    The log holds transcripts. _open() runs on first open and again after every
    rollover, so creating the file with 0o600 here (and fchmod-ing a file that
    already existed) covers agent.log itself; the numbered backups are renames
    of it and keep the same mode.
    """
    def _open(self):
        fd = os.open(self.baseFilename, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.fchmod(fd, 0o600)
        except OSError:
            pass  # not our file; still usable
        return os.fdopen(fd, self.mode, encoding=self.encoding, errors=self.errors)

LOG_BACKUP_COUNT = 3  # agent.log.1 .. agent.log.3

def _setup_logging():
    handlers = [logging.StreamHandler(sys.stdout)]
    file_error = None
    try:
        handlers.append(_PrivateRotatingFileHandler(
            BASE_DIR / "agent.log", maxBytes=1_000_000, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"))
    except OSError as e:
        file_error = e
    # Root at WARNING so library warnings land in the log too, without the
    # per-request INFO chatter from httpx/anthropic; our own logger is INFO.
    logging.basicConfig(level=logging.WARNING, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger("bmo")
    logger.setLevel(logging.INFO)
    if file_error:
        logger.warning(f"[LOG] Could not open agent.log ({file_error}); logging to stdout only")
    return logger

log = _setup_logging()

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
import json
import subprocess
import random
import re
import select
import atexit
import datetime
import warnings
import wave
import queue
import collections
import signal
import stat
import shutil
import tempfile
import fcntl

# Suppress harmless library warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="ddgs")

# Core dependencies
import sounddevice as sd
import numpy as np
import scipy.signal 

# --- AI ENGINES ---
from openwakeword.model import Model
import anthropic
from piper import PiperVoice
from dotenv import load_dotenv
import base64

# Load environment variables from the repo's .env file
load_dotenv(BASE_DIR / ".env")

# --- WEB SEARCH (Using your working import) ---
from ddgs import DDGS

# =========================================================================
# 1. CONFIGURATION & CONSTANTS
# =========================================================================

CONFIG_FILE = "config.json"
MEMORY_FILE = "memory.json"
BMO_IMAGE_FILE = "current_image.jpg"  # written inside the per-run temp dir
WAKE_WORD_MODEL = "./wakeword.onnx"
# Files holding the API key or conversation transcripts: owner read/write only.
# Includes the rotated log backups, which may predate the 0600 handler.
PRIVATE_FILES = [".env", MEMORY_FILE, "agent.log"] + \
    [f"agent.log.{i}" for i in range(1, LOG_BACKUP_COUNT + 1)]

DEFAULT_CONFIG = {
    "text_model": "claude-haiku-4-5",
    "voice_model": "piper/en_US-bmo-medium.onnx",
    "whisper_model": "./whisper.cpp/models/ggml-base.en.bin",
    "whisper_threads": 3,
    "audio_energy_threshold": 0.002,
    "silence_threshold": 0.006,
    # Raise either of these only if you observe genuine false wakes; 0.5/1 is the
    # long-standing openwakeword default. "consecutive" requires N frames in a row
    # over threshold, which suppresses single-frame noise spikes.
    "wake_word_threshold": 0.5,
    "wake_word_consecutive": 1,
    "chat_memory": True,
    # Write what you said ("Heard: ...") and what the bot said ("[PIPER
    # SPEAKING] ...") to agent.log. false logs only their length, at DEBUG.
    "log_transcripts": True,
    "camera_rotation": 0,
    "system_prompt_extras": "",
    "input_device": None,
    "input_sample_rate": None,
    # Most recent chat messages resent to Claude each turn (user + assistant).
    "max_history_messages": 20,
    # The camera tool is offered to Claude only on turns where the user says one
    # of these (whole words/phrases; case and punctuation ignored). [] = never.
    "camera_trigger_words": [
        "see", "seeing", "look", "looking", "looks", "photo", "photograph",
        "picture", "pic", "snapshot", "selfie", "camera", "view",
        "what is this", "what's this", "what am i holding", "describe this"
    ],
    # Sampling temperature, 0.0-1.0. null omits it (required for Claude 5.x
    # models such as claude-haiku-5-5, which reject non-default values).
    "temperature": 0.7,
    # Thinking effort: "low", "medium" or "high". null omits it (required for
    # claude-haiku-4-5, which rejects it). "low" keeps 5.x voice replies fast.
    "effort": None
}

# LLM SETTINGS
MAX_TOOL_ROUNDS = 5
TTS_MIN_SENTENCE_LENGTH = 80

# Keys that also accept null, with the types accepted otherwise. Every other key
# must match the type of its DEFAULT_CONFIG value.
NULLABLE_CONFIG_TYPES = {
    "input_device": (int, str),
    "input_sample_rate": (int,),
    "temperature": (float,),
    "effort": (str,),
}

# Allowed ranges, checked after the type check: (predicate, description).
CONFIG_RANGES = {
    "wake_word_threshold": (lambda v: 0 < v <= 1, "greater than 0 and at most 1"),
    "wake_word_consecutive": (lambda v: v >= 1, "at least 1"),
    "max_history_messages": (lambda v: v >= 2, "at least 2"),
    "whisper_threads": (lambda v: 1 <= v <= 8, "between 1 and 8"),
    "camera_rotation": (lambda v: v in (0, 90, 180, 270), "one of 0, 90, 180, 270"),
    "audio_energy_threshold": (lambda v: v >= 0, "0 or more"),
    "silence_threshold": (lambda v: v >= 0, "0 or more"),
    "input_sample_rate": (lambda v: v > 0, "a positive number"),
    "temperature": (lambda v: 0.0 <= v <= 1.0, "between 0.0 and 1.0, or null"),
    "effort": (lambda v: v in ("low", "medium", "high"), '"low", "medium", "high", or null'),
}

def _coerce_config_value(key, value, expected):
    """Return value as the expected type, or raise ValueError."""
    if expected is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            coerced = value.strip().lower() == "true"
            log.warning(f"[CONFIG] WARNING: '{key}' is the string {value!r}; treating it as "
                  f"{str(coerced).lower()}. Write it without quotes.")
            return coerced
        raise ValueError("expected true or false")
    if isinstance(value, bool):
        # bool is a subclass of int; true/false is never a valid number here
        raise ValueError("expected a number, not true/false")
    if expected is int:
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise ValueError("expected a whole number")
    if expected is float:
        if isinstance(value, (int, float)):
            return float(value)
        raise ValueError("expected a number")
    if expected is str:
        if isinstance(value, str):
            return value
        raise ValueError("expected a string")
    if expected is list:
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return value
        raise ValueError("expected a list of strings")
    raise ValueError(f"unsupported type {expected.__name__}")

def validate_config(user_cfg):
    """Return the valid entries of user_cfg; warn about and drop everything else."""
    unknown = sorted(k for k in user_cfg if k not in DEFAULT_CONFIG)
    if unknown:
        log.warning(f"[CONFIG] WARNING: ignoring unknown key(s) in {CONFIG_FILE} "
              f"(typo?): {', '.join(unknown)}")

    valid = {}
    for key, value in user_cfg.items():
        if key not in DEFAULT_CONFIG:
            continue
        default = DEFAULT_CONFIG[key]
        try:
            if key in NULLABLE_CONFIG_TYPES and value is None:
                valid[key] = None
                continue
            errors = []
            for expected in NULLABLE_CONFIG_TYPES.get(key, (type(default),)):
                try:
                    coerced = _coerce_config_value(key, value, expected)
                    break
                except ValueError as e:
                    errors.append(str(e))
            else:
                raise ValueError(" or ".join(errors))
            check = CONFIG_RANGES.get(key)
            if check and not check[0](coerced):
                raise ValueError(f"must be {check[1]}")
        except ValueError as e:
            log.warning(f"[CONFIG] WARNING: invalid '{key}': {value!r} ({e}); "
                  f"using default {default!r}")
            continue
        valid[key] = coerced
    return valid

def load_config():
    config = DEFAULT_CONFIG.copy()
    if not os.path.exists(CONFIG_FILE):
        return config
    try:
        with open(CONFIG_FILE, "r") as f:
            user_config = json.load(f)
    except (OSError, ValueError) as e:  # json.JSONDecodeError is a ValueError
        user_config = e
    if not isinstance(user_config, dict):
        reason = user_config if isinstance(user_config, Exception) else \
            f"top level is a {type(user_config).__name__}, not an object"
        log.error("!" * 72)
        log.error(f"[CONFIG] ERROR: {CONFIG_FILE} is unusable: {reason}")
        log.error("[CONFIG] ERROR: IGNORING THE WHOLE FILE. Every setting is at its default.")
        log.error("!" * 72)
        return config
    config.update(validate_config(user_config))
    return config

CURRENT_CONFIG = load_config()
TEXT_MODEL = os.environ.get("ANTHROPIC_MODEL", CURRENT_CONFIG["text_model"])
WAKE_WORD_THRESHOLD = CURRENT_CONFIG["wake_word_threshold"]
WAKE_WORD_CONSECUTIVE = CURRENT_CONFIG["wake_word_consecutive"]
MAX_HISTORY_MESSAGES = CURRENT_CONFIG["max_history_messages"]
CHAT_MEMORY_ENABLED = CURRENT_CONFIG["chat_memory"]
LOG_TRANSCRIPTS = CURRENT_CONFIG["log_transcripts"]

def log_transcript(prefix, text):
    """Log a line carrying the user's or the bot's words, honouring log_transcripts."""
    if LOG_TRANSCRIPTS:
        log.info(f"{prefix} '{text}'")
    else:
        log.debug(f"{prefix} <redacted, {len(text)} chars>")

def _err_summary(e):
    """An exception for the log: the full message only when log_transcripts is
    on, since some messages (e.g. ddgs errors) embed the user's query."""
    return f"{type(e).__name__}: {e}" if LOG_TRANSCRIPTS else type(e).__name__

# Optional request parameters, passed to every Claude call. A null config value
# leaves the parameter out so the model's own default applies.
LLM_KWARGS = {}
if CURRENT_CONFIG["temperature"] is not None:
    LLM_KWARGS["temperature"] = CURRENT_CONFIG["temperature"]
if CURRENT_CONFIG["effort"] is not None:
    LLM_KWARGS["output_config"] = {"effort": CURRENT_CONFIG["effort"]}

# Cap for each chat response. Thinking tokens count toward it, so allow more
# when effort is set; otherwise a long think could leave no room for the reply.
MAX_RESPONSE_TOKENS = 4096 if CURRENT_CONFIG["effort"] else 1024

def restrict_private_files():
    """chmod 600 any existing file that may hold the API key or transcripts."""
    for name in PRIVATE_FILES:
        path = BASE_DIR / name
        try:
            if path.exists() and stat.S_IMODE(path.stat().st_mode) != 0o600:
                os.chmod(path, 0o600)
                log.info(f"[INIT] Restricted permissions on {name} to 600")
        except OSError as e:
            log.warning(f"[WARN] Could not restrict permissions on {name}: {e}")

# whisper.cpp prefixes every segment with "[hh:mm:ss.mmm --> hh:mm:ss.mmm]".
# Match only that prefix so bracketed non-speech markers survive intact.
WHISPER_TIMESTAMP_RE = re.compile(r"^\s*\[[\d:.]+\s*-->\s*[\d:.]+\]\s*")

# Whisper annotates non-speech audio instead of returning nothing: "[BLANK_AUDIO]",
# "(wind blowing)", "*coughs*". Strip these; if nothing is left, there was no speech.
NON_SPEECH_MARKER_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*")

def normalize_words(text):
    """Lowercase, drop punctuation, collapse whitespace: "What's THIS?" -> "whats this".
    Hyphens and dashes split words first: "Bye-bye!" -> "bye bye"."""
    text = re.sub(r"[-–—]", " ", text.lower())
    return " ".join(re.sub(r"[^\w\s]", "", text).split())

# Well-known whisper hallucinations on silent/near-silent input. Only ever compared
# against the *entire* transcript, so a real sentence containing these is unaffected.
# Stored normalized ("amara.org" -> "amaraorg") to match normalize_words() output.
WHISPER_HALLUCINATIONS = {normalize_words(p) for p in (
    "you", "thank you", "thanks for watching", "thank you for watching",
    "bye", "bye bye", "okay", "oh", "so", "um", "uh", "hmm", "mm",
    "please subscribe", "subtitles by the amara.org community",
)}

def is_valid_memory(history):
    """True if history is a well-formed memory.json: a system entry first, then
    user/assistant entries only, each with non-empty string content."""
    if not isinstance(history, list) or not history:
        return False
    for i, m in enumerate(history):
        if not isinstance(m, dict):
            return False
        if m.get("role") not in ("system", "user", "assistant"):
            return False
        if (m["role"] == "system") != (i == 0):
            return False  # system first, and only first
        content = m.get("content")
        if not isinstance(content, str) or not content.strip():
            return False
    return True

# Memory wipe fires only when the *whole* utterance is the command (after a
# short lead-in such as "okay" or the wake word), so "don't forget everything"
# never wipes anything. Matched after normalize.
MEMORY_WIPE_RE = re.compile(
    r"^(?:(?:ok|okay|hey|hi|um|uh|so|please|bmo|jarvis|robot)\s+)*"
    r"(?:forget everything|reset (?:your )?memory)(?:\s+(?:please|now))?$")

class MissingAPIKeyError(RuntimeError):
    pass

# Anthropic client — reads ANTHROPIC_API_KEY from environment / .env. Built on
# first use by get_llm_client(), not at import, so a missing key shows up on the
# ERROR face instead of killing the process before the GUI exists.
llm_client = None

def get_llm_client():
    global llm_client
    if llm_client is None:
        try:
            # The SDK default timeout is 10 minutes; 60s bounds each request (and
            # each idle gap between streamed chunks). With one retry, a black-holed
            # network stalls a turn for at most 2 attempts x 60s plus backoff,
            # about 2 minutes, during which wake word and PTT are not listened to.
            client = anthropic.Anthropic(timeout=60.0, max_retries=1)
        except anthropic.AnthropicError as e:
            raise MissingAPIKeyError(f"Could not create the Anthropic client: {e}. "
                                     "Set ANTHROPIC_API_KEY in .env") from e
        # Recent SDKs don't raise here without a key; the first request then fails
        # with an opaque TypeError. Check up front for a clear message instead.
        # Falsy, not None: an empty "ANTHROPIC_API_KEY=" line counts as missing.
        if (not getattr(client, "api_key", None) and not getattr(client, "auth_token", None)
                and getattr(client, "credentials", None) is None):
            raise MissingAPIKeyError("ANTHROPIC_API_KEY is not set. Add it to .env")
        llm_client = client
    return llm_client

def resolve_input_device(config):
    requested = config.get("input_device")
    if requested in (None, "", "default"):
        return None

    try:
        devices = sd.query_devices()
    except Exception as e:
        log.warning(f"[AUDIO] Device query failed: {e}")
        return None

    if isinstance(requested, int) or (isinstance(requested, str) and requested.isdigit()):
        index = int(requested)
        if 0 <= index < len(devices):
            return index
        log.warning(f"[AUDIO] Input device index not found: {index}")
        return None

    requested_lower = str(requested).lower()
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0 and requested_lower in dev.get("name", "").lower():
            return idx

    log.warning(f"[AUDIO] Input device name not found: {requested}")
    return None

INPUT_DEVICE_NAME = resolve_input_device(CURRENT_CONFIG)
if INPUT_DEVICE_NAME is not None:
    try:
        device_info = sd.query_devices(INPUT_DEVICE_NAME)
        log.info(f"[AUDIO] Using input device: {device_info.get('name', INPUT_DEVICE_NAME)}")
    except Exception:
        log.info(f"[AUDIO] Using input device index: {INPUT_DEVICE_NAME}")

def choose_input_samplerate(device, preferred=None):
    """Pick a sample rate the input device actually supports."""
    candidates = []
    if preferred:
        candidates.append(int(preferred))
    try:
        device_info = sd.query_devices(device, kind='input') if device is not None else sd.query_devices(kind='input')
        if "default_samplerate" in device_info:
            candidates.append(int(device_info["default_samplerate"]))
    except Exception as e:
        log.warning(f"[AUDIO] Input device query failed: {e}")

    candidates.extend([48000, 44100, 32000, 16000])
    seen = set()
    for rate in candidates:
        if not rate or rate in seen:
            continue
        seen.add(rate)
        try:
            sd.check_input_settings(device=device, samplerate=rate, channels=1, dtype="int16")
            return rate
        except Exception:
            continue

    return int(candidates[0]) if candidates else 44100

class BotStates:
    IDLE = "idle"             
    LISTENING = "listening"   
    THINKING = "thinking"     
    SPEAKING = "speaking"     
    ERROR = "error"           
    CAPTURING = "capturing" 
    WARMUP = "warmup"       

# --- SYSTEM PROMPT ---
BASE_SYSTEM_PROMPT = """You are a helpful robot assistant running on a Raspberry Pi.
Personality: Cute, helpful, robot.
Style: Short sentences. Enthusiastic.

You have real tools: you can check the time and search the web. When the user
asks you to look at or see something, you can also take a photo with your
camera (that tool is only offered on those turns). Use a tool whenever the
user asks for something a tool provides; otherwise just chat.

Your replies are spoken aloud, so keep them short and conversational.
You cannot erase your own memory; if asked to, tell the user to say "forget everything".
"""

SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + "\n\n" + CURRENT_CONFIG.get("system_prompt_extras", "")

# --- TOOL DEFINITIONS (Anthropic native tool use) ---
TOOLS = [
    {
        "name": "get_time",
        "description": "Get the current local time.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "search_web",
        "description": "Search the web for news or current information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
            },
            "required": ["query"]
        }
    },
    {
        "name": "capture_image",
        "description": "Take a photo with the onboard camera so you can see the surroundings.",
        "input_schema": {"type": "object", "properties": {}}
    }
]

def build_trigger_re(phrases):
    """Whole-word/phrase matcher for normalized text, or None if phrases is empty."""
    phrases = sorted({p for p in map(normalize_words, phrases) if p}, key=len, reverse=True)
    if not phrases:
        return None
    return re.compile(r"\b(?:" + "|".join(re.escape(p) for p in phrases) + r")\b")

# The camera tool is offered only on turns where the user asked to look at
# something, so a misheard phrase, a TV in the room, or text planted in a
# search result can't make Claude take a photo.
CAMERA_TRIGGER_RE = build_trigger_re(CURRENT_CONFIG["camera_trigger_words"])
TOOLS_WITHOUT_CAMERA = [t for t in TOOLS if t["name"] != "capture_image"]

# Symbols Piper can't read, spelled out before speak() strips non-speech
# characters. Applied in order: "$12.50" must match before a bare "$".
TTS_SYMBOL_SUBS = [
    (re.compile(r"[‘’ʼ]"), "'"),  # curly apostrophes -> '
    # "$5 million" -> "5 million dollars", not "5 dollars million"
    (re.compile(r"\$\s*(\d(?:[\d,]*\d)?(?:\.\d+)?)(\s*(?i:thousand|million|billion|trillion|k|m|bn))?\b"),
     r"\1\2 dollars"),
    (re.compile(r"\$"), " dollars "),
    # Numeric ranges: "15–20" -> "15 to 20", not "15, 20"
    (re.compile(r"(?<=\d)\s*[–—]\s*(?=\d)"), " to "),
    # Dashes are stripped below; read them as a pause instead of gluing words
    (re.compile(r"\s*[–—]\s*"), ", "),
    (re.compile(r"%"), " percent"),
    (re.compile(r"\s*°\s*C\b"), " degrees celsius"),
    (re.compile(r"\s*°\s*F\b"), " degrees fahrenheit"),
    (re.compile(r"\s*°"), " degrees"),
    (re.compile(r"&"), " and "),
    (re.compile(r"\+"), " plus "),
    (re.compile(r"="), " equals "),
    (re.compile(r"(?<=\d)\s*/\s*(?=\d)"), " slash "),
    (re.compile(r"/"), " "),  # and/or -> "and or", not "andor"
    (re.compile(r"#(?=\d)"), " number "),
]

def normalize_for_tts(text):
    """Make text speakable: spell out symbols, then drop emoji and other non-speech characters."""
    for pattern, replacement in TTS_SYMBOL_SUBS:
        text = pattern.sub(replacement, text)
    text = re.sub(r"[^\w\s,.!?:'-]", "", text)
    return re.sub(r"[ \t]+", " ", text).strip()

# Sound Directories
greeting_sounds_dir = "sounds/greeting_sounds"
ack_sounds_dir = "sounds/ack_sounds"
thinking_sounds_dir = "sounds/thinking_sounds"
error_sounds_dir = "sounds/error_sounds"

# =========================================================================
# 2. GUI CLASS
# =========================================================================

class BotGUI:
    BG_WIDTH, BG_HEIGHT = 800, 480 
    OVERLAY_WIDTH, OVERLAY_HEIGHT = 400, 300 

    def __init__(self, master):
        self.master = master
        master.title("Pi Assistant")
        master.attributes('-fullscreen', True) 
        master.bind('<Escape>', self.exit_fullscreen)
        
        # Inputs
        master.bind('<Return>', self.handle_ptt_toggle)
        master.bind('<space>', self.handle_speaking_interrupt)

        # Per-run scratch space for input.wav and camera photos; removed in safe_exit.
        # A flock on .lock inside it, held for the life of the process, marks it
        # as live so another instance's sweep leaves it alone.
        self._sweep_stale_tmp_dirs()
        self.tmp_dir = tempfile.mkdtemp(prefix="be-more-agent-")
        self.tmp_lock_fd = None
        self._lock_tmp_dir()
        # Shown in the IDLE status until a turn succeeds (set by warm_up_logic)
        self.startup_error = None
        self.memory_lock = threading.Lock()

        # State
        self.current_state = BotStates.WARMUP
        self.animations = {}
        self.current_frame_index = 0
        self.current_overlay_image = None
        
        log.info(f"[INIT] Persistent chat memory: {'ON' if CHAT_MEMORY_ENABLED else 'OFF'}")
        self.permanent_memory = self.load_chat_history()
        self.session_memory = []
        self.thinking_sound_active = threading.Event()
        self.thinking_sound_thread = None
        
        self.last_ptt_time = 0 
        self.ptt_event = threading.Event()       
        self.recording_active = threading.Event() 
        self.interrupted = threading.Event() 
        
        self.tts_queue = queue.Queue()
        self.tts_thread = None
        self.tts_active = threading.Event()
        self.exiting = False
        self.turn_cam_path = None
        # True while the text box ends in an unfinished "BOT: ..." line. Only
        # the main-loop thread appends conversation text, so it owns this flag.
        self._bot_line_open = False
        # Worker threads must not touch Tk directly: they queue zero-arg
        # callables here and _process_ui_queue runs them on the Tk thread.
        self.ui_queue = queue.Queue()

        # --- PIPER TTS INITIALIZATION ---
        log.info("[INIT] Loading Piper voice...")
        self.piper_voice = None
        voice_model = CURRENT_CONFIG.get("voice_model", "piper/en_US-bmo-medium.onnx")
        try:
            self.piper_voice = PiperVoice.load(voice_model)
            log.info(f"[INIT] Piper voice loaded: {voice_model}")
        except Exception as e:
            log.error(f"[CRITICAL] Failed to load Piper voice '{voice_model}': {e}")

        # --- WAKE WORD INITIALIZATION ---
        log.info("[INIT] Loading Wake Word...")
        self.oww_model = None
        if os.path.exists(WAKE_WORD_MODEL):
            try:
                self.oww_model = Model(wakeword_model_paths=[WAKE_WORD_MODEL])
                log.info("[INIT] Wake Word Loaded.")
            except TypeError:
                try:
                    self.oww_model = Model(wakeword_models=[WAKE_WORD_MODEL])
                    log.info("[INIT] Wake Word Loaded (New API).")
                except Exception as e:
                    log.error(f"[CRITICAL] Failed to load model: {e}")
            except Exception as e:
                log.error(f"[CRITICAL] Failed to load model: {e}")
        else:
            log.error(f"[CRITICAL] Model not found: {WAKE_WORD_MODEL}")

        # GUI Setup
        self.background_label = tk.Label(master)
        self.background_label.place(x=0, y=0, width=self.BG_WIDTH, height=self.BG_HEIGHT)
        self.background_label.bind('<Button-1>', self.toggle_hud_visibility) 
        
        self.overlay_label = tk.Label(master, bg='black')
        self.overlay_label.bind('<Button-1>', self.toggle_hud_visibility)
        
        self.response_text = tk.Text(master, height=6, width=60, wrap=tk.WORD, 
                                     state=tk.DISABLED, bg="#ffffff", fg="#000000", font=('Arial', 12)) 
        
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(master, textvariable=self.status_var, background="#2e2e2e", foreground="white")
        
        self.exit_button = ttk.Button(master, text="Exit & Save", command=self.safe_exit)

        self.load_animations()
        self.update_animation()
        self._process_ui_queue()

        # Registered last: safe_exit reads exiting, the Events, both memories,
        # memory_lock, tmp_dir and master, all of which now exist. A signal
        # during the (slow) Piper/wake-word loading above gets Python's default
        # handling; any temp dir it leaves is swept on the next start.
        atexit.register(self.safe_exit)
        # SIGTERM (systemd stop, shutdown) and SIGINT (Ctrl+C) take the same
        # save-and-quit path. Python runs these on the main (tkinter) thread
        # the next time it executes bytecode, which the animation timer
        # guarantees at least every 500ms.
        signal.signal(signal.SIGTERM, self._handle_exit_signal)
        signal.signal(signal.SIGINT, self._handle_exit_signal)

        threading.Thread(target=self.safe_main_execution, daemon=True).start()

    # --- HELPERS ---

    def _lock_tmp_dir(self):
        """(Re)create tmp_dir/.lock and hold an exclusive flock on it."""
        fd = None
        try:
            fd = os.open(os.path.join(self.tmp_dir, ".lock"), os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            log.warning(f"[INIT] Could not lock temp dir: {e}")
            if fd is not None:
                os.close(fd)
            return
        if self.tmp_lock_fd is not None:
            os.close(self.tmp_lock_fd)
        self.tmp_lock_fd = fd

    def _ensure_tmp_dir(self):
        """Recreate tmp_dir if it vanished (systemd-tmpfiles ages out idle /tmp
        entries after 10 days), along with the lock file that marks it live."""
        if not os.path.isdir(self.tmp_dir):
            log.warning(f"[TMP] {self.tmp_dir} vanished; recreating it")
        os.makedirs(self.tmp_dir, mode=0o700, exist_ok=True)
        if not os.path.exists(os.path.join(self.tmp_dir, ".lock")):
            self._lock_tmp_dir()

    def _sweep_stale_tmp_dirs(self):
        """Remove temp dirs left by earlier runs that died without safe_exit
        (SIGKILL, power loss). Called before this run creates its own. A dir
        whose .lock is still flocked belongs to a running instance: skip it."""
        removed = 0
        try:
            tmp_root = tempfile.gettempdir()
            uid = os.getuid()
            for entry in os.scandir(tmp_root):
                try:
                    if not (entry.name.startswith("be-more-agent-")
                            and entry.is_dir(follow_symlinks=False)
                            and entry.stat(follow_symlinks=False).st_uid == uid):
                        continue
                    lock_fd = None
                    try:
                        lock_fd = os.open(os.path.join(entry.path, ".lock"), os.O_RDONLY)
                        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except FileNotFoundError:
                        pass  # no lock file: a pre-lock run, or one that died early
                    except OSError:
                        # EWOULDBLOCK: another instance holds it; anything else: leave it
                        if lock_fd is not None:
                            os.close(lock_fd)
                        continue
                    try:
                        shutil.rmtree(entry.path, ignore_errors=True)
                        removed += 1
                    finally:
                        if lock_fd is not None:
                            os.close(lock_fd)  # releases the lock
                except OSError:
                    continue
        except OSError as e:
            log.warning(f"[INIT] Could not sweep old temp dirs: {e}")
        if removed:
            log.info(f"[INIT] Removed {removed} temp dir(s) left by earlier runs")

    def safe_exit(self):
        # Reachable from the Exit button, Escape, and atexit — run once only
        if self.exiting:
            return
        self.exiting = True
        log.info("--- SHUTDOWN SEQUENCE ---")
        self.recording_active.clear()
        self.thinking_sound_active.clear()
        self.tts_active.clear()

        self.save_chat_history()

        try:
            sd.stop()
        except Exception:
            pass
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        # Released only after the dir is gone, so no sweep can race the rmtree
        if self.tmp_lock_fd is not None:
            try:
                os.close(self.tmp_lock_fd)
            except OSError:
                pass
            self.tmp_lock_fd = None
        try:
            self.master.quit()
        except Exception:
            pass

    def _handle_exit_signal(self, signum, frame):
        log.info(f"[SIGNAL] Received {signal.Signals(signum).name}")
        self.safe_exit()
        # quit() is a no-op if the signal landed before mainloop() started
        # (e.g. Ctrl+C while Piper loads); queue one more so it exits on entry
        try:
            self.master.after(0, self.master.quit)
        except Exception:
            pass

    def _remove_file(self, path):
        try:
            os.remove(path)
        except OSError:
            pass

    def _ui(self, fn, *args):
        """Run fn(*args) on the Tk thread. Safe to call from any thread; FIFO."""
        self.ui_queue.put(lambda: fn(*args))

    def _process_ui_queue(self):
        # Tk thread only: drain everything queued since the last pass
        did_work = False
        while True:
            try:
                fn = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            did_work = True
            try:
                fn()
            except tk.TclError:
                if self.exiting:
                    return  # widgets are being torn down
                log.exception("[UI] Update failed")
            except Exception:
                log.exception("[UI] Update failed")
        if self.exiting:
            return
        try:
            # Poll fast while updates are flowing (streamed text), slower when idle
            self.master.after(30 if did_work else 100, self._process_ui_queue)
        except tk.TclError:
            pass  # window destroyed

    def exit_fullscreen(self, event=None):
        self.master.attributes('-fullscreen', False)
        self.safe_exit()

    def toggle_hud_visibility(self, event=None):
        try:
            if self.response_text.winfo_ismapped():
                self.response_text.place_forget()
                self.status_label.place_forget()
                self.exit_button.place_forget()
            else:
                self.response_text.place(relx=0.5, rely=0.82, anchor=tk.S)
                self.status_label.place(relx=0.5, rely=1.0, anchor=tk.S, relwidth=1)
                self.exit_button.place(x=10, y=10)
        except tk.TclError: pass

    def handle_ptt_toggle(self, event=None):
        current_time = time.time()
        if current_time - self.last_ptt_time < 0.5: 
            return 
        self.last_ptt_time = current_time

        if self.recording_active.is_set():
            log.info("[PTT] Toggle OFF")
            self.recording_active.clear() 
        else:
            if self.current_state == BotStates.IDLE or "Wait" in self.status_var.get():
                log.info("[PTT] Toggle ON")
                self.recording_active.set() 
                self.ptt_event.set()

    def handle_speaking_interrupt(self, event=None):
        # current_state is written by the main-loop thread; reading one
        # attribute is atomic in CPython, and a stale value only skips a press.
        if self.current_state == BotStates.SPEAKING or self.current_state == BotStates.THINKING:
            self.interrupted.set()
            self.thinking_sound_active.clear()
            while not self.tts_queue.empty():
                try:
                    self.tts_queue.get_nowait()
                    self.tts_queue.task_done()
                except queue.Empty:
                    break
            self.set_state(BotStates.IDLE, "Interrupted.")

    def load_animations(self):
        base_path = "faces"
        states = ["idle", "listening", "thinking", "speaking", "error", "capturing", "warmup"] 
        for state in states:
            folder = os.path.join(base_path, state)
            self.animations[state] = []
            if os.path.exists(folder):
                files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])
                for f in files:
                    img = Image.open(os.path.join(folder, f)).resize((self.BG_WIDTH, self.BG_HEIGHT))
                    self.animations[state].append(ImageTk.PhotoImage(img))
            if not self.animations[state]:
                # No images for this state: reuse the idle animation if it loaded
                if self.animations.get("idle"):
                    self.animations[state] = self.animations["idle"]
                else:
                    # Blue screen fallback
                    blank = Image.new('RGB', (self.BG_WIDTH, self.BG_HEIGHT), color='#0000FF')
                    self.animations[state].append(ImageTk.PhotoImage(blank))

    def update_animation(self):
        frames = self.animations.get(self.current_state, []) or self.animations.get(BotStates.IDLE, [])
        if not frames:
            self.master.after(500, self.update_animation)
            return

        if self.current_state == BotStates.SPEAKING:
            if len(frames) > 1:
                self.current_frame_index = random.randint(1, len(frames) - 1)
            else:
                self.current_frame_index = 0 
        else:
            self.current_frame_index = (self.current_frame_index + 1) % len(frames)

        self.background_label.config(image=frames[self.current_frame_index])
        
        speed = 50 if self.current_state == BotStates.SPEAKING else 500
        self.master.after(speed, self.update_animation)

    def set_state(self, state, msg="", cam_path=None):
        # current_state changes immediately on the calling thread, because the
        # worker reads it right after this call; only the drawing is queued.
        # update_animation() takes the frame index modulo the new frame count,
        # so it is safe for it to see the new state before _update runs.
        if msg: log.info(f"[STATE] {state.upper()}: {msg}")
        changed = self.current_state != state
        self.current_state = state

        def _update():
            if changed:
                self.current_frame_index = 0
            if msg: self.status_var.set(msg)
            if cam_path and os.path.exists(cam_path) and state in [BotStates.THINKING, BotStates.SPEAKING]:
                try:
                    img = Image.open(cam_path).resize((self.OVERLAY_WIDTH, self.OVERLAY_HEIGHT))
                    self.current_overlay_image = ImageTk.PhotoImage(img)
                    self.overlay_label.config(image=self.current_overlay_image)
                    self.overlay_label.place(x=200, y=90)
                except Exception as e:
                    log.warning(f"[UI] Could not show camera overlay: {e}")
            else:
                self.overlay_label.place_forget()
        self._ui(_update)

    def append_to_text(self, text, newline=True):
        def _update():
            self.response_text.config(state=tk.NORMAL)
            if newline:
                self.response_text.insert(tk.END, text + "\n")
            else:
                self.response_text.insert(tk.END, text)

            self.response_text.see(tk.END)
            self.response_text.config(state=tk.DISABLED)

        self._ui(_update)

    def _end_bot_line(self):
        """Terminate the "BOT: ..." line, if one is open. Main-loop thread only."""
        if self._bot_line_open:
            self.append_to_text("")
            self._bot_line_open = False

    # =========================================================================
    # 3. TOOL EXECUTION
    # =========================================================================

    def execute_tool(self, name, tool_input):
        """Run a tool requested by Claude. Returns (tool_result content, is_error)."""
        log.info(f"TOOL: {name}")

        if name == "get_time":
            now = datetime.datetime.now().strftime("%I:%M %p")
            return f"The current time is {now}.", False

        if name == "search_web":
            query = tool_input.get("query", "")
            log_transcript("Searching web for:", query)
            try:
                # 'us-en' region is often more stable for CLI queries
                with DDGS() as ddgs:
                    results = []
                    search_errors = 0
                    # 1. News search
                    try:
                        results = list(ddgs.news(query, region='us-en', max_results=1))
                        if results:
                            log.debug(f"[SEARCH] Found News: {results[0].get('title')}")
                    except Exception as e:
                        search_errors += 1
                        log.warning(f"[SEARCH] News Search Error: {_err_summary(e)}")

                    # 2. Text fallback
                    if not results:
                        log.debug("[SEARCH] No news found, trying text search...")
                        try:
                            results = list(ddgs.text(query, region='us-en', max_results=1))
                            if results:
                                log.debug(f"[SEARCH] Found Text: {results[0].get('title')}")
                        except Exception as e:
                            search_errors += 1
                            log.warning(f"[SEARCH] Text Search Error: {_err_summary(e)}")

                    if results:
                        r = results[0]
                        # Safe get
                        title = r.get('title', 'No Title')
                        body = r.get('body', r.get('snippet', 'No Body'))
                        return f"SEARCH RESULTS for '{query}':\nTitle: {title}\nSnippet: {body[:300]}", False
                    if search_errors == 2:
                        # Both searches raised: a failure, not an empty result
                        return "Search failed: the search service is unreachable right now.", True
                    log.info("[SEARCH] Search returned 0 results.")
                    return "The search returned no results.", False
            except Exception as e:
                log.error(f"[SEARCH] Connection/Library Error: {_err_summary(e)}")
                return "Search failed: the internet is unreachable right now.", True

        if name == "capture_image":
            img_path = self.capture_image()
            if not img_path:
                return "Camera error: no image could be captured.", True
            self.turn_cam_path = img_path
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            return [{
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}
            }], False

        return f"Error: unknown tool '{name}'. Available tools: get_time, search_web, capture_image.", True

    # =========================================================================
    # 4. CORE LOGIC
    # =========================================================================

    def safe_main_execution(self):
        try:
            self.warm_up_logic()
            self.tts_active.clear()
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()
            
            while True:
                # One bad turn must not stop the bot listening for good
                try:
                    trigger_source = self.detect_wake_word_or_ptt()
                    if self.interrupted.is_set():
                        self.interrupted.clear()
                        self.set_state(BotStates.IDLE, "Resetting...")
                        continue

                    self.set_state(BotStates.LISTENING, "I'm listening!")
                
                    audio_file = None
                    if trigger_source == "PTT":
                        audio_file = self.record_voice_ptt()
                    else:
                        audio_file = self.record_voice_adaptive()
                
                    if not audio_file: 
                        self.set_state(BotStates.IDLE, "Heard nothing.")
                        continue
                
                    try:
                        user_text = self.transcribe_audio(audio_file)
                    finally:
                        # The recording is only needed for whisper; don't leave it on disk
                        self._remove_file(audio_file)
                    if not user_text:
                        self.set_state(BotStates.IDLE, "Transcription empty.")
                        continue
                
                    self.append_to_text(f"YOU: {user_text}")
                    self.interrupted.clear()
                    self.chat_and_respond(user_text)
                except Exception:
                    log.exception("[MAIN] Turn failed; recovering")
                    self.thinking_sound_active.clear()
                    self._end_bot_line()
                    self.set_state(BotStates.ERROR, "Something went wrong")
                    time.sleep(2)
                    
        except Exception as e:
            log.exception("[CRITICAL] Main loop crashed")
            self.set_state(BotStates.ERROR, f"Fatal Error: {str(e)[:40]}")

    def warm_up_logic(self):
        self.set_state(BotStates.WARMUP, "Warming up brains...")
        try:
            # Reachability check. Same optional params as a real turn, so a model
            # that rejects them fails here, at startup. max_tokens leaves room for
            # models that think by default (thinking counts toward the cap).
            get_llm_client().messages.create(
                model=TEXT_MODEL, max_tokens=32,
                messages=[{"role": "user", "content": "hi"}], **LLM_KWARGS
            )
            log.info(f"Anthropic API is reachable. Model: {TEXT_MODEL}")
        except Exception as e:
            log.error(f"Failed to reach Anthropic API: {e}")
            # Make the failure visible and audible instead of greeting as if all
            # is well. The main loop still starts, so PTT keeps working; the
            # message stays in the status line (and replaces "Brain Freeze!" on
            # a failed turn) until a turn succeeds.
            if isinstance(e, MissingAPIKeyError):
                self.startup_error = "No API key: set ANTHROPIC_API_KEY in .env"
            else:
                self.startup_error = "Can't reach Claude: check API key / network"
            self.set_state(BotStates.ERROR, self.startup_error)
            self.play_sound(self.get_random_sound(error_sounds_dir))
            time.sleep(3)
            return
        self.play_sound(self.get_random_sound(greeting_sounds_dir))
        log.info("Ready.")

    def detect_wake_word_or_ptt(self):
        # Keep a failed warm-up visible until a turn succeeds
        status = f"Waiting... ({self.startup_error})" if self.startup_error else "Waiting..."
        self.set_state(BotStates.IDLE, status)
        self.ptt_event.clear()
        
        if self.oww_model: self.oww_model.reset()

        if self.oww_model is None:
            self.ptt_event.wait()
            self.ptt_event.clear()
            return "PTT"

        CHUNK_SIZE = 1280
        OWW_SAMPLE_RATE = 16000

        input_rate = choose_input_samplerate(INPUT_DEVICE_NAME, CURRENT_CONFIG.get("input_sample_rate"))
        use_resampling = (input_rate != OWW_SAMPLE_RATE)
        input_chunk_size = int(CHUNK_SIZE * (input_rate / OWW_SAMPLE_RATE)) if use_resampling else CHUNK_SIZE

        stream_args = {
            "samplerate": input_rate,
            "channels": 1,
            "dtype": 'int16',
            "blocksize": input_chunk_size,
            "device": INPUT_DEVICE_NAME
        }

        try:
            return self._listen_loop(stream_args, input_chunk_size, CHUNK_SIZE, use_resampling)
        except Exception as e:
            log.warning(f"[AUDIO] Wake stream failed: {e}. Retrying with fallback settings...")
            try:
                # Only the stream buffering changes; keep reading a full 80ms
                # frame and resampling exactly as computed above.
                stream_args["blocksize"] = 1024
                stream_args["latency"] = "high"
                return self._listen_loop(stream_args, input_chunk_size, CHUNK_SIZE, use_resampling)
            except Exception as e2:
                log.error(f"[CRITICAL] Wake Word Stream Error: {e2}")
                self.ptt_event.wait()
                self.ptt_event.clear()
                return "PTT"

    def _listen_loop(self, stream_args, input_chunk_size, target_chunk_size, use_resampling):
        with sd.InputStream(**stream_args) as stream:
            log.info(f"[AUDIO] Listening at {stream_args['samplerate']} Hz, block {stream_args['blocksize']}")
            consecutive_overflows = 0
            try:
                stdin_is_interactive = sys.stdin is not None and sys.stdin.isatty()
            except (ValueError, AttributeError):
                stdin_is_interactive = False
            while True:
                if self.ptt_event.is_set():
                    self.ptt_event.clear()
                    return "PTT"

                # Only poll stdin when it is an interactive terminal. Under the
                # desktop autostart / systemd there is no tty, and select() reports
                # an EOF stdin as permanently readable — which fired a bogus "CLI"
                # trigger on every pass of this loop.
                if stdin_is_interactive:
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
                    if rlist:
                        if sys.stdin.readline() == "":
                            stdin_is_interactive = False  # EOF: stop polling
                        else:
                            return "CLI"

                data, overflow = stream.read(input_chunk_size)
                if overflow:
                    # Counted, not logged per read; the caller logs the raise
                    consecutive_overflows += 1
                    if consecutive_overflows >= 5:
                        raise RuntimeError(f"Persistent audio buffer overflow ({consecutive_overflows} reads in a row)")
                else:
                    if consecutive_overflows:
                        log.warning(f"[AUDIO] Input overflowed on {consecutive_overflows} read(s), then recovered")
                    consecutive_overflows = 0

                audio_data = np.frombuffer(data, dtype=np.int16)
                if audio_data.ndim > 1:
                    audio_data = audio_data.flatten()

                if use_resampling:
                    # Nearest-neighbor slicing — scipy's FFT resample per 80ms
                    # chunk overloads the Pi CPU and causes buffer overflows
                    step = len(audio_data) / target_chunk_size
                    indices = np.arange(0, len(audio_data), step)[:target_chunk_size].astype(int)
                    audio_data = audio_data[indices]

                # Skip inference on near-silence to save CPU
                if np.max(np.abs(audio_data)) <= 200:
                    continue

                self.oww_model.predict(audio_data)
                for mdl in self.oww_model.prediction_buffer.keys():
                    scores = list(self.oww_model.prediction_buffer[mdl])
                    # Require several consecutive frames over threshold. A single
                    # noise spike can clear the bar for one 80ms frame; a real wake
                    # word stays above it for the length of the utterance.
                    recent = scores[-WAKE_WORD_CONSECUTIVE:]
                    if len(recent) < WAKE_WORD_CONSECUTIVE:
                        continue
                    if all(score > WAKE_WORD_THRESHOLD for score in recent):
                        log.info(f"[WAKE] {mdl} triggered ({max(recent):.2f})")
                        self.oww_model.reset()
                        return "WAKE"

    def record_voice_adaptive(self, filename=None):
        filename = filename or os.path.join(self.tmp_dir, "input.wav")
        log.info("Recording (Adaptive)...")
        time.sleep(0.5)
        samplerate = choose_input_samplerate(INPUT_DEVICE_NAME, CURRENT_CONFIG.get("input_sample_rate"))

        silence_threshold = CURRENT_CONFIG.get("silence_threshold", 0.006)
        silence_duration = 1.5
        max_record_time = 30.0
        buffer = []
        silent_chunks = 0
        chunk_duration = 0.05
        chunk_size = int(samplerate * chunk_duration)

        num_silent_chunks = int(silence_duration / chunk_duration)
        num_leading_silence_chunks = int(5.0 / chunk_duration)
        max_chunks = int(max_record_time / chunk_duration)
        recorded_chunks = 0
        silence_started = False
        had_speech = False
        volume_history = []
        recent_volumes = collections.deque(maxlen=5)

        def callback(indata, frames, time_info, status):
            nonlocal silent_chunks, recorded_chunks, silence_started, had_speech
            volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
            buffer.append(indata.copy())
            recorded_chunks += 1
            volume_history.append(volume_norm)
            recent_volumes.append(volume_norm)
            if recorded_chunks < 5: return
            # Room noise often sits above the static threshold, which used to
            # force every recording to run the full max_record_time. Track the
            # ambient floor (20th percentile) and compare a smoothed volume
            # against it, capped so loud early speech can't inflate the bar.
            noise_floor = np.percentile(volume_history, 20)
            dynamic_threshold = max(silence_threshold, min(noise_floor * 2.2, 0.015))
            smoothed = sum(recent_volumes) / len(recent_volumes)
            if smoothed < dynamic_threshold:
                silent_chunks += 1
                # Stop 1.5s after speech ends; allow a longer pause before
                # speech starts so slow starters aren't cut off
                limit = num_silent_chunks if had_speech else num_leading_silence_chunks
                if silent_chunks >= limit: silence_started = True
            else:
                had_speech = True
                silent_chunks = 0

        try:
            # Release any prior stream first — hardware contention freezes the Pi 5
            sd.stop()
            time.sleep(0.2)

            with sd.InputStream(samplerate=samplerate, channels=1, callback=callback,
                                device=INPUT_DEVICE_NAME, blocksize=chunk_size):
                while not silence_started and recorded_chunks < max_chunks:
                    sd.sleep(int(chunk_duration * 1000))
        except Exception as e:
            log.error(f"[AUDIO ERROR] Adaptive Recording Failed: {e}")
            return None

        return self.save_audio_buffer(buffer, filename, samplerate)

    def record_voice_ptt(self, filename=None):
        filename = filename or os.path.join(self.tmp_dir, "input.wav")
        log.info("Recording (PTT)...")
        time.sleep(0.5)
        samplerate = choose_input_samplerate(INPUT_DEVICE_NAME, CURRENT_CONFIG.get("input_sample_rate"))

        buffer = []
        def callback(indata, frames, time_info, status): buffer.append(indata.copy())

        try:
            # Release any prior stream first — hardware contention freezes the Pi 5
            sd.stop()
            time.sleep(0.2)

            with sd.InputStream(samplerate=samplerate, channels=1, callback=callback, device=INPUT_DEVICE_NAME):
                while self.recording_active.is_set(): sd.sleep(50)
        except Exception as e:
            log.error(f"[AUDIO ERROR] PTT Recording Failed: {e}")
            return None

        return self.save_audio_buffer(buffer, filename, samplerate)

    def save_audio_buffer(self, buffer, filename, samplerate=16000):
        if not buffer: return None
        audio_data = np.concatenate(buffer, axis=0).flatten()
        audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
        # Clip first: samples beyond +/-1.0 would wrap around in int16
        audio_data = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
        try:
            self._ensure_tmp_dir()
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(samplerate)
                wf.writeframes(audio_data.tobytes())
        except OSError as e:
            log.error(f"[AUDIO ERROR] Could not save recording: {e}")
            self._remove_file(filename)
            return None
        self.play_sound(self.get_random_sound(ack_sounds_dir))
        return filename

    def _check_audio_energy(self, filename):
        """Return True if audio has enough energy to be worth transcribing."""
        threshold = CURRENT_CONFIG.get("audio_energy_threshold", 0.002)
        try:
            with wave.open(filename, 'rb') as wf:
                data = wf.readframes(wf.getnframes())
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            rms = np.sqrt(np.mean(audio ** 2))
            log.info(f"[AUDIO] RMS energy: {rms:.6f} (threshold: {threshold})")
            return rms >= threshold
        except Exception as e:
            log.warning(f"[AUDIO] Energy check failed: {e}")
            return True  # transcribe on error to be safe

    def _is_speech(self, transcription):
        """False for whisper's non-speech annotations and silence hallucinations.

        A false wake word fire on room noise used to reach the LLM as "(wind
        blowing)" and get answered out loud. Whisper is already telling us there
        was no speech — honour it instead of forwarding it to Claude.
        """
        if not transcription:
            return False

        # Drop bracketed annotations; if nothing survives, it was pure non-speech.
        residue = NON_SPEECH_MARKER_RE.sub("", transcription)
        if not re.search(r"[A-Za-z0-9]", residue):
            log.info("[AUDIO] Non-speech audio — ignoring.")
            return False

        if normalize_words(residue) in WHISPER_HALLUCINATIONS:
            # The text itself was already logged (or redacted) as "Heard:"
            log.info("[AUDIO] Silence hallucination — ignoring.")
            return False

        return True

    def transcribe_audio(self, filename):
        if not self._check_audio_energy(filename):
            log.info("[AUDIO] Skipping transcription — audio too quiet.")
            return ""

        log.info("Transcribing...")
        whisper_model = CURRENT_CONFIG.get("whisper_model", "./whisper.cpp/models/ggml-base.en.bin")
        whisper_threads = str(CURRENT_CONFIG.get("whisper_threads", 3))
        try:
            result = subprocess.run(
                ["./whisper.cpp/build/bin/whisper-cli", "-m", whisper_model, "-l", "en", "-t", whisper_threads, "-f", filename],
                capture_output=True, text=True, timeout=90
            )
            if result.returncode != 0:
                log.error(f"Transcription Error: whisper-cli exited {result.returncode}: {result.stderr.strip()[-300:]}")
                return ""
            # whisper-cli prints one timestamped line per segment; join them all
            segments = [WHISPER_TIMESTAMP_RE.sub("", line).strip() for line in result.stdout.split('\n')]
            transcription = " ".join(seg for seg in segments if seg)
            log_transcript("Heard:", transcription)
            if not self._is_speech(transcription):
                return ""
            return transcription.strip()
        except subprocess.TimeoutExpired:
            log.error("Transcription Error: whisper-cli timed out after 90s")
            return ""
        except Exception as e:
            log.error(f"Transcription Error: {e}")
            return ""

    def capture_image(self):
        self.set_state(BotStates.CAPTURING, "Watching...")
        img_path = os.path.join(self.tmp_dir, BMO_IMAGE_FILE)
        try:
            self._ensure_tmp_dir()
            subprocess.run(["rpicam-still", "-t", "500", "-n", "--width", "640", "--height", "480", "-o", img_path],
                           check=True, timeout=20)
            rotation = CURRENT_CONFIG.get("camera_rotation", 0)
            if rotation != 0:
                img = Image.open(img_path)
                img = img.rotate(rotation, expand=True) 
                img.save(img_path)
            return img_path
        except subprocess.TimeoutExpired:
            log.error("Camera Error: rpicam-still timed out after 20s")
            return None
        except Exception as e:
            log.error(f"Camera Error: {e}")
            return None

    # =========================================================================
    # 5. CHAT & RESPOND
    # =========================================================================

    def _history_messages(self):
        """Chat history as Anthropic messages (skips legacy system entries)."""
        history = []
        recent = [m for m in (self.permanent_memory + self.session_memory) if m["role"] != "system"]
        for msg in recent[-MAX_HISTORY_MESSAGES:]:
            if not history and msg["role"] != "user":
                continue  # API conversations must start with a user message
            history.append(msg)
        return history

    def chat_and_respond(self, text):
        # Drain any stale items from a previous response
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
            except queue.Empty:
                break

        normalized = normalize_words(text)
        if MEMORY_WIPE_RE.match(normalized):
            self.session_memory = []
            self.permanent_memory = [{"role": "system", "content": SYSTEM_PROMPT}]
            if CHAT_MEMORY_ENABLED:
                self.save_chat_history()
            else:
                # save_chat_history() is a no-op with memory off; don't leave a
                # memory.json from an earlier run that had it on
                with self.memory_lock:
                    self._remove_file(MEMORY_FILE)
                    self._remove_file(MEMORY_FILE + ".tmp")
            self.tts_queue.put("Okay. Memory wiped.")
            self.wait_for_tts()
            self.interrupted.clear()
            self.set_state(BotStates.IDLE, "Memory Wiped")
            return

        self.set_state(BotStates.THINKING, "Thinking...")
        self.turn_cam_path = None

        messages = self._history_messages() + [{"role": "user", "content": text}]

        # Decided once from the user's own words; the same tool list is used for
        # every round below, so a tool result can't unlock the camera mid-turn.
        if CAMERA_TRIGGER_RE and CAMERA_TRIGGER_RE.search(normalized):
            turn_tools = TOOLS
            log.info("[VISION] camera tool enabled for this turn")
        else:
            turn_tools = TOOLS_WITHOUT_CAMERA
        turn_tool_names = {t["name"] for t in turn_tools}

        self._start_thinking_sound()

        full_response_text = ""

        try:
            client = get_llm_client()
            for _ in range(MAX_TOOL_ROUNDS):
                sentence_buffer = ""
                final_message = None
                round_started = False

                with client.messages.stream(
                    model=TEXT_MODEL, messages=messages, system=SYSTEM_PROMPT,
                    tools=turn_tools, max_tokens=MAX_RESPONSE_TOKENS, **LLM_KWARGS
                ) as stream:
                    for content in stream.text_stream:
                        if self.interrupted.is_set(): break
                        if not content: continue
                        if not round_started:
                            round_started = True
                            # Text after a tool result continues the same "BOT: "
                            # line; keep it from gluing onto the text before the
                            # tool call ("Let me check.It is...")
                            if full_response_text and not full_response_text[-1].isspace():
                                content = " " + content
                        full_response_text += content

                        self.thinking_sound_active.clear()
                        if self.current_state != BotStates.SPEAKING:
                            self.set_state(BotStates.SPEAKING, "Speaking...", cam_path=self.turn_cam_path)
                        # Tracked separately from current_state: a tool round
                        # switches back to THINKING mid-reply, and that must
                        # not start a second "BOT: " on the same line.
                        if not self._bot_line_open:
                            self.append_to_text("BOT: ", newline=False)
                            self._bot_line_open = True

                        self.append_to_text(content, newline=False)

                        sentence_buffer += content
                        if re.search(r'[.!?][\s\n]*$', sentence_buffer) and len(sentence_buffer.strip()) >= TTS_MIN_SENTENCE_LENGTH:
                            clean_sentence = sentence_buffer.strip()
                            if re.search(r'[a-zA-Z0-9]', clean_sentence):
                                self.tts_queue.put(clean_sentence)
                            sentence_buffer = ""

                    if not self.interrupted.is_set():
                        final_message = stream.get_final_message()

                if self.interrupted.is_set() or final_message is None:
                    break

                # Flush trailing text that never hit sentence-ending punctuation
                clean_sentence = sentence_buffer.strip()
                if clean_sentence and re.search(r'[a-zA-Z0-9]', clean_sentence):
                    self.tts_queue.put(clean_sentence)

                if final_message.stop_reason != "tool_use":
                    break

                # Execute requested tools, then loop so Claude sees the results
                # with the full conversation context
                messages.append({"role": "assistant", "content": final_message.content})
                tool_results = []
                for block in final_message.content:
                    if block.type != "tool_use":
                        continue
                    if block.name not in turn_tool_names:
                        log.warning(f"[TOOL] Refused {block.name}: not offered this turn")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Tool {block.name} is not available right now.",
                            "is_error": True
                        })
                        continue
                    try:
                        result_content, is_error = self.execute_tool(block.name, block.input)
                        tool_result = {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_content
                        }
                        if is_error:
                            tool_result["is_error"] = True
                        tool_results.append(tool_result)
                    except Exception as e:
                        # Report the failure to Claude instead of aborting the turn
                        log.exception(f"[TOOL] {block.name} failed: {e!r}")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Tool {block.name} failed: {e}",
                            "is_error": True
                        })
                messages.append({"role": "user", "content": tool_results})

                self.set_state(BotStates.THINKING, "Reading...", cam_path=self.turn_cam_path)
                self._start_thinking_sound()

            self.thinking_sound_active.clear()
            # Claude answered, so whatever failed at warm-up has recovered
            self.startup_error = None

            if not self.interrupted.is_set():
                self.session_memory.append({"role": "user", "content": text})
                if full_response_text.strip():
                    self.session_memory.append({"role": "assistant", "content": full_response_text})
                if len(self.session_memory) > MAX_HISTORY_MESSAGES:
                    del self.session_memory[:-MAX_HISTORY_MESSAGES]
                self.save_chat_history()

            self.wait_for_tts()
            # Playback is finished; a spacebar interrupt of this response must
            # not discard the next wake word
            self.interrupted.clear()
            self.set_state(BotStates.IDLE, "Ready")

        except Exception as e:
            log.error(f"LLM Error: {e}")
            self.thinking_sound_active.clear()
            # Don't reopen the mic while queued speech is still playing,
            # or the bot records and answers its own voice
            self.wait_for_tts()
            self.interrupted.clear()
            self.set_state(BotStates.ERROR, self.startup_error or "Brain Freeze!")

        # Finish the reply's line on every exit path (done, interrupted, error)
        # so the next "YOU:" starts on a line of its own.
        self._end_bot_line()

        # The photo overlay was built from this file. set_state() applies its
        # updates through self.ui_queue, which runs callbacks FIFO, so
        # deleting through the same queue happens only after every pending
        # overlay update and the IDLE/ERROR state above have been applied.
        if self.turn_cam_path:
            cam_path, self.turn_cam_path = self.turn_cam_path, None
            self._ui(self._remove_file, cam_path)

    def wait_for_tts(self):
        self.tts_queue.join()

    def _tts_worker(self):
        while True:
            try:
                text = self.tts_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if self.interrupted.is_set():
                self.tts_queue.task_done()
                continue
            self.tts_active.set()
            self.speak(text)
            self.tts_active.clear()
            self.tts_queue.task_done()

    def speak(self, text):
        clean = normalize_for_tts(text)
        if not clean.strip(): return
        if self.piper_voice is None:
            log.warning("[PIPER] Voice not loaded; skipping playback.")
            return

        log_transcript("[PIPER SPEAKING]", clean)

        try:
            device_info = sd.query_devices(kind='output')
            native_rate = int(device_info['default_samplerate'])
        except Exception:
            native_rate = 48000

        stream = None
        playback_rate = None
        resample = False
        chunk_size = 2048
        byte_step = chunk_size * 2  # 2 bytes per int16 sample

        try:
            for chunk in self.piper_voice.synthesize(clean):
                if self.interrupted.is_set():
                    break

                # Lazily open the output stream on the first chunk so we know the voice's sample rate.
                if stream is None:
                    voice_rate = chunk.sample_rate
                    try:
                        sd.check_output_settings(device=None, samplerate=voice_rate)
                        playback_rate = voice_rate
                    except Exception:
                        playback_rate = native_rate
                        resample = True
                    stream = sd.RawOutputStream(
                        samplerate=playback_rate, channels=1, dtype='int16',
                        device=None, latency='low', blocksize=chunk_size,
                    )
                    stream.start()

                audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
                if resample:
                    num_samples = int(len(audio) * (playback_rate / chunk.sample_rate))
                    audio = scipy.signal.resample(audio, num_samples).astype(np.int16)

                audio_bytes = audio.tobytes()
                for i in range(0, len(audio_bytes), byte_step):
                    if self.interrupted.is_set():
                        break
                    stream.write(audio_bytes[i:i + byte_step])

            if stream is not None and not self.interrupted.is_set():
                time.sleep(0.5)

        except Exception as e:
            log.error(f"Audio Error: {e}")
        finally:
            if stream is not None:
                try: stream.stop()
                except Exception: pass  # best-effort cleanup
                try: stream.close()
                except Exception: pass  # best-effort cleanup

    def _start_thinking_sound(self):
        # One loop thread at most: two threads calling sd.play() crash PortAudio
        self.thinking_sound_active.set()
        if self.thinking_sound_thread is None or not self.thinking_sound_thread.is_alive():
            self.thinking_sound_thread = threading.Thread(target=self._run_thinking_sound_loop, daemon=True)
            self.thinking_sound_thread.start()

    def _run_thinking_sound_loop(self):
        time.sleep(0.5)
        while self.thinking_sound_active.is_set():
            # Thinking sounds are voice clips — never play one while TTS is
            # speaking or has speech queued, or it sounds like two voices
            if self.tts_active.is_set() or not self.tts_queue.empty():
                time.sleep(0.1)
                continue
            sound = self.get_random_sound(thinking_sounds_dir)
            if sound: self._play_clip_interruptible(sound)
            for _ in range(50):
                if not self.thinking_sound_active.is_set(): return
                time.sleep(0.1)

    def _play_clip_interruptible(self, file_path):
        # Like play_sound, but cuts the clip as soon as thinking ends or TTS
        # speech starts. All sd.play/sd.stop calls stay on this thread —
        # stopping another thread's playback crashes PortAudio.
        if not file_path or not os.path.exists(file_path): return
        try:
            with wave.open(file_path, 'rb') as wf:
                file_sr = wf.getframerate()
                data = wf.readframes(wf.getnframes())
                audio = np.frombuffer(data, dtype=np.int16)

            playback_rate = file_sr
            try:
                sd.check_output_settings(device=None, samplerate=file_sr)
            except Exception:
                try:
                    device_info = sd.query_devices(kind='output')
                    playback_rate = int(device_info['default_samplerate'])
                except Exception:
                    playback_rate = 48000
                num_samples = int(len(audio) * (playback_rate / file_sr))
                audio = scipy.signal.resample(audio, num_samples).astype(np.int16)

            sd.play(audio, playback_rate)
            remaining = len(audio) / playback_rate
            while remaining > 0:
                # Let the clip finish naturally; only cut it if speech is
                # actually queued or playing, so voices never overlap
                if self.tts_active.is_set() or not self.tts_queue.empty():
                    sd.stop()
                    return
                time.sleep(0.05)
                remaining -= 0.05
            sd.wait()
        except Exception as e:
            log.warning(f"[AUDIO] Could not play {file_path}: {e}")

    def get_random_sound(self, directory):
        if os.path.exists(directory):
            files = [f for f in os.listdir(directory) if f.endswith(".wav")]
            return os.path.join(directory, random.choice(files)) if files else None
        return None

    def play_sound(self, file_path):
        if not file_path or not os.path.exists(file_path): return
        try:
            with wave.open(file_path, 'rb') as wf:
                file_sr = wf.getframerate()
                data = wf.readframes(wf.getnframes())
                audio = np.frombuffer(data, dtype=np.int16)

            try:
                device_info = sd.query_devices(kind='output')
                native_rate = int(device_info['default_samplerate'])
            except Exception:
                native_rate = 48000 

            playback_rate = file_sr
            try:
                sd.check_output_settings(device=None, samplerate=file_sr)
            except Exception:
                playback_rate = native_rate
                num_samples = int(len(audio) * (native_rate / file_sr))
                audio = scipy.signal.resample(audio, num_samples).astype(np.int16)

            sd.play(audio, playback_rate)
            sd.wait() 
        except Exception as e:
            log.warning(f"[AUDIO] Could not play {file_path}: {e}")

    def load_chat_history(self):
        fresh = [{"role": "system", "content": SYSTEM_PROMPT}]
        if not CHAT_MEMORY_ENABLED or not os.path.exists(MEMORY_FILE):
            return fresh
        try:
            with open(MEMORY_FILE, "r") as f: history = json.load(f)
        except Exception as e:
            log.warning(f"[WARN] Could not read {MEMORY_FILE}: {e}. Starting fresh.")
            return fresh
        if not is_valid_memory(history):
            log.warning(f"[WARN] {MEMORY_FILE} is malformed. Starting fresh.")
            return fresh
        return history

    def save_chat_history(self):
        if not CHAT_MEMORY_ENABLED: return
        tmp_path = MEMORY_FILE + ".tmp"
        # Called from the main loop after each turn and from safe_exit on the
        # tkinter thread; the lock keeps the two from sharing the temp file,
        # and taking the snapshot under it means saves land in the order taken.
        with self.memory_lock:
            # The system entry is rebuilt from the current prompt rather than
            # trusting position 0; only the conversation itself is carried over.
            conv = [m for m in (self.permanent_memory + self.session_memory) if m.get("role") != "system"]
            if len(conv) > 10: conv = conv[-10:]
            data = [{"role": "system", "content": SYSTEM_PROMPT}] + conv
            try:
                # Write a private temp file, then atomically swap it in, so a
                # crash or power loss mid-write never leaves a truncated memory
                fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w") as f:
                    os.fchmod(f.fileno(), 0o600)
                    json.dump(data, f, indent=4)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, MEMORY_FILE)
            except Exception as e:
                log.warning(f"[WARN] Could not save chat history: {e}")

if __name__ == "__main__":
    log.info("--- SYSTEM STARTING ---")
    restrict_private_files()
    root = tk.Tk()
    app = BotGUI(root)
    root.mainloop()
