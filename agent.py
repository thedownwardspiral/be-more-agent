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
# sounds/, piper/, models/, wakeword.onnx, and the whisper.cpp paths in config.json) is
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
import math
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
import socket
import stat
import shutil
import tempfile
import traceback
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
# Speech-to-text (Parakeet) and voice activity detection (Silero) run in-process
# through sherpa-onnx. Guarded so a .venv that predates it (start_agent.sh runs
# .venv/bin/python directly, without `uv sync`) still starts, on whisper.cpp.
try:
    import sherpa_onnx
    SHERPA_IMPORT_ERROR = None
except ImportError as e:
    sherpa_onnx = None
    SHERPA_IMPORT_ERROR = e
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
    # Claude Haiku 5.5, paired with temperature null and effort "low" below.
    "text_model": "claude-haiku-5-5",
    "voice_model": "piper/en_US-bmo-medium.onnx",
    # Speech-to-text engine: "parakeet" (NVIDIA Parakeet-TDT 0.6B v2, in-process
    # via sherpa-onnx; models/ is filled by setup.sh step 5) or "whisper" (the
    # whisper.cpp fallback below). A missing Parakeet model falls back to whisper.
    "stt_engine": "parakeet",
    # CPU threads for Parakeet. 2 is fastest on a Pi 5; 3 or 4 are no faster.
    "stt_threads": 2,
    # Used only when stt_engine is "whisper" (or Parakeet could not load)
    "whisper_model": "./whisper.cpp/models/ggml-base.en.bin",
    "whisper_threads": 3,
    "audio_energy_threshold": 0.002,
    # Silero VAD closes a speech segment after vad_min_silence seconds of
    # quiet, then waits vad_resume_window more for the speaker to go on (0 =
    # don't wait); speech in that window is merged into the same utterance.
    # So a recording ends ~1.5 s after the last word by default; raise
    # vad_min_silence if you get cut off mid-sentence, lower either key for
    # snappier turns. Audio counts as speech above vad_threshold (0.1-0.9;
    # raise it in a noisy room).
    "vad_min_silence": 0.9,
    "vad_resume_window": 0.6,
    "vad_threshold": 0.5,
    # Used only if models/silero_vad.onnx is missing (RMS silence detection)
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
    # Sampling temperature, 0.0-1.0. Default null omits it, as Claude 5.x models
    # such as the default claude-haiku-5-5 reject non-default values.
    "temperature": None,
    # Thinking effort: "low", "medium" or "high". Default "low" keeps voice
    # replies fast (thinking is on by default). null omits it.
    "effort": "low"
}

# LLM SETTINGS
MAX_TOOL_ROUNDS = 5
# How often a blocking span (opening the API stream, a web search, the camera)
# checks for a spacebar interrupt, in seconds
INTERRUPT_POLL = 0.1
# Overall cap on one search_web call (both DDG searches together), and the
# per-backend request timeout passed to ddgs
SEARCH_TIMEOUT = 20.0
SEARCH_BACKEND_TIMEOUT = 5
CAMERA_TIMEOUT = 20.0
TTS_MIN_SENTENCE_LENGTH = 80
# Lower bar for the first chunk of a turn, and of the text after each tool
# round: Piper synthesizes a short sentence in ~0.2 s, so waiting for 80 chars
# would dominate time-to-first-audio. Once the bot is talking, later chunks use
# TTS_MIN_SENTENCE_LENGTH to avoid choppy fragments.
TTS_FIRST_SENTENCE_MIN_LENGTH = 20
# A sentence end anywhere in the buffer: punctuation (plus closing quotes or
# brackets) followed by whitespace. Stream deltas rarely end exactly on
# punctuation ("! Let"), so the buffer is split at the last match rather than
# only checked at its end. The end of the buffer is not a sentence end: "3."
# may be followed by "14" in the next delta. Whatever is left when a round
# ends is flushed anyway.
TTS_SENTENCE_END_RE = re.compile(r'[.!?…]+["\'”’)\]]*(?=\s)')
# A word that a period follows without ending the sentence: common
# abbreviations, single letters and dotted initials ("J.", "U.S.", "p.m."),
# and 1-2 digit list numbers ("1. Go hiking").
TTS_ABBREVIATION_RE = re.compile(
    r'(?i)^(?:(?:[a-z]\.)*[a-z]|\d{1,2}|mr|mrs|ms|dr|st|jr|sr|vs|etc|prof|mt|no)$'
)
def last_sentence_end(buffer):
    """Index just past the last sentence end in buffer, or None.

    A single period after an abbreviation, initial or list number
    ("St. Louis", "3 p.m. on", "1. Go") is not a sentence end.
    """
    end = None
    for m in TTS_SENTENCE_END_RE.finditer(buffer):
        if m.group().rstrip('"\'”’)]') == ".":
            token = re.search(r'\S*$', buffer[:m.start()]).group().lstrip('"\'“‘([')
            if TTS_ABBREVIATION_RE.match(token):
                continue
        end = m.end()
    return end


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
    "stt_engine": (lambda v: v in ("parakeet", "whisper"), '"parakeet" or "whisper"'),
    "stt_threads": (lambda v: 1 <= v <= 4, "between 1 and 4"),
    "vad_min_silence": (lambda v: 0.2 <= v <= 3.0, "between 0.2 and 3.0"),
    "vad_resume_window": (lambda v: 0.0 <= v <= 2.0, "between 0.0 and 2.0"),
    "vad_threshold": (lambda v: 0.1 <= v <= 0.9, "between 0.1 and 0.9"),
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
STT_ENGINE = CURRENT_CONFIG["stt_engine"]
STT_THREADS = CURRENT_CONFIG["stt_threads"]

# Both STT engines take 16 kHz mono; recordings are converted to this rate.
STT_SAMPLE_RATE = 16000
PARAKEET_MODEL_DIR = BASE_DIR / "models" / "parakeet-tdt-0.6b-v2-int8"
PARAKEET_FILES = {
    "encoder": "encoder.int8.onnx",
    "decoder": "decoder.int8.onnx",
    "joiner": "joiner.int8.onnx",
    "tokens": "tokens.txt",
}
SILERO_VAD_MODEL = BASE_DIR / "models" / "silero_vad.onnx"
VAD_THRESHOLD = CURRENT_CONFIG["vad_threshold"]
VAD_MIN_SILENCE = CURRENT_CONFIG["vad_min_silence"]
VAD_RESUME_WINDOW = CURRENT_CONFIG["vad_resume_window"]
# Kept around the VAD's speech segment, in 16 kHz samples: Silero puts the
# start ~50 ms into the first word, and the end needs no more than a breath
VAD_PRE_ROLL = int(0.3 * STT_SAMPLE_RATE)
VAD_POST_ROLL = int(0.2 * STT_SAMPLE_RATE)
# Wake-word recordings: hard cap, and how long to wait for speech to start
MAX_RECORD_TIME = 30.0
LEADING_SILENCE_TIME = 5.0
# The VAD reports speech only after min_speech_duration (0.25 s) plus ~0.35 s of
# model lag, so speech starting just before LEADING_SILENCE_TIME is still being
# confirmed at the cutoff; this grace keeps it from being dropped.
LEADING_SILENCE_GRACE = 0.5
RECORD_CHUNK_DURATION = 0.05  # 50 ms input blocks

def resample_to_stt_rate(audio, rate, padtype="constant"):
    """float32 mono audio at `rate` -> float32 at STT_SAMPLE_RATE (polyphase)."""
    rate = int(rate)
    if rate == STT_SAMPLE_RATE:
        return np.asarray(audio, dtype=np.float32)
    g = math.gcd(rate, STT_SAMPLE_RATE)
    return scipy.signal.resample_poly(audio, STT_SAMPLE_RATE // g, rate // g,
                                      padtype=padtype).astype(np.float32)

def log_transcript(prefix, text):
    """Log a line carrying the user's or the bot's words, honouring log_transcripts."""
    if LOG_TRANSCRIPTS:
        log.info(f"{prefix} '{text}'")
    else:
        # Still at INFO, so the log shows the turn happened, just not its words
        log.info(f"{prefix} <redacted, {len(text)} chars>")

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

# Cap for each chat response. Thinking tokens count toward it, so allow 4096
# when effort is set or temperature is null (as in the default setup, where
# thinking is on) so a long think still leaves room for the reply; otherwise 1024.
MAX_RESPONSE_TOKENS = 4096 if (CURRENT_CONFIG["effort"] or CURRENT_CONFIG["temperature"] is None) else 1024

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

# Both sets are only ever compared against the *entire* transcript, so a real
# sentence containing these is unaffected. Stored normalized ("amara.org" ->
# "amaraorg") to match normalize_words() output.
# Pure fillers and non-word sounds: a false wake on a cough or a hesitant "um"
# with nothing after it, from either engine. Never worth sending to Claude.
FILLER_UTTERANCES = {normalize_words(p) for p in (
    "um", "uh", "hmm", "mm", "oh", "so", "ah", "er", "erm",
)}
# Conversational phrases whisper invents on silent/near-silent input. Parakeet
# doesn't, so a real "thank you" or "okay" to it must reach Claude.
WHISPER_ONLY_HALLUCINATIONS = {normalize_words(p) for p in (
    "you", "thank you", "thanks for watching", "thank you for watching",
    "bye", "bye bye", "okay", "please subscribe",
    "subtitles by the amara.org community",
)}

# A saved reply's tool calls ("tool_calls" in memory.json), replayed to Claude
# as real tool_use / tool_result blocks so it can see that an earlier answer
# came from a tool. Kept small: inputs and results are capped, and photos are
# never stored. Ids and names must fit the API's tool_use patterns, or every
# later request would be rejected.
TOOL_RECORD_INPUT_MAX = 200   # per string value in the tool input
TOOL_RECORD_RESULT_MAX = 300
TOOL_RECORD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _cap_text(text, limit):
    return text if len(text) <= limit else text[:limit - 1] + "…"


def make_tool_record(name, tool_input, result_content, is_error):
    """A compact, JSON-safe record of one executed tool call."""
    if isinstance(result_content, str):
        result = result_content
    elif name == "capture_image" and not is_error:
        result = "photo captured (image not retained)"
    else:
        result = "(result not retained)"
    if not isinstance(tool_input, dict):
        tool_input = {}
    return {
        "id": f"toolu_mem_{os.urandom(4).hex()}",
        "name": name,
        "input": {k: _cap_text(v, TOOL_RECORD_INPUT_MAX) if isinstance(v, str) else v
                  for k, v in tool_input.items()},
        "result": _cap_text(result, TOOL_RECORD_RESULT_MAX) or "(no output)",
        "error": bool(is_error),
    }


def is_valid_tool_record(r):
    return (isinstance(r, dict)
            and isinstance(r.get("id"), str) and bool(TOOL_RECORD_ID_RE.match(r["id"]))
            and isinstance(r.get("name"), str) and bool(TOOL_RECORD_ID_RE.match(r["name"]))
            and isinstance(r.get("input"), dict)
            and isinstance(r.get("result"), str)
            and isinstance(r.get("error"), bool))


def is_valid_memory(history):
    """True if history is a well-formed memory.json: a system entry first, then
    user/assistant entries only, each with non-empty string content. An
    assistant entry may carry an optional "tool_calls" list of tool records
    (see make_tool_record; ids unique across the file). Any entry may still
    carry a legacy "tools" list of strings from older builds, which
    load_chat_history() drops."""
    if not isinstance(history, list) or not history:
        return False
    seen_ids = set()
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
        if "tools" in m:
            tools = m["tools"]
            if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
                return False
        if "tool_calls" in m:
            calls = m["tool_calls"]
            if m["role"] != "assistant" or not isinstance(calls, list):
                return False
            for r in calls:
                if not is_valid_tool_record(r) or r["id"] in seen_ids:
                    return False
                seen_ids.add(r["id"])
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
            # about 2 minutes, during which wake word and PTT are not listened to
            # (the spacebar cancels the wait; see _open_stream).
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
Earlier turns may include the tool calls and results that produced your replies.
"""

SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + "\n\n" + CURRENT_CONFIG["system_prompt_extras"]

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

# Marks the end of one tts_queue item's audio on BotGUI.audio_queue
TTS_ITEM_END = object()
# TTS playback writes this many int16 frames at a time and checks for an
# interrupt between writes (~93 ms at 22050 Hz)
TTS_WRITE_FRAMES = 2048
TTS_WRITE_BYTES = TTS_WRITE_FRAMES * 2

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
        self.exiting = False
        self.tmp_lock_fd = None
        # Serializes tmp_lock_fd between _lock_tmp_dir (main loop) and safe_exit
        self.tmp_lock_guard = threading.Lock()
        self._sweep_stale_tmp_dirs()
        # Lock immediately after mkdtemp: until .lock exists, another
        # instance's sweep would take this fresh dir for a dead run's
        self.tmp_dir = tempfile.mkdtemp(prefix="be-more-agent-")
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
        # Set only while detect_wake_word_or_ptt waits for a trigger; Enter
        # starts PTT only then, never while a turn is still finishing
        self.awaiting_input = threading.Event()
        # Makes handle_ptt_toggle's check-and-set atomic against the main
        # loop clearing a press that landed on a wake word
        self.ptt_lock = threading.Lock()
        self.interrupted = threading.Event() 
        # The open Claude response stream, set by the main loop while it reads
        # one; a spacebar interrupt closes it from the Tk thread
        self.active_stream = None
        
        # Two-stage TTS: the synth thread turns tts_queue text into PCM on
        # audio_queue while the playback thread plays the previous sentence
        self.tts_queue = queue.Queue()
        self.tts_thread = None
        # Set while the synth thread works on an item
        self.tts_active = threading.Event()
        # (int16 bytes, sample rate) chunks and TTS_ITEM_END markers; bounded
        # so synthesis can't run far ahead of what is being heard
        self.audio_queue = queue.Queue(maxsize=16)
        self.playback_thread = None
        # Clear while the playback thread has its output stream open
        self.playback_idle = threading.Event()
        self.playback_idle.set()
        # voice sample rate -> (stream rate, resample?), see _playback_format
        self._playback_formats = {}
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
        voice_model = CURRENT_CONFIG["voice_model"]
        try:
            self.piper_voice = PiperVoice.load(voice_model)
            log.info(f"[INIT] Piper voice loaded: {voice_model}")
        except Exception as e:
            log.error(f"[CRITICAL] Failed to load Piper voice '{voice_model}': {e}")

        # --- SPEECH-TO-TEXT INITIALIZATION ---
        # None means whisper.cpp transcribes (configured, or the fallback)
        self.recognizer = None
        # Shown in the IDLE status for the whole run when Parakeet was wanted
        # but could not load, so the slower fallback is not silent
        self.stt_notice = None
        if STT_ENGINE == "parakeet":
            self._load_parakeet()
        else:
            log.info("[INIT] Speech-to-text: whisper.cpp (stt_engine is \"whisper\")")
        # Ends wake-word recordings; None means RMS silence detection
        self.vad = self._load_vad()

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

    def _load_parakeet(self):
        """Load Parakeet into self.recognizer, or log why not and leave whisper on."""
        paths = {k: PARAKEET_MODEL_DIR / f for k, f in PARAKEET_FILES.items()}
        missing = [p for p in paths.values() if not p.is_file()]
        if sherpa_onnx is None:
            reason = f"sherpa-onnx is not importable ({SHERPA_IMPORT_ERROR}); run `uv sync`"
        elif missing:
            reason = (f"model file {missing[0].relative_to(BASE_DIR)} is missing; "
                      "run setup.sh (step 5 downloads it)")
        else:
            log.info(f"[INIT] Loading Parakeet speech-to-text ({STT_THREADS} threads, ~1 GB RAM)...")
            t0 = time.monotonic()
            try:
                self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                    **{k: str(p) for k, p in paths.items()},
                    model_type="nemo_transducer", num_threads=STT_THREADS)
                log.info(f"[INIT] Parakeet loaded in {time.monotonic() - t0:.1f}s")
                return
            except Exception as e:
                reason = f"loading failed: {e}"
        log.error(f"[STT] ERROR: Parakeet unavailable: {reason}. "
                  "Falling back to whisper.cpp for this run.")
        self.stt_notice = "Parakeet unavailable: using whisper"

    def _load_vad(self):
        """Silero VAD for record_voice_adaptive, or None (logged) to use RMS silence."""
        if sherpa_onnx is None:
            reason = "sherpa-onnx is not importable; run `uv sync`"
        elif not SILERO_VAD_MODEL.is_file():
            reason = f"{SILERO_VAD_MODEL.relative_to(BASE_DIR)} is missing; run setup.sh (step 5 downloads it)"
        else:
            try:
                config = sherpa_onnx.VadModelConfig(
                    silero_vad=sherpa_onnx.SileroVadModelConfig(
                        model=str(SILERO_VAD_MODEL), threshold=VAD_THRESHOLD,
                        min_silence_duration=VAD_MIN_SILENCE, min_speech_duration=0.25,
                        # Past this Silero force-splits a segment; never before our own cap
                        max_speech_duration=MAX_RECORD_TIME + 1),
                    sample_rate=STT_SAMPLE_RATE, num_threads=1)
                vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=MAX_RECORD_TIME + 10)
                log.info(f"[INIT] Silero VAD loaded: recordings end {VAD_MIN_SILENCE}s "
                         f"+ {VAD_RESUME_WINDOW}s resume window after speech")
                return vad
            except Exception as e:
                reason = f"loading failed: {e}"
        log.warning(f"[VAD] Silero VAD unavailable: {reason}. "
                    "Ending recordings on RMS silence (1.5 s tail) instead.")
        return None

    def _lock_tmp_dir(self):
        """(Re)create tmp_dir/.lock and hold an exclusive flock on it."""
        with self.tmp_lock_guard:
            # safe_exit has closed (or is closing) the lock fd; don't open another
            if self.exiting:
                return
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
        entries after 10 days), along with the lock file that marks it live.
        Callers must build paths from self.tmp_dir after this, since it can
        move to a fresh dir."""
        # safe_exit removed the dir; don't bring it back behind its rmtree
        if self.exiting:
            raise OSError("shutting down")
        try:
            os.mkdir(self.tmp_dir, 0o700)
            log.warning(f"[TMP] {self.tmp_dir} vanished; recreated it")
        except FileExistsError:
            # Ours, unless the name was freed and someone else (another local
            # user) took it: then never write there, start a fresh dir instead
            st = os.stat(self.tmp_dir, follow_symlinks=False)
            if st.st_uid != os.getuid() or not stat.S_ISDIR(st.st_mode):
                old_dir = self.tmp_dir
                self.tmp_dir = tempfile.mkdtemp(prefix="be-more-agent-")
                self._lock_tmp_dir()  # at once, before any sweep can see it unlocked
                log.warning(f"[TMP] {old_dir} is no longer ours; using {self.tmp_dir}")
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
        # The playback thread sees exiting within ~0.1 s and aborts/closes
        # its own stream (never closed from here: another thread's stream)
        t = self.playback_thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=0.5)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        # Released only after the dir is gone, so no sweep can race the rmtree.
        # Under the guard: the main loop may be in _lock_tmp_dir right now.
        with self.tmp_lock_guard:
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

        with self.ptt_lock:
            if self.recording_active.is_set():
                # Always allowed: ends a PTT recording
                log.info("[PTT] Toggle OFF")
                self.recording_active.clear()
                return
            # ON only while the main loop is waiting for a trigger. current_state
            # alone is not enough: a spacebar interrupt shows IDLE while the
            # worker may still be stuck finishing the turn (a slow API call or
            # tool), and an ON+OFF pair in that window would record nothing.
            if self.current_state == BotStates.IDLE and self.awaiting_input.is_set():
                log.info("[PTT] Toggle ON")
                self.recording_active.set()
                self.ptt_event.set()
                return
        log.info("[PTT] Ignored: still finishing the previous turn")
        # Tk thread: set the status directly, without touching current_state,
        # and put the old text back unless something newer replaced it
        previous = self.status_var.get()
        self.status_var.set("Finishing...")
        def _restore():
            if self.status_var.get() == "Finishing...":
                self.status_var.set(previous)
        try:
            self.master.after(1500, _restore)
        except tk.TclError:
            pass

    def handle_speaking_interrupt(self, event=None):
        # current_state is written by the main-loop thread; reading one
        # attribute is atomic in CPython, and a stale value only skips a press.
        # CAPTURING too: the tool loop drops the photo once the camera returns.
        if self.current_state in (BotStates.SPEAKING, BotStates.THINKING, BotStates.CAPTURING):
            self.interrupted.set()
            # After interrupted.set(): the main loop publishes a new stream and
            # then checks interrupted, so one side always sees the other
            self._cancel_active_stream()
            self.thinking_sound_active.clear()
            while not self.tts_queue.empty():
                try:
                    self.tts_queue.get_nowait()
                    self.tts_queue.task_done()
                except queue.Empty:
                    break
            self.set_state(BotStates.IDLE, "Interrupted.")

    def _cancel_active_stream(self):
        """Shut down the socket of the response stream the main loop is
        reading, so a reader blocked waiting for the next chunk wakes at once
        (with an httpx.RemoteProtocolError or ReadError, which chat_and_respond
        treats as the interrupt). The main loop's own `with stream:` then
        closes the connection; closing it here, from the Tk thread, could race
        the reader's fd (it could be reused before the reader's recv). close()
        also wouldn't help: on Linux, closing a socket fd does not wake a
        thread blocked in poll/recv on it, so a stalled stream would sit until
        the 60s read timeout. shutdown() does wake it."""
        s = self.active_stream
        if s is None:
            return
        sock = None
        try:
            sock = s.response.extensions["network_stream"].get_extra_info("socket")
        except Exception as e:
            log.debug(f"[LLM] No socket to shut down on interrupt: {e!r}")
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
                return
            except Exception as e:
                log.debug(f"[LLM] Shutting the socket down on interrupt failed: {e!r}")
        # No socket to shut down: close() is the best available, though it
        # won't wake a reader already blocked on the socket
        try:
            s.close()
        except Exception as e:
            log.debug(f"[LLM] Closing the stream on interrupt failed: {e!r}")

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

            def search():
                try:
                    # 'us-en' region is often more stable for CLI queries
                    with DDGS(timeout=SEARCH_BACKEND_TIMEOUT) as ddgs:
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

            # On a helper thread, so a spacebar interrupt doesn't wait out a
            # stalled DDG backend
            status, value = self._run_interruptible(search, SEARCH_TIMEOUT, "search_web")
            if status == "done":
                return value
            if status == "interrupted":
                return "interrupted", True
            log.error(f"[SEARCH] Gave up after {SEARCH_TIMEOUT:.0f}s")
            return "Search failed: the search service is not responding right now.", True

        if name == "capture_image":
            img_path = self.capture_image()
            if not img_path:
                if self.interrupted.is_set():
                    return "interrupted", True
                return "Camera error: no image could be captured.", True
            self.turn_cam_path = img_path
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            return [{
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}
            }], False

        return f"Error: unknown tool '{name}'. Available tools: get_time, search_web, capture_image.", True

    def _run_interruptible(self, fn, timeout, label):
        """Run fn() on a short-lived daemon thread while this thread watches for
        a spacebar interrupt. Returns ("done", fn's result), ("interrupted",
        None) or ("timeout", None); re-raises fn's exception. On an interrupt
        or timeout the thread is abandoned: it finishes on its own (bounded by
        fn's own timeouts) and its result is discarded."""
        box = {}

        def run():
            try:
                box["result"] = fn()
            except BaseException as e:
                box["error"] = e

        thread = threading.Thread(target=run, daemon=True, name=f"tool-{label}")
        deadline = time.monotonic() + timeout
        thread.start()
        while thread.is_alive():
            if self.interrupted.is_set():
                log.info(f"[TOOL] {label} cancelled by interrupt; abandoning its thread")
                return "interrupted", None
            if time.monotonic() >= deadline:
                log.warning(f"[TOOL] {label} still running after {timeout:.0f}s; abandoning its thread")
                return "timeout", None
            thread.join(INTERRUPT_POLL)
        if "error" in box:
            raise box["error"]
        return "done", box.get("result")

    @staticmethod
    def _stop_process(proc):
        """Terminate a child process, killing it if it ignores SIGTERM for 1s."""
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    # =========================================================================
    # 4. CORE LOGIC
    # =========================================================================

    def safe_main_execution(self):
        try:
            self.warm_up_logic()
            self._start_tts_threads()
            
            while True:
                # One bad turn must not stop the bot listening for good
                try:
                    trigger_source = self.detect_wake_word_or_ptt()
                    with self.ptt_lock:
                        # Enter from here on is refused until the next wait
                        self.awaiting_input.clear()
                        if trigger_source != "PTT":
                            # An Enter that landed together with the wake word
                            # (current_state was still IDLE) must not leave PTT
                            # armed, or the NEXT turn would start recording
                            # unasked; this turn records adaptively instead
                            self.recording_active.clear()
                            self.ptt_event.clear()
                    if self.interrupted.is_set():
                        self.interrupted.clear()
                        self.set_state(BotStates.IDLE, "Resetting...")
                        continue

                    self.set_state(BotStates.LISTENING, "I'm listening!")
                
                    # 16 kHz float32 mono samples, or None
                    if trigger_source == "PTT":
                        audio = self.record_voice_ptt()
                    else:
                        audio = self.record_voice_adaptive()

                    if audio is None:
                        self.set_state(BotStates.IDLE, "Heard nothing.")
                        continue

                    user_text = self.transcribe_audio(audio)
                    if not user_text:
                        self.set_state(BotStates.IDLE, "Transcription empty.")
                        continue
                
                    self.append_to_text(f"YOU: {user_text}")
                    self.interrupted.clear()
                    self.chat_and_respond(user_text)
                except Exception:
                    log.exception("[MAIN] Turn failed; recovering")
                    # detect_wake_word_or_ptt may have raised mid-wait
                    self.awaiting_input.clear()
                    self.thinking_sound_active.clear()
                    # Never reopen the mic with the TTS stream still open
                    self.wait_for_tts()
                    self.interrupted.clear()
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
        # Keep a failed warm-up visible until a turn succeeds, and an STT
        # fallback for the whole run
        notes = [n for n in (self.startup_error, self.stt_notice) if n]
        status = f"Waiting... ({'; '.join(notes)})" if notes else "Waiting..."
        self.set_state(BotStates.IDLE, status)
        # From here Enter may toggle PTT on (handle_ptt_toggle); the caller
        # clears this once a trigger is returned. Set after IDLE, so the two
        # are never seen as awaiting-but-busy.
        self.awaiting_input.set()
        # Clear first, then check: an Enter between the two is then caught by
        # the check (or by _listen_loop's ptt_event poll), never cleared away.
        self.ptt_event.clear()
        # PTT already toggled on (e.g. an Enter between awaiting_input.set()
        # and here, or one kept across a "Resetting..." pass): record now.
        # Listening instead would make the next Enter toggle it off.
        if self.recording_active.is_set():
            return "PTT"
        
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

    def _wait_for_thinking_sound(self):
        """Let an ending thinking-sound thread finish before this thread calls
        sd.stop(): stopping another thread's playback crashes PortAudio. Its
        clip stops within ~50 ms of thinking_sound_active being cleared."""
        # Every turn clears this on the way out; make sure, since a set flag
        # would keep the loop (and the join below) going
        self.thinking_sound_active.clear()
        t = self.thinking_sound_thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=1.0)
            if t.is_alive():
                log.warning("[AUDIO] Thinking sound still playing; recording anyway")

    def record_voice_adaptive(self):
        """Record until the speaker stops. Returns 16 kHz float32 samples or None."""
        log.info("Recording (Adaptive)...")
        # Nothing plays between the trigger and here (no sound on wake), and the
        # wake-word stream is already closed; the device settle is the 0.2 s
        # after sd.stop() below. Any longer wait only drops the first words.
        time.sleep(0.2)
        samplerate = choose_input_samplerate(INPUT_DEVICE_NAME, CURRENT_CONFIG.get("input_sample_rate"))

        try:
            # Release any prior stream first — hardware contention freezes the Pi 5
            self._wait_for_thinking_sound()
            sd.stop()
            time.sleep(0.2)

            if self.vad is not None:
                audio = self._record_until_vad_end(samplerate)
            else:
                audio = self._record_until_rms_silence(samplerate)
        except Exception as e:
            log.error(f"[AUDIO ERROR] Adaptive Recording Failed: {e}")
            return None

        return self._finish_recording(audio)

    def _record_until_vad_end(self, samplerate):
        """Endpoint with Silero VAD. Returns 16 kHz float32 audio spanning every
        speech segment of the utterance (padded), or None if there was no speech.

        States, advanced once per 50 ms block:
          waiting   - no speech yet. Give up after LEADING_SILENCE_TIME, plus
                      LEADING_SILENCE_GRACE while the VAD is still confirming
                      speech that began just before the cutoff (returns None).
          speaking  - the VAD is inside a segment. Silero closes it after
                      vad_min_silence of quiet.
          resuming  - a segment just closed. Keep listening vad_resume_window
                      longer: if speech starts again ("...weather like <pause>
                      in Boston"), go back to speaking and merge the segments;
                      otherwise stop. With no resumption the recording ends
                      vad_min_silence + vad_resume_window (0.9 + 0.6 = 1.5 s by
                      default) after the last word. A window of 0 stops at once.
        MAX_RECORD_TIME caps the whole recording in any state; the VAD is then
        flushed so speech still in progress counts as a segment.
        """
        chunk_size = int(samplerate * RECORD_CHUNK_DURATION)
        max_samples = int(MAX_RECORD_TIME * STT_SAMPLE_RATE)
        leading_samples = int(LEADING_SILENCE_TIME * STT_SAMPLE_RATE)
        grace_samples = int(LEADING_SILENCE_GRACE * STT_SAMPLE_RATE)
        resume_samples = int(VAD_RESUME_WINDOW * STT_SAMPLE_RATE)
        blocks = queue.Queue()
        raw = []

        def callback(indata, frames, time_info, status):
            # PortAudio thread: only hand the block over; the VAD runs below
            blocks.put(indata.copy())

        vad = self.vad
        vad.reset()
        fed = 0               # 16 kHz samples given to the VAD so far
        heard = False         # the VAD has reported speech at least once
        segments = []         # finished segments as (start, end), 16 kHz samples
        resume_until = None   # set while resuming: stop when fed reaches it
        ended = False
        # Bounds the loop even if the stream stops delivering blocks
        deadline = time.monotonic() + MAX_RECORD_TIME + 5
        with sd.InputStream(samplerate=samplerate, channels=1, callback=callback,
                            device=INPUT_DEVICE_NAME, blocksize=chunk_size):
            while time.monotonic() < deadline:
                try:
                    block = blocks.get(timeout=0.5)
                except queue.Empty:
                    continue
                raw.append(block)
                block = np.clip(np.nan_to_num(block.ravel(), nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
                # Per-block resample just for the VAD ("line" padding keeps the
                # block edges smooth); the utterance itself is resampled whole below
                samples = resample_to_stt_rate(block, samplerate, padtype="line")
                vad.accept_waveform(samples)
                fed += len(samples)

                closed = False  # a segment (speech, then the silence tail) finished
                while not vad.empty():
                    seg = vad.front
                    segments.append((seg.start, seg.start + len(seg.samples)))
                    vad.pop()
                    closed = True
                in_speech = vad.is_speech_detected()
                if in_speech and not heard:
                    heard = True
                    log.info(f"[VAD] Speech started ({fed / STT_SAMPLE_RATE:.2f}s)")

                if closed and not in_speech:
                    # speaking -> resuming (or stop, with no resume window)
                    if resume_samples == 0:
                        ended = True
                        break
                    resume_until = fed + resume_samples
                elif in_speech and resume_until is not None:
                    # resuming -> speaking: the speaker went on after a pause
                    log.info(f"[VAD] Speech resumed ({fed / STT_SAMPLE_RATE:.2f}s)")
                    resume_until = None
                if resume_until is not None and fed >= resume_until:
                    ended = True  # resume window passed in silence
                    break
                if fed >= max_samples:
                    break
                # waiting: no speech by the cutoff, and none being confirmed now
                if not heard and not in_speech and fed >= leading_samples + grace_samples:
                    break

        if not ended:
            vad.flush()  # finalize speech that ran into the time limit
            while not vad.empty():
                seg = vad.front
                segments.append((seg.start, seg.start + len(seg.samples)))
                vad.pop()
        if not segments:
            log.info(f"[VAD] No speech in {fed / STT_SAMPLE_RATE:.1f}s")
            return None
        start = segments[0][0]
        end = segments[-1][1]
        log.info(f"[VAD] Speech {start / STT_SAMPLE_RATE:.2f}-{end / STT_SAMPLE_RATE:.2f}s "
                 f"({len(segments)} segment(s)), stopped at {fed / STT_SAMPLE_RATE:.2f}s")
        audio = self._buffer_to_stt_rate(raw, samplerate)
        if audio is None:
            return None
        # Silero places the start a little into the first word; widen the cut
        # from the full recording rather than using segment.samples as-is
        return audio[max(0, start - VAD_PRE_ROLL):end + VAD_POST_ROLL]

    def _record_until_rms_silence(self, samplerate):
        """Fallback endpointing when the VAD model is missing: stop 1.5 s after
        the volume drops to the ambient noise floor. Returns 16 kHz float32 or None."""
        silence_threshold = CURRENT_CONFIG["silence_threshold"]
        silence_duration = 1.5
        buffer = []
        silent_chunks = 0
        chunk_duration = RECORD_CHUNK_DURATION
        chunk_size = int(samplerate * chunk_duration)

        num_silent_chunks = int(silence_duration / chunk_duration)
        num_leading_silence_chunks = int(LEADING_SILENCE_TIME / chunk_duration)
        max_chunks = int(MAX_RECORD_TIME / chunk_duration)
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
            # force every recording to run the full MAX_RECORD_TIME. Track the
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

        with sd.InputStream(samplerate=samplerate, channels=1, callback=callback,
                            device=INPUT_DEVICE_NAME, blocksize=chunk_size):
            while not silence_started and recorded_chunks < max_chunks:
                sd.sleep(int(chunk_duration * 1000))

        return self._buffer_to_stt_rate(buffer, samplerate)

    def record_voice_ptt(self):
        """Record until Enter is pressed again. Returns 16 kHz float32 samples or None."""
        log.info("Recording (PTT)...")
        # As in record_voice_adaptive: the settle is the 0.2 s after sd.stop()
        time.sleep(0.2)
        samplerate = choose_input_samplerate(INPUT_DEVICE_NAME, CURRENT_CONFIG.get("input_sample_rate"))

        buffer = []
        def callback(indata, frames, time_info, status): buffer.append(indata.copy())

        try:
            # Release any prior stream first — hardware contention freezes the Pi 5
            self._wait_for_thinking_sound()
            sd.stop()
            time.sleep(0.2)

            with sd.InputStream(samplerate=samplerate, channels=1, callback=callback, device=INPUT_DEVICE_NAME):
                while self.recording_active.is_set(): sd.sleep(50)
        except Exception as e:
            log.error(f"[AUDIO ERROR] PTT Recording Failed: {e}")
            # End the PTT session, or detect_wake_word_or_ptt would retry at once
            self.recording_active.clear()
            return None

        return self._finish_recording(self._buffer_to_stt_rate(buffer, samplerate))

    def _buffer_to_stt_rate(self, buffer, samplerate):
        """InputStream callback blocks -> one clean 16 kHz float32 array, or None."""
        if not buffer:
            return None
        audio = np.concatenate(buffer, axis=0).flatten()
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        return resample_to_stt_rate(np.clip(audio, -1.0, 1.0), samplerate)

    def _finish_recording(self, audio):
        """Acknowledge a usable recording with the ack sound; pass it through."""
        if audio is None or not len(audio):
            return None
        self.play_sound(self.get_random_sound(ack_sounds_dir))
        return audio

    def _write_wav(self, audio):
        """16 kHz float32 samples -> tmp_dir/input.wav for whisper-cli. Path or None."""
        filename = None
        try:
            self._ensure_tmp_dir()
            # Resolved after _ensure_tmp_dir, which may have moved tmp_dir
            filename = os.path.join(self.tmp_dir, "input.wav")
            # Clip first: samples beyond +/-1.0 would wrap around in int16
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(STT_SAMPLE_RATE)
                wf.writeframes(pcm.tobytes())
        except OSError as e:
            log.error(f"[AUDIO ERROR] Could not save recording: {e}")
            if filename:
                self._remove_file(filename)
            return None
        return filename

    def _check_audio_energy(self, audio):
        """Return True if audio has enough energy to be worth transcribing."""
        threshold = CURRENT_CONFIG["audio_energy_threshold"]
        try:
            rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
            log.info(f"[AUDIO] RMS energy: {rms:.6f} (threshold: {threshold})")
            return rms >= threshold
        except Exception as e:
            log.warning(f"[AUDIO] Energy check failed: {e}")
            return True  # transcribe on error to be safe

    def _is_speech(self, transcription, engine):
        """False for non-speech annotations, bare fillers, and whisper's silence
        hallucinations.

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

        # The text itself was already logged (or redacted) as "Heard:"
        words = normalize_words(residue)
        if words in FILLER_UTTERANCES:
            log.info("[AUDIO] Filler only — ignoring.")
            return False
        # Whisper's silence hallucinations. Parakeet doesn't invent text on
        # silence, so a real "thank you" or "okay" to it must get through.
        if engine == "whisper" and words in WHISPER_ONLY_HALLUCINATIONS:
            log.info("[AUDIO] Silence hallucination — ignoring.")
            return False

        return True

    def transcribe_audio(self, audio):
        """16 kHz float32 samples -> the user's words, or "" for nothing usable."""
        if not self._check_audio_energy(audio):
            log.info("[AUDIO] Skipping transcription — audio too quiet.")
            return ""

        log.info("Transcribing...")
        engine = "parakeet" if self.recognizer is not None else "whisper"
        t0 = time.monotonic()
        if engine == "parakeet":
            transcription = self._transcribe_parakeet(audio)
        else:
            transcription = self._transcribe_whisper(audio)
        log.info(f"[STT] {engine} {time.monotonic() - t0:.2f}s "
                 f"({len(audio) / STT_SAMPLE_RATE:.1f}s of audio)")
        if transcription is None:
            return ""  # the engine failed; already logged
        log_transcript("Heard:", transcription)
        if not self._is_speech(transcription, engine):
            return ""
        return transcription.strip()

    def _transcribe_parakeet(self, audio):
        """Decode in-process on this (main-loop) thread: ~0.2-1.2 s on a Pi 5."""
        try:
            stream = self.recognizer.create_stream()
            stream.accept_waveform(STT_SAMPLE_RATE, audio)
            self.recognizer.decode_stream(stream)
            return stream.result.text.strip()
        except Exception as e:
            log.error(f"Transcription Error: Parakeet failed: {e}")
            return None

    def _transcribe_whisper(self, audio):
        """Run whisper-cli on a temp WAV. Returns the text, or None on failure."""
        filename = self._write_wav(audio)
        if filename is None:
            return None
        whisper_model = CURRENT_CONFIG["whisper_model"]
        whisper_threads = str(CURRENT_CONFIG["whisper_threads"])
        try:
            result = subprocess.run(
                ["./whisper.cpp/build/bin/whisper-cli", "-m", whisper_model, "-l", "en", "-t", whisper_threads, "-f", filename],
                capture_output=True, text=True, timeout=90
            )
            if result.returncode != 0:
                log.error(f"Transcription Error: whisper-cli exited {result.returncode}: {result.stderr.strip()[-300:]}")
                return None
            # whisper-cli prints one timestamped line per segment; join them all
            segments = [WHISPER_TIMESTAMP_RE.sub("", line).strip() for line in result.stdout.split('\n')]
            return " ".join(seg for seg in segments if seg)
        except subprocess.TimeoutExpired:
            log.error("Transcription Error: whisper-cli timed out after 90s")
            return None
        except Exception as e:
            log.error(f"Transcription Error: {e}")
            return None
        finally:
            # The WAV exists only for whisper-cli; don't leave the recording on disk
            self._remove_file(filename)

    def capture_image(self):
        self.set_state(BotStates.CAPTURING, "Watching...")
        try:
            self._ensure_tmp_dir()
            # Built after _ensure_tmp_dir, which may have moved tmp_dir
            img_path = os.path.join(self.tmp_dir, BMO_IMAGE_FILE)
            # Polled rather than subprocess.run(), so a spacebar interrupt
            # stops the camera at once instead of waiting for it to finish
            cmd = ["rpicam-still", "-t", "500", "-n", "--width", "640", "--height", "480", "-o", img_path]
            proc = subprocess.Popen(cmd)
            deadline = time.monotonic() + CAMERA_TIMEOUT
            while True:
                try:
                    proc.wait(timeout=INTERRUPT_POLL)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if self.interrupted.is_set() or time.monotonic() >= deadline:
                    self._stop_process(proc)
                    # Don't leave a half-written photo behind
                    self._remove_file(img_path)
                    if self.interrupted.is_set():
                        log.info("[CAMERA] Capture cancelled by interrupt")
                        return None
                    raise subprocess.TimeoutExpired(cmd, CAMERA_TIMEOUT)
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd)
            rotation = CURRENT_CONFIG["camera_rotation"]
            if rotation != 0:
                img = Image.open(img_path)
                img = img.rotate(rotation, expand=True) 
                img.save(img_path)
            return img_path
        except subprocess.TimeoutExpired:
            log.error(f"Camera Error: rpicam-still timed out after {CAMERA_TIMEOUT:.0f}s")
            return None
        except Exception as e:
            log.error(f"Camera Error: {e}")
            return None

    # =========================================================================
    # 5. CHAT & RESPOND
    # =========================================================================

    def _history_messages(self):
        """Chat history as Anthropic messages (skips legacy system entries).
        A reply with "tool_calls" is replayed as the calls that produced it:
        an assistant message of tool_use blocks, a user message of the
        matching tool_result blocks, then the reply text. No thinking blocks
        are stored or replayed, so there is nothing for the API to bind."""
        history = []
        recent = [m for m in (self.permanent_memory + self.session_memory) if m["role"] != "system"]
        # MAX_HISTORY_MESSAGES counts memory entries, not rendered API
        # messages: a reply with tool calls renders as three messages.
        for msg in recent[-MAX_HISTORY_MESSAGES:]:
            if not history and msg["role"] != "user":
                continue  # API conversations must start with a user message
            if msg["role"] == "assistant" and history[-1]["role"] == "assistant":
                continue  # memory never has two replies in a row; don't send one
            calls = msg.get("tool_calls") if msg["role"] == "assistant" else None
            if calls:
                history.append({"role": "assistant", "content": [
                    {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["input"]}
                    for c in calls
                ]})
                results = []
                for c in calls:
                    block = {"type": "tool_result", "tool_use_id": c["id"], "content": c["result"]}
                    if c["error"]:
                        block["is_error"] = True
                    results.append(block)
                history.append({"role": "user", "content": results})
            history.append({"role": msg["role"], "content": msg["content"]})
        return history

    def _open_stream(self, client, **kwargs):
        """Send a streaming Messages request and wait for the response to start,
        on a helper thread, so a spacebar interrupt doesn't wait out a stalled
        connection (up to 60s per attempt, with a retry). Returns the open
        MessageStream, or None if interrupted first; an abandoned helper
        closes the stream itself if it opens later. Re-raises request errors."""
        if self.interrupted.is_set():
            log.info("[LLM] Interrupted before the request was sent; not sending it")
            return None
        box = {}
        lock = threading.Lock()

        def run():
            # Last check before anything goes on the wire: the main loop may
            # already have given up on this helper
            with lock:
                if box.get("abandoned") or self.interrupted.is_set():
                    box["abandoned"] = True
                    return
            try:
                opened = client.messages.stream(**kwargs).__enter__()
            except BaseException as e:
                with lock:
                    if not box.get("abandoned"):
                        box["error"] = e
                        return
                log.debug(f"[LLM] Abandoned request failed: {_err_summary(e)}")
                return
            with lock:
                if not box.get("abandoned"):
                    box["stream"] = opened
                    return
            log.info("[LLM] Abandoned request opened late; closing it")
            try:
                opened.close()
            except Exception as e:
                log.debug(f"[LLM] Closing the abandoned stream failed: {e!r}")

        thread = threading.Thread(target=run, daemon=True, name="llm-open")
        thread.start()
        while True:
            thread.join(INTERRUPT_POLL)
            with lock:
                if "stream" in box:
                    return box["stream"]  # the caller handles a late interrupt
                # Before the error check: a request failure that coincides
                # with a spacebar press is shown as the interrupt
                if self.interrupted.is_set():
                    box["abandoned"] = True
                    if "error" in box:
                        log.debug(f"[LLM] Request failed during an interrupt: {_err_summary(box['error'])}")
                    log.info("[LLM] Request cancelled by interrupt before the response started")
                    return None
                if "error" in box:
                    raise box["error"]

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
        # Records of the tools actually run this turn (make_tool_record), in
        # call order; saved with the reply and replayed in later turns.
        tool_calls = []
        # Once anything has gone to TTS, later chunks use the longer
        # TTS_MIN_SENTENCE_LENGTH. Reset after each tool round, so the first
        # sentence after a (silent) tool call also starts quickly.
        first_chunk_sent = False
        # Set when the tool-free last round below still produced no text
        gave_up = False
        # Set once any round got a response from the API
        api_answered = False

        try:
            client = get_llm_client()
            # Up to MAX_TOOL_ROUNDS rounds may run tools; one more round then
            # forbids them, so a turn never ends on a tool call with no reply
            for round_num in range(MAX_TOOL_ROUNDS + 1):
                final_round = round_num == MAX_TOOL_ROUNDS
                sentence_buffer = ""
                final_message = None
                round_started = False

                # The final round keeps tools= (the API rejects tool_use and
                # tool_result blocks in messages when no tools are defined)
                # but tool_choice "none" makes Claude answer in text
                round_kwargs = {"tool_choice": {"type": "none"}} if final_round else {}
                stream = self._open_stream(
                    client, model=TEXT_MODEL, messages=messages, system=SYSTEM_PROMPT,
                    tools=turn_tools, max_tokens=MAX_RESPONSE_TOKENS, **round_kwargs, **LLM_KWARGS
                )
                if stream is None:
                    break  # interrupted before the response started
                # The response headers arrived, so the API is reachable; set
                # now so an interrupted reply still clears a stale startup_error
                api_answered = True
                # Published for handle_speaking_interrupt, which shuts it down to
                # wake the read below; then re-check, in case the interrupt
                # landed before the stream was published
                self.active_stream = stream
                try:
                    with stream:
                        if not self.interrupted.is_set():
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
                                min_len = TTS_MIN_SENTENCE_LENGTH if first_chunk_sent else TTS_FIRST_SENTENCE_MIN_LENGTH
                                split_at = last_sentence_end(sentence_buffer)
                                if split_at is not None and len(sentence_buffer[:split_at].strip()) >= min_len:
                                    clean_sentence = sentence_buffer[:split_at].strip()
                                    sentence_buffer = sentence_buffer[split_at:]
                                    if re.search(r'[a-zA-Z0-9]', clean_sentence):
                                        self.tts_queue.put(clean_sentence)
                                        first_chunk_sent = True

                            if not self.interrupted.is_set():
                                final_message = stream.get_final_message()
                except Exception as e:
                    # Shutting the socket down under a blocked read raises (an
                    # httpx.RemoteProtocolError or ReadError); that is the
                    # interrupt, not a failure. Any exception while interrupted
                    # is classified this way, so log it in full at debug in
                    # case it was a real bug in the UI/TTS code above.
                    if not self.interrupted.is_set():
                        raise
                    tb = traceback.extract_tb(e.__traceback__)
                    where = f" at {os.path.basename(tb[-1].filename)}:{tb[-1].lineno}" if tb else ""
                    log.debug(f"[LLM] Exception during interrupted stream{where}: {_err_summary(e)}")
                    log.info(f"[LLM] Response stream closed by interrupt ({type(e).__name__})")
                finally:
                    self.active_stream = None

                if self.interrupted.is_set() or final_message is None:
                    break

                # Flush trailing text that never hit sentence-ending punctuation
                clean_sentence = sentence_buffer.strip()
                if clean_sentence and re.search(r'[a-zA-Z0-9]', clean_sentence):
                    self.tts_queue.put(clean_sentence)
                    first_chunk_sent = True

                if final_round:
                    if not round_started:
                        gave_up = True
                    break
                if final_message.stop_reason != "tool_use":
                    break

                # Execute requested tools, then loop so Claude sees the results
                # with the full conversation context
                messages.append({"role": "assistant", "content": final_message.content})
                tool_results = []
                for block in final_message.content:
                    # Spacebar during a slow tool (search_web, the camera): run no more
                    if self.interrupted.is_set():
                        break
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
                        is_error = True
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Tool {block.name} failed: {e}",
                            "is_error": True
                        })
                    tool_calls.append(make_tool_record(
                        block.name, block.input, tool_results[-1]["content"], is_error))
                # Interrupted: don't send the results or restart the thinking
                # sound; the code after the loop finishes the turn quietly
                if self.interrupted.is_set():
                    break
                messages.append({"role": "user", "content": tool_results})
                first_chunk_sent = False

                self.set_state(BotStates.THINKING, "Reading...", cam_path=self.turn_cam_path)
                self._start_thinking_sound()

            self.thinking_sound_active.clear()
            # Claude answered, so whatever failed at warm-up has recovered (an
            # interrupt before any reply proves nothing either way)
            if api_answered:
                self.startup_error = None

            if gave_up and not self.interrupted.is_set():
                log.warning(f"[TOOL] No reply after {MAX_TOOL_ROUNDS} tool rounds; giving up")
                fallback = "I couldn't finish that one."
                self.set_state(BotStates.SPEAKING, "Speaking...", cam_path=self.turn_cam_path)
                if not self._bot_line_open:
                    self.append_to_text("BOT: ", newline=False)
                    self._bot_line_open = True
                elif full_response_text and not full_response_text[-1].isspace():
                    fallback = " " + fallback
                self.append_to_text(fallback, newline=False)
                self.tts_queue.put(fallback.strip())
                # Save what was said, not the unfinished reply on its own
                full_response_text += fallback

            if not self.interrupted.is_set():
                self.session_memory.append({"role": "user", "content": text})
                if full_response_text.strip():
                    reply = {"role": "assistant", "content": full_response_text}
                    if tool_calls:
                        reply["tool_calls"] = tool_calls
                    self.session_memory.append(reply)
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
        """Block until everything queued so far has been SPOKEN: synthesized,
        written to the device, heard, and the stream closed. The paths that
        end a turn call it, which keeps the mic closed while TTS plays.
        Interrupts can't deadlock it: the synth stage skips items while
        interrupted is set and the playback stage drains audio_queue and goes
        idle on its own."""
        self.tts_queue.join()
        self.audio_queue.join()
        if not self.playback_idle.wait(timeout=10):
            log.warning("[TTS] Playback did not go idle within 10 s")

    def _tts_busy(self):
        """True while speech is queued, being synthesized, or playing."""
        return (self.tts_queue.unfinished_tasks != 0
                or not self.audio_queue.empty() or not self.playback_idle.is_set())

    def _start_tts_threads(self):
        self.tts_active.clear()
        self.playback_idle.set()
        self.tts_thread = threading.Thread(target=self._tts_worker, name="tts-synth", daemon=True)
        self.playback_thread = threading.Thread(target=self._playback_worker, name="tts-playback", daemon=True)
        self.tts_thread.start()
        self.playback_thread.start()

    def _playback_format(self, voice_rate):
        """(rate to open the output stream at, whether to resample), decided
        once per voice sample rate: the voice's own rate if the device takes
        it, else the device's default rate."""
        fmt = self._playback_formats.get(voice_rate)
        if fmt is None:
            try:
                sd.check_output_settings(device=None, samplerate=voice_rate)
                fmt = (voice_rate, False)
            except Exception:
                try:
                    native_rate = int(sd.query_devices(kind='output')['default_samplerate'])
                except Exception:
                    native_rate = 48000
                fmt = (native_rate, True)
            self._playback_formats[voice_rate] = fmt
        return fmt

    def _put_audio(self, item):
        """Enqueue for playback, blocking while audio_queue is full. Gives up
        (returns False) only on exit; on an interrupt the playback stage
        drains the queue, so a blocked put always gets through."""
        while not self.exiting:
            try:
                self.audio_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _tts_worker(self):
        """Synthesis stage: text from tts_queue -> PCM chunks on audio_queue.
        Runs ahead of playback, so the next sentence is ready by the time the
        current one finishes playing."""
        while not self.exiting:
            try:
                text = self.tts_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                if self.interrupted.is_set():
                    continue
                self.tts_active.set()
                self._synthesize(text)
            except Exception as e:
                log.error(f"[PIPER] Synthesis failed: {e}")
            finally:
                self.tts_active.clear()
                # Only now: every chunk of this item is already on audio_queue,
                # so wait_for_tts's audio_queue.join() covers all of it
                self.tts_queue.task_done()

    def _synthesize(self, text):
        clean = normalize_for_tts(text)
        if not clean.strip(): return
        if self.piper_voice is None:
            log.warning("[PIPER] Voice not loaded; skipping playback.")
            return

        log_transcript("[PIPER SPEAKING]", clean)

        started = False
        try:
            for chunk in self.piper_voice.synthesize(clean):
                if self.interrupted.is_set() or self.exiting:
                    break
                rate, resample = self._playback_format(chunk.sample_rate)
                audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
                if resample:
                    num_samples = int(len(audio) * (rate / chunk.sample_rate))
                    audio = scipy.signal.resample(audio, num_samples).astype(np.int16)
                if not self._put_audio((audio.tobytes(), rate)):
                    break
                started = True
        finally:
            # Tells playback this item is complete, so a slow synthesis
            # mid-item doesn't make it close the stream
            if started:
                self._put_audio(TTS_ITEM_END)

    def _close_stream(self, stream, drain_frames=0):
        """drain_frames > 0: let everything written be heard, then stop().
        0: abort() drops the buffer at once (interrupt, exit, error).

        stop() alone does NOT wait for buffered audio here: on the Pi's ALSA
        "default" device (PulseAudio plugin) it returns at once and cuts the
        last ~0.25 s. So draining writes silence first: once a buffer's worth
        plus a margin of zeros has gone in, the real audio has been played
        and stop() drops only silence. The writes block, so this also times
        the end of the reply for wait_for_tts."""
        drain = drain_frames > 0
        try:
            if drain:
                silence = bytes(2 * drain_frames)
                for i in range(0, len(silence), TTS_WRITE_BYTES):
                    if self.interrupted.is_set() or self.exiting:
                        drain = False
                        break
                    stream.write(silence[i:i + TTS_WRITE_BYTES])
            if drain:
                stream.stop()
            else:
                stream.abort()
        except Exception:
            pass  # best-effort cleanup
        try:
            stream.close()
        except Exception:
            pass  # best-effort cleanup

    def _playback_worker(self):
        """Playback stage: owns the only RawOutputStream TTS uses. Opens it on
        the first chunk of a reply, keeps it open across sentences, and
        drains and closes it once nothing more is queued or being
        synthesized. All stream calls, including the abort on an interrupt,
        happen on this thread."""
        stream = None
        stream_rate = None
        drain_frames = 0
        in_item = False  # a chunk was played and its item's end hasn't arrived

        while True:
            if self.exiting:
                if stream is not None:
                    self._close_stream(stream)
                self.playback_idle.set()
                return

            if self.interrupted.is_set() and (stream is not None or in_item):
                if stream is not None:
                    self._close_stream(stream)
                stream = None
                in_item = False
                while True:
                    try:
                        self.audio_queue.get_nowait()
                        self.audio_queue.task_done()
                    except queue.Empty:
                        break
                self.playback_idle.set()

            try:
                item = self.audio_queue.get(timeout=0.02)
            except queue.Empty:
                # End of the reply (or of what has been streamed so far):
                # _close_stream returns once everything written has been heard
                # unfinished_tasks covers both queued text and the item being
                # synthesized: task_done runs only after its TTS_ITEM_END is
                # on audio_queue
                if (stream is not None and not in_item
                        and self.tts_queue.unfinished_tasks == 0):
                    self._close_stream(stream, drain_frames)
                    stream = None
                    self.playback_idle.set()
                continue

            try:
                if item is TTS_ITEM_END:
                    in_item = False
                    continue
                if self.interrupted.is_set() or self.exiting:
                    continue  # stale audio from an interrupted reply
                in_item = True
                audio_bytes, rate = item
                if stream is not None and rate != stream_rate:
                    self._close_stream(stream, drain_frames)
                    stream = None
                if stream is None:
                    self.playback_idle.clear()
                    stream = sd.RawOutputStream(
                        samplerate=rate, channels=1, dtype='int16',
                        device=None, latency='low', blocksize=TTS_WRITE_FRAMES,
                    )
                    stream_rate = rate
                    stream.start()
                    # Silence _close_stream writes to flush the device buffer
                    # before stop(): the buffer, or the reported latency if
                    # larger, plus 0.15 s for the sound server behind it
                    drain_frames = (max(stream.write_available, int(stream.latency * rate))
                                    + int(0.15 * rate))
                for i in range(0, len(audio_bytes), TTS_WRITE_BYTES):
                    if self.interrupted.is_set() or self.exiting:
                        break
                    stream.write(audio_bytes[i:i + TTS_WRITE_BYTES])
            except Exception as e:
                log.error(f"Audio Error: {e}")
                if stream is not None:
                    self._close_stream(stream)
                    stream = None
                self.playback_idle.set()
            finally:
                self.audio_queue.task_done()

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
            if self._tts_busy():
                time.sleep(0.1)
                continue
            sound = self.get_random_sound(thinking_sounds_dir)
            if sound: self._play_clip_interruptible(sound)
            for _ in range(50):
                if not self.thinking_sound_active.is_set(): return
                time.sleep(0.1)

    def _play_clip_interruptible(self, file_path):
        # Like play_sound, but cuts the clip (checked every 50 ms) as soon as
        # thinking ends or TTS speech starts. All sd.play/sd.stop calls stay on
        # this thread — stopping another thread's playback crashes PortAudio.
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
                # Cut the clip once speech is queued or playing, so voices
                # never overlap, and once thinking has ended (reply text
                # arrived, a spacebar interrupt, the end of the turn), so no
                # clip outlives the turn into the next recording
                if self._tts_busy() or not self.thinking_sound_active.is_set():
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
        # Older builds saved a "tools" list of labels; replies now carry
        # "tool_calls" records instead, so the labels are dropped.
        dropped = False
        for m in history:
            if m.pop("tools", None) is not None:
                dropped = True
        if dropped:
            log.info(f"{MEMORY_FILE}: dropped legacy tool labels")
        return history

    def save_chat_history(self):
        if not CHAT_MEMORY_ENABLED: return
        tmp_path = MEMORY_FILE + ".tmp"
        # Called from the main loop after each turn and from safe_exit on the
        # tkinter thread; the lock keeps the two from sharing the temp file,
        # and taking the snapshot under it means saves land in the order taken.
        with self.memory_lock:
            # The system entry is rebuilt from the current prompt rather than
            # trusting position 0; only the conversation itself is carried over
            # (entries as-is, so a reply's optional "tool_calls" are kept).
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
