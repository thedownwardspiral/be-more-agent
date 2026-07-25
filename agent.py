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

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
import json
import subprocess
import random
import re
import sys
import select
import traceback
import atexit
import datetime
import warnings
import wave
import struct
import queue
import collections

# Suppress harmless library warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="ddgs")

# Core dependencies
import sounddevice as sd
import numpy as np
import scipy.signal 

# --- AI ENGINES ---
import openwakeword
from openwakeword.model import Model
import anthropic
from piper import PiperVoice
from dotenv import load_dotenv
import base64

# Load environment variables from .env file
load_dotenv()

# --- WEB SEARCH (Using your working import) ---
from ddgs import DDGS

# =========================================================================
# 1. CONFIGURATION & CONSTANTS
# =========================================================================

CONFIG_FILE = "config.json"
MEMORY_FILE = "memory.json"
BMO_IMAGE_FILE = "current_image.jpg"
WAKE_WORD_MODEL = "./wakeword.onnx"

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
    "camera_rotation": 0,
    "system_prompt_extras": "",
    "input_device": None,
    "input_sample_rate": None
}

# LLM SETTINGS
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.9
MAX_TOOL_ROUNDS = 5
TTS_MIN_SENTENCE_LENGTH = 80

def load_config():
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                user_config = json.load(f)
                config.update(user_config)
        except Exception as e:
            print(f"Config Error: {e}. Using defaults.")
    return config

CURRENT_CONFIG = load_config()
TEXT_MODEL = os.environ.get("ANTHROPIC_MODEL", CURRENT_CONFIG["text_model"])
WAKE_WORD_THRESHOLD = CURRENT_CONFIG["wake_word_threshold"]
WAKE_WORD_CONSECUTIVE = max(1, int(CURRENT_CONFIG["wake_word_consecutive"]))

# whisper.cpp prefixes every segment with "[hh:mm:ss.mmm --> hh:mm:ss.mmm]".
# Match only that prefix so bracketed non-speech markers survive intact.
WHISPER_TIMESTAMP_RE = re.compile(r"^\s*\[[\d:.]+\s*-->\s*[\d:.]+\]\s*")

# Whisper annotates non-speech audio instead of returning nothing: "[BLANK_AUDIO]",
# "(wind blowing)", "*coughs*". Strip these; if nothing is left, there was no speech.
NON_SPEECH_MARKER_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*")

# Well-known whisper hallucinations on silent/near-silent input. Only ever compared
# against the *entire* transcript, so a real sentence containing these is unaffected.
WHISPER_HALLUCINATIONS = {
    "you", "thank you", "thanks for watching", "thank you for watching",
    "bye", "bye bye", "okay", "oh", "so", "um", "uh", "hmm", "mm",
    "please subscribe", "subtitles by the amara.org community",
}

# Anthropic client — reads ANTHROPIC_API_KEY from environment / .env
llm_client = anthropic.Anthropic()

def resolve_input_device(config):
    requested = config.get("input_device")
    if requested in (None, "", "default"):
        return None

    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"[AUDIO] Device query failed: {e}", flush=True)
        return None

    if isinstance(requested, int) or (isinstance(requested, str) and requested.isdigit()):
        index = int(requested)
        if 0 <= index < len(devices):
            return index
        print(f"[AUDIO] Input device index not found: {index}", flush=True)
        return None

    requested_lower = str(requested).lower()
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0 and requested_lower in dev.get("name", "").lower():
            return idx

    print(f"[AUDIO] Input device name not found: {requested}", flush=True)
    return None

INPUT_DEVICE_NAME = resolve_input_device(CURRENT_CONFIG)
if INPUT_DEVICE_NAME is not None:
    try:
        device_info = sd.query_devices(INPUT_DEVICE_NAME)
        print(f"[AUDIO] Using input device: {device_info.get('name', INPUT_DEVICE_NAME)}", flush=True)
    except Exception:
        print(f"[AUDIO] Using input device index: {INPUT_DEVICE_NAME}", flush=True)

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
        print(f"[AUDIO] Input device query failed: {e}", flush=True)

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

You have real tools: you can check the time, search the web, and take a photo
with your camera to see your surroundings. Use a tool whenever the user asks
for something a tool provides; otherwise just chat.

Your replies are spoken aloud, so keep them short and conversational.
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
        atexit.register(self.safe_exit)
        
        # State
        self.current_state = BotStates.WARMUP
        self.current_volume = 0 
        self.animations = {}
        self.current_frame_index = 0
        self.current_overlay_image = None
        
        self.permanent_memory = self.load_chat_history()
        self.session_memory = []
        self.thinking_sound_active = threading.Event()
        
        self.last_ptt_time = 0 
        self.ptt_event = threading.Event()       
        self.recording_active = threading.Event() 
        self.interrupted = threading.Event() 
        
        self.tts_queue = queue.Queue()
        self.tts_thread = None
        self.tts_active = threading.Event()
        self.exiting = False
        self.turn_cam_path = None

        # --- PIPER TTS INITIALIZATION ---
        print("[INIT] Loading Piper voice...", flush=True)
        self.piper_voice = None
        voice_model = CURRENT_CONFIG.get("voice_model", "piper/en_US-bmo-medium.onnx")
        try:
            self.piper_voice = PiperVoice.load(voice_model)
            print(f"[INIT] Piper voice loaded: {voice_model}", flush=True)
        except Exception as e:
            print(f"[CRITICAL] Failed to load Piper voice '{voice_model}': {e}")

        # --- WAKE WORD INITIALIZATION ---
        print("[INIT] Loading Wake Word...", flush=True)
        self.oww_model = None
        if os.path.exists(WAKE_WORD_MODEL):
            try:
                self.oww_model = Model(wakeword_model_paths=[WAKE_WORD_MODEL])
                print("[INIT] Wake Word Loaded.", flush=True)
            except TypeError:
                try:
                    self.oww_model = Model(wakeword_models=[WAKE_WORD_MODEL])
                    print("[INIT] Wake Word Loaded (New API).", flush=True)
                except Exception as e:
                    print(f"[CRITICAL] Failed to load model: {e}")
            except Exception as e:
                print(f"[CRITICAL] Failed to load model: {e}")
        else:
            print(f"[CRITICAL] Model not found: {WAKE_WORD_MODEL}")

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
        
        threading.Thread(target=self.safe_main_execution, daemon=True).start()

    # --- HELPERS ---

    def safe_exit(self):
        # Reachable from the Exit button, Escape, and atexit — run once only
        if self.exiting:
            return
        self.exiting = True
        print("\n--- SHUTDOWN SEQUENCE ---", flush=True)
        self.recording_active.clear()
        self.thinking_sound_active.clear()
        self.tts_active.clear()

        self.save_chat_history()

        try:
            sd.stop()
        except Exception:
            pass
        try:
            self.master.quit()
        except Exception:
            pass
        
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
            print("[PTT] Toggle OFF", flush=True)
            self.recording_active.clear() 
        else:
            if self.current_state == BotStates.IDLE or "Wait" in self.status_var.get():
                print("[PTT] Toggle ON", flush=True)
                self.recording_active.set() 
                self.ptt_event.set()

    def handle_speaking_interrupt(self, event=None):
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
                if state in self.animations.get("idle", []):
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
        def _update():
            if msg: print(f"[STATE] {state.upper()}: {msg}", flush=True)
            if self.current_state != state:
                self.current_state = state
                self.current_frame_index = 0
            if msg: self.status_var.set(msg)
            if cam_path and os.path.exists(cam_path) and state in [BotStates.THINKING, BotStates.SPEAKING]:
                try:
                    img = Image.open(cam_path).resize((self.OVERLAY_WIDTH, self.OVERLAY_HEIGHT))
                    self.current_overlay_image = ImageTk.PhotoImage(img)
                    self.overlay_label.config(image=self.current_overlay_image)
                    self.overlay_label.place(x=200, y=90)
                except: pass
            else:
                self.overlay_label.place_forget()
        self.master.after(0, _update)

    def append_to_text(self, text, newline=True):
        def _update():
            self.response_text.config(state=tk.NORMAL)
            if newline: 
                self.response_text.insert(tk.END, text + "\n")
            else: 
                self.response_text.insert(tk.END, text)
            
            self.response_text.see(tk.END)
            self.response_text.config(state=tk.DISABLED)
            
        self.master.after(0, _update)

    def _stream_to_text(self, chunk):
        def update_text_stream():
            self.response_text.config(state=tk.NORMAL)
            self.response_text.insert(tk.END, chunk)
            self.response_text.see(tk.END) 
            self.response_text.config(state=tk.DISABLED)
        self.master.after(0, update_text_stream)

    # =========================================================================
    # 3. TOOL EXECUTION
    # =========================================================================

    def execute_tool(self, name, tool_input):
        """Run a tool requested by Claude and return tool_result content."""
        print(f"TOOL: {name} {tool_input}", flush=True)

        if name == "get_time":
            now = datetime.datetime.now().strftime("%I:%M %p")
            return f"The current time is {now}."

        if name == "search_web":
            query = tool_input.get("query", "")
            print(f"Searching web for: {query}...", flush=True)
            try:
                # 'us-en' region is often more stable for CLI queries
                with DDGS() as ddgs:
                    results = []
                    # 1. News search
                    try:
                        results = list(ddgs.news(query, region='us-en', max_results=1))
                        if results:
                            print(f"[DEBUG] Found News: {results[0].get('title')}", flush=True)
                    except Exception as e:
                        print(f"[DEBUG] News Search Error: {e}", flush=True)

                    # 2. Text fallback
                    if not results:
                        print("[DEBUG] No news found, trying text search...", flush=True)
                        try:
                            results = list(ddgs.text(query, region='us-en', max_results=1))
                            if results:
                                print(f"[DEBUG] Found Text: {results[0].get('title')}", flush=True)
                        except Exception as e:
                             print(f"[DEBUG] Text Search Error: {e}", flush=True)

                    if results:
                        r = results[0]
                        # Safe get
                        title = r.get('title', 'No Title')
                        body = r.get('body', r.get('snippet', 'No Body'))
                        return f"SEARCH RESULTS for '{query}':\nTitle: {title}\nSnippet: {body[:300]}"
                    else:
                        print(f"[DEBUG] Search returned 0 results.", flush=True)
                        return "The search returned no results."
            except Exception as e:
                print(f"[DEBUG] Connection/Library Error: {e}", flush=True)
                return "Search failed: the internet is unreachable right now."

        if name == "capture_image":
            img_path = self.capture_image()
            if not img_path:
                return "Camera error: no image could be captured."
            self.turn_cam_path = img_path
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            return [{
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}
            }]

        return f"Unknown tool: {name}"

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
                
                user_text = self.transcribe_audio(audio_file)
                if not user_text:
                    self.set_state(BotStates.IDLE, "Transcription empty.")
                    continue
                
                self.append_to_text(f"YOU: {user_text}")
                self.interrupted.clear()
                self.chat_and_respond(user_text)
                    
        except Exception as e:
            traceback.print_exc()
            self.set_state(BotStates.ERROR, f"Fatal Error: {str(e)[:40]}")

    def warm_up_logic(self):
        self.set_state(BotStates.WARMUP, "Warming up brains...")
        try:
            llm_client.messages.create(
                model=TEXT_MODEL, max_tokens=1,
                messages=[{"role": "user", "content": "hi"}]
            )
            print(f"Anthropic API is reachable. Model: {TEXT_MODEL}", flush=True)
        except Exception as e:
            print(f"Failed to reach Anthropic API: {e}", flush=True)
        self.play_sound(self.get_random_sound(greeting_sounds_dir))
        print("Ready.", flush=True)

    def detect_wake_word_or_ptt(self):
        self.set_state(BotStates.IDLE, "Waiting...")
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
            print(f"[AUDIO] Wake stream failed: {e}. Retrying with fallback settings...", flush=True)
            try:
                stream_args["blocksize"] = 1024
                stream_args["latency"] = "high"
                return self._listen_loop(stream_args, 1024, CHUNK_SIZE, True)
            except Exception as e2:
                print(f"[CRITICAL] Wake Word Stream Error: {e2}", flush=True)
                self.ptt_event.wait()
                self.ptt_event.clear()
                return "PTT"

    def _listen_loop(self, stream_args, input_chunk_size, target_chunk_size, use_resampling):
        with sd.InputStream(**stream_args) as stream:
            print(f"[AUDIO] Listening at {stream_args['samplerate']} Hz, block {stream_args['blocksize']}", flush=True)
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
                    print("!", end="", flush=True)
                    consecutive_overflows += 1
                    if consecutive_overflows >= 5:
                        raise RuntimeError("Persistent audio buffer overflow")
                else:
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
                        print(f"[WAKE] {mdl} triggered ({max(recent):.2f})", flush=True)
                        self.oww_model.reset()
                        return "WAKE"

    def record_voice_adaptive(self, filename="input.wav"):
        print("Recording (Adaptive)...", flush=True)
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
            print(f"[AUDIO ERROR] Adaptive Recording Failed: {e}", flush=True)
            return None

        return self.save_audio_buffer(buffer, filename, samplerate)

    def record_voice_ptt(self, filename="input.wav"):
        print("Recording (PTT)...", flush=True)
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
            print(f"[AUDIO ERROR] PTT Recording Failed: {e}", flush=True)
            return None

        return self.save_audio_buffer(buffer, filename, samplerate)

    def save_audio_buffer(self, buffer, filename, samplerate=16000):
        if not buffer: return None
        audio_data = np.concatenate(buffer, axis=0).flatten()
        audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
        audio_data = (audio_data * 32767).astype(np.int16)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio_data.tobytes())
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
            print(f"[AUDIO] RMS energy: {rms:.6f} (threshold: {threshold})", flush=True)
            return rms >= threshold
        except Exception as e:
            print(f"[AUDIO] Energy check failed: {e}", flush=True)
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
            print("[AUDIO] Non-speech audio — ignoring.", flush=True)
            return False

        normalized = re.sub(r"[^a-z0-9 ]", "", residue.lower()).strip()
        if normalized in WHISPER_HALLUCINATIONS:
            print(f"[AUDIO] Silence hallucination '{transcription}' — ignoring.", flush=True)
            return False

        return True

    def transcribe_audio(self, filename):
        if not self._check_audio_energy(filename):
            print("[AUDIO] Skipping transcription — audio too quiet.", flush=True)
            return ""

        print("Transcribing...", flush=True)
        whisper_model = CURRENT_CONFIG.get("whisper_model", "./whisper.cpp/models/ggml-base.en.bin")
        whisper_threads = str(CURRENT_CONFIG.get("whisper_threads", 3))
        try:
            result = subprocess.run(
                ["./whisper.cpp/build/bin/whisper-cli", "-m", whisper_model, "-l", "en", "-t", whisper_threads, "-f", filename],
                capture_output=True, text=True
            )
            transcription_lines = result.stdout.strip().split('\n')
            if transcription_lines and transcription_lines[-1].strip():
                last_line = transcription_lines[-1].strip()
                transcription = WHISPER_TIMESTAMP_RE.sub("", last_line).strip()
            else: transcription = ""
            print(f"Heard: '{transcription}'", flush=True)
            if not self._is_speech(transcription):
                return ""
            return transcription.strip()
        except Exception as e:
            print(f"Transcription Error: {e}")
            return ""

    def capture_image(self):
        self.set_state(BotStates.CAPTURING, "Watching...")
        try:
            subprocess.run(["rpicam-still", "-t", "500", "-n", "--width", "640", "--height", "480", "-o", BMO_IMAGE_FILE], check=True)
            rotation = CURRENT_CONFIG.get("camera_rotation", 0)
            if rotation != 0:
                img = Image.open(BMO_IMAGE_FILE)
                img = img.rotate(rotation, expand=True) 
                img.save(BMO_IMAGE_FILE)
            return BMO_IMAGE_FILE
        except Exception as e:
            print(f"Camera Error: {e}")
            return None

    # =========================================================================
    # 5. CHAT & RESPOND
    # =========================================================================

    def _history_messages(self):
        """Chat history as Anthropic messages (skips legacy system entries)."""
        history = []
        for msg in (self.permanent_memory + self.session_memory):
            if msg["role"] == "system":
                continue
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

        if "forget everything" in text.lower() or "reset memory" in text.lower():
            self.session_memory = []
            self.permanent_memory = [{"role": "system", "content": SYSTEM_PROMPT}]
            self.save_chat_history()
            self.tts_queue.put("Okay. Memory wiped.")
            self.wait_for_tts()
            self.set_state(BotStates.IDLE, "Memory Wiped")
            return

        self.set_state(BotStates.THINKING, "Thinking...")
        self.turn_cam_path = None

        messages = self._history_messages() + [{"role": "user", "content": text}]

        self.thinking_sound_active.set()
        threading.Thread(target=self._run_thinking_sound_loop, daemon=True).start()

        full_response_text = ""

        try:
            for _ in range(MAX_TOOL_ROUNDS):
                sentence_buffer = ""
                final_message = None

                with llm_client.messages.stream(
                    model=TEXT_MODEL, messages=messages, system=SYSTEM_PROMPT,
                    tools=TOOLS, max_tokens=1024, temperature=LLM_TEMPERATURE
                ) as stream:
                    for content in stream.text_stream:
                        if self.interrupted.is_set(): break
                        if not content: continue
                        full_response_text += content

                        self.thinking_sound_active.clear()
                        if self.current_state != BotStates.SPEAKING:
                            self.set_state(BotStates.SPEAKING, "Speaking...", cam_path=self.turn_cam_path)
                            self.append_to_text("BOT: ", newline=False)

                        self._stream_to_text(content)

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
                    if block.type == "tool_use":
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": self.execute_tool(block.name, block.input)
                        })
                messages.append({"role": "user", "content": tool_results})

                self.set_state(BotStates.THINKING, "Reading...", cam_path=self.turn_cam_path)
                self.thinking_sound_active.set()
                threading.Thread(target=self._run_thinking_sound_loop, daemon=True).start()

            self.thinking_sound_active.clear()

            if not self.interrupted.is_set():
                self.session_memory.append({"role": "user", "content": text})
                if full_response_text.strip():
                    self.append_to_text("")
                    self.session_memory.append({"role": "assistant", "content": full_response_text})

            self.wait_for_tts()
            self.set_state(BotStates.IDLE, "Ready")

        except Exception as e:
            print(f"LLM Error: {e}")
            self.thinking_sound_active.clear()
            # Don't reopen the mic while queued speech is still playing,
            # or the bot records and answers its own voice
            self.wait_for_tts()
            self.set_state(BotStates.ERROR, "Brain Freeze!")

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
        clean = re.sub(r"[^\w\s,.!?:-]", "", text)
        if not clean.strip(): return
        if self.piper_voice is None:
            print("[PIPER] Voice not loaded; skipping playback.", flush=True)
            return

        print(f"[PIPER SPEAKING] '{clean}'", flush=True)

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
                    sub = audio_bytes[i:i + byte_step]
                    sub_arr = np.frombuffer(sub, dtype=np.int16)
                    if len(sub_arr) > 0:
                        self.current_volume = np.max(np.abs(sub_arr))
                    stream.write(sub)

            if stream is not None and not self.interrupted.is_set():
                time.sleep(0.5)

        except Exception as e:
            print(f"Audio Error: {e}")
        finally:
            self.current_volume = 0
            if stream is not None:
                try: stream.stop()
                except: pass
                try: stream.close()
                except: pass

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
        except Exception:
            pass

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
            except:
                native_rate = 48000 

            playback_rate = file_sr
            try:
                sd.check_output_settings(device=None, samplerate=file_sr)
            except:
                playback_rate = native_rate
                num_samples = int(len(audio) * (native_rate / file_sr))
                audio = scipy.signal.resample(audio, num_samples).astype(np.int16)

            sd.play(audio, playback_rate)
            sd.wait() 
        except: pass

    def load_chat_history(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r") as f: return json.load(f)
            except: pass
        return [{"role": "system", "content": SYSTEM_PROMPT}]

    def save_chat_history(self):
        full = self.permanent_memory + self.session_memory
        conv = full[1:]
        if len(conv) > 10: conv = conv[-10:]
        with open(MEMORY_FILE, "w") as f: 
            json.dump([full[0]] + conv, f, indent=4)

if __name__ == "__main__":
    print("--- SYSTEM STARTING ---", flush=True)
    root = tk.Tk()
    app = BotGUI(root)
    root.mainloop()
