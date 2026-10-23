"""config.json defaults, validation and loading."""

import json
import logging
import os

log = logging.getLogger("bmo")

CONFIG_FILE = "config.json"

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
    # SPEAKING] ...") to agent.log. Off by default: those lines are then
    # logged as "<redacted, N chars>", so the log shows activity, not words.
    "log_transcripts": False,
    "camera_rotation": 0,
    "system_prompt_extras": "",
    "input_device": None,
    "input_sample_rate": None,
    # Most recent chat messages resent to Claude each turn (user + assistant),
    # and kept in memory.json across restarts.
    "max_history_messages": 20,
    # The camera tool is offered to Claude only on turns where the user says one
    # of these (whole words/phrases; case and punctuation ignored). [] = never.
    # Phrases, not bare "see"/"look": "look up the news", "let's see" and "see
    # you later" must not offer the camera, least of all on a search turn.
    "camera_trigger_words": [
        "look at", "looking at", "take a look", "have a look", "how do i look",
        "how does this look", "can you see", "do you see", "what do you see",
        "what can you see", "see this", "see me", "see my", "photo", "photograph",
        "picture", "snapshot", "selfie", "camera", "what is this", "what's this",
        "who is this", "what am i holding", "describe this"
    ],
    # Sampling temperature, 0.0-1.0. Default null omits it, as Claude 5.x models
    # such as the default claude-haiku-5-5 reject non-default values.
    "temperature": None,
    # Thinking effort: "low", "medium" or "high". Default "low" keeps voice
    # replies fast (thinking is on by default). null omits it.
    "effort": "low"
}

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


def load_config(path=CONFIG_FILE):
    """DEFAULT_CONFIG with the valid entries of the JSON file at path merged over it."""
    config = DEFAULT_CONFIG.copy()
    if not os.path.exists(path):
        return config
    try:
        with open(path, "r") as f:
            user_config = json.load(f)
    except (OSError, ValueError) as e:  # json.JSONDecodeError is a ValueError
        user_config = e
    if not isinstance(user_config, dict):
        reason = user_config if isinstance(user_config, Exception) else \
            f"top level is a {type(user_config).__name__}, not an object"
        log.error("!" * 72)
        log.error(f"[CONFIG] ERROR: {path} is unusable: {reason}")
        log.error("[CONFIG] ERROR: IGNORING THE WHOLE FILE. Every setting is at its default.")
        log.error("!" * 72)
        return config
    config.update(validate_config(user_config))
    return config
