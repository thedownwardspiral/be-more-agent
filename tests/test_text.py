import pytest

from bmo.config import DEFAULT_CONFIG
from bmo.text import (MEMORY_WIPE_RE, build_trigger_re, last_sentence_end, normalize_for_tts,
                      normalize_words, speech_rejection_reason)

CAMERA_RE = build_trigger_re(DEFAULT_CONFIG["camera_trigger_words"])


def camera_offered(text):
    return bool(CAMERA_RE.search(normalize_words(text)))


@pytest.mark.parametrize("text", [
    "Can you look at this?",
    "What do you see?",
    "Take a picture of me",
    "What's this?",
    "What am I holding?",
    "How do I look today?",
    "Snap a selfie",
    "Can you see my cat?",
])
def test_camera_offered_when_asked_to_look(text):
    assert camera_offered(text)


@pytest.mark.parametrize("text", [
    "Can you look up the news?",
    "Look up the weather in Boston",
    "Let's see, what's the weather?",
    "See you later",
    "Looks like rain today?",
    "I see what you mean",
    "What's the view count on that video?",
])
def test_camera_not_offered_on_everyday_phrases(text):
    assert not camera_offered(text)


def test_empty_trigger_list_never_offers_camera():
    assert build_trigger_re([]) is None


@pytest.mark.parametrize("text,wipe", [
    ("forget everything", True),
    ("Please forget everything now.", True),
    ("Okay Jarvis, reset memory", True),
    ("reset your memory please", True),
    ("don't forget everything", False),
    ("I never forget everything I learn", False),
])
def test_memory_wipe_needs_the_whole_utterance(text, wipe):
    assert bool(MEMORY_WIPE_RE.match(normalize_words(text))) is wipe


def test_normalize_words():
    assert normalize_words("What's THIS?") == "whats this"
    assert normalize_words("Bye-bye!") == "bye bye"


@pytest.mark.parametrize("text,expected", [
    ("It's 72°F and 50% humidity — nice!", "It's 72 degrees fahrenheit and 50 percent humidity, nice!"),
    ("He'll pay $5 million.", "He'll pay 5 million dollars."),
    ("Open 9–5 & weekends", "Open 9 to 5 and weekends"),
    ("Ready 🤖!", "Ready !"),
])
def test_normalize_for_tts(text, expected):
    assert normalize_for_tts(text) == expected


@pytest.mark.parametrize("buffer,expected", [
    ("Hello there. How", len("Hello there.")),
    ("Meet me at 3 p.m. on Friday", None),
    ("I live in St. Louis", None),
    ("1. Go hiking", None),
    ("Wow! Really? Yes", len("Wow! Really?")),
    ("Pi is 3.14", None),
    ("The end.", None),  # the buffer's end is never a sentence end
])
def test_last_sentence_end(buffer, expected):
    assert last_sentence_end(buffer) == expected


@pytest.mark.parametrize("text,engine,reason", [
    ("", "parakeet", "empty"),
    ("[BLANK_AUDIO]", "whisper", "non-speech audio"),
    ("(wind blowing)", "whisper", "non-speech audio"),
    ("Um.", "parakeet", "filler only"),
    ("Thank you.", "whisper", "silence hallucination"),
    ("Thank you.", "parakeet", None),
    ("What time is it?", "whisper", None),
])
def test_speech_rejection_reason(text, engine, reason):
    assert speech_rejection_reason(text, engine) == reason
