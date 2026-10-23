"""Text handling: transcript filtering, trigger phrases, TTS sentence splitting
and making reply text speakable."""

import re

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


def speech_rejection_reason(transcription, engine):
    """Why a transcript is not worth sending to Claude, or None if it is.

    A false wake word fire on room noise used to reach the LLM as "(wind
    blowing)" and get answered out loud. Whisper is already telling us there
    was no speech; honour it instead of forwarding it to Claude.
    """
    if not transcription:
        return "empty"
    # Drop bracketed annotations; if nothing survives, it was pure non-speech.
    residue = NON_SPEECH_MARKER_RE.sub("", transcription)
    if not re.search(r"[A-Za-z0-9]", residue):
        return "non-speech audio"
    words = normalize_words(residue)
    if words in FILLER_UTTERANCES:
        return "filler only"
    # Whisper's silence hallucinations. Parakeet doesn't invent text on
    # silence, so a real "thank you" or "okay" to it must get through.
    if engine == "whisper" and words in WHISPER_ONLY_HALLUCINATIONS:
        return "silence hallucination"
    return None


# Memory wipe fires only when the *whole* utterance is the command (after a
# short lead-in such as "okay" or the wake word), so "don't forget everything"
# never wipes anything. Matched after normalize_words.
MEMORY_WIPE_RE = re.compile(
    r"^(?:(?:ok|okay|hey|hi|um|uh|so|please|bmo|jarvis|robot)\s+)*"
    r"(?:forget everything|reset (?:your )?memory)(?:\s+(?:please|now))?$")


def build_trigger_re(phrases):
    """Whole-word/phrase matcher for normalized text, or None if phrases is empty."""
    phrases = sorted({p for p in map(normalize_words, phrases) if p}, key=len, reverse=True)
    if not phrases:
        return None
    return re.compile(r"\b(?:" + "|".join(re.escape(p) for p in phrases) + r")\b")


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
