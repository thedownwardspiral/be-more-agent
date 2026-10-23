"""Speech-to-text audio helpers: resampling and splitting long audio."""

import math

import numpy as np
import scipy.signal

# Both STT engines take 16 kHz mono; recordings are converted to this rate.
STT_SAMPLE_RATE = 16000
# Longer audio is decoded by Parakeet in chunks of at most this many seconds.
# Each cut is the centre of the quietest STT_SPLIT_WINDOW whose start lies
# within STT_SPLIT_SEARCH seconds of a target STT_SPLIT_SEARCH before the cap;
# the window must also end by the cap, and that clamp (not the target) is what
# keeps every chunk within STT_CHUNK_SECONDS.
STT_CHUNK_SECONDS = 25.0
STT_SPLIT_SEARCH = 3.0
STT_SPLIT_WINDOW = 0.2
STT_SPLIT_FRAME = 0.02
# No chunk after the last cut may be shorter than this: Parakeet invents a
# filler ("Okay.", "Mm-hmm.") for a lone sub-second near-silent chunk.
STT_MIN_LAST_CHUNK = 2.0
# Chunks shorter than this are not decoded at all (safety net for the above;
# cannot fire with the current constants)
STT_MIN_DECODE_CHUNK = 1.0


def stt_split_points(audio, rate=STT_SAMPLE_RATE):
    """Sample indices at which to cut `audio` into chunks of at most
    STT_CHUNK_SECONDS, the last at least STT_MIN_LAST_CHUNK. Each cut is the
    centre of the STT_SPLIT_WINDOW with the lowest mean energy (mean square of
    the samples, over STT_SPLIT_FRAME frames) in its search range: window
    starts from STT_SPLIT_SEARCH before the target up to the earliest of
    STT_SPLIT_SEARCH after it, a window end at the cap, and a cut
    STT_MIN_LAST_CHUNK before the end of the audio. That is a pause when the
    range has one; continuous speech can still be cut between syllables.
    The range is never empty with the current constants (asserted). Returns []
    when the audio fits in one chunk."""
    n = len(audio)
    max_len = int(STT_CHUNK_SECONDS * rate)
    if n <= max_len:
        return []
    frame = int(STT_SPLIT_FRAME * rate)
    n_frames = n // frame
    energy = np.mean(np.square(audio[:n_frames * frame].astype(np.float64)).reshape(n_frames, frame), axis=1)
    win = max(1, int(round(STT_SPLIT_WINDOW / STT_SPLIT_FRAME)))
    # Mean energy of the window starting at each frame
    csum = np.concatenate(([0.0], np.cumsum(energy)))
    win_energy = (csum[win:] - csum[:-win]) / win
    search = int(STT_SPLIT_SEARCH / STT_SPLIT_FRAME)
    cap = int(STT_CHUNK_SECONDS / STT_SPLIT_FRAME)
    min_last = int(round(STT_MIN_LAST_CHUNK / STT_SPLIT_FRAME))

    def quietest(lo, hi, target):
        """Start frame of the quietest window in [lo, hi]; among (near-)equal
        ones (within 5% in energy) the one nearest `target`."""
        cand = win_energy[lo:hi + 1]
        quiet = np.flatnonzero(cand <= cand.min() * 1.05 + 1e-12) + lo
        return int(quiet[np.argmin(np.abs(quiet - target))])

    points = []
    prev = 0  # in frames
    while n - prev * frame > max_len:
        target = prev + cap - search
        lo = max(prev + 1, target - search)
        # Window must end by prev + cap (chunk within the cap), and the cut
        # must leave at least STT_MIN_LAST_CHUNK of audio after it
        hi = min(len(win_energy) - 1, prev + cap - win, target + search,
                 n_frames - min_last - win // 2)
        # Unreachable: while > STT_CHUNK_SECONDS remain the range always holds
        # a window (proven for all lengths with the current constants)
        assert hi >= lo, "stt_split_points: empty search range with > STT_CHUNK_SECONDS left"
        cut = quietest(lo, hi, target) + win // 2
        points.append(cut * frame)
        prev = cut
    return points


def resample_to_stt_rate(audio, rate, padtype="constant"):
    """float32 mono audio at `rate` -> float32 at STT_SAMPLE_RATE (polyphase)."""
    rate = int(rate)
    if rate == STT_SAMPLE_RATE:
        return np.asarray(audio, dtype=np.float32)
    g = math.gcd(rate, STT_SAMPLE_RATE)
    return scipy.signal.resample_poly(audio, STT_SAMPLE_RATE // g, rate // g,
                                      padtype=padtype).astype(np.float32)
