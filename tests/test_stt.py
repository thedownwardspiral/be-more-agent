import numpy as np
import pytest

from bmo.stt import (STT_CHUNK_SECONDS, STT_MIN_LAST_CHUNK, STT_SAMPLE_RATE, resample_to_stt_rate,
                     stt_split_points)

RATE = STT_SAMPLE_RATE


def speechlike(seconds, pauses=()):
    """Noise with near-silent gaps at the given (start, end) seconds."""
    rng = np.random.default_rng(0)
    audio = rng.uniform(-0.5, 0.5, int(seconds * RATE)).astype(np.float32)
    for start, end in pauses:
        audio[int(start * RATE):int(end * RATE)] *= 0.001
    return audio


def chunk_lengths(audio, points):
    bounds = [0] + points + [len(audio)]
    return [(b - a) / RATE for a, b in zip(bounds, bounds[1:])]


def test_short_audio_is_not_split():
    assert stt_split_points(speechlike(STT_CHUNK_SECONDS)) == []


@pytest.mark.parametrize("seconds", [25.5, 27.0, 40.0, 55.7, 60.0, 90.0])
def test_chunks_respect_the_cap_and_the_minimum_last_chunk(seconds):
    audio = speechlike(seconds)
    lengths = chunk_lengths(audio, stt_split_points(audio))
    assert all(length <= STT_CHUNK_SECONDS + 1e-9 for length in lengths)
    assert lengths[-1] >= STT_MIN_LAST_CHUNK


def test_cut_lands_in_a_pause():
    audio = speechlike(40.0, pauses=[(21.0, 21.6)])
    (cut,) = stt_split_points(audio)
    assert 21.0 <= cut / RATE <= 21.6


def test_resample_to_stt_rate():
    one_second_48k = np.zeros(48000, dtype=np.float32)
    out = resample_to_stt_rate(one_second_48k, 48000)
    assert out.dtype == np.float32 and len(out) == RATE
    same = resample_to_stt_rate(np.ones(10), RATE)
    assert same.dtype == np.float32 and len(same) == 10
