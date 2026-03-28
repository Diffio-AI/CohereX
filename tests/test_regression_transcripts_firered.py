import gc
import json
import os
import re
from pathlib import Path

import pytest
import torch

from coherex.asr import load_model
from coherex.audio import load_audio


FIXTURES_DIR = Path(__file__).parent / "fixtures"
AUDIO_DIR = FIXTURES_DIR / "audio"
BASELINE_PATH = FIXTURES_DIR / "regression_transcripts.json"
MAX_WORD_ERROR_RATE = 0.04


def _normalize_words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9']+", " ", text.lower()).split()


def _joined_words(result: dict) -> list[str]:
    return _normalize_words(" ".join(segment["text"] for segment in result["segments"]))


def _levenshtein_distance(actual: list[str], expected: list[str]) -> int:
    dp = list(range(len(expected) + 1))
    for i, actual_word in enumerate(actual, 1):
        previous = dp[0]
        dp[0] = i
        for j, expected_word in enumerate(expected, 1):
            current = dp[j]
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                previous + int(actual_word != expected_word),
            )
            previous = current
    return dp[-1]


@pytest.fixture(scope="session")
def regression_baselines():
    return json.loads(BASELINE_PATH.read_text())


@pytest.fixture(scope="session")
def firered_regression_model():
    device = os.environ.get("COHEREX_TEST_DEVICE", "cpu")
    try:
        model = load_model(
            "CohereLabs/cohere-transcribe-03-2026",
            device=device,
            language="en",
            vad_method="firered",
            local_files_only=True,
            asr_options={"max_new_tokens": 448},
        )
    except Exception as exc:
        pytest.skip(f"FireRed regression model unavailable: {exc}")

    yield model

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "filename",
    [
        "hur.mp3",
        "amelia_earhart_noisy.c431d09f.wav",
        "david-gooding-noisy.mp3",
    ],
)
def test_firered_transcription_regression_matches_baseline_words(
    firered_regression_model,
    regression_baselines,
    filename,
):
    audio = load_audio(str(AUDIO_DIR / filename))
    result = firered_regression_model.transcribe(audio, batch_size=1, print_progress=False, verbose=False)
    expected = regression_baselines[filename]

    actual_words = _joined_words(result)
    expected_words = _normalize_words(expected["word_sequence"])
    word_error_rate = _levenshtein_distance(actual_words, expected_words) / len(expected_words)

    assert result["language"] == expected["language"]
    assert result["segments"]
    assert word_error_rate <= MAX_WORD_ERROR_RATE
