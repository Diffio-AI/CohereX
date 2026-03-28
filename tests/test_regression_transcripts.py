import gc
import json
import os
from pathlib import Path

import pytest
import torch

from coherex.asr import load_model
from coherex.audio import load_audio


FIXTURES_DIR = Path(__file__).parent / "fixtures"
AUDIO_DIR = FIXTURES_DIR / "audio"
BASELINE_PATH = FIXTURES_DIR / "regression_transcripts.json"


def _normalize_words(text: str) -> str:
    return " ".join(text.split())


def _joined_words(result: dict) -> str:
    return _normalize_words(" ".join(segment["text"] for segment in result["segments"]))


@pytest.fixture(scope="session")
def regression_baselines():
    return json.loads(BASELINE_PATH.read_text())


@pytest.fixture(scope="session")
def regression_model():
    device = os.environ.get("COHEREX_TEST_DEVICE", "cpu")
    try:
        model = load_model(
            "CohereLabs/cohere-transcribe-03-2026",
            device=device,
            language="en",
            vad_method="none",
            local_files_only=True,
            asr_options={"max_new_tokens": 448},
        )
    except Exception as exc:
        pytest.skip(f"Regression model unavailable: {exc}")

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
def test_transcription_regression_matches_baseline_words(regression_model, regression_baselines, filename):
    audio = load_audio(str(AUDIO_DIR / filename))
    result = regression_model.transcribe(audio, batch_size=1, print_progress=False, verbose=False)
    expected = regression_baselines[filename]

    assert result["language"] == expected["language"]
    assert len(result["segments"]) == len(expected["segments"])
    assert _joined_words(result) == _normalize_words(expected["word_sequence"])
