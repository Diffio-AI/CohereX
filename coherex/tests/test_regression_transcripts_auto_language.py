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
def transcript_baselines():
    return json.loads(BASELINE_PATH.read_text())


@pytest.fixture(scope="session")
def regression_model_auto_language():
    device = os.environ.get("COHEREX_TEST_DEVICE", "cpu")
    try:
        model = load_model(
            "CohereLabs/cohere-transcribe-03-2026",
            device=device,
            language=None,
            vad_method="none",
            local_files_only=True,
            asr_options={"max_new_tokens": 448},
        )
    except Exception as exc:
        pytest.skip(f"Automatic-language regression model unavailable: {exc}")

    yield model

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_transcription_regression_matches_baseline_with_auto_language(regression_model_auto_language, transcript_baselines):
    audio = load_audio(str(AUDIO_DIR / "hur.mp3"))
    result = regression_model_auto_language.transcribe(audio, batch_size=1, print_progress=False, verbose=False)
    expected = transcript_baselines["hur.mp3"]

    assert result["language"] == "en"
    assert len(result["segments"]) == len(expected["segments"])
    assert _joined_words(result) == _normalize_words(expected["word_sequence"])
