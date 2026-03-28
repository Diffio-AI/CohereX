import gc
import json
from pathlib import Path

import pytest
import torch

from coherex.alignment import align, load_align_model
from coherex.audio import load_audio


FIXTURES_DIR = Path(__file__).parent / "fixtures"
AUDIO_DIR = FIXTURES_DIR / "audio"
TRANSCRIPT_BASELINE_PATH = FIXTURES_DIR / "regression_transcripts.json"
ALIGNMENT_BASELINE_PATH = FIXTURES_DIR / "regression_word_alignment.json"
ALIGNMENT_TOLERANCE_S = 0.02
ALIGNMENT_SCORE_TOLERANCE = 0.005


@pytest.fixture(scope="session")
def transcript_baselines():
    return json.loads(TRANSCRIPT_BASELINE_PATH.read_text())


@pytest.fixture(scope="session")
def alignment_baselines():
    return json.loads(ALIGNMENT_BASELINE_PATH.read_text())


@pytest.fixture(scope="session")
def alignment_model():
    try:
        model, metadata = load_align_model("en", "cpu")
    except Exception as exc:
        pytest.skip(f"Alignment model unavailable: {exc}")

    yield model, metadata

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
def test_word_alignment_regression_matches_whisperx(
    alignment_model,
    transcript_baselines,
    alignment_baselines,
    filename,
):
    model, metadata = alignment_model
    audio = load_audio(str(AUDIO_DIR / filename))
    transcript_segments = transcript_baselines[filename]["segments"]

    result = align(
        transcript_segments,
        model,
        metadata,
        audio,
        "cpu",
        return_char_alignments=False,
    )
    expected = alignment_baselines[filename]

    assert len(result["segments"]) == len(expected["segments"])
    assert len(result["word_segments"]) == len(expected["word_segments"])

    for actual_segment, expected_segment in zip(result["segments"], expected["segments"]):
        assert actual_segment["text"] == expected_segment["text"]
        assert len(actual_segment["words"]) == len(expected_segment["words"])

    for actual_word, expected_word in zip(result["word_segments"], expected["word_segments"]):
        assert actual_word["word"] == expected_word["word"]
        assert actual_word["start"] == pytest.approx(expected_word["start"], abs=ALIGNMENT_TOLERANCE_S)
        assert actual_word["end"] == pytest.approx(expected_word["end"], abs=ALIGNMENT_TOLERANCE_S)
        assert actual_word["score"] == pytest.approx(expected_word["score"], abs=ALIGNMENT_SCORE_TOLERANCE)
