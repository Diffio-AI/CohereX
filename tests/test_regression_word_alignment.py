import gc
import json
import statistics
import unicodedata
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
QWEN_ALIGNMENT_TOLERANCE_S = 0.12
NEMO_MEAN_START_ERROR_TOLERANCE_S = 0.09
NEMO_MEAN_END_ERROR_TOLERANCE_S = 0.16
NEMO_P95_START_ERROR_TOLERANCE_S = 0.18
NEMO_P95_END_ERROR_TOLERANCE_S = 0.40
NEMO_MAX_START_ERROR_TOLERANCE_S = 0.30
NEMO_MAX_END_ERROR_TOLERANCE_S = 0.55


def _normalize_qwen_word(word: str) -> str:
    kept = []
    for ch in word:
        if ch == "'" or unicodedata.category(ch).startswith(("L", "N")):
            kept.append(ch)
    return "".join(kept)


def _normalized_ground_truth_word_segments(word_segments):
    normalized = []
    for word in word_segments:
        normalized_word = _normalize_qwen_word(word["word"])
        if not normalized_word:
            continue
        normalized.append(
            {
                "word": normalized_word,
                "start": word["start"],
                "end": word["end"],
            }
        )
    return normalized


def _percentile(values, fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


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


@pytest.fixture(scope="session")
def qwen_alignment_model():
    try:
        model, metadata = load_align_model("en", "cpu", backend="qwen3")
    except Exception as exc:
        pytest.skip(f"Qwen3 alignment model unavailable: {exc}")

    yield model, metadata

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def nemo_alignment_model():
    try:
        model, metadata = load_align_model("en", "cpu", backend="nemo")
    except Exception as exc:
        pytest.skip(f"NeMo alignment model unavailable: {exc}")

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


@pytest.mark.parametrize(
    "filename",
    [
        "hur.mp3",
        "amelia_earhart_noisy.c431d09f.wav",
        "david-gooding-noisy.mp3",
    ],
)
def test_qwen3_word_alignment_regression_matches_ground_truth(
    qwen_alignment_model,
    transcript_baselines,
    alignment_baselines,
    filename,
):
    model, metadata = qwen_alignment_model
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
    expected = _normalized_ground_truth_word_segments(
        alignment_baselines[filename]["word_segments"]
    )

    assert len(result["word_segments"]) == len(expected)

    for actual_word, expected_word in zip(result["word_segments"], expected):
        assert actual_word["word"] == expected_word["word"]
        assert actual_word["start"] == pytest.approx(
            expected_word["start"], abs=QWEN_ALIGNMENT_TOLERANCE_S
        )
        assert actual_word["end"] == pytest.approx(
            expected_word["end"], abs=QWEN_ALIGNMENT_TOLERANCE_S
        )


def test_nemo_word_alignment_regression_matches_ground_truth(
    nemo_alignment_model,
    transcript_baselines,
    alignment_baselines,
):
    filename = "amelia_earhart_noisy.c431d09f.wav"
    model, metadata = nemo_alignment_model
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

    assert len(result["word_segments"]) == len(expected["word_segments"])

    start_errors = []
    end_errors = []
    for actual_word, expected_word in zip(result["word_segments"], expected["word_segments"]):
        assert actual_word["word"] == expected_word["word"]
        start_errors.append(abs(actual_word["start"] - expected_word["start"]))
        end_errors.append(abs(actual_word["end"] - expected_word["end"]))

    # NeMo emits word boundaries at a coarser timestep than wav2vec2, so
    # we validate aggregate timing quality instead of requiring every word to
    # land inside the WhisperX baseline tolerance.
    assert statistics.mean(start_errors) <= NEMO_MEAN_START_ERROR_TOLERANCE_S
    assert statistics.mean(end_errors) <= NEMO_MEAN_END_ERROR_TOLERANCE_S
    assert _percentile(start_errors, 0.95) <= NEMO_P95_START_ERROR_TOLERANCE_S
    assert _percentile(end_errors, 0.95) <= NEMO_P95_END_ERROR_TOLERANCE_S
    assert max(start_errors) <= NEMO_MAX_START_ERROR_TOLERANCE_S
    assert max(end_errors) <= NEMO_MAX_END_ERROR_TOLERANCE_S
