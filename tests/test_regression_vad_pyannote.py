import json
from pathlib import Path

import pytest

from coherex.audio import SAMPLE_RATE, load_audio
from coherex.vads.pyannote import Pyannote


FIXTURES_DIR = Path(__file__).parent / "fixtures"
AUDIO_DIR = FIXTURES_DIR / "audio"
BASELINE_PATH = FIXTURES_DIR / "regression_vad_pyannote_whisperx.json"
VAD_TOLERANCE_S = 1e-6


@pytest.fixture(scope="session")
def vad_baseline():
    return json.loads(BASELINE_PATH.read_text())


def test_pyannote_vad_regression_matches_whisperx_ground_truth(vad_baseline):
    audio = load_audio(str(AUDIO_DIR / vad_baseline["audio"]))
    vad = Pyannote(
        device="cpu",
        vad_onset=vad_baseline["vad_onset"],
        vad_offset=vad_baseline["vad_offset"],
    )
    waveform = vad.preprocess_audio(audio)
    scores = vad({"waveform": waveform, "sample_rate": SAMPLE_RATE})
    result = vad.merge_chunks(
        scores,
        vad_baseline["chunk_size"],
        onset=vad_baseline["vad_onset"],
        offset=vad_baseline["vad_offset"],
    )

    expected_segments = vad_baseline["segments"]
    assert len(result) == len(expected_segments)

    for actual_segment, expected_segment in zip(result, expected_segments):
        assert actual_segment["start"] == pytest.approx(expected_segment["start"], abs=VAD_TOLERANCE_S)
        assert actual_segment["end"] == pytest.approx(expected_segment["end"], abs=VAD_TOLERANCE_S)
        assert len(actual_segment["segments"]) == len(expected_segment["segments"])
        for actual_span, expected_span in zip(actual_segment["segments"], expected_segment["segments"]):
            assert actual_span[0] == pytest.approx(expected_span[0], abs=VAD_TOLERANCE_S)
            assert actual_span[1] == pytest.approx(expected_span[1], abs=VAD_TOLERANCE_S)
