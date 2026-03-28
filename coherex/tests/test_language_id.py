import gc
import json
import os
from pathlib import Path

import pytest
import torch

from coherex import load_lid_model
from coherex.audio import load_audio
from coherex.lids.base import select_supported_language, supported_language_scores


FIXTURES_DIR = Path(__file__).parent / "fixtures"
AUDIO_DIR = FIXTURES_DIR / "audio"
BASELINE_PATH = FIXTURES_DIR / "regression_language_id.json"


@pytest.fixture(scope="session")
def language_id_baselines():
    return json.loads(BASELINE_PATH.read_text())


@pytest.fixture(scope="session")
def speechbrain_lid():
    device = torch.device(os.environ.get("COHEREX_TEST_DEVICE", "cpu"))
    try:
        model = load_lid_model(
            method="speechbrain",
            device=device,
            local_files_only=True,
        )
    except Exception as exc:
        pytest.skip(f"SpeechBrain language-id model unavailable: {exc}")

    yield model

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def taltech_lid():
    device = torch.device(os.environ.get("COHEREX_TEST_DEVICE", "cpu"))
    try:
        model = load_lid_model(
            method="taltech",
            device=device,
            local_files_only=True,
        )
    except Exception as exc:
        pytest.skip(f"TalTech language-id model unavailable: {exc}")

    yield model

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_select_supported_language_falls_back_to_best_supported_score():
    scores = supported_language_scores(
        ["sw: Swahili", "en: English", "fr: French"],
        torch.tensor([-0.1, -0.2, -0.3]),
    )

    assert select_supported_language("sw: Swahili", scores) == "en"


@pytest.mark.parametrize(
    "filename",
    [
        "hur.mp3",
        "amelia_earhart_noisy.c431d09f.wav",
        "david-gooding-noisy.mp3",
        "suppress-numerals.mp3",
    ],
)
def test_speechbrain_language_id_regression(speechbrain_lid, language_id_baselines, filename):
    prediction = speechbrain_lid.detect(load_audio(str(AUDIO_DIR / filename)))
    expected = language_id_baselines[filename]["speechbrain"]

    assert prediction.language == expected["language"]
    assert prediction.raw_label
    assert max(prediction.scores, key=prediction.scores.get) == expected["language"]


@pytest.mark.parametrize(
    "filename",
    [
        "hur.mp3",
        "amelia_earhart_noisy.c431d09f.wav",
        "david-gooding-noisy.mp3",
        "suppress-numerals.mp3",
    ],
)
def test_taltech_language_id_regression(taltech_lid, language_id_baselines, filename):
    prediction = taltech_lid.detect(load_audio(str(AUDIO_DIR / filename)))
    expected = language_id_baselines[filename]["taltech"]

    assert prediction.language == expected["language"]
    assert prediction.raw_label
    assert max(prediction.scores, key=prediction.scores.get) == expected["language"]
