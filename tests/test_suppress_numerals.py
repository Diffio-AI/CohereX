import gc
import os
import re
from pathlib import Path

import pytest
import torch

from coherex.asr import find_numeral_symbol_tokens, load_model
from coherex.audio import load_audio
from coherex.tokenization_cohere_asr import CohereAsrTokenizer


FIXTURES_DIR = Path(__file__).parent / "fixtures"
AUDIO_PATH = FIXTURES_DIR / "audio" / "suppress-numerals.mp3"
COHEREX_OFF_PATH = FIXTURES_DIR / "coherex_suppress_numerals_off.txt"
COHEREX_ON_PATH = FIXTURES_DIR / "coherex_suppress_numerals_on.txt"
WHISPERX_OFF_PATH = FIXTURES_DIR / "whisperx_suppress_numerals_off.txt"
WHISPERX_ON_PATH = FIXTURES_DIR / "whisperx_suppress_numerals_on.txt"
NUMERAL_PATTERN = re.compile(r"[0-9%$£]")

# The WhisperX pipeline aborts with std::bad_alloc in this environment, so these
# reference texts were generated with faster-whisper using WhisperX's decoded-token
# suppression heuristic from whisperx/asr.py.


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _expected_numeral_symbol_tokens(tokenizer: CohereAsrTokenizer) -> list[int]:
    special_ids = set(tokenizer.all_special_ids)
    numeral_symbol_tokens = []
    for token_id in sorted(set(tokenizer.get_vocab().values())):
        if token_id in special_ids:
            continue
        token = tokenizer.decode([token_id], skip_special_tokens=False).removeprefix(" ")
        if any(char in "0123456789%$£" for char in token):
            numeral_symbol_tokens.append(token_id)
    return numeral_symbol_tokens


def _transcribed_text(result: dict) -> str:
    return _normalize_text(" ".join(segment["text"] for segment in result["segments"]))


@pytest.fixture(scope="session")
def suppress_numerals_audio():
    return load_audio(str(AUDIO_PATH))


@pytest.fixture(scope="session")
def suppress_numerals_models():
    device = os.environ.get("COHEREX_TEST_DEVICE", "cpu")
    try:
        models = {
            False: load_model(
                "CohereLabs/cohere-transcribe-03-2026",
                device=device,
                language="en",
                vad_method="none",
                local_files_only=True,
                asr_options={"max_new_tokens": 448, "suppress_numerals": False},
            ),
            True: load_model(
                "CohereLabs/cohere-transcribe-03-2026",
                device=device,
                language="en",
                vad_method="none",
                local_files_only=True,
                asr_options={"max_new_tokens": 448, "suppress_numerals": True},
            ),
        }
    except Exception as exc:
        pytest.skip(f"Suppress numerals regression model unavailable: {exc}")

    yield models

    for model in models.values():
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def suppress_numerals_results(suppress_numerals_models, suppress_numerals_audio):
    return {
        suppress: model.transcribe(
            suppress_numerals_audio,
            batch_size=1,
            print_progress=False,
            verbose=False,
        )
        for suppress, model in suppress_numerals_models.items()
    }


def test_find_numeral_symbol_tokens_matches_whisperx_decoded_token_scan():
    tokenizer = CohereAsrTokenizer.from_pretrained(
        "CohereLabs/cohere-transcribe-03-2026",
        local_files_only=True,
    )

    actual = find_numeral_symbol_tokens(tokenizer)
    expected = _expected_numeral_symbol_tokens(tokenizer)

    assert actual == expected
    assert 255 not in actual


def test_transcription_regression_without_suppress_numerals(
    suppress_numerals_results,
):
    expected = _normalize_text(COHEREX_OFF_PATH.read_text())

    assert _transcribed_text(suppress_numerals_results[False]) == expected


def test_transcription_regression_with_suppress_numerals(
    suppress_numerals_results,
):
    expected = _normalize_text(COHEREX_ON_PATH.read_text())

    assert _transcribed_text(suppress_numerals_results[True]) == expected


def test_suppress_numerals_behavior_matches_whisperx_reference(
    suppress_numerals_results,
):
    whisperx_default = _normalize_text(WHISPERX_OFF_PATH.read_text())
    whisperx_suppressed = _normalize_text(WHISPERX_ON_PATH.read_text())
    default_text = _transcribed_text(suppress_numerals_results[False])
    suppressed_text = _transcribed_text(suppress_numerals_results[True])

    assert NUMERAL_PATTERN.search(whisperx_default)
    assert NUMERAL_PATTERN.search(default_text)
    assert NUMERAL_PATTERN.search(whisperx_suppressed) is None
    assert NUMERAL_PATTERN.search(suppressed_text) is None

    assert default_text != suppressed_text
    assert whisperx_default != whisperx_suppressed
    assert "100" in whisperx_default
    assert "100" in default_text
    assert "one hundred" in whisperx_suppressed.lower()
    assert "one hundred" in suppressed_text.lower()
    assert "five hundred thousand" in whisperx_suppressed.lower()
    assert "five hundred thousand" in suppressed_text.lower()
