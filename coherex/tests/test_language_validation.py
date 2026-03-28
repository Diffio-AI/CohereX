import sys

import pytest

from coherex.asr import load_model
from coherex.configuration_cohere_asr import normalize_language_code, supported_languages_help_text


def test_normalize_language_code_accepts_supported_names_and_codes():
    assert normalize_language_code("en") == "en"
    assert normalize_language_code("English") == "en"
    assert normalize_language_code("french") == "fr"
    assert normalize_language_code("Chinese (Mandarin)") == "zh"
    assert normalize_language_code("mandarin") == "zh"
    assert normalize_language_code("Korean") == "ko"


def test_supported_languages_help_text_lists_regions_and_codes():
    help_text = supported_languages_help_text()

    assert "European:" in help_text
    assert "APAC:" in help_text
    assert "MENA:" in help_text
    assert "English (`en`)" in help_text
    assert "Chinese (Mandarin) (`zh`)" in help_text
    assert "Arabic (`ar`)" in help_text


def test_load_model_rejects_unsupported_language():
    with pytest.raises(ValueError, match="Unsupported language"):
        load_model(
            "CohereLabs/cohere-transcribe-03-2026",
            device="cpu",
            language="swedish",
            vad_method="none",
            local_files_only=True,
        )


def test_cli_reports_supported_languages_for_invalid_language(monkeypatch, capsys):
    from coherex.__main__ import cli

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "coherex",
            "--language",
            "swedish",
            "dummy.wav",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli()

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "Unsupported language: 'swedish'" in stderr
    assert "European:" in stderr
    assert "APAC:" in stderr
    assert "MENA:" in stderr
    assert "English (`en`)" in stderr
