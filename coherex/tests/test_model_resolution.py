from pathlib import Path

import pytest

from coherex.asr import _resolve_model_source
from coherex.lids.base import (
    SPEECHBRAIN_LID_REQUIRED_FILES,
    TALTECH_LID_REQUIRED_FILES,
    resolve_lid_model_source,
)
from coherex.vads.firered import _resolve_model_dir


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")


def test_resolve_model_source_prefers_local_snapshot_dir(tmp_path, monkeypatch):
    model_dir = tmp_path / "cohere-snapshot"
    for filename in ("config.json", "model.safetensors", "preprocessor_config.json", "tokenizer_config.json"):
        _touch(model_dir / filename)

    called = False

    def fake_snapshot_download(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("snapshot_download should not be called for a local snapshot dir")

    monkeypatch.setattr("coherex.asr.snapshot_download", fake_snapshot_download)

    resolved = _resolve_model_source(
        model_name=str(model_dir),
        download_root=None,
        local_files_only=False,
        token=None,
    )

    assert resolved == str(model_dir)
    assert called is False


def test_resolve_model_source_uses_download_root_snapshot_dir(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "cached-snapshot"
    for filename in ("config.json", "model.safetensors", "preprocessor_config.json", "tokenizer_config.json"):
        _touch(snapshot_dir / filename)

    called = False

    def fake_snapshot_download(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("snapshot_download should not be called when --model_dir is already a snapshot dir")

    monkeypatch.setattr("coherex.asr.snapshot_download", fake_snapshot_download)

    resolved = _resolve_model_source(
        model_name="CohereLabs/cohere-transcribe-03-2026",
        download_root=str(snapshot_dir),
        local_files_only=False,
        token=None,
    )

    assert resolved == str(snapshot_dir)
    assert called is False


def test_resolve_model_source_downloads_when_snapshot_missing(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    downloaded_snapshot = tmp_path / "downloaded"
    downloaded_snapshot.mkdir()

    calls = []

    def fake_snapshot_download(*args, **kwargs):
        calls.append(kwargs)
        return str(downloaded_snapshot)

    monkeypatch.setattr("coherex.asr.snapshot_download", fake_snapshot_download)

    resolved = _resolve_model_source(
        model_name="CohereLabs/cohere-transcribe-03-2026",
        download_root=str(cache_root),
        local_files_only=False,
        token="token",
    )

    assert resolved == str(downloaded_snapshot)
    assert calls == [{"cache_dir": str(cache_root), "local_files_only": False, "token": "token"}]


def test_resolve_model_source_raises_clear_error_for_missing_cached_files(tmp_path, monkeypatch):
    def fake_snapshot_download(*args, **kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr("coherex.asr.snapshot_download", fake_snapshot_download)

    with pytest.raises(RuntimeError, match="Unable to load cached Cohere model files"):
        _resolve_model_source(
            model_name="CohereLabs/cohere-transcribe-03-2026",
            download_root=str(tmp_path / "cache"),
            local_files_only=True,
            token=None,
        )


def test_resolve_firered_model_dir_prefers_local_vad_dir(tmp_path, monkeypatch):
    model_dir = tmp_path / "VAD"
    for filename in ("cmvn.ark", "model.pth.tar"):
        _touch(model_dir / filename)

    called = False

    def fake_snapshot_download(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("snapshot_download should not be called for a local FireRed dir")

    monkeypatch.setattr("coherex.vads.firered.snapshot_download", fake_snapshot_download)

    resolved = _resolve_model_dir(
        model_dir=str(model_dir),
        cache_dir=None,
        local_files_only=False,
        token=None,
    )

    assert resolved == str(model_dir)
    assert called is False


def test_resolve_firered_model_dir_accepts_snapshot_root(tmp_path, monkeypatch):
    snapshot_root = tmp_path / "snapshot"
    vad_dir = snapshot_root / "VAD"
    for filename in ("cmvn.ark", "model.pth.tar"):
        _touch(vad_dir / filename)

    called = False

    def fake_snapshot_download(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("snapshot_download should not be called for an existing FireRed snapshot root")

    monkeypatch.setattr("coherex.vads.firered.snapshot_download", fake_snapshot_download)

    resolved = _resolve_model_dir(
        model_dir=str(snapshot_root),
        cache_dir=None,
        local_files_only=False,
        token=None,
    )

    assert resolved == str(vad_dir)
    assert called is False


def test_resolve_firered_model_dir_downloads_when_missing(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    snapshot_root = tmp_path / "downloaded"
    (snapshot_root / "VAD").mkdir(parents=True)
    for filename in ("cmvn.ark", "model.pth.tar"):
        _touch(snapshot_root / "VAD" / filename)

    calls = []

    def fake_snapshot_download(*args, **kwargs):
        calls.append(kwargs)
        return str(snapshot_root)

    monkeypatch.setattr("coherex.vads.firered.snapshot_download", fake_snapshot_download)

    resolved = _resolve_model_dir(
        model_dir=str(cache_root),
        cache_dir=None,
        local_files_only=False,
        token="token",
    )

    assert resolved == str(snapshot_root / "VAD")
    assert calls == [{
        "allow_patterns": ["VAD/*"],
        "cache_dir": str(cache_root),
        "local_files_only": False,
        "token": "token",
    }]


@pytest.mark.parametrize(
    ("required_files", "model_name", "log_name"),
    [
        (SPEECHBRAIN_LID_REQUIRED_FILES, "speechbrain/lang-id-voxlingua107-ecapa", "SpeechBrain language-id"),
        (TALTECH_LID_REQUIRED_FILES, "TalTechNLP/voxlingua107-xls-r-300m-wav2vec", "TalTech language-id"),
    ],
)
def test_resolve_lid_model_source_prefers_local_snapshot_dir(tmp_path, monkeypatch, required_files, model_name, log_name):
    model_dir = tmp_path / "lid-snapshot"
    for filename in required_files:
        _touch(model_dir / filename)

    called = False

    def fake_snapshot_download(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("snapshot_download should not be called for a local lid snapshot dir")

    monkeypatch.setattr("coherex.lids.base.snapshot_download", fake_snapshot_download)

    resolved = resolve_lid_model_source(
        model_name=str(model_dir),
        model_dir=None,
        local_files_only=False,
        token=None,
        required_files=required_files,
        log_name=log_name,
    )

    assert resolved == str(model_dir)
    assert called is False


@pytest.mark.parametrize(
    ("required_files", "model_name", "log_name"),
    [
        (SPEECHBRAIN_LID_REQUIRED_FILES, "speechbrain/lang-id-voxlingua107-ecapa", "SpeechBrain language-id"),
        (TALTECH_LID_REQUIRED_FILES, "TalTechNLP/voxlingua107-xls-r-300m-wav2vec", "TalTech language-id"),
    ],
)
def test_resolve_lid_model_source_downloads_when_missing(tmp_path, monkeypatch, required_files, model_name, log_name):
    cache_root = tmp_path / "cache"
    downloaded_snapshot = tmp_path / "downloaded"
    downloaded_snapshot.mkdir()
    for filename in required_files:
        _touch(downloaded_snapshot / filename)

    calls = []

    def fake_snapshot_download(*args, **kwargs):
        calls.append(kwargs)
        return str(downloaded_snapshot)

    monkeypatch.setattr("coherex.lids.base.snapshot_download", fake_snapshot_download)

    resolved = resolve_lid_model_source(
        model_name=model_name,
        model_dir=str(cache_root),
        local_files_only=False,
        token="token",
        required_files=required_files,
        log_name=log_name,
    )

    assert resolved == str(downloaded_snapshot)
    assert calls == [{"cache_dir": str(cache_root), "local_files_only": False, "token": "token"}]


@pytest.mark.parametrize(
    ("required_files", "model_name", "log_name"),
    [
        (SPEECHBRAIN_LID_REQUIRED_FILES, "speechbrain/lang-id-voxlingua107-ecapa", "SpeechBrain language-id"),
        (TALTECH_LID_REQUIRED_FILES, "TalTechNLP/voxlingua107-xls-r-300m-wav2vec", "TalTech language-id"),
    ],
)
def test_resolve_lid_model_source_raises_clear_error_for_missing_cached_files(tmp_path, monkeypatch, required_files, model_name, log_name):
    def fake_snapshot_download(*args, **kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr("coherex.lids.base.snapshot_download", fake_snapshot_download)

    with pytest.raises(RuntimeError, match=f"Unable to load cached {log_name} model files"):
        resolve_lid_model_source(
            model_name=model_name,
            model_dir=str(tmp_path / "cache"),
            local_files_only=True,
            token=None,
            required_files=required_files,
            log_name=log_name,
        )
