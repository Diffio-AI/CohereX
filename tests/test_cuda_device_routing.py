import argparse
import sys
import types

import numpy as np

from coherex.alignment import _resolve_device
from coherex.transcribe import transcribe_task


def test_alignment_resolves_cuda_device_index():
    assert str(_resolve_device("cuda", device_index=1)) == "cuda:1"
    assert str(_resolve_device("cuda:1", device_index=0)) == "cuda:1"


def test_transcribe_task_passes_device_index_to_alignment(monkeypatch, tmp_path):
    calls = {"load_align_model": [], "align": []}

    class FakeModel:
        def transcribe(self, audio, batch_size, chunk_size, print_progress, verbose):
            return {
                "segments": [{"start": 0.0, "end": 0.5, "text": "hello"}],
                "language": "en",
            }

    def fake_load_model(*args, **kwargs):
        return FakeModel()

    def fake_load_audio(path):
        return np.zeros(16000, dtype=np.float32)

    def fake_get_writer(output_format, output_dir):
        def writer(result, audio_path, writer_args):
            return None

        return writer

    def fake_load_align_model(language_code, device, **kwargs):
        calls["load_align_model"].append(
            {
                "language_code": language_code,
                "device": device,
                "device_index": kwargs.get("device_index"),
                "backend": kwargs.get("backend"),
            }
        )
        return object(), {"language": language_code, "dictionary": {}, "type": "torchaudio"}

    def fake_align(transcript, model, align_model_metadata, audio, device, **kwargs):
        calls["align"].append(
            {
                "device": device,
                "device_index": kwargs.get("device_index"),
            }
        )
        return {"segments": transcript, "word_segments": []}

    fake_alignment_module = types.ModuleType("coherex.alignment")
    fake_alignment_module.load_align_model = fake_load_align_model
    fake_alignment_module.align = fake_align

    monkeypatch.setattr("coherex.transcribe.load_model", fake_load_model)
    monkeypatch.setattr("coherex.transcribe.load_audio", fake_load_audio)
    monkeypatch.setattr("coherex.transcribe.get_writer", fake_get_writer)
    monkeypatch.setitem(sys.modules, "coherex.alignment", fake_alignment_module)

    args = {
        "model": "CohereLabs/cohere-transcribe-03-2026",
        "batch_size": 1,
        "model_dir": None,
        "model_cache_only": True,
        "output_dir": str(tmp_path),
        "output_format": "json",
        "device": "cuda",
        "device_index": 1,
        "compute_type": "default",
        "verbose": False,
        "align_model": None,
        "align_backend": "qwen3",
        "interpolate_method": "nearest",
        "no_align": False,
        "return_char_alignments": False,
        "language": "en",
        "lid_method": "speechbrain",
        "lid_model": None,
        "lid_model_dir": None,
        "hf_token": None,
        "vad_method": "firered",
        "vad_model_dir": None,
        "vad_onset": 0.5,
        "vad_offset": 0.363,
        "chunk_size": 35.0,
        "suppress_numerals": False,
        "no_punctuation": False,
        "print_progress": False,
        "max_new_tokens": 128,
        "threads": 0,
        "highlight_words": False,
        "max_line_count": None,
        "max_line_width": None,
        "audio": ["fixture.wav"],
    }

    transcribe_task(args, argparse.ArgumentParser())

    assert calls["load_align_model"] == [
        {"language_code": "en", "device": "cuda", "device_index": 1, "backend": "qwen3"}
    ]
    assert calls["align"] == [
        {"device": "cuda", "device_index": 1}
    ]
