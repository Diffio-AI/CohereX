import sys
import types

import numpy as np
import pytest

from coherex.alignment import QWEN3_FORCED_ALIGNER_MODEL, align, load_align_model
from coherex.audio import SAMPLE_RATE


class _FakeAlignItem:
    def __init__(self, text, start_time, end_time):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time


class _FakeAlignResult:
    def __init__(self, items):
        self.items = items

    def __iter__(self):
        return iter(self.items)


def test_load_align_model_qwen_backend(monkeypatch, tmp_path):
    calls = []

    class FakeQwen3ForcedAligner:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append((model_name, kwargs))
            model = cls()
            model.aligner_processor = types.SimpleNamespace(
                encode_timestamp=lambda text, language: (text.split(), text)
            )
            return model

    fake_module = types.ModuleType("qwen_asr")
    fake_module.Qwen3ForcedAligner = FakeQwen3ForcedAligner
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_module)

    model, metadata = load_align_model(
        "en",
        "cpu",
        model_name=QWEN3_FORCED_ALIGNER_MODEL,
        model_dir=str(tmp_path),
        model_cache_only=True,
    )

    assert model is not None
    assert metadata == {
        "language": "en",
        "language_name": "English",
        "dictionary": None,
        "type": "qwen3",
    }
    assert calls == [
        (
            QWEN3_FORCED_ALIGNER_MODEL,
            {
                "cache_dir": str(tmp_path),
                "local_files_only": True,
                "device_map": "cpu",
            },
        )
    ]


def test_load_align_model_qwen_rejects_unsupported_language():
    with pytest.raises(ValueError, match="does not support language"):
        load_align_model("ar", "cpu", backend="qwen3")


def test_qwen_alignment_chunks_and_offsets_windows():
    class FakeQwenModel:
        def __init__(self):
            self.aligner_processor = types.SimpleNamespace(
                encode_timestamp=lambda text, language: (text.split(), text)
            )
            self.calls = []

        def align(self, audio, text, language):
            self.calls.append(
                {
                    "duration": round(len(audio[0]) / SAMPLE_RATE, 3),
                    "text": text,
                    "language": language,
                }
            )
            outputs = {
                "hello world again there": [
                    _FakeAlignItem("hello", 0.1, 0.6),
                    _FakeAlignItem("world", 0.6, 1.0),
                    _FakeAlignItem("again", 150.2, 150.7),
                    _FakeAlignItem("there", 150.7, 151.0),
                ],
                "final bit": [
                    _FakeAlignItem("final", 0.2, 0.5),
                    _FakeAlignItem("bit", 0.5, 0.8),
                ],
            }
            return [_FakeAlignResult(outputs[text])]

    transcript = [
        {"start": 0.0, "end": 100.0, "text": "hello world", "avg_logprob": -0.1},
        {"start": 100.0, "end": 260.0, "text": "again there"},
        {"start": 260.0, "end": 320.0, "text": "final bit"},
    ]
    audio = np.zeros(int(321 * SAMPLE_RATE), dtype=np.float32)
    model = FakeQwenModel()

    result = align(
        transcript=transcript,
        model=model,
        align_model_metadata={
            "language": "en",
            "language_name": "English",
            "dictionary": None,
            "type": "qwen3",
        },
        audio=audio,
        device="cpu",
        return_char_alignments=True,
    )

    assert model.calls == [
        {"duration": 260.0, "text": "hello world again there", "language": "English"},
        {"duration": 60.0, "text": "final bit", "language": "English"},
    ]
    assert [segment["text"] for segment in result["segments"]] == [
        "hello world",
        "again there",
        "final bit",
    ]
    assert result["segments"][0]["start"] == pytest.approx(0.1)
    assert result["segments"][0]["end"] == pytest.approx(1.0)
    assert result["segments"][0]["avg_logprob"] == pytest.approx(-0.1)
    assert result["segments"][0]["chars"] == []
    assert result["segments"][1]["words"] == [
        {"word": "again", "start": 150.2, "end": 150.7, "score": 1.0},
        {"word": "there", "start": 150.7, "end": 151.0, "score": 1.0},
    ]
    assert result["segments"][2]["words"] == [
        {"word": "final", "start": 260.2, "end": 260.5, "score": 1.0},
        {"word": "bit", "start": 260.5, "end": 260.8, "score": 1.0},
    ]
    assert result["word_segments"][-2:] == [
        {"word": "final", "start": 260.2, "end": 260.5, "score": 1.0},
        {"word": "bit", "start": 260.5, "end": 260.8, "score": 1.0},
    ]
