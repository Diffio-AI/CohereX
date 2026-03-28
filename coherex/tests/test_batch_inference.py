from types import SimpleNamespace

import numpy as np
import pytest
import torch

import coherex.asr as asr_module
from coherex.asr import Chunk, CohereTranscriptionPipeline, NullVad, SpeechSpan


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99
    all_special_ids = [0, 99]

    def get_vocab(self):
        return {"<pad>": 0, "<eos>": 99, "zero": 1000, "one": 1001, "two": 1002}

    def convert_ids_to_tokens(self, token_id):
        reverse = {
            0: "<pad>",
            99: "<eos>",
            1000: "zero",
            1001: "one",
            1002: "two",
        }
        return reverse[token_id]

    def batch_decode(self, token_ids, skip_special_tokens=True):
        mapping = {1000: "zero", 1001: "one", 1002: "two"}
        return [mapping[ids[0]] for ids in token_ids]


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.batch_lengths = []

    def __call__(self, audio, text, sampling_rate, return_tensors):
        self.batch_lengths.append([len(sample) for sample in audio])
        markers = torch.tensor([int(sample[0]) for sample in audio], dtype=torch.long)
        return {
            "input_features": markers.unsqueeze(1),
            "input_ids": torch.full((len(audio), 1), 42, dtype=torch.long),
        }


class FakeModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(max_audio_clip_s=35, overlap_chunk_second=5, batch_size=4)

    def build_prompt(self, language, punctuation):
        return f"{language}:{punctuation}"

    def generate(self, input_features=None, decoder_input_ids=None, **kwargs):
        rows = []
        for idx in range(input_features.shape[0]):
            marker = int(input_features[idx, 0].item())
            rows.append([42, 1000 + marker, 99])
        return torch.tensor(rows, dtype=torch.long)


def _make_pipeline(batch_size=4):
    return CohereTranscriptionPipeline(
        model=FakeModel(),
        processor=FakeProcessor(),
        vad=NullVad(),
        vad_params={"vad_onset": 0.5, "vad_offset": 0.363},
        language="en",
        punctuation=True,
        suppress_numerals=False,
        max_new_tokens=32,
        batch_size=batch_size,
    )


def test_batch_inference_sorts_by_length_and_restores_original_order():
    pipeline = _make_pipeline(batch_size=2)
    chunks = [
        Chunk(0.0, 1.0, [SpeechSpan(0.0, 1.0)], np.full(5, 0, dtype=np.float32)),
        Chunk(1.0, 2.0, [SpeechSpan(1.0, 2.0)], np.full(10, 1, dtype=np.float32)),
        Chunk(2.0, 3.0, [SpeechSpan(2.0, 3.0)], np.full(7, 2, dtype=np.float32)),
    ]

    texts = pipeline._transcribe_chunks(chunks, batch_size=2, print_progress=False, progress_callback=None)

    assert pipeline.processor.batch_lengths == [[10, 7], [5]]
    assert texts == ["zero", "one", "two"]


def test_batch_inference_uses_pipeline_default_batch_size(monkeypatch):
    pipeline = _make_pipeline(batch_size=3)

    def fake_chunk_spans(**kwargs):
        return [Chunk(0.0, 1.0, [SpeechSpan(0.0, 1.0)], np.zeros(10, dtype=np.float32))]

    captured = {}

    def fake_transcribe_chunks(chunks, batch_size, print_progress, progress_callback):
        captured["batch_size"] = batch_size
        return ["ok"]

    monkeypatch.setattr(asr_module, "_chunk_spans", fake_chunk_spans)
    monkeypatch.setattr(pipeline, "_transcribe_chunks", fake_transcribe_chunks)

    result = pipeline.transcribe(np.zeros(16000, dtype=np.float32), batch_size=None)

    assert captured["batch_size"] == 3
    assert result["segments"][0]["text"] == "ok"


def test_batch_inference_reports_progress_by_processed_chunks():
    pipeline = _make_pipeline(batch_size=2)
    chunks = [
        Chunk(0.0, 1.0, [SpeechSpan(0.0, 1.0)], np.full(5, 0, dtype=np.float32)),
        Chunk(1.0, 2.0, [SpeechSpan(1.0, 2.0)], np.full(10, 1, dtype=np.float32)),
        Chunk(2.0, 3.0, [SpeechSpan(2.0, 3.0)], np.full(7, 2, dtype=np.float32)),
    ]
    progress = []

    pipeline._transcribe_chunks(chunks, batch_size=2, print_progress=False, progress_callback=progress.append)

    assert progress == pytest.approx([66.6666667, 100.0])


def test_batch_inference_rejects_non_positive_batch_size():
    pipeline = _make_pipeline(batch_size=2)

    with pytest.raises(ValueError, match="batch_size must be > 0"):
        pipeline._transcribe_chunks([], batch_size=0, print_progress=False, progress_callback=None)
