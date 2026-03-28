import numpy as np
import pytest

from coherex.alignment import (
    NEMO_CONFORMER_CTC_DEFAULT_ALIGN_MODELS,
    NEMO_CTC_OR_HYBRID_DEFAULT_ALIGN_MODELS,
    align,
    load_align_model,
)


class _FakeWord:
    def __init__(self, text, t_start, t_end):
        self.text = text
        self.t_start = t_start
        self.t_end = t_end
        self.tokens = []


class _FakeSegment:
    def __init__(self, words):
        self.words_and_tokens = words


class _FakeUtterance:
    def __init__(self, words):
        self.segments_and_tokens = [_FakeSegment(words)]


def test_load_align_model_nemo_ctc_or_hybrid_backend(monkeypatch):
    calls = []

    def fake_loader(model_name, resolved_device):
        calls.append((model_name, str(resolved_device)))
        return object()

    monkeypatch.setattr("coherex.alignment._load_nemo_align_model", fake_loader)

    model, metadata = load_align_model("en", "cpu", backend="nemo_ctc_or_hybrid")

    assert model is not None
    assert metadata == {
        "language": "en",
        "dictionary": None,
        "type": "nemo",
        "batch_size": 1,
        "model_name": NEMO_CTC_OR_HYBRID_DEFAULT_ALIGN_MODELS["en"],
    }
    assert calls == [(NEMO_CTC_OR_HYBRID_DEFAULT_ALIGN_MODELS["en"], "cpu")]


def test_load_align_model_nemo_conformer_ctc_backend(monkeypatch):
    calls = []

    def fake_loader(model_name, resolved_device):
        calls.append((model_name, str(resolved_device)))
        return object()

    monkeypatch.setattr("coherex.alignment._load_nemo_align_model", fake_loader)

    model, metadata = load_align_model("en", "cpu", backend="nemo_conformer_ctc")

    assert model is not None
    assert metadata == {
        "language": "en",
        "dictionary": None,
        "type": "nemo",
        "batch_size": 1,
        "model_name": NEMO_CONFORMER_CTC_DEFAULT_ALIGN_MODELS["en"],
    }
    assert calls == [(NEMO_CONFORMER_CTC_DEFAULT_ALIGN_MODELS["en"], "cpu")]


def test_load_align_model_infers_nemo_conformer_ctc_from_model_name(monkeypatch):
    calls = []

    def fake_loader(model_name, resolved_device):
        calls.append((model_name, str(resolved_device)))
        return object()

    monkeypatch.setattr("coherex.alignment._load_nemo_align_model", fake_loader)

    model, metadata = load_align_model(
        "en",
        "cpu",
        backend="wav2vec2",
        model_name=NEMO_CONFORMER_CTC_DEFAULT_ALIGN_MODELS["en"],
    )

    assert model is not None
    assert metadata["type"] == "nemo"
    assert metadata["model_name"] == NEMO_CONFORMER_CTC_DEFAULT_ALIGN_MODELS["en"]
    assert calls == [(NEMO_CONFORMER_CTC_DEFAULT_ALIGN_MODELS["en"], "cpu")]


@pytest.mark.parametrize(
    ("backend", "error_message"),
    [
        ("nemo_ctc_or_hybrid", "No default NeMo CTC/Hybrid align-model"),
        ("nemo_conformer_ctc", "No default NeMo Conformer CTC align-model"),
    ],
)
def test_load_align_model_nemo_backends_require_model_for_non_default_language(backend, error_message):
    with pytest.raises(ValueError, match=error_message):
        load_align_model("fr", "cpu", backend=backend)


@pytest.mark.parametrize(
    ("legacy_backend", "canonical_backend"),
    [
        ("nemo", "nemo_ctc_or_hybrid"),
        ("nemo_conformer", "nemo_conformer_ctc"),
    ],
)
def test_load_align_model_accepts_legacy_nemo_backend_aliases(
    monkeypatch,
    legacy_backend,
    canonical_backend,
):
    calls = []

    def fake_loader(model_name, resolved_device):
        calls.append((model_name, str(resolved_device)))
        return object()

    monkeypatch.setattr("coherex.alignment._load_nemo_align_model", fake_loader)

    with pytest.deprecated_call():
        model, metadata = load_align_model("en", "cpu", backend=legacy_backend)

    assert model is not None
    expected_model_name = {
        "nemo_ctc_or_hybrid": NEMO_CTC_OR_HYBRID_DEFAULT_ALIGN_MODELS["en"],
        "nemo_conformer_ctc": NEMO_CONFORMER_CTC_DEFAULT_ALIGN_MODELS["en"],
    }[canonical_backend]
    assert metadata["model_name"] == expected_model_name
    assert calls == [(expected_model_name, "cpu")]


def test_align_with_nemo_preserves_segment_order(monkeypatch):
    utterances = {
        "hello world": _FakeUtterance(
            [_FakeWord("hello", 0.1, 0.4), _FakeWord("world", 0.4, 0.9)]
        ),
        "again now": _FakeUtterance(
            [_FakeWord("again", 0.05, 0.25), _FakeWord("now", 0.25, 0.5)]
        ),
    }

    def fake_get_nemo_alignment_utils():
        def fake_get_batch_variables(
            audio,
            model,
            segment_separators,
            word_separator,
            align_using_pred_text,
            audio_filepath_parts_in_utt_id,
            gt_text_batch,
            output_timestep_duration,
        ):
            utt_batch = [utterances[text] for text in gt_text_batch]
            size = len(gt_text_batch)
            return None, None, None, None, utt_batch, 0.02 if output_timestep_duration is None else output_timestep_duration

        def fake_viterbi_decoding(*args, **kwargs):
            return [[0], [0]]

        def fake_add_t_start_end_to_utt_obj(utt_obj, alignment_utt, output_timestep_duration):
            return utt_obj

        return fake_add_t_start_end_to_utt_obj, fake_get_batch_variables, fake_viterbi_decoding

    monkeypatch.setattr(
        "coherex.alignment._get_nemo_alignment_utils",
        fake_get_nemo_alignment_utils,
    )

    transcript = [
        {"start": 0.0, "end": 1.5, "text": "hello world"},
        {"start": 1.5, "end": 2.0, "text": "   "},
        {"start": 2.0, "end": 3.0, "text": "again now", "avg_logprob": -0.2},
    ]
    audio = np.zeros(4 * 16000, dtype=np.float32)

    result = align(
        transcript=transcript,
        model=object(),
        align_model_metadata={
            "language": "en",
            "dictionary": None,
            "type": "nemo",
            "batch_size": 2,
            "model_name": NEMO_CTC_OR_HYBRID_DEFAULT_ALIGN_MODELS["en"],
        },
        audio=audio,
        device="cpu",
        return_char_alignments=True,
    )

    assert [segment["text"] for segment in result["segments"]] == [
        "hello world",
        "   ",
        "again now",
    ]
    assert result["segments"][0]["words"] == [
        {"word": "hello", "start": 0.1, "end": 0.4, "score": 1.0},
        {"word": "world", "start": 0.4, "end": 0.9, "score": 1.0},
    ]
    assert result["segments"][1]["words"] == []
    assert result["segments"][1]["chars"] == []
    assert result["segments"][2]["words"] == [
        {"word": "again", "start": 2.05, "end": 2.25, "score": 1.0},
        {"word": "now", "start": 2.25, "end": 2.5, "score": 1.0},
    ]
    assert result["segments"][2]["avg_logprob"] == pytest.approx(-0.2)
