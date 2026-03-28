import numpy as np

from coherex.asr import SpeechSpan, _chunk_spans, _join_chunk_texts, _split_oversized_span


def test_chunking_prefers_latest_valid_boundary():
    audio = np.zeros(16000 * 80, dtype=np.float32)
    spans = [
        SpeechSpan(0.0, 10.0),
        SpeechSpan(12.0, 20.0),
        SpeechSpan(22.0, 30.0),
        SpeechSpan(40.0, 50.0),
    ]

    chunks = _chunk_spans(audio, spans, 16000, 25.0, 5.0, 1600)

    assert [(round(chunk.start, 1), round(chunk.end, 1)) for chunk in chunks] == [
        (0.0, 20.0),
        (22.0, 50.0),
    ]
    assert [round(sum(span.duration for span in chunk.spans), 1) for chunk in chunks] == [18.0, 18.0]


def test_oversized_span_is_split_below_limit():
    audio = np.zeros(16000 * 50, dtype=np.float32)
    spans = _split_oversized_span(
        audio=audio,
        span=SpeechSpan(0.0, 50.0),
        sample_rate=16000,
        chunk_size=35.0,
        boundary_context=5.0,
        min_energy_window_samples=1600,
    )

    assert len(spans) == 2
    assert all(span.duration <= 35.0 for span in spans)


def test_join_chunk_texts_respects_no_space_languages():
    assert _join_chunk_texts(["hello", "world"], "en") == "hello world"
    assert _join_chunk_texts(["你", "好"], "zh") == "你好"
