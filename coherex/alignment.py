"""
Forced Alignment with Whisper
C. Max Bain
"""
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Iterable, Optional, Union, List

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
sys.modules.setdefault("tensorflow", None)
sys.modules.setdefault("tensorflow_text", None)
sys.modules.setdefault("keras", None)

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from coherex.audio import SAMPLE_RATE, load_audio
from coherex.utils import interpolate_nans, PUNKT_LANGUAGES
from coherex.schema import (
    AlignedTranscriptionResult,
    SingleSegment,
    SingleAlignedSegment,
    SingleWordSegment,
    SegmentData,
    ProgressCallback,
)
import nltk
from nltk.data import load as nltk_load
from coherex.log_utils import get_logger

logger = get_logger(__name__)

LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]
QWEN3_FORCED_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
QWEN3_MAX_ALIGN_SECONDS = 300.0
QWEN3_SUPPORTED_LANGUAGE_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
}
NEMO_DEFAULT_ALIGN_MODELS = {
    "en": "stt_en_fastconformer_hybrid_large_pc",
}

DEFAULT_ALIGN_MODELS_TORCH = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
}

DEFAULT_ALIGN_MODELS_HF = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
    "da": "saattrupdan/wav2vec2-xls-r-300m-ftspeech",
    "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
    "vi": 'nguyenvulebinh/wav2vec2-base-vi-vlsp2020',
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "ur": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
    "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "ca": "softcatala/wav2vec2-large-xlsr-catala",
    "ml": "gvs/wav2vec2-large-xlsr-malayalam",
    "no": "NbAiLab/nb-wav2vec2-1b-bokmaal-v2",
    "nn": "NbAiLab/nb-wav2vec2-1b-nynorsk",
    "sk": "comodoro/wav2vec2-xls-r-300m-sk-cv8",
    "sl": "anton-l/wav2vec2-large-xlsr-53-slovenian",
    "hr": "classla/wav2vec2-xls-r-parlaspeech-hr",
    "ro": "gigant/romanian-wav2vec2",
    "eu": "stefan-it/wav2vec2-large-xlsr-53-basque",
    "gl": "ifrz/wav2vec2-large-xlsr-galician",
    "ka": "xsway/wav2vec2-large-xlsr-georgian",
    "lv": "jimregan/wav2vec2-large-xlsr-latvian-cv",
    "tl": "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official",
    "sv": "KBLab/wav2vec2-large-voxrex-swedish",
}


def _resolve_device(device: Union[str, torch.device], device_index: int = 0) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "cuda":
        return torch.device(f"cuda:{device_index}")
    return torch.device(device)


def _infer_align_backend(model_name: Optional[str], backend: str) -> str:
    if model_name is not None and "qwen3-forcedaligner" in model_name.lower().replace("_", "-"):
        return "qwen3"
    return backend


def _build_fallback_segment(
    segment: SingleSegment,
    return_char_alignments: bool = False,
) -> SingleAlignedSegment:
    aligned_seg: SingleAlignedSegment = {
        "start": segment["start"],
        "end": segment["end"],
        "text": segment["text"],
        "words": [],
        "chars": [] if return_char_alignments else None,
    }
    if "avg_logprob" in segment:
        aligned_seg["avg_logprob"] = segment["avg_logprob"]
    return aligned_seg


def _qwen_supported_language_codes() -> str:
    return ", ".join(sorted(QWEN3_SUPPORTED_LANGUAGE_NAMES))


def _qwen_language_name(language_code: str) -> str:
    try:
        return QWEN3_SUPPORTED_LANGUAGE_NAMES[language_code]
    except KeyError as exc:
        raise ValueError(
            "Qwen3 forced alignment does not support language "
            f"{language_code!r}. Supported language codes: {_qwen_supported_language_codes()}"
        ) from exc


def _group_segments_for_qwen(
    transcript: Iterable[SingleSegment],
    max_chunk_seconds: float = QWEN3_MAX_ALIGN_SECONDS,
) -> List[List[SingleSegment]]:
    groups: List[List[SingleSegment]] = []
    current_group: List[SingleSegment] = []
    current_start: Optional[float] = None

    for segment in transcript:
        segment_duration = segment["end"] - segment["start"]
        if segment_duration > max_chunk_seconds:
            raise ValueError(
                "Qwen3 forced alignment only supports chunks up to "
                f"{max_chunk_seconds:.0f} seconds, but one transcript segment spans "
                f"{segment_duration:.3f} seconds. Reduce ASR chunking or VAD window size."
            )

        if not current_group:
            current_group = [segment]
            current_start = segment["start"]
            continue

        assert current_start is not None
        candidate_duration = segment["end"] - current_start
        if candidate_duration <= max_chunk_seconds:
            current_group.append(segment)
            continue

        groups.append(current_group)
        current_group = [segment]
        current_start = segment["start"]

    if current_group:
        groups.append(current_group)

    return groups


def _build_qwen_window_text(segments: List[SingleSegment], model_lang: str) -> str:
    separator = "" if model_lang in LANGUAGES_WITHOUT_SPACES else " "
    texts = [segment["text"].strip() for segment in segments if segment["text"].strip()]
    return separator.join(texts)


def _tokenize_qwen_segment(model, text: str, language_name: str) -> List[str]:
    if not text.strip():
        return []
    tokens, _ = model.aligner_processor.encode_timestamp(text, language_name)
    return tokens


def _align_with_qwen3(
    transcript: Iterable[SingleSegment],
    model,
    align_model_metadata: dict,
    audio: torch.Tensor,
    return_char_alignments: bool = False,
) -> AlignedTranscriptionResult:
    if return_char_alignments:
        logger.warning(
            "Qwen3 forced alignment does not expose WhisperX-style character alignments; "
            "returning empty char lists."
        )

    model_lang = align_model_metadata["language"]
    language_name = align_model_metadata["language_name"]
    max_duration = audio.shape[1] / SAMPLE_RATE

    aligned_segments: List[SingleAlignedSegment] = []

    for group in _group_segments_for_qwen(transcript):
        group_start = group[0]["start"]
        group_end = group[-1]["end"]

        if group_start >= max_duration:
            for segment in group:
                aligned_segments.append(
                    _build_fallback_segment(segment, return_char_alignments=return_char_alignments)
                )
            continue

        combined_text = _build_qwen_window_text(group, model_lang)
        if not combined_text:
            for segment in group:
                aligned_segments.append(
                    _build_fallback_segment(segment, return_char_alignments=return_char_alignments)
                )
            continue

        f1 = int(group_start * SAMPLE_RATE)
        f2 = int(min(group_end, max_duration) * SAMPLE_RATE)
        waveform_segment = audio[:, f1:f2].squeeze(0).cpu().numpy()

        tokenized_segments = [
            _tokenize_qwen_segment(model, segment["text"], language_name) for segment in group
        ]

        results = model.align(
            audio=(waveform_segment, SAMPLE_RATE),
            text=combined_text,
            language=language_name,
        )
        if len(results) != 1:
            raise ValueError(
                f"Expected one Qwen3 alignment result for a single window, got {len(results)}"
            )

        aligned_items = list(results[0])
        expected_item_count = sum(len(tokens) for tokens in tokenized_segments)
        if len(aligned_items) != expected_item_count:
            logger.warning(
                "Qwen3 alignment token count mismatch for transcript window %.3f-%.3f; "
                "expected %d items, got %d. Falling back to original timestamps for that window.",
                group_start,
                group_end,
                expected_item_count,
                len(aligned_items),
            )
            for segment in group:
                aligned_segments.append(
                    _build_fallback_segment(segment, return_char_alignments=return_char_alignments)
                )
            continue

        item_offset = 0
        for segment, token_list in zip(group, tokenized_segments):
            word_items = []
            for item in aligned_items[item_offset:item_offset + len(token_list)]:
                word_items.append(
                    {
                        "word": item.text,
                        "start": round(item.start_time + group_start, 3),
                        "end": round(item.end_time + group_start, 3),
                        "score": 1.0,
                    }
                )
            item_offset += len(token_list)

            aligned_segment: SingleAlignedSegment = {
                "start": word_items[0]["start"] if word_items else segment["start"],
                "end": word_items[-1]["end"] if word_items else segment["end"],
                "text": segment["text"],
                "words": word_items,
                "chars": [] if return_char_alignments else None,
            }
            if "avg_logprob" in segment:
                aligned_segment["avg_logprob"] = segment["avg_logprob"]
            aligned_segments.append(aligned_segment)

    word_segments: List[SingleWordSegment] = []
    for segment in aligned_segments:
        word_segments += segment["words"]

    return {"segments": aligned_segments, "word_segments": word_segments}


def _load_nemo_align_model(
    model_name: str,
    resolved_device: torch.device,
):
    try:
        from omegaconf import OmegaConf
        from nemo.collections.asr.models.ctc_models import EncDecCTCModel
        from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel
        from nemo.collections.asr.parts.utils.transcribe_utils import setup_model
    except ImportError as exc:
        raise ImportError(
            "NeMo forced alignment requires the optional NeMo ASR dependencies. "
            "Install them with `pip install 'nemo-toolkit[core,asr-only,common-only]>=2.5.0,<3.0.0'`, "
            "`uv pip install 'nemo-toolkit[core,asr-only,common-only]>=2.5.0,<3.0.0'`, "
            "or sync the project with `uv sync --extra nemo`."
        ) from exc

    cfg = OmegaConf.create(
        {
            "pretrained_name": None if os.path.exists(model_name) else model_name,
            "model_path": model_name if os.path.exists(model_name) else None,
        }
    )
    align_model, _ = setup_model(cfg, resolved_device)
    align_model.eval()

    if isinstance(align_model, EncDecHybridRNNTCTCModel):
        align_model.change_decoding_strategy(decoder_type="ctc")

    if hasattr(align_model, "change_attention_model"):
        try:
            align_model.change_attention_model(
                self_attention_model="rel_pos_local_attn",
                att_context_size=[64, 64],
            )
        except Exception:
            logger.debug("Failed to switch NeMo aligner to local attention; continuing with model defaults.")

    if not isinstance(align_model, (EncDecCTCModel, EncDecHybridRNNTCTCModel)):
        raise NotImplementedError(
            "NeMo forced alignment currently supports only EncDecCTCModel and "
            "EncDecHybridRNNTCTCModel checkpoints."
        )

    return align_model


def _get_nemo_alignment_utils():
    try:
        from nemo.collections.asr.parts.utils.aligner_utils import (
            add_t_start_end_to_utt_obj,
            get_batch_variables,
            viterbi_decoding,
        )
    except ImportError as exc:
        raise ImportError(
            "NeMo forced alignment utilities are unavailable. "
            "Install the optional NeMo extra with `uv sync --extra nemo`."
        ) from exc

    return add_t_start_end_to_utt_obj, get_batch_variables, viterbi_decoding


def _convert_nemo_utt_obj_to_segment(
    source_segment: SingleSegment,
    utt_obj,
    offset_sec: float,
    return_char_alignments: bool = False,
) -> SingleAlignedSegment:
    words: List[SingleWordSegment] = []

    for segment_or_token in getattr(utt_obj, "segments_and_tokens", []):
        if not hasattr(segment_or_token, "words_and_tokens"):
            continue
        for word_or_token in segment_or_token.words_and_tokens:
            if not hasattr(word_or_token, "tokens"):
                continue
            if word_or_token.t_start is None or word_or_token.t_end is None:
                continue
            if word_or_token.t_start < 0 or word_or_token.t_end < 0:
                continue
            words.append(
                {
                    "word": word_or_token.text,
                    "start": round(word_or_token.t_start + offset_sec, 3),
                    "end": round(word_or_token.t_end + offset_sec, 3),
                    "score": 1.0,
                }
            )

    aligned_segment: SingleAlignedSegment = {
        "start": words[0]["start"] if words else source_segment["start"],
        "end": words[-1]["end"] if words else source_segment["end"],
        "text": source_segment["text"],
        "words": words,
        "chars": [] if return_char_alignments else None,
    }
    if "avg_logprob" in source_segment:
        aligned_segment["avg_logprob"] = source_segment["avg_logprob"]
    return aligned_segment


def _align_with_nemo(
    transcript: Iterable[SingleSegment],
    model,
    align_model_metadata: dict,
    audio: torch.Tensor,
    resolved_device: torch.device,
    return_char_alignments: bool = False,
) -> AlignedTranscriptionResult:
    if return_char_alignments:
        logger.warning(
            "NeMo forced alignment does not expose WhisperX-style character alignments; "
            "returning empty char lists."
        )

    add_t_start_end_to_utt_obj, get_batch_variables, viterbi_decoding = _get_nemo_alignment_utils()

    transcript = list(transcript)
    max_duration = audio.shape[1] / SAMPLE_RATE
    aligned_segments: List[Optional[SingleAlignedSegment]] = [None] * len(transcript)
    jobs = []

    with tempfile.TemporaryDirectory(prefix="coherex-nemo-align-") as tmpdir:
        for idx, segment in enumerate(transcript):
            t1 = segment["start"]
            t2 = segment["end"]

            if not segment["text"].strip():
                aligned_segments[idx] = _build_fallback_segment(
                    segment, return_char_alignments=return_char_alignments
                )
                continue

            if t1 >= max_duration or t2 <= t1:
                aligned_segments[idx] = _build_fallback_segment(
                    segment, return_char_alignments=return_char_alignments
                )
                continue

            f1 = int(t1 * SAMPLE_RATE)
            f2 = int(min(t2, max_duration) * SAMPLE_RATE)
            waveform_segment = audio[:, f1:f2].squeeze(0).cpu().numpy()
            if waveform_segment.size == 0:
                aligned_segments[idx] = _build_fallback_segment(
                    segment, return_char_alignments=return_char_alignments
                )
                continue

            audio_path = os.path.join(tmpdir, f"segment_{idx}.wav")
            sf.write(audio_path, waveform_segment, SAMPLE_RATE)
            jobs.append(
                {
                    "index": idx,
                    "segment": segment,
                    "audio_path": audio_path,
                    "start": t1,
                }
            )

        output_timestep_duration = None
        batch_size = max(1, int(align_model_metadata.get("batch_size", 1)))

        for batch_start in range(0, len(jobs), batch_size):
            batch_jobs = jobs[batch_start:batch_start + batch_size]
            (
                log_probs_batch,
                y_batch,
                T_batch,
                U_batch,
                utt_obj_batch,
                output_timestep_duration,
            ) = get_batch_variables(
                audio=[job["audio_path"] for job in batch_jobs],
                model=model,
                segment_separators=None,
                word_separator=" ",
                align_using_pred_text=False,
                audio_filepath_parts_in_utt_id=1,
                gt_text_batch=[job["segment"]["text"] for job in batch_jobs],
                output_timestep_duration=output_timestep_duration,
            )

            alignments_batch = viterbi_decoding(
                log_probs_batch,
                y_batch,
                T_batch,
                U_batch,
                viterbi_device=resolved_device,
            )

            for job, utt_obj, alignment_utt in zip(batch_jobs, utt_obj_batch, alignments_batch):
                if not getattr(utt_obj, "segments_and_tokens", None):
                    aligned_segments[job["index"]] = _build_fallback_segment(
                        job["segment"],
                        return_char_alignments=return_char_alignments,
                    )
                    continue

                utt_obj = add_t_start_end_to_utt_obj(
                    utt_obj,
                    alignment_utt,
                    output_timestep_duration,
                )
                aligned_segment = _convert_nemo_utt_obj_to_segment(
                    source_segment=job["segment"],
                    utt_obj=utt_obj,
                    offset_sec=job["start"],
                    return_char_alignments=return_char_alignments,
                )
                aligned_segments[job["index"]] = aligned_segment

    word_segments: List[SingleWordSegment] = []
    final_segments = [segment for segment in aligned_segments if segment is not None]
    for segment in final_segments:
        word_segments += segment["words"]

    return {"segments": final_segments, "word_segments": word_segments}


def load_align_model(
    language_code: str,
    device: Union[str, torch.device],
    model_name: Optional[str] = None,
    backend: str = "wav2vec2",
    model_dir=None,
    model_cache_only: bool = False,
    device_index: int = 0,
):
    resolved_device = _resolve_device(device, device_index=device_index)
    backend = _infer_align_backend(model_name=model_name, backend=backend)

    if backend == "qwen3":
        language_name = _qwen_language_name(language_code)
        if model_name is None:
            model_name = QWEN3_FORCED_ALIGNER_MODEL

        try:
            from qwen_asr import Qwen3ForcedAligner
        except ImportError as exc:
            raise ImportError(
                "Qwen3 forced alignment requires the optional `qwen-asr` package. "
                "Install it with `pip install qwen-asr`, `uv pip install qwen-asr`, "
                "or sync the project with `uv sync --extra qwen`."
            ) from exc

        align_model = Qwen3ForcedAligner.from_pretrained(
            model_name,
            cache_dir=model_dir,
            local_files_only=model_cache_only,
            device_map=str(resolved_device),
        )
        align_metadata = {
            "language": language_code,
            "language_name": language_name,
            "dictionary": None,
            "type": "qwen3",
        }
        return align_model, align_metadata

    if backend == "nemo":
        if model_name is None:
            if language_code not in NEMO_DEFAULT_ALIGN_MODELS:
                raise ValueError(
                    "No default NeMo align-model for language: "
                    f"{language_code}. Pass a NeMo CTC or hybrid CTC checkpoint via --align_model."
                )
            model_name = NEMO_DEFAULT_ALIGN_MODELS[language_code]

        if model_cache_only and not os.path.exists(model_name):
            raise ValueError(
                "NeMo forced alignment does not support cache-only loading by model name. "
                "Pass a local `.nemo` model path or disable --model_cache_only."
            )

        align_model = _load_nemo_align_model(
            model_name=model_name,
            resolved_device=resolved_device,
        )
        align_metadata = {
            "language": language_code,
            "dictionary": None,
            "type": "nemo",
            "batch_size": 1,
            "model_name": model_name,
        }
        return align_model, align_metadata

    if backend != "wav2vec2":
        raise ValueError(f"Unsupported alignment backend: {backend}")

    if model_name is None:
        # use default model
        if language_code in DEFAULT_ALIGN_MODELS_TORCH:
            model_name = DEFAULT_ALIGN_MODELS_TORCH[language_code]
        elif language_code in DEFAULT_ALIGN_MODELS_HF:
            model_name = DEFAULT_ALIGN_MODELS_HF[language_code]
        else:
            logger.error(f"No default alignment model for language: {language_code}. "
                         f"Please find a wav2vec2.0 model finetuned on this language at https://huggingface.co/models, "
                         f"then pass the model name via --align_model [MODEL_NAME]")
            raise ValueError(f"No default align-model for language: {language_code}")

    if model_name in torchaudio.pipelines.__all__:
        pipeline_type = "torchaudio"
        bundle = torchaudio.pipelines.__dict__[model_name]
        align_model = bundle.get_model(dl_kwargs={"model_dir": model_dir}).to(resolved_device)
        labels = bundle.get_labels()
        align_dictionary = {c.lower(): i for i, c in enumerate(labels)}
    else:
        try:
            processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=model_dir, local_files_only=model_cache_only)
            align_model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=model_dir, local_files_only=model_cache_only)
        except Exception as e:
            print(e)
            print(f"Error loading model from huggingface, check https://huggingface.co/models for finetuned wav2vec2.0 models")
            raise ValueError(f'The chosen align_model "{model_name}" could not be found in huggingface (https://huggingface.co/models) or torchaudio (https://pytorch.org/audio/stable/pipelines.html#id14)')
        pipeline_type = "huggingface"
        align_model = align_model.to(resolved_device)
        labels = processor.tokenizer.get_vocab()
        align_dictionary = {char.lower(): code for char,code in processor.tokenizer.get_vocab().items()}

    align_metadata = {"language": language_code, "dictionary": align_dictionary, "type": pipeline_type}

    return align_model, align_metadata


def align(
    transcript: Iterable[SingleSegment],
    model: torch.nn.Module,
    align_model_metadata: dict,
    audio: Union[str, np.ndarray, torch.Tensor],
    device: Union[str, torch.device],
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    print_progress: bool = False,
    combined_progress: bool = False,
    progress_callback: ProgressCallback = None,
    device_index: int = 0,
) -> AlignedTranscriptionResult:
    """
    Align phoneme recognition predictions to known transcription.
    """

    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)
    resolved_device = _resolve_device(device, device_index=device_index)

    if align_model_metadata["type"] == "qwen3":
        return _align_with_qwen3(
            transcript=transcript,
            model=model,
            align_model_metadata=align_model_metadata,
            audio=audio,
            return_char_alignments=return_char_alignments,
        )
    if align_model_metadata["type"] == "nemo":
        return _align_with_nemo(
            transcript=transcript,
            model=model,
            align_model_metadata=align_model_metadata,
            audio=audio,
            resolved_device=resolved_device,
            return_char_alignments=return_char_alignments,
        )

    MAX_DURATION = audio.shape[1] / SAMPLE_RATE

    model_dictionary = align_model_metadata["dictionary"]
    model_lang = align_model_metadata["language"]
    model_type = align_model_metadata["type"]

    # 1. Preprocess to keep only characters in dictionary
    total_segments = len(transcript)
    # Store temporary processing values
    segment_data: dict[int, SegmentData] = {}
    for sdx, segment in enumerate(transcript):
        # strip spaces at beginning / end, but keep track of the amount.
        if print_progress:
            base_progress = ((sdx + 1) / total_segments) * 100
            percent_complete = (50 + base_progress / 2) if combined_progress else base_progress
            print(f"Progress: {percent_complete:.2f}%...")

        num_leading = len(segment["text"]) - len(segment["text"].lstrip())
        num_trailing = len(segment["text"]) - len(segment["text"].rstrip())
        text = segment["text"]

        # split into words
        if model_lang not in LANGUAGES_WITHOUT_SPACES:
            per_word = text.split(" ")
        else:
            per_word = text

        clean_char, clean_cdx = [], []
        for cdx, char in enumerate(text):
            char_ = char.lower()
            # wav2vec2 models use "|" character to represent spaces
            if model_lang not in LANGUAGES_WITHOUT_SPACES:
                char_ = char_.replace(" ", "|")

            # ignore whitespace at beginning and end of transcript
            if cdx < num_leading:
                pass
            elif cdx > len(text) - num_trailing - 1:
                pass
            elif char_ in model_dictionary.keys():
                clean_char.append(char_)
                clean_cdx.append(cdx)
            elif char_ not in (" ", "|"):
                # unknown char (digit, symbol, foreign script) — use wildcard
                clean_char.append(char_)
                clean_cdx.append(cdx)

        clean_wdx = list(range(len(per_word)))

        # Use language-specific Punkt model if available otherwise we fallback to English.
        punkt_lang = PUNKT_LANGUAGES.get(model_lang, 'english')
        try:
            sentence_splitter = nltk_load(f'tokenizers/punkt_tab/{punkt_lang}.pickle')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
            sentence_splitter = nltk_load(f'tokenizers/punkt_tab/{punkt_lang}.pickle')
        sentence_spans = list(sentence_splitter.span_tokenize(text))

        segment_data[sdx] = {
            "clean_char": clean_char,
            "clean_cdx": clean_cdx,
            "clean_wdx": clean_wdx,
            "sentence_spans": sentence_spans
        }

    aligned_segments: List[SingleAlignedSegment] = []

    # 2. Get prediction matrix from alignment model & align
    for sdx, segment in enumerate(transcript):

        t1 = segment["start"]
        t2 = segment["end"]
        text = segment["text"]
        avg_logprob = segment.get("avg_logprob")

        aligned_seg: SingleAlignedSegment = {
            "start": t1,
            "end": t2,
            "text": text,
            "words": [],
            "chars": None,
        }

        if avg_logprob is not None:
            aligned_seg["avg_logprob"] = avg_logprob

        if return_char_alignments:
            aligned_seg["chars"] = []

        # check we can align
        if len(segment_data[sdx]["clean_char"]) == 0:
            logger.warning(f'Failed to align segment ("{segment["text"]}"): no characters in this segment found in model dictionary, resorting to original')
            aligned_segments.append(aligned_seg)
            continue

        if t1 >= MAX_DURATION:
            logger.warning(f'Failed to align segment ("{segment["text"]}"): original start time longer than audio duration, skipping')
            aligned_segments.append(aligned_seg)
            continue

        text_clean = "".join(segment_data[sdx]["clean_char"])

        f1 = int(t1 * SAMPLE_RATE)
        f2 = int(t2 * SAMPLE_RATE)

        # TODO: Probably can get some speedup gain with batched inference here
        waveform_segment = audio[:, f1:f2]
        # Handle the minimum input length for wav2vec2 models
        if waveform_segment.shape[-1] < 400:
            lengths = torch.as_tensor([waveform_segment.shape[-1]]).to(resolved_device)
            waveform_segment = torch.nn.functional.pad(
                waveform_segment, (0, 400 - waveform_segment.shape[-1])
            )
        else:
            lengths = None

        with torch.inference_mode():
            if model_type == "torchaudio":
                emissions, _ = model(waveform_segment.to(resolved_device), lengths=lengths)
            elif model_type == "huggingface":
                emissions = model(waveform_segment.to(resolved_device)).logits
            else:
                raise NotImplementedError(f"Align model of type {model_type} not supported.")
            emissions = torch.log_softmax(emissions, dim=-1)

        emission = emissions[0].cpu().detach()

        blank_id = 0
        for char, code in model_dictionary.items():
            if char == '[pad]' or char == '<pad>':
                blank_id = code

        # Build tokens, mapping unknown chars to a wildcard column
        has_wildcard = any(c not in model_dictionary for c in text_clean)
        if has_wildcard:
            # Extend emission with a wildcard column: max non-blank score per frame
            non_blank_mask = torch.ones(emission.size(1), dtype=torch.bool)
            non_blank_mask[blank_id] = False
            wildcard_col = emission[:, non_blank_mask].max(dim=1).values
            emission = torch.cat([emission, wildcard_col.unsqueeze(1)], dim=1)
            wildcard_id = emission.size(1) - 1
            tokens = [model_dictionary.get(c, wildcard_id) for c in text_clean]
        else:
            tokens = [model_dictionary[c] for c in text_clean]

        trellis = get_trellis(emission, tokens, blank_id)
        path = backtrack(trellis, emission, tokens, blank_id)

        if path is None:
            logger.warning(f'Failed to align segment ("{segment["text"]}"): backtrack failed, resorting to original')
            aligned_segments.append(aligned_seg)
            continue

        char_segments = merge_repeats(path, text_clean)

        duration = t2 - t1
        ratio = duration * waveform_segment.size(0) / (trellis.size(0) - 1)

        # assign timestamps to aligned characters
        char_segments_arr = []
        word_idx = 0
        for cdx, char in enumerate(text):
            start, end, score = None, None, None
            if cdx in segment_data[sdx]["clean_cdx"]:
                char_seg = char_segments[segment_data[sdx]["clean_cdx"].index(cdx)]
                start = round(char_seg.start * ratio + t1, 3)
                end = round(char_seg.end * ratio + t1, 3)
                score = round(char_seg.score, 3)

            char_segments_arr.append(
                {
                    "char": char,
                    "start": start,
                    "end": end,
                    "score": score,
                    "word-idx": word_idx,
                }
            )

            # increment word_idx, nltk word tokenization would probably be more robust here, but us space for now...
            if model_lang in LANGUAGES_WITHOUT_SPACES:
                word_idx += 1
            elif cdx == len(text) - 1 or text[cdx+1] == " ":
                word_idx += 1

        char_segments_arr = pd.DataFrame(char_segments_arr)

        aligned_subsegments = []
        # assign sentence_idx to each character index
        char_segments_arr["sentence-idx"] = None
        for sdx2, (sstart, send) in enumerate(segment_data[sdx]["sentence_spans"]):
            curr_chars = char_segments_arr.loc[(char_segments_arr.index >= sstart) & (char_segments_arr.index <= send)]
            char_segments_arr.loc[(char_segments_arr.index >= sstart) & (char_segments_arr.index <= send), "sentence-idx"] = sdx2

            sentence_text = text[sstart:send]
            sentence_start = curr_chars["start"].min()
            end_chars = curr_chars[curr_chars["char"] != ' ']
            sentence_end = end_chars["end"].max()
            sentence_words = []

            for word_idx in curr_chars["word-idx"].unique():
                word_chars = curr_chars.loc[curr_chars["word-idx"] == word_idx]
                word_text = "".join(word_chars["char"].tolist()).strip()
                if len(word_text) == 0:
                    continue

                # dont use space character for alignment
                word_chars = word_chars[word_chars["char"] != " "]

                word_start = word_chars["start"].min()
                word_end = word_chars["end"].max()
                word_score = round(word_chars["score"].mean(), 3)

                # -1 indicates unalignable
                word_segment = {"word": word_text}

                if not np.isnan(word_start):
                    word_segment["start"] = word_start
                if not np.isnan(word_end):
                    word_segment["end"] = word_end
                if not np.isnan(word_score):
                    word_segment["score"] = word_score

                sentence_words.append(word_segment)

            # Interpolate timestamps for words with no alignable characters
            if sentence_words:
                _starts = pd.Series([w.get("start", np.nan) for w in sentence_words])
                _ends = pd.Series([w.get("end", np.nan) for w in sentence_words])
                if _starts.isna().any() and _starts.notna().any():
                    _starts = interpolate_nans(_starts, method=interpolate_method)
                    _ends = interpolate_nans(_ends, method=interpolate_method)
                    for i, w in enumerate(sentence_words):
                        if "start" not in w and pd.notna(_starts.iloc[i]):
                            w["start"] = _starts.iloc[i]
                        if "end" not in w and pd.notna(_ends.iloc[i]):
                            w["end"] = _ends.iloc[i]

            subsegment = {
                "text": sentence_text,
                "start": sentence_start,
                "end": sentence_end,
                "words": sentence_words,
            }
            if avg_logprob is not None:
                subsegment["avg_logprob"] = avg_logprob
            aligned_subsegments.append(subsegment)

            if return_char_alignments:
                curr_chars = curr_chars[["char", "start", "end", "score"]]
                curr_chars.fillna(-1, inplace=True)
                curr_chars = curr_chars.to_dict("records")
                curr_chars = [{key: val for key, val in char.items() if val != -1} for char in curr_chars]
                aligned_subsegments[-1]["chars"] = curr_chars

        aligned_subsegments = pd.DataFrame(aligned_subsegments)
        aligned_subsegments["start"] = interpolate_nans(aligned_subsegments["start"], method=interpolate_method)
        aligned_subsegments["end"] = interpolate_nans(aligned_subsegments["end"], method=interpolate_method)
        # concatenate sentences with same timestamps
        agg_dict = {"text": " ".join, "words": "sum"}
        if model_lang in LANGUAGES_WITHOUT_SPACES:
            agg_dict["text"] = "".join
        if return_char_alignments:
            agg_dict["chars"] = "sum"
        if avg_logprob is not None:
            agg_dict["avg_logprob"] = "first"
        aligned_subsegments= aligned_subsegments.groupby(["start", "end"], as_index=False).agg(agg_dict)
        aligned_subsegments = aligned_subsegments.to_dict('records')
        if progress_callback is not None:
            progress_callback(((sdx + 1) / total_segments) * 100)

        aligned_segments += aligned_subsegments

    # create word_segments list
    word_segments: List[SingleWordSegment] = []
    for segment in aligned_segments:
        word_segments += segment["words"]

    return {"segments": aligned_segments, "word_segments": word_segments}

"""
source: https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html
"""


def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra dimensions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, blank_id], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else blank_id].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        # failed
        return None

    return path[::-1]


# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start

def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments

def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words
