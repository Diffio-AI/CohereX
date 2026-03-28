import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
sys.modules.setdefault("tensorflow", None)
sys.modules.setdefault("tensorflow_text", None)
sys.modules.setdefault("keras", None)

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers.generation.logits_process import LogitsProcessorList, SuppressTokensLogitsProcessor

from coherex.audio import SAMPLE_RATE, load_audio
from coherex.configuration_cohere_asr import CohereAsrConfig, normalize_language_code
from coherex.log_utils import get_logger
from coherex.modeling_cohere_asr import (
    CohereAsrForConditionalGeneration,
    _batched_indices,
    get_chunk_separator,
    split_audio_chunks_energy,
)
from coherex.processing_cohere_asr import CohereAsrFeatureExtractor, CohereAsrProcessor
from coherex.schema import ProgressCallback, SingleSegment, TranscriptionResult
from coherex.tokenization_cohere_asr import CohereAsrTokenizer
from coherex.vads.binarize import Binarize

logger = get_logger(__name__)
COHERE_MODEL_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer_config.json",
)


@dataclass
class SpeechSpan:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Chunk:
    start: float
    end: float
    spans: List[SpeechSpan]
    audio: np.ndarray


class NullVad:
    @staticmethod
    def preprocess_audio(audio):
        return audio

    def __call__(self, audio_input, **kwargs):
        return audio_input


def find_numeral_symbol_tokens(tokenizer: CohereAsrTokenizer) -> List[int]:
    numeral_symbol_tokens = []
    special_ids = set(tokenizer.all_special_ids)
    for token_id in sorted(set(tokenizer.get_vocab().values())):
        if token_id in special_ids:
            continue
        token = tokenizer.decode([token_id], skip_special_tokens=False)
        token = token.removeprefix(" ")
        if any(char in "0123456789%$£" for char in token):
            numeral_symbol_tokens.append(token_id)
    return numeral_symbol_tokens


def _resolve_device(device: str, device_index: int) -> torch.device:
    if device == "cuda":
        return torch.device(f"cuda:{device_index}")
    return torch.device(device)


def _resolve_dtype(compute_type: str, device: torch.device) -> torch.dtype:
    if compute_type == "default":
        return torch.float16 if device.type == "cuda" else torch.float32
    if compute_type == "float16":
        return torch.float16
    if compute_type == "bfloat16":
        return torch.bfloat16
    if compute_type == "float32":
        return torch.float32
    raise ValueError(f"Unsupported compute_type: {compute_type}")


def _has_required_files(path: Path, required_files: tuple[str, ...]) -> bool:
    return path.is_dir() and all((path / filename).exists() for filename in required_files)


def _resolve_model_source(
    model_name: str,
    download_root: Optional[str],
    local_files_only: bool,
    token,
) -> str:
    candidate_paths = []
    if model_name is not None:
        candidate_paths.append(Path(model_name).expanduser())
    if download_root is not None:
        candidate_paths.append(Path(download_root).expanduser())

    for path in candidate_paths:
        if _has_required_files(path, COHERE_MODEL_REQUIRED_FILES):
            logger.info("Using local Cohere model files from %s", path)
            return str(path)

    cache_dir = None
    if download_root is not None:
        cache_dir = str(Path(download_root).expanduser())

    if local_files_only:
        logger.info("Loading cached Cohere model files for %s", model_name)
    else:
        logger.info("Resolving Cohere model files for %s", model_name)

    try:
        snapshot_dir = snapshot_download(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )
    except Exception as exc:
        action = "load cached" if local_files_only else "download"
        raise RuntimeError(
            f"Unable to {action} Cohere model files for {model_name!r}. "
            "Pass a local snapshot path with --model, or set --model_dir to a cache directory "
            "or a resolved snapshot directory."
        ) from exc

    logger.info("Resolved Cohere model files at %s", snapshot_dir)
    return snapshot_dir


def _segments_from_vad(scores, onset: float, offset: Optional[float]) -> List[SpeechSpan]:
    speech = Binarize(max_duration=float("inf"), onset=onset, offset=offset)(scores)
    return [SpeechSpan(turn.start, turn.end) for turn in speech.get_timeline()]


def _segments_from_firered(result: dict) -> List[SpeechSpan]:
    return [SpeechSpan(start, end) for start, end in result.get("timestamps", [])]


def _split_oversized_span(
    audio: np.ndarray,
    span: SpeechSpan,
    sample_rate: int,
    chunk_size: float,
    boundary_context: float,
    min_energy_window_samples: int,
) -> List[SpeechSpan]:
    if span.duration <= chunk_size:
        return [span]

    start_sample = int(round(span.start * sample_rate))
    end_sample = int(round(span.end * sample_rate))
    waveform = audio[start_sample:end_sample]
    chunks = split_audio_chunks_energy(
        waveform=waveform,
        sample_rate=sample_rate,
        max_audio_clip_s=chunk_size,
        overlap_chunk_second=boundary_context,
        min_energy_window_samples=min_energy_window_samples,
    )

    spans = []
    offset_samples = 0
    for chunk in chunks:
        chunk_start = span.start + (offset_samples / sample_rate)
        offset_samples += len(chunk)
        chunk_end = span.start + (offset_samples / sample_rate)
        spans.append(SpeechSpan(chunk_start, min(chunk_end, span.end)))
    return spans


def _chunk_spans(
    audio: np.ndarray,
    spans: List[SpeechSpan],
    sample_rate: int,
    chunk_size: float,
    boundary_context: float,
    min_energy_window_samples: int,
) -> List[Chunk]:
    atomic_spans: List[SpeechSpan] = []
    for span in spans:
        atomic_spans.extend(
            _split_oversized_span(
                audio=audio,
                span=span,
                sample_rate=sample_rate,
                chunk_size=chunk_size,
                boundary_context=boundary_context,
                min_energy_window_samples=min_energy_window_samples,
            )
        )

    chunks: List[Chunk] = []
    current_spans: List[SpeechSpan] = []
    current_duration = 0.0
    for span in atomic_spans:
        if current_spans and current_duration + span.duration > chunk_size:
            chunks.append(_build_chunk(audio, current_spans, sample_rate))
            current_spans = []
            current_duration = 0.0
        current_spans.append(span)
        current_duration += span.duration

    if current_spans:
        chunks.append(_build_chunk(audio, current_spans, sample_rate))

    return chunks


def _build_chunk(audio: np.ndarray, spans: List[SpeechSpan], sample_rate: int) -> Chunk:
    parts = []
    for span in spans:
        start_sample = int(round(span.start * sample_rate))
        end_sample = int(round(span.end * sample_rate))
        parts.append(audio[start_sample:end_sample])
    trimmed_audio = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
    return Chunk(start=spans[0].start, end=spans[-1].end, spans=spans, audio=trimmed_audio)


def _join_chunk_texts(texts: List[str], language: str) -> str:
    parts = [piece.strip() for piece in texts if piece and piece.strip()]
    if not parts:
        return ""
    return get_chunk_separator(language).join(parts)


class CohereTranscriptionPipeline:
    def __init__(
        self,
        model: CohereAsrForConditionalGeneration,
        processor: CohereAsrProcessor,
        vad,
        vad_params: dict,
        language: str,
        punctuation: bool,
        suppress_numerals: bool,
        max_new_tokens: int,
        batch_size: int,
    ):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.vad_model = vad
        self.vad_params = vad_params
        self.language = language
        self.punctuation = punctuation
        self.suppress_numerals = suppress_numerals
        self.max_new_tokens = max_new_tokens
        self._batch_size = batch_size
        self._suppressed_tokens = (
            find_numeral_symbol_tokens(self.tokenizer) if suppress_numerals else []
        )

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        batch_size: Optional[int] = None,
        chunk_size: Optional[float] = None,
        print_progress: bool = False,
        verbose: bool = False,
        progress_callback: ProgressCallback = None,
    ) -> TranscriptionResult:
        if isinstance(audio, str):
            audio = load_audio(audio)

        waveform = self.vad_model.preprocess_audio(audio)
        if isinstance(self.vad_model, NullVad):
            speech_spans = [SpeechSpan(0.0, len(audio) / SAMPLE_RATE)]
        else:
            vad_output = self.vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
            if isinstance(vad_output, dict) and "timestamps" in vad_output:
                speech_spans = _segments_from_firered(vad_output)
            else:
                speech_spans = _segments_from_vad(
                    vad_output,
                    onset=self.vad_params["vad_onset"],
                    offset=self.vad_params["vad_offset"],
                )
        if not speech_spans:
            logger.warning("No active speech found in audio")
            return {"segments": [], "language": self.language}

        chunk_limit = float(chunk_size or self.model.config.max_audio_clip_s)
        boundary_context = float(self.model.config.overlap_chunk_second)
        min_energy_window_samples = int(getattr(self.model.config, "min_energy_window_samples", 1600))
        chunks = _chunk_spans(
            audio=audio,
            spans=speech_spans,
            sample_rate=SAMPLE_RATE,
            chunk_size=chunk_limit,
            boundary_context=boundary_context,
            min_energy_window_samples=min_energy_window_samples,
        )

        texts = self._transcribe_chunks(
            chunks=chunks,
            batch_size=batch_size or self._batch_size,
            print_progress=print_progress,
            progress_callback=progress_callback,
        )

        segments: List[SingleSegment] = []
        for idx, (chunk, text) in enumerate(zip(chunks, texts)):
            text = text.strip()
            if verbose:
                print(f"Transcript: [{round(chunk.start, 3)} --> {round(chunk.end, 3)}] {text}")
            segments.append(
                {
                    "start": round(chunk.start, 3),
                    "end": round(chunk.end, 3),
                    "text": text,
                }
            )

        if verbose:
            logger.info("Transcribed %s segments", len(segments))

        return {"segments": segments, "language": self.language}

    def _transcribe_chunks(
        self,
        chunks: List[Chunk],
        batch_size: int,
        print_progress: bool,
        progress_callback: ProgressCallback,
    ) -> List[str]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if not chunks:
            return []

        transcriptions = [""] * len(chunks)
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id
        ordered_indices = sorted(range(len(chunks)), key=lambda idx: chunks[idx].audio.shape[0], reverse=True)
        prompt_text = self.model.build_prompt(language=self.language, punctuation=self.punctuation)
        logits_processor = LogitsProcessorList()
        if self._suppressed_tokens:
            logits_processor.append(SuppressTokensLogitsProcessor(self._suppressed_tokens))

        total_chunks = len(ordered_indices)
        for batch_order_indices in _batched_indices(total_chunks, batch_size):
            batch_indices = [ordered_indices[idx] for idx in batch_order_indices]
            batch_waves = [chunks[i].audio for i in batch_indices]
            inputs = self.processor(
                audio=batch_waves,
                text=[prompt_text] * len(batch_waves),
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            inputs["decoder_input_ids"] = inputs.pop("input_ids")
            if pad_token_id is None:
                decoder_attention_mask = torch.ones_like(inputs["decoder_input_ids"], dtype=torch.long)
            else:
                decoder_attention_mask = inputs["decoder_input_ids"].ne(pad_token_id).long()
            inputs["decoder_attention_mask"] = decoder_attention_mask

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    decoder_start_token_id=int(inputs["decoder_input_ids"][0, 0].item()),
                    use_cache=True,
                    logits_processor=logits_processor,
                )

            prompt_lens = decoder_attention_mask.sum(dim=1).cpu().tolist()
            generated_ids = generated_ids.cpu().tolist()
            decoder_input_ids = inputs["decoder_input_ids"].cpu().tolist()

            trimmed_ids = []
            for row_idx, prompt_len in enumerate(prompt_lens):
                token_ids = generated_ids[row_idx]
                prompt_ids = decoder_input_ids[row_idx][:prompt_len]
                if prompt_len > 0 and token_ids[:prompt_len] == prompt_ids:
                    token_ids = token_ids[prompt_len:]
                if eos_token_id is not None and eos_token_id in token_ids:
                    token_ids = token_ids[: token_ids.index(eos_token_id)]
                trimmed_ids.append(token_ids)

            batch_texts = self.tokenizer.batch_decode(trimmed_ids, skip_special_tokens=True)
            for row_idx, text in enumerate(batch_texts):
                transcriptions[batch_indices[row_idx]] = text.strip()

            progress = (batch_order_indices[-1] + 1) / total_chunks * 100
            if print_progress:
                print(f"Progress: {progress:.2f}%...")
            if progress_callback is not None:
                progress_callback(progress)

        return transcriptions


def load_model(
    model_name: str,
    device: str,
    device_index: int = 0,
    compute_type: str = "default",
    language: Optional[str] = None,
    vad_method: str = "pyannote",
    vad_options: Optional[dict] = None,
    download_root: Optional[str] = None,
    local_files_only: bool = False,
    threads: int = 4,
    use_auth_token: Optional[Union[str, bool]] = None,
    asr_options: Optional[dict] = None,
) -> CohereTranscriptionPipeline:
    language = normalize_language_code(language)

    torch_device = _resolve_device(device, device_index)
    torch_dtype = _resolve_dtype(compute_type, torch_device)
    if threads > 0 and torch_device.type == "cpu":
        torch.set_num_threads(threads)

    resolved_model_source = _resolve_model_source(
        model_name=model_name,
        download_root=download_root,
        local_files_only=local_files_only,
        token=use_auth_token,
    )

    config = CohereAsrConfig.from_pretrained(
        resolved_model_source,
        local_files_only=True,
        token=use_auth_token,
    )
    model = CohereAsrForConditionalGeneration.from_pretrained(
        resolved_model_source,
        config=config,
        local_files_only=True,
        token=use_auth_token,
        dtype=torch_dtype,
    )
    model = model.to(torch_device)
    model.eval()

    feature_extractor = CohereAsrFeatureExtractor.from_pretrained(
        resolved_model_source,
        local_files_only=True,
        token=use_auth_token,
        device=str(torch_device),
    )
    tokenizer = CohereAsrTokenizer.from_pretrained(
        resolved_model_source,
        local_files_only=True,
        token=use_auth_token,
    )
    processor = CohereAsrProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    default_asr_options = {
        "punctuation": True,
        "suppress_numerals": False,
        "max_new_tokens": 448,
    }
    if asr_options is not None:
        default_asr_options.update(asr_options)

    default_vad_options = {
        "vad_onset": 0.500,
        "vad_offset": 0.363,
        "model_dir": None,
    }
    if vad_options is not None:
        default_vad_options.update(vad_options)

    if vad_method == "none":
        vad_model = NullVad()
    elif vad_method == "pyannote":
        from coherex.vads.pyannote import Pyannote

        vad_model = Pyannote(device=torch_device, token=use_auth_token, **default_vad_options)
    elif vad_method == "firered":
        from coherex.vads.firered import FireRed

        vad_model = FireRed(
            device=torch_device,
            token=use_auth_token,
            cache_dir=download_root,
            local_files_only=local_files_only,
            **default_vad_options,
        )
    else:
        raise ValueError(f"Unsupported vad_method: {vad_method}")

    return CohereTranscriptionPipeline(
        model=model,
        processor=processor,
        vad=vad_model,
        vad_params=default_vad_options,
        language=language,
        punctuation=default_asr_options["punctuation"],
        suppress_numerals=default_asr_options["suppress_numerals"],
        max_new_tokens=int(default_asr_options["max_new_tokens"]),
        batch_size=int(config.batch_size),
    )
