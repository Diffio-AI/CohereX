import argparse
import gc
import os

import torch

from coherex.asr import load_model
from coherex.audio import load_audio
from coherex.log_utils import get_logger
from coherex.schema import TranscriptionResult
from coherex.utils import get_writer

logger = get_logger(__name__)


def transcribe_task(args: dict, parser: argparse.ArgumentParser):
    model_name: str = args.pop("model")
    batch_size: int = args.pop("batch_size")
    model_dir: str = args.pop("model_dir")
    model_cache_only: bool = args.pop("model_cache_only")
    output_dir: str = args.pop("output_dir")
    output_format: str = args.pop("output_format")
    device: str = args.pop("device")
    device_index: int = args.pop("device_index")
    compute_type: str = args.pop("compute_type")
    verbose: bool = args.pop("verbose")

    os.makedirs(output_dir, exist_ok=True)

    align_model_name: str = args.pop("align_model")
    interpolate_method: str = args.pop("interpolate_method")
    no_align: bool = args.pop("no_align")
    return_char_alignments: bool = args.pop("return_char_alignments")

    language: str = args.pop("language")
    hf_token: str = args.pop("hf_token")
    vad_method: str = args.pop("vad_method")
    vad_model_dir: str = args.pop("vad_model_dir")
    vad_onset: float = args.pop("vad_onset")
    vad_offset: float = args.pop("vad_offset")
    chunk_size: float = args.pop("chunk_size")
    suppress_numerals: bool = args.pop("suppress_numerals")
    no_punctuation: bool = args.pop("no_punctuation")
    print_progress: bool = args.pop("print_progress")
    max_new_tokens: int = args.pop("max_new_tokens")

    threads = args.pop("threads")
    if threads > 0:
        torch.set_num_threads(threads)

    writer = get_writer(output_format, output_dir)
    writer_args = {
        "highlight_words": args.pop("highlight_words"),
        "max_line_count": args.pop("max_line_count"),
        "max_line_width": args.pop("max_line_width"),
    }

    model = load_model(
        model_name,
        device=device,
        device_index=device_index,
        compute_type=compute_type,
        language=language,
        vad_method=vad_method,
        vad_options={"vad_onset": vad_onset, "vad_offset": vad_offset, "model_dir": vad_model_dir},
        asr_options={
            "punctuation": not no_punctuation,
            "suppress_numerals": suppress_numerals,
            "max_new_tokens": max_new_tokens,
        },
        download_root=model_dir,
        local_files_only=model_cache_only,
        threads=threads if threads > 0 else 4,
        use_auth_token=hf_token,
    )

    results = []
    for audio_path in args.pop("audio"):
        audio = load_audio(audio_path)
        logger.info("Performing transcription...")
        result: TranscriptionResult = model.transcribe(
            audio,
            batch_size=batch_size,
            chunk_size=chunk_size,
            print_progress=print_progress,
            verbose=verbose,
        )
        results.append((result, audio_path, audio))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not no_align:
        from coherex.alignment import align, load_align_model

        align_model, align_metadata = load_align_model(
            language,
            device,
            model_name=align_model_name,
            model_dir=model_dir,
            model_cache_only=model_cache_only,
        )
        aligned_results = []
        for result, audio_path, audio in results:
            if align_model is not None and len(result["segments"]) > 0:
                logger.info("Performing alignment...")
                aligned = align(
                    result["segments"],
                    align_model,
                    align_metadata,
                    audio,
                    device,
                    interpolate_method=interpolate_method,
                    return_char_alignments=return_char_alignments,
                    print_progress=print_progress,
                )
                aligned_results.append((aligned, audio_path))
            else:
                aligned_results.append((result, audio_path))
        results_to_write = aligned_results
    else:
        results_to_write = [(result, audio_path) for result, audio_path, _ in results]

    for result, audio_path in results_to_write:
        result["language"] = language
        writer(result, audio_path, writer_args)
