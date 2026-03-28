import argparse
import importlib.metadata
import platform
import sys

sys.modules.setdefault("tensorflow", None)
sys.modules.setdefault("tensorflow_text", None)
sys.modules.setdefault("keras", None)

import torch

from coherex.configuration_cohere_asr import DEFAULT_SUPPORTED_LANGUAGES
from coherex.log_utils import setup_logging
from coherex.utils import optional_int, str2bool


def _package_version() -> str:
    try:
        return importlib.metadata.version("coherex")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"


def cli():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", nargs="+", type=str, help="audio file(s) to transcribe")
    parser.add_argument("--model", default="CohereLabs/cohere-transcribe-03-2026", help="name of the Cohere model to use")
    parser.add_argument("--model_cache_only", type=str2bool, default=False, help="If True, only use cached model files from --model_dir")
    parser.add_argument("--model_dir", type=str, default=None, help="directory for cached model files")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device type to use for inference")
    parser.add_argument("--device_index", default=0, type=int, help="device index to use for CUDA inference")
    parser.add_argument("--batch_size", default=8, type=int, help="preferred batch size for inference")
    parser.add_argument("--compute_type", default="default", type=str, choices=["default", "float16", "bfloat16", "float32"], help="compute type for inference")
    parser.add_argument("--language", required=True, type=str, choices=sorted(DEFAULT_SUPPORTED_LANGUAGES), help="language spoken in the audio")

    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--output_format", "-f", type=str, default="all", choices=["all", "srt", "vtt", "txt", "tsv", "json"], help="format of the output file")
    parser.add_argument("--verbose", type=str2bool, default=True, help="whether to print progress and debug messages")
    parser.add_argument("--log-level", type=str, default=None, choices=["debug", "info", "warning", "error", "critical"], help="logging level")

    parser.add_argument("--align_model", default=None, help="name of phoneme-level ASR model to use for alignment")
    parser.add_argument("--interpolate_method", default="nearest", choices=["nearest", "linear", "ignore"], help="method to assign timestamps to non-aligned words")
    parser.add_argument("--no_align", action="store_true", help="do not perform phoneme alignment")
    parser.add_argument("--return_char_alignments", action="store_true", help="return character-level alignments in JSON output")

    parser.add_argument("--vad_method", type=str, default="pyannote", choices=["pyannote", "firered", "none"], help="voice activity detection method")
    parser.add_argument("--vad_model_dir", type=str, default=None, help="optional model directory for the selected VAD backend")
    parser.add_argument("--vad_onset", type=float, default=0.500, help="onset threshold for VAD")
    parser.add_argument("--vad_offset", type=float, default=0.363, help="offset threshold for VAD")
    parser.add_argument("--chunk_size", type=float, default=35.0, help="maximum trimmed speech duration per ASR chunk")

    parser.add_argument("--suppress_numerals", action="store_true", help="suppress numeric and currency-like tokens during generation")
    parser.add_argument("--no_punctuation", action="store_true", help="strip punctuation in the transcription prompt")
    parser.add_argument("--max_new_tokens", type=int, default=448, help="maximum number of generated tokens per chunk")

    parser.add_argument("--max_line_width", type=optional_int, default=None, help="maximum number of characters in a subtitle line")
    parser.add_argument("--max_line_count", type=optional_int, default=None, help="maximum number of lines in a subtitle segment")
    parser.add_argument("--highlight_words", type=str2bool, default=False, help="underline each word as it is spoken in srt and vtt")

    parser.add_argument("--threads", type=optional_int, default=0, help="number of CPU threads used by torch")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face access token for gated model downloads")
    parser.add_argument("--print_progress", type=str2bool, default=False, help="if True, print transcription and alignment progress")
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {_package_version()}", help="show coherex version information and exit")
    parser.add_argument("--python-version", "-P", action="version", version=f"Python {platform.python_version()} ({platform.python_implementation()})", help="show python version information and exit")

    args = parser.parse_args().__dict__

    log_level = args.get("log_level")
    verbose = args.get("verbose")
    if log_level is not None:
        setup_logging(level=log_level)
    elif verbose:
        setup_logging(level="info")
    else:
        setup_logging(level="warning")

    from coherex.transcribe import transcribe_task

    transcribe_task(args, parser)


if __name__ == "__main__":
    cli()
