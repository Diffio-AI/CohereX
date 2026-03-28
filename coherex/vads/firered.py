import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from huggingface_hub import snapshot_download

from coherex.log_utils import get_logger
from coherex.vads.vad import Vad

logger = get_logger(__name__)
FIRERED_REQUIRED_FILES = ("cmvn.ark", "model.pth.tar")


def _load_firered_module():
    repo_root = Path(__file__).resolve().parents[3]
    firered_repo = repo_root / "FireRedVAD"
    if str(firered_repo) not in sys.path and firered_repo.exists():
        sys.path.insert(0, str(firered_repo))

    try:
        from fireredvad import FireRedVad, FireRedVadConfig
    except ImportError as exc:
        raise RuntimeError(
            "FireRedVAD support requires the local FireRedVAD repo on PYTHONPATH "
            "and its Python dependencies installed."
        ) from exc

    return FireRedVad, FireRedVadConfig


def _resolve_model_dir(
    model_dir: Optional[str],
    cache_dir: Optional[str],
    local_files_only: bool,
    token,
) -> str:
    candidate_paths = []
    if model_dir is not None:
        candidate_paths.append(Path(model_dir).expanduser())
    if cache_dir is not None:
        candidate_paths.append(Path(cache_dir).expanduser())

    for path in candidate_paths:
        if path.is_dir() and all((path / name).exists() for name in FIRERED_REQUIRED_FILES):
            logger.info("Using local FireRedVAD model files from %s", path)
            return str(path)
        vad_subdir = path / "VAD"
        if vad_subdir.is_dir() and all((vad_subdir / name).exists() for name in FIRERED_REQUIRED_FILES):
            logger.info("Using local FireRedVAD model files from %s", vad_subdir)
            return str(vad_subdir)

    resolved_cache_dir = None
    if model_dir is not None:
        resolved_cache_dir = str(Path(model_dir).expanduser())
    elif cache_dir is not None:
        resolved_cache_dir = str(Path(cache_dir).expanduser())

    if local_files_only:
        logger.info("Loading cached FireRedVAD model files")
    else:
        logger.info("Resolving FireRedVAD model files")

    try:
        snapshot_dir = snapshot_download(
            "FireRedTeam/FireRedVAD",
            allow_patterns=["VAD/*"],
            cache_dir=resolved_cache_dir,
            local_files_only=local_files_only,
            token=token,
        )
    except Exception as exc:
        action = "load cached" if local_files_only else "download"
        raise RuntimeError(
            f"Unable to {action} FireRedVAD model files. Pass --vad_model_dir pointing to a local "
            "VAD directory, a snapshot directory, or a cache directory."
        ) from exc
    resolved_model_dir = Path(snapshot_dir) / "VAD"
    if not resolved_model_dir.exists():
        raise FileNotFoundError(f"FireRedVAD model directory not found at {resolved_model_dir}")
    logger.info("Resolved FireRedVAD model files at %s", resolved_model_dir)
    return str(resolved_model_dir)


class FireRed(Vad):
    def __init__(
        self,
        device: torch.device,
        token=None,
        model_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        **kwargs,
    ):
        logger.info("Performing voice activity detection using FireRedVAD...")
        super().__init__(kwargs.get("vad_onset", 0.5))

        FireRedVad, FireRedVadConfig = _load_firered_module()
        resolved_model_dir = _resolve_model_dir(
            model_dir=model_dir,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )

        use_gpu = device.type == "cuda"
        if use_gpu:
            torch.cuda.set_device(device)

        config = FireRedVadConfig(
            use_gpu=use_gpu,
            smooth_window_size=int(kwargs.get("smooth_window_size", 5)),
            speech_threshold=float(kwargs.get("speech_threshold", 0.4)),
            min_speech_frame=int(kwargs.get("min_speech_frame", 20)),
            max_speech_frame=int(kwargs.get("max_speech_frame", 2000)),
            min_silence_frame=int(kwargs.get("min_silence_frame", 20)),
            merge_silence_frame=int(kwargs.get("merge_silence_frame", 0)),
            extend_speech_frame=int(kwargs.get("extend_speech_frame", 0)),
            chunk_max_frame=int(kwargs.get("chunk_max_frame", 30000)),
        )
        self.vad_model = FireRedVad.from_pretrained(resolved_model_dir, config)

    def __call__(self, audio_input, **kwargs):
        if isinstance(audio_input, dict):
            waveform = audio_input["waveform"]
        else:
            waveform = audio_input
        result, _ = self.vad_model.detect(waveform)
        return result

    @staticmethod
    def preprocess_audio(audio):
        if torch.is_tensor(audio):
            audio = audio.cpu().numpy()
        if audio.dtype == np.int16:
            return audio
        audio = np.asarray(audio)
        return np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
