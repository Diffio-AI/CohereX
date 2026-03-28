import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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

from coherex.audio import SAMPLE_RATE, load_audio
from coherex.configuration_cohere_asr import maybe_normalize_language_code
from coherex.log_utils import get_logger

logger = get_logger(__name__)

SPEECHBRAIN_LID_REQUIRED_FILES = (
    "hyperparams.yaml",
    "label_encoder.txt",
    "embedding_model.ckpt",
    "classifier.ckpt",
)
TALTECH_LID_REQUIRED_FILES = (
    "inference_wav2vec.yaml",
    "encoder_wav2vec_classifier.py",
    "label_encoder.txt",
    "wav2vec2.ckpt",
    "classifier.ckpt",
    "model.ckpt",
)
DEFAULT_LID_MODELS = {
    "speechbrain": "speechbrain/lang-id-voxlingua107-ecapa",
    "taltech": "TalTechNLP/voxlingua107-xls-r-300m-wav2vec",
}
DEFAULT_LID_METHOD = "speechbrain"
MAX_LANGUAGE_ID_AUDIO_SECONDS = 30.0


@dataclass
class LanguagePrediction:
    language: str
    raw_label: str
    score: float
    scores: dict[str, float]
    method: str
    model_name: str


class LanguageIdentifier(ABC):
    def __init__(self, method: str, model_name: str, device: torch.device):
        self.method = method
        self.model_name = model_name
        self.device = device

    @abstractmethod
    def detect(self, audio) -> LanguagePrediction:
        raise NotImplementedError

    def _prepare_audio(self, audio) -> np.ndarray:
        if isinstance(audio, str):
            audio = load_audio(audio)
        if not isinstance(audio, np.ndarray):
            audio = np.asarray(audio, dtype=np.float32)
        audio = audio.astype(np.float32, copy=False)
        max_samples = int(MAX_LANGUAGE_ID_AUDIO_SECONDS * SAMPLE_RATE)
        return audio[:max_samples]


def _has_required_files(path: Path, required_files: tuple[str, ...]) -> bool:
    return path.is_dir() and all((path / filename).exists() for filename in required_files)


def resolve_lid_model_source(
    model_name: str,
    model_dir: Optional[str],
    local_files_only: bool,
    token,
    required_files: tuple[str, ...],
    log_name: str,
) -> str:
    candidate_paths = []
    if model_name is not None:
        candidate_paths.append(Path(model_name).expanduser())
    if model_dir is not None:
        candidate_paths.append(Path(model_dir).expanduser())

    for path in candidate_paths:
        if _has_required_files(path, required_files):
            logger.info("Using local %s model files from %s", log_name, path)
            return str(path)

    cache_dir = None
    if model_dir is not None:
        cache_dir = str(Path(model_dir).expanduser())

    if local_files_only:
        logger.info("Loading cached %s model files for %s", log_name, model_name)
    else:
        logger.info("Resolving %s model files for %s", log_name, model_name)

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
            f"Unable to {action} {log_name} model files for {model_name!r}. "
            "Pass a local model snapshot path or set the model directory to a cache directory "
            "or a resolved snapshot directory."
        ) from exc

    logger.info("Resolved %s model files at %s", log_name, snapshot_dir)
    return snapshot_dir


def load_label_list(model_source: str) -> list[str]:
    labels = {}
    label_path = Path(model_source) / "label_encoder.txt"
    pattern = re.compile(r"^'([^']+)' => (\d+)$")
    for line in label_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match is None:
            continue
        label, index = match.groups()
        labels[int(index)] = label
    return [labels[idx] for idx in sorted(labels)]


def normalize_score_tensor(scores) -> torch.Tensor:
    scores_tensor = torch.as_tensor(scores).detach().cpu().float()
    if scores_tensor.ndim == 3 and scores_tensor.shape[1] == 1:
        scores_tensor = scores_tensor.squeeze(1)
    elif scores_tensor.ndim == 1:
        scores_tensor = scores_tensor.unsqueeze(0)
    return scores_tensor


def flatten_label(raw_label) -> str:
    while isinstance(raw_label, (list, tuple)) and raw_label:
        raw_label = raw_label[0]
    return str(raw_label)


def extract_language_code(label: str) -> str:
    return flatten_label(label).split(":", 1)[0].strip().strip("'\"")


def supported_language_scores(labels: list[str], scores: torch.Tensor) -> dict[str, float]:
    supported_scores = {}
    for label, value in zip(labels, scores.tolist()):
        code = maybe_normalize_language_code(extract_language_code(label))
        if code is None:
            continue
        supported_scores[code] = max(supported_scores.get(code, float("-inf")), float(value))
    return supported_scores


def select_supported_language(raw_label: str, scores: dict[str, float]) -> str:
    raw_code = maybe_normalize_language_code(extract_language_code(raw_label))
    if raw_code is not None:
        return raw_code
    if scores:
        return max(scores, key=scores.get)
    raise ValueError(f"Unable to map detected language label {raw_label!r} to a supported Cohere language")
