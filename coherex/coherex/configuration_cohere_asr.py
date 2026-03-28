import torch
import re
from transformers import PretrainedConfig

DEFAULT_SUPPORTED_LANGUAGES = ["ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh"]
NO_SPACE_LANGS = {"ja", "zh"}
SUPPORTED_LANGUAGE_GROUPS = {
    "European": [
        ("en", "English"),
        ("fr", "French"),
        ("de", "German"),
        ("it", "Italian"),
        ("es", "Spanish"),
        ("pt", "Portuguese"),
        ("el", "Greek"),
        ("nl", "Dutch"),
        ("pl", "Polish"),
    ],
    "APAC": [
        ("zh", "Chinese (Mandarin)"),
        ("ja", "Japanese"),
        ("ko", "Korean"),
        ("vi", "Vietnamese"),
    ],
    "MENA": [
        ("ar", "Arabic"),
    ],
}

SUPPORTED_LANGUAGE_ALIASES = {
    "arabic": "ar",
    "ar": "ar",
    "chinese": "zh",
    "chinese mandarin": "zh",
    "mandarin": "zh",
    "mandarin chinese": "zh",
    "zh": "zh",
    "dutch": "nl",
    "nl": "nl",
    "english": "en",
    "en": "en",
    "french": "fr",
    "fr": "fr",
    "german": "de",
    "de": "de",
    "greek": "el",
    "el": "el",
    "italian": "it",
    "it": "it",
    "japanese": "ja",
    "ja": "ja",
    "korean": "ko",
    "ko": "ko",
    "polish": "pl",
    "pl": "pl",
    "portuguese": "pt",
    "pt": "pt",
    "spanish": "es",
    "es": "es",
    "vietnamese": "vi",
    "vi": "vi",
}


def supported_languages_help_text() -> str:
    lines = []
    for region, languages in SUPPORTED_LANGUAGE_GROUPS.items():
        items = ", ".join(f"{name} (`{code}`)" for code, name in languages)
        lines.append(f"{region}: {items}")
    return "\n".join(lines)


def normalize_language_code(language: str) -> str:
    if language is None or not str(language).strip():
        raise ValueError(
            "language is required. Supported languages:\n"
            f"{supported_languages_help_text()}"
        )

    normalized = str(language).strip().lower()
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized)
    normalized = " ".join(normalized.split())
    code = SUPPORTED_LANGUAGE_ALIASES.get(normalized)
    if code is None:
        raise ValueError(
            f"Unsupported language: {language!r}. Supported languages:\n"
            f"{supported_languages_help_text()}"
        )
    return code


class CohereAsrConfig(PretrainedConfig):
    """Configuration for the Cohere ASR remote-code model."""

    model_type = "cohere_asr"

    def __init__(
        self,
        vocab_size=16384,
        encoder=None,
        transf_decoder=None,
        head=None,
        preprocessor=None,
        max_audio_clip_s=35,
        overlap_chunk_second=5,
        min_energy_window_samples=1600,
        batch_size=64,
        sample_rate=16000,
        supported_languages=None,
        **kwargs,
    ):
        kwargs.setdefault("is_encoder_decoder", True)
        self.vocab_size = vocab_size
        self.encoder = encoder
        self.transf_decoder = transf_decoder
        self.head = head
        self.preprocessor = preprocessor
        self.max_audio_clip_s = max_audio_clip_s
        self.overlap_chunk_second = overlap_chunk_second
        self.min_energy_window_samples = min_energy_window_samples
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.supported_languages = (
            list(supported_languages) if supported_languages is not None else list(DEFAULT_SUPPORTED_LANGUAGES)
        )
        super().__init__(**kwargs)

    @property
    def num_hidden_layers(self):
        return self.transf_decoder["config_dict"]["num_layers"]


if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "disable"):
    _dynamo_disable = torch._dynamo.disable
else:

    def _dynamo_disable(fn):
        return fn
