import sys
import types

import torch

from coherex.lids.base import (
    DEFAULT_LID_MODELS,
    LanguageIdentifier,
    LanguagePrediction,
    TALTECH_LID_REQUIRED_FILES,
    flatten_label,
    load_label_list,
    normalize_score_tensor,
    resolve_lid_model_source,
    select_supported_language,
    supported_language_scores,
)


def _install_taltech_compatibility_shim():
    import speechbrain.lobes.models
    from speechbrain.lobes.models.huggingface_transformers.wav2vec2 import Wav2Vec2

    shim_name = "speechbrain.lobes.models.huggingface_wav2vec"
    shim = sys.modules.get(shim_name)
    if shim is None:
        shim = types.ModuleType(shim_name)
        shim.__package__ = "speechbrain.lobes.models"
        sys.modules[shim_name] = shim

    shim.HuggingFaceWav2Vec2 = Wav2Vec2
    speechbrain.lobes.models.huggingface_wav2vec = shim


class TaltechLanguageIdentifier(LanguageIdentifier):
    def __init__(
        self,
        device: torch.device,
        model_name: str | None = None,
        model_dir: str | None = None,
        local_files_only: bool = False,
        token=None,
    ):
        resolved_model_name = model_name or DEFAULT_LID_MODELS["taltech"]
        super().__init__(method="taltech", model_name=resolved_model_name, device=device)
        model_source = resolve_lid_model_source(
            model_name=resolved_model_name,
            model_dir=model_dir,
            local_files_only=local_files_only,
            token=token,
            required_files=TALTECH_LID_REQUIRED_FILES,
            log_name="TalTech language-id",
        )

        _install_taltech_compatibility_shim()
        from speechbrain.inference.interfaces import foreign_class

        self.classifier = foreign_class(
            source=model_source,
            pymodule_file="encoder_wav2vec_classifier.py",
            hparams_file="inference_wav2vec.yaml",
            classname="EncoderWav2vecClassifier",
            run_opts={"device": str(device)},
        )
        self.labels = load_label_list(model_source)

    def detect(self, audio) -> LanguagePrediction:
        waveform = self._prepare_audio(audio)
        wavs = torch.from_numpy(waveform).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            out_prob, score, _, text_lab = self.classifier.classify_batch(wavs)

        score_tensor = normalize_score_tensor(out_prob)
        raw_label = flatten_label(text_lab)
        raw_scores = supported_language_scores(self.labels, score_tensor[0])
        language = select_supported_language(raw_label, raw_scores)

        return LanguagePrediction(
            language=language,
            raw_label=raw_label,
            score=float(torch.as_tensor(score).reshape(-1)[0].item()),
            scores=raw_scores,
            method=self.method,
            model_name=self.model_name,
        )
