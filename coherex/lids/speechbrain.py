import torch

from coherex.lids.base import (
    DEFAULT_LID_MODELS,
    LanguageIdentifier,
    LanguagePrediction,
    flatten_label,
    load_label_list,
    normalize_score_tensor,
    resolve_lid_model_source,
    select_supported_language,
    supported_language_scores,
    SPEECHBRAIN_LID_REQUIRED_FILES,
)


class SpeechBrainLanguageIdentifier(LanguageIdentifier):
    def __init__(
        self,
        device: torch.device,
        model_name: str | None = None,
        model_dir: str | None = None,
        local_files_only: bool = False,
        token=None,
    ):
        resolved_model_name = model_name or DEFAULT_LID_MODELS["speechbrain"]
        super().__init__(method="speechbrain", model_name=resolved_model_name, device=device)
        model_source = resolve_lid_model_source(
            model_name=resolved_model_name,
            model_dir=model_dir,
            local_files_only=local_files_only,
            token=token,
            required_files=SPEECHBRAIN_LID_REQUIRED_FILES,
            log_name="SpeechBrain language-id",
        )

        from speechbrain.inference.classifiers import EncoderClassifier

        self.classifier = EncoderClassifier.from_hparams(
            source=model_source,
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
