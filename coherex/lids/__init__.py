from coherex.lids.base import DEFAULT_LID_METHOD, DEFAULT_LID_MODELS, LanguagePrediction
from coherex.lids.speechbrain import SpeechBrainLanguageIdentifier
from coherex.lids.taltech import TaltechLanguageIdentifier


def load_lid_model(
    method: str = DEFAULT_LID_METHOD,
    device=None,
    model_name: str | None = None,
    model_dir: str | None = None,
    local_files_only: bool = False,
    use_auth_token=None,
):
    if method == "speechbrain":
        return SpeechBrainLanguageIdentifier(
            device=device,
            model_name=model_name or DEFAULT_LID_MODELS["speechbrain"],
            model_dir=model_dir,
            local_files_only=local_files_only,
            token=use_auth_token,
        )
    if method == "taltech":
        return TaltechLanguageIdentifier(
            device=device,
            model_name=model_name or DEFAULT_LID_MODELS["taltech"],
            model_dir=model_dir,
            local_files_only=local_files_only,
            token=use_auth_token,
        )
    raise ValueError(f"Unsupported lid_method: {method}")
