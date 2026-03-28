import os
from dataclasses import dataclass
from typing import Any, Callable, Optional, Text, Union

import numpy as np
import torch
from pyannote.core import Annotation, SlidingWindowFeature
from pyannote.core import Segment

from coherex.vads.vad import Vad
from coherex.vads.binarize import Binarize
from coherex.log_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SegmentX:
    start: float
    end: float
    speaker: Optional[str] = None


def load_vad_model(device, vad_onset=0.500, vad_offset=0.363, token=None, model_fp=None):
    from pyannote.audio import Model

    model_dir = torch.hub._get_torch_home()

    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    os.makedirs(model_dir, exist_ok = True)
    if model_fp is None:
        # Dynamically resolve the path to the model file
        model_fp = os.path.join(main_dir, "assets", "pytorch_model.bin")
        model_fp = os.path.abspath(model_fp)  # Ensure the path is absolute
    else:
        model_fp = os.path.abspath(model_fp)  # Ensure any provided path is absolute

    # Check if the resolved model file exists
    if not os.path.exists(model_fp):
        raise FileNotFoundError(f"Model file not found at {model_fp}")

    if os.path.exists(model_fp) and not os.path.isfile(model_fp):
        raise RuntimeError(f"{model_fp} exists and is not a regular file")

    vad_model = Model.from_pretrained(model_fp, token=token)
    hyperparameters = {"onset": vad_onset,
                    "offset": vad_offset,
                    "min_duration_on": 0.1,
                    "min_duration_off": 0.1}
    vad_pipeline = VoiceActivitySegmentation(segmentation=vad_model, device=torch.device(device))
    vad_pipeline.instantiate(hyperparameters)

    return vad_pipeline


class VoiceActivitySegmentation:
    def __init__(
            self,
            segmentation: Any = "pyannote/segmentation",
            fscore: bool = False,
            token: Union[Text, None] = None,
            **inference_kwargs,
    ):
        from pyannote.audio.pipelines import VoiceActivityDetection

        self.pipeline = VoiceActivityDetection(segmentation=segmentation, fscore=fscore, token=token, **inference_kwargs)

    def instantiate(self, hyperparameters):
        self.pipeline.instantiate(hyperparameters)

    def __call__(self, file, hook: Optional[Callable] = None):
        return self.apply(file, hook=hook)

    def apply(self, file, hook: Optional[Callable] = None) -> Annotation:
        """Apply voice activity detection

        Parameters
        ----------
        file : AudioFile
            Processed file.
        hook : callable, optional
            Hook called after each major step of the pipeline with the following
            signature: hook("step_name", step_artefact, file=file)

        Returns
        -------
        speech : Annotation
            Speech regions.
        """

        # setup hook (e.g. for debugging purposes)
        hook = self.pipeline.setup_hook(file, hook=hook)

        # apply segmentation model (only if needed)
        # output shape is (num_chunks, num_frames, 1)
        if self.pipeline.training:
            if self.pipeline.CACHED_SEGMENTATION in file:
                segmentations = file[self.pipeline.CACHED_SEGMENTATION]
            else:
                segmentations = self.pipeline._segmentation(file)
                file[self.pipeline.CACHED_SEGMENTATION] = segmentations
        else:
            segmentations: SlidingWindowFeature = self.pipeline._segmentation(file)

        return segmentations


class Pyannote(Vad):

    def __init__(self, device, token=None, model_fp=None, **kwargs):
        logger.info("Performing voice activity detection using Pyannote...")
        super().__init__(kwargs['vad_onset'])
        self.vad_pipeline = load_vad_model(device, token=token, model_fp=model_fp)

    def __call__(self, audio, **kwargs):
        return self.vad_pipeline(audio)

    @staticmethod
    def preprocess_audio(audio):
        return torch.from_numpy(audio).unsqueeze(0)

    @staticmethod
    def merge_chunks(segments,
                     chunk_size,
                     onset: float = 0.5,
                     offset: Optional[float] = None,
                     ):
        assert chunk_size > 0
        binarize = Binarize(max_duration=chunk_size, onset=onset, offset=offset)
        segments = binarize(segments)
        segments_list = []
        for speech_turn in segments.get_timeline():
            segments_list.append(SegmentX(speech_turn.start, speech_turn.end, "UNKNOWN"))

        if len(segments_list) == 0:
            logger.warning("No active speech found in audio")
            return []
        assert segments_list, "segments_list is empty."
        return Vad.merge_chunks(segments_list, chunk_size, onset, offset)
