import importlib


def _lazy_import(name):
    return importlib.import_module(f"coherex.{name}")


def load_align_model(*args, **kwargs):
    alignment = _lazy_import("alignment")
    return alignment.load_align_model(*args, **kwargs)


def align(*args, **kwargs):
    alignment = _lazy_import("alignment")
    return alignment.align(*args, **kwargs)


def load_model(*args, **kwargs):
    asr = _lazy_import("asr")
    return asr.load_model(*args, **kwargs)


def load_lid_model(*args, **kwargs):
    lids = _lazy_import("lids")
    return lids.load_lid_model(*args, **kwargs)


def load_audio(*args, **kwargs):
    audio = _lazy_import("audio")
    return audio.load_audio(*args, **kwargs)


def setup_logging(*args, **kwargs):
    logging_module = _lazy_import("log_utils")
    return logging_module.setup_logging(*args, **kwargs)


def get_logger(*args, **kwargs):
    logging_module = _lazy_import("log_utils")
    return logging_module.get_logger(*args, **kwargs)
