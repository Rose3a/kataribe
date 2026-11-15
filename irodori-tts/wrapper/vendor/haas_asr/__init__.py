"""Inference for the Haas speech recognition models.

Importing this package lets NeMo build the model's custom decoder, so import
it before calling ``ASRModel.from_pretrained`` or ``restore_from`` directly.
"""

from importlib.metadata import PackageNotFoundError, version

from haas_asr._nemo import allow_package_targets
from haas_asr.audio import SAMPLE_RATE, load_audio
from haas_asr.longform import split_at_quiet_points, transcribe_longform
from haas_asr.model import load_model
from haas_asr.transcription import transcribe
from haas_asr.vocabulary import Vocabulary

allow_package_targets()

try:
    __version__ = version("haas-asr")
except PackageNotFoundError:  # running from a source checkout, not installed
    __version__ = "0+unknown"
__all__ = [
    "SAMPLE_RATE",
    "Vocabulary",
    "load_audio",
    "load_model",
    "split_at_quiet_points",
    "transcribe",
    "transcribe_longform",
]
