from __future__ import annotations

import logging
from collections.abc import Iterable

import torch

logger = logging.getLogger(__name__)

IRODORI_WATERMARK_PAYLOAD = (73, 82, 68, 84, 83)  # "IRDTS"

# Pinned SilentCipher weights. Setup downloads exactly these files; the runtime
# reads them from the Hugging Face cache and only goes online if they are missing.
SILENTCIPHER_REPO = "Sony/SilentCipher"
SILENTCIPHER_REVISION = "a1c4d021905e0dc5b24be5f68db5fc4dba410ee1"
SILENTCIPHER_CKPT_DIR = "44_1_khz/73999_iteration"
SILENTCIPHER_FILES = tuple(
    f"{SILENTCIPHER_CKPT_DIR}/{name}"
    for name in ("hparams.yaml", "enc_c.ckpt", "dec_c.ckpt", "dec_m_0.ckpt")
)


def silentcipher_checkpoint_dir(*, local_files_only: bool = True) -> str:
    """Return the folder holding the pinned 44.1 kHz SilentCipher checkpoint."""
    from huggingface_hub import snapshot_download

    folder = snapshot_download(
        repo_id=SILENTCIPHER_REPO,
        revision=SILENTCIPHER_REVISION,
        allow_patterns=list(SILENTCIPHER_FILES),
        local_files_only=local_files_only,
    )
    return f"{folder}/{SILENTCIPHER_CKPT_DIR}"


def _as_single_channel_vector(audio: torch.Tensor) -> torch.Tensor | None:
    squeezed = audio.detach().float().squeeze()
    if squeezed.ndim == 0 or squeezed.numel() == 0:
        return None
    if squeezed.ndim == 1:
        return squeezed
    return squeezed.reshape(-1)


def _match_original_rank(audio: torch.Tensor, *, reference: torch.Tensor) -> torch.Tensor:
    if reference.ndim == 2:
        return audio.reshape(1, -1)
    return audio.reshape(-1)


class SilentCipherWatermarker:
    def __init__(self, *, device: str, model_type: str = "44.1k") -> None:
        self.model = self._load_backend(device=device, model_type=model_type)

    @staticmethod
    def _load_backend(*, device: str, model_type: str):
        try:
            import silentcipher
        except ImportError:
            logger.warning(
                "SilentCipher package is unavailable; generated audio will not be watermarked."
            )
            return None

        try:
            try:
                ckpt_dir = silentcipher_checkpoint_dir(local_files_only=True)
            except Exception:
                ckpt_dir = silentcipher_checkpoint_dir(local_files_only=False)
            return silentcipher.get_model(
                model_type=model_type,
                ckpt_path=ckpt_dir,
                config_path=f"{ckpt_dir}/hparams.yaml",
                device=device,
            )
        except Exception as exc:
            logger.warning(
                "SilentCipher model could not be loaded (%s); generated audio will not be "
                "watermarked.",
                exc,
            )
            return None

    @property
    def ready(self) -> bool:
        return self.model is not None

    def encode_one(
        self,
        audio: torch.Tensor,
        *,
        sample_rate: int,
        payload: Iterable[int] = IRODORI_WATERMARK_PAYLOAD,
    ) -> torch.Tensor:
        if self.model is None:
            return audio

        vector = _as_single_channel_vector(audio)
        if vector is None:
            return audio

        encoded, _ = self.model.encode_wav(
            vector.to(self.model.device),
            int(sample_rate),
            list(payload),
            calc_sdr=False,
        )
        encoded_audio = torch.as_tensor(encoded, dtype=torch.float32, device="cpu")
        return _match_original_rank(encoded_audio, reference=audio)

    def encode_batch(self, audios: list[torch.Tensor], *, sample_rate: int) -> list[torch.Tensor]:
        if self.model is None:
            return audios
        return [self.encode_one(audio, sample_rate=sample_rate) for audio in audios]
