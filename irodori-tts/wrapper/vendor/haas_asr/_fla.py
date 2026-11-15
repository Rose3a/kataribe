"""Import Kimi Delta Attention from flash-linear-attention, with a strict version check."""

from __future__ import annotations

from importlib import metadata

FLA_DISTRIBUTION = "flash-linear-attention"
FLA_REQUIRED_VERSION = "0.5.2"


def check_fla_version() -> None:
    """Raise unless the tested flash-linear-attention release is installed."""
    try:
        installed = metadata.version(FLA_DISTRIBUTION)
    except metadata.PackageNotFoundError as e:
        raise ImportError(
            f"{FLA_DISTRIBUTION}=={FLA_REQUIRED_VERSION} is required but not installed."
        ) from e
    if installed != FLA_REQUIRED_VERSION:
        raise ImportError(
            f"{FLA_DISTRIBUTION}=={FLA_REQUIRED_VERSION} is required, found {installed}. "
            "Its layer APIs change between releases, so other versions are refused."
        )


def load_kda_class() -> type:
    """Return fla's ``KimiDeltaAttention`` layer class.

    Imported on first use rather than at package import, because fla needs
    Triton, which is unavailable without a CUDA build (e.g. on macOS).
    """
    check_fla_version()
    from fla.layers.kda import KimiDeltaAttention

    return KimiDeltaAttention
