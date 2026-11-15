"""Let NeMo instantiate this package's classes from a model config."""

from __future__ import annotations

from nemo.core.classes import common

PACKAGE_PREFIX = f"{__package__}."


def allow_package_targets() -> None:
    """Add this package to NeMo's ``_target_`` allowlist.

    NeMo refuses to instantiate a config ``_target_`` unless its import path
    starts with an allowlisted prefix (``nemo.``, ``torch.nn.``, ...), and
    then also requires the class to be an ``nn.Module`` or ``ModelPT``. This
    adds our prefix; the class check still applies. Idempotent.
    """
    prefixes = getattr(common, "ALLOWED_TARGET_PREFIXES", None)
    if not isinstance(prefixes, list):
        raise ImportError(
            "nemo.core.classes.common.ALLOWED_TARGET_PREFIXES is missing; "
            "this NeMo version is not supported."
        )
    if PACKAGE_PREFIX not in prefixes:
        prefixes.append(PACKAGE_PREFIX)
