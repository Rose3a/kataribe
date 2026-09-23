"""Shared speaker metadata for torch, TensorRT and the VOICEVOX adapter."""
from __future__ import annotations

import base64
import io
import json
import mimetypes
from functools import lru_cache
from pathlib import Path


FALLBACK_ICON_PATH = (
    Path(__file__).resolve().parents[2] / "speakers" / "thumbnails" / "default-speaker.png"
)

# Stable ASCII identifiers are retained in filenames, API IDs, and saved
# projects. The editor must use the approved name for any user-facing label.
SPEAKER_DISPLAY_NAMES = {
    "tsukuyomi": "つくよみちゃん",
}


def display_name_for(speaker_name: str) -> str:
    """Return the user-facing name for a stable internal speaker identifier."""
    return SPEAKER_DISPLAY_NAMES.get(speaker_name, speaker_name)


def speaker_stem(path: Path) -> str:
    name = path.name
    return name[: -len(".speaker.safetensors")] if name.endswith(".speaker.safetensors") else path.stem


def thumbnail_for(path: Path) -> tuple[str, str] | None:
    """Find a raster icon/sidecar, falling back to the bundled default."""
    stem = speaker_stem(path)
    extensions = (".png", ".jpg", ".jpeg", ".webp")
    # A speaker may provide its UI icon in an ``icon`` directory.  Prefer it
    # over legacy sidecars so custom icons are used when both are present.
    icon_dirs = (path.parent / "icon", path.parent / stem / "icon")
    # The common layout keeps ``icon.png`` beside the speaker embedding.
    candidates = [path.with_name("icon" + ext) for ext in extensions]
    candidates += [directory / (stem + ext) for directory in icon_dirs for ext in extensions]
    candidates += [directory / ("icon" + ext) for directory in icon_dirs for ext in extensions]
    candidates += [path.with_name(stem + ext) for ext in extensions]
    candidates = [path.parent / "thumbnails" / (stem + ext) for ext in extensions] + candidates
    candidates += [path.with_suffix(ext) for ext in extensions]
    for image in candidates:
        if image.is_file() and image.stat().st_size <= 2_000_000:
            mime = "image/png"
            raw = image.read_bytes()
            try:
                from PIL import Image
                source = Image.open(io.BytesIO(raw)).convert("RGBA")
                alpha = source.getchannel("A")
                bbox = alpha.getbbox()
                if bbox:
                    left, top, right, bottom = bbox
                    width, height = right - left, bottom - top
                    # Prefer the face/upper body for the square UI thumbnail.
                    side = max(width, int(height * 0.72))
                    center_x = (left + right) // 2
                    center_y = top + int(height * 0.36)
                    crop_left = max(0, min(source.width - side, center_x - side // 2))
                    crop_top = max(0, min(source.height - side, center_y - side // 2))
                    source = source.crop((crop_left, crop_top, crop_left + side, crop_top + side))
                source.thumbnail((512, 512), Image.Resampling.LANCZOS)
                canvas = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
                canvas.alpha_composite(source, ((512 - source.width) // 2, (512 - source.height) // 2))
                output = io.BytesIO()
                canvas.save(output, format="PNG", optimize=True)
                raw = output.getvalue()
            except Exception:
                pass
            encoded = base64.b64encode(raw).decode("ascii")
            return mime, encoded
    return _fallback_icon()


# 母音ごとの口パーツ（あいうえお）。ファイル名は <stem>_mouth_<vowel>.png など。
MOUTH_SHAPES = ("n", "a", "i", "u", "e", "o")


def mouth_parts_for(path: Path) -> dict[str, str] | None:
    """Find optional per-vowel mouth parts beside the portrait.

    Returns a shape -> base64 payload map, or None when nothing matches, so the
    editor can fall back to the single open-mouth image.
    """
    stem = speaker_stem(path)
    extensions = (".png", ".jpg", ".jpeg", ".webp")
    parts: dict[str, str] = {}
    for shape in MOUTH_SHAPES:
        suffixes = (f"_mouth_{shape}", f".mouth-{shape}", f"_mouth{shape}")
        candidates = [path.parent / "thumbnails" / f"{stem}{suffix}{ext}"
                      for suffix in suffixes for ext in extensions]
        candidates += [path.with_name(f"{stem}{suffix}{ext}")
                       for suffix in suffixes for ext in extensions]
        for image in candidates:
            if image.is_file() and image.stat().st_size <= 2_000_000:
                mime = mimetypes.guess_type(image.name)[0] or "image/png"
                parts[shape] = base64.b64encode(image.read_bytes()).decode("ascii")
                break
    return parts or None


def mouth_open_thumbnail_for(path: Path) -> tuple[str, str] | None:
    """Find an optional open-mouth sidecar using simple, portable names."""
    stem = speaker_stem(path)
    suffixes = ("_open", ".open", ".mouth-open", "_mouth")
    extensions = (".png", ".jpg", ".jpeg", ".webp")
    candidates = [path.parent / "thumbnails" / f"{stem}{suffix}{ext}"
                  for suffix in suffixes for ext in extensions]
    candidates += [path.with_name(f"{stem}{suffix}{ext}")
                   for suffix in suffixes for ext in extensions]
    for image in candidates:
        if image.is_file() and image.stat().st_size <= 2_000_000:
            mime = mimetypes.guess_type(image.name)[0] or "image/png"
            return mime, base64.b64encode(image.read_bytes()).decode("ascii")
    return None


def blink_thumbnail_for(path: Path) -> tuple[str, str] | None:
    """Find an optional blink/eyes-closed sidecar."""
    stem = speaker_stem(path)
    suffixes = ("_blink", ".blink", "_eyes-closed", ".eyes-closed", "_eye-close", ".eye-close", "_close", "_me")
    extensions = (".png", ".jpg", ".jpeg", ".webp")
    candidates = [path.parent / "thumbnails" / f"{stem}{suffix}{ext}"
                  for suffix in suffixes for ext in extensions]
    candidates += [path.with_name(f"{stem}{suffix}{ext}")
                   for suffix in suffixes for ext in extensions]
    for image in candidates:
        if image.is_file() and image.stat().st_size <= 2_000_000:
            raw = image.read_bytes()
            mime = mimetypes.guess_type(image.name)[0] or "image/png"
            return mime, base64.b64encode(raw).decode("ascii")
    return None


def portrait_for(path: Path) -> tuple[str, str] | None:
    """Return the original portrait, or the bundled SVG icon when absent."""
    extensions = (".png", ".jpg", ".jpeg", ".webp")
    stem = speaker_stem(path)
    for image in [path.with_name(stem + ext) for ext in extensions]:
        if image.is_file() and image.stat().st_size <= 8_000_000:
            mime = "image/png" if image.suffix.lower() == ".png" else (mimetypes.guess_type(image.name)[0] or "image/png")
            return mime, base64.b64encode(image.read_bytes()).decode("ascii")
    return _fallback_icon()


def credit_for(path: Path) -> str | None:
    """Show a speaker's credit.txt below its portrait when provided."""
    return _read_credit(path, ("credit.txt",))


def policy_for(path: Path) -> str | None:
    """Prefer the full credit notice over abbreviated JSON artwork metadata."""
    return _read_credit(path, ("credit.txt", "credits.json"))


def _read_credit(path: Path, filenames: tuple[str, ...]) -> str | None:
    candidates = (path.parent / filename for filename in filenames)
    for metadata in candidates:
        if not metadata.is_file() or metadata.stat().st_size > 64_000:
            continue
        try:
            if metadata.suffix.lower() == ".json":
                value = json.loads(metadata.read_text(encoding="utf-8"))
                if isinstance(value, str):
                    credit = value.strip()
                elif isinstance(value, dict):
                    fields = ("credit", "text", "illustrator", "artist")
                    values = [str(value[key]).strip() for key in fields
                              if value.get(key)]
                    credit = "\n".join(values)
                else:
                    credit = ""
            else:
                credit = metadata.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if credit:
            return credit
    return None


def _fallback_icon() -> tuple[str, str] | None:
    """Return the bundled default portrait for speakers without artwork."""
    if not FALLBACK_ICON_PATH.is_file() or FALLBACK_ICON_PATH.stat().st_size > 8_000_000:
        return None
    stat = FALLBACK_ICON_PATH.stat()
    return _cached_fallback_icon(stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=2)
def _cached_fallback_icon(_mtime_ns: int, _size: int) -> tuple[str, str]:
    return "image/png", base64.b64encode(FALLBACK_ICON_PATH.read_bytes()).decode("ascii")


def speaker_catalog(embed_dirs, speaker_paths=None):
    """Return deduplicated speakers with optional VOICEVOX image data."""
    result = []
    seen = set()
    if speaker_paths is None:
        speaker_paths = (
            (speaker_stem(path), path)
            for directory in embed_dirs
            for path in sorted(Path(directory).rglob("*.safetensors"))
        )
    for name, path in speaker_paths:
        if name in seen:
            continue
        seen.add(name)
        result.append((name, thumbnail_for(path)))
    return result
