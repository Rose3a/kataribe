"""List and delete downloaded models and generated TensorRT caches.

Models and plans pile up quickly (every source update makes older TensorRT
plans unusable), so the editor shows what each folder is, how big it is and
whether anything still uses it.  Deletion only ever touches entries produced
by the latest scan and marked deletable, never an arbitrary path.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import struct
import threading
import time

BOX = Path(__file__).resolve().parents[2]

# Labels for Hugging Face repos this app downloads.
KNOWN_REPOS = {
    "Aratako/Irodori-TTS-v4.1-Small": "Irodori-TTS v4.1 Small",
    "Aratako/Irodori-TTS-v4.1-Small-MF": "Irodori-TTS v4.1 Small MF（MeanFlow）",
    "j-llm/Irodori-TTS-v4.1-Small-Yomi-Tech-tuned": "Irodori-TTS v4.1 Small Yomi Tech",
    "phasefield-audio/Irodori-TTS-v4.1-Anime": "Irodori-TTS v4.1 Anime",
    "Aratako/Irodori-TTS-v4-Large": "Irodori-TTS v4 Large（bf16）",
    "Aratako/Irodori-TTS-v4-Large-Quantized": "Irodori-TTS v4 Large 量子化版",
    "Aratako/Irodori-TTS-500M-v3": "Irodori-TTS 500M v3（旧版）",
}
# Shared parts every synthesis needs; shown but never deletable.
REQUIRED_REPOS = {
    "Aratako/Semantic-DACVAE-Japanese-32dim": "音声コーデック（DACVAE）",
    "Sony/SilentCipher": "透かし（SilentCipher、参照音声での生成に使用）",
}


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                stat = os.lstat(os.path.join(root, name))
            except OSError:
                continue
            # HF cache snapshots may be symlinks into blobs/; count blobs once.
            if not os.path.islink(os.path.join(root, name)):
                total += stat.st_size
    return total


def _size(path: Path) -> int:
    try:
        return _dir_size(path) if path.is_dir() else path.stat().st_size
    except OSError:
        return 0


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(BOX.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _safetensors_config(path: Path) -> dict:
    try:
        with path.open("rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            if not 0 < header_len <= 64 * 1024 * 1024:
                return {}
            meta = json.loads(handle.read(header_len).decode("utf-8", "replace")).get(
                "__metadata__") or {}
        config = json.loads(meta.get("config_json") or "{}")
        quant = meta.get("irodori_quantization_json")
        if quant:
            config["_quantization"] = json.loads(quant).get("quantization_type") or "unknown"
        return config
    except (OSError, ValueError, struct.error):
        return {}


def _describe_config(config: dict) -> str:
    parts = []
    if config.get("flow_parameterization") == "meanflow":
        parts.append("MeanFlow")
    elif config:
        parts.append("RF")
    if config.get("_quantization"):
        parts.append(f"量子化 {config['_quantization']}")
    if config.get("speaker_dim"):
        parts.append(f"話者 {config['speaker_dim']} 次元")
    return " · ".join(parts)


def hf_repo_of(path: str | Path) -> str | None:
    """repo_id for a file inside the HF hub cache (models--owner--name/...)."""
    for part in Path(path).parts:
        if part.startswith("models--"):
            return part[len("models--"):].replace("--", "/")
    return None


def checkpoint_label(path: str | Path) -> str:
    """Human name for a checkpoint path (HF cache or models folder)."""
    path = Path(path)
    repo = hf_repo_of(path)
    if repo:
        parts = path.parts
        # Quantized repos keep each variant in a subfolder of the snapshot.
        try:
            snap = parts.index("snapshots")
            sub = "/".join(parts[snap + 2:-1])
        except ValueError:
            sub = ""
        label = KNOWN_REPOS.get(repo, repo)
        return f"{label}（{sub}）" if sub else label
    return _rel(path)


class ModelStorage:
    """Scan result cache and deletion for the editor's storage panel."""

    def __init__(self, model_dir: Path, hf_home: Path, asr_dir: Path):
        self.model_dir = Path(model_dir)
        self.hub_dir = Path(hf_home) / "hub"
        self.asr_dir = Path(asr_dir)
        self.trt_dir = BOX / ".cache" / "trt"
        self.codec_dir = BOX / ".cache" / "trt-codec"
        self.legacy_plan = BOX / "irodori-tts" / "bf16-fallback" / "fallback_bf16.plan"
        self._identify_lock = threading.Lock()
        self._identifying = False
        self._last: dict[str, dict] = {}

    # ------------------------------------------------------------ scanning
    def scan(self, *, selected_model: str, active_checkpoint: Path | None,
             active_plan: Path | None, active_codec_plan: Path | None,
             asr_loaded: bool, busy: bool) -> dict:
        entries: list[dict] = []
        entries += self._hf_entries(selected_model)
        entries += self._local_model_entries(active_checkpoint)
        entries += self._asr_entries(asr_loaded)
        entries += self._plan_entries(active_plan, busy)
        entries += self._codec_entries(active_codec_plan, busy)
        entries += self._legacy_entries()
        self._last = {entry["id"]: entry for entry in entries}
        return dict(entries=entries, totalBytes=sum(e["bytes"] for e in entries),
                    identifying=self._identifying, scannedAt=time.time())

    @staticmethod
    def _entry(kind, path, label, detail, status, deletable, note="", bytes_=None):
        return dict(id=f"{kind}:{_rel(path)}", kind=kind, path=str(path), label=label,
                    detail=detail, status=status, deletable=deletable, note=note,
                    bytes=_size(path) if bytes_ is None else bytes_, modified=_mtime(path))

    def _hf_entries(self, selected_model: str) -> list[dict]:
        if not self.hub_dir.is_dir():
            return []
        selected_repo = "/".join(str(selected_model).strip().strip("/").split("/")[:2])
        result = []
        for folder in sorted(self.hub_dir.glob("models--*")):
            repo = hf_repo_of(folder)
            if repo in REQUIRED_REPOS:
                result.append(self._entry("hf", folder, REQUIRED_REPOS[repo], repo,
                                          "required", False, "常に必要な部品です"))
                continue
            is_tts = repo in KNOWN_REPOS or "irodori-tts" in repo.lower()
            if not is_tts:
                # Text tokenizers and other small helpers the models pull in.
                result.append(self._entry("hf", folder, repo, "モデルが使う補助ファイル（トークナイザなど）",
                                          "required", False, "モデルの読み込みに使います"))
                continue
            checkpoints = sorted(folder.glob("snapshots/*/**/model.safetensors"))
            detail = " / ".join(filter(None, (
                repo, _describe_config(_safetensors_config(checkpoints[0])) if checkpoints else "")))
            variants = sorted({checkpoint_label(p) for p in checkpoints})
            note = "含まれる版: " + "、".join(variants) if len(variants) > 1 else ""
            in_use = repo.lower() == selected_repo.lower()
            result.append(self._entry(
                "hf", folder, KNOWN_REPOS.get(repo, repo), detail,
                "in_use" if in_use else "unused", not in_use,
                note or ("選択中のモデルです" if in_use else "再び選ぶと自動でダウンロードし直します")))
        return result

    def _local_model_entries(self, active_checkpoint: Path | None) -> list[dict]:
        if not self.model_dir.is_dir():
            return []
        active = active_checkpoint.resolve() if active_checkpoint else None
        result = []
        for path in sorted(self.model_dir.rglob("*.safetensors")):
            if path.name.endswith(".speaker.safetensors"):
                continue
            detail = _describe_config(_safetensors_config(path))
            note = ""
            if path.resolve() == (self.model_dir / "model.safetensors").resolve():
                note = ("セットアップで取得した v4.1 Small。エディタは Hugging Face 版を使うため、"
                        "旧 CLI（run.bat など）や Radeon の codec 変換を使わなければ不要です")
            in_use = active is not None and path.resolve() == active
            result.append(self._entry(
                "local", path, _rel(path), detail, "in_use" if in_use else "unused",
                not in_use, note or ("選択中のモデルです" if in_use else "")))
        return result

    def _asr_entries(self, asr_loaded: bool) -> list[dict]:
        if not self.asr_dir.is_dir():
            return []
        return [self._entry("asr", self.asr_dir, "口パク用 音声認識（ASR）", _rel(self.asr_dir),
                            "in_use" if asr_loaded else "unused", not asr_loaded,
                            "削除しても口パクを使うと自動で再取得します")]

    def _plan_entries(self, active_plan: Path | None, busy: bool) -> list[dict]:
        if not self.trt_dir.is_dir():
            return []
        try:
            from trt_cache import env_identity, known_model_digests
        except Exception:  # noqa: BLE001
            env_identity = None
            known_model_digests = dict
        try:
            current = env_identity() if env_identity else None
        except Exception:  # noqa: BLE001 - no CUDA / TensorRT on this PC
            current = None
        digests = known_model_digests()
        active_dir = active_plan.resolve().parent if active_plan else None
        result, unknown = [], False
        for folder in sorted(self.trt_dir.iterdir()):
            if not folder.is_dir():
                continue
            if folder.name.startswith("build-"):
                result.append(self._build_entry("trt-build", folder, busy))
                continue
            try:
                identity = json.loads((folder / "ready.json").read_text(encoding="utf-8"))["identity"]
            except (OSError, ValueError, KeyError, TypeError):
                result.append(self._entry("trt", folder, "TensorRT plan（壊れている / 未完成）",
                                          _rel(folder), "stale", not busy,
                                          "使われません"))
                continue
            model = digests.get(identity.get("model"))
            if model is None:
                unknown = True
            label = ("TensorRT plan: " + checkpoint_label(model)) if model else "TensorRT plan: モデル照合中 / 不明"
            detail = f"{identity.get('gpu', '?')} · TensorRT {identity.get('trt', '?')}"
            # INT4 の plan（trt_int4.py）は identity に variant と変換スクリプトの内容ハッシュが加わる
            int4 = identity.get("variant") == "int4"
            env = {k: v for k, v in identity.items() if k not in ("model", "variant", "int4_sources")}
            if int4:
                try:
                    from trt_int4 import int4_sources
                    int4_current = identity.get("int4_sources") == int4_sources()
                except Exception:  # noqa: BLE001
                    int4_current = False
            else:
                int4_current = True
            if int4:
                label += "（INT4）"
            if active_dir is not None and folder.resolve() == active_dir:
                status, note = "in_use", "読み込み中のモデルが使っています"
            elif current is not None and env == current and int4_current:
                status, note = "unused", "このモデルを TensorRT で使うときに再利用します（消すと次回変換し直し）"
            else:
                status, note = "stale", "プログラムや GPU 環境が変わったため、もう使われません"
            result.append(self._entry("trt", folder, label, detail, status,
                                      status != "in_use" and not busy, note))
        if unknown:
            self._identify_in_background()
        return result

    def _codec_entries(self, active_codec_plan: Path | None, busy: bool) -> list[dict]:
        if not self.codec_dir.is_dir():
            return []
        current = None
        try:
            from trt_codec import cache_identity, codec_weights
            current = cache_identity(codec_weights())
        except Exception:  # noqa: BLE001
            pass
        active_dir = active_codec_plan.resolve().parent if active_codec_plan else None
        result = []
        for folder in sorted(self.codec_dir.iterdir()):
            if not folder.is_dir():
                continue
            if folder.name.startswith("build-"):
                result.append(self._build_entry("codec-build", folder, busy))
                continue
            try:
                identity = json.loads((folder / "ready.json").read_text(encoding="utf-8"))["identity"]
            except (OSError, ValueError, KeyError, TypeError):
                identity = None
            detail = (f"{identity.get('gpu', '?')} · TensorRT {identity.get('trt', '?')}"
                      if identity else _rel(folder))
            if active_dir is not None and folder.resolve() == active_dir:
                status, note = "in_use", "TensorRT で音声の復元に使っています"
            elif identity is not None and identity == current:
                status, note = "unused", "TensorRT を使うときに再利用します（消すと次回変換し直し）"
            else:
                status, note = "stale", "プログラムや GPU 環境が変わったため、もう使われません"
            result.append(self._entry("codec", folder, "TensorRT codec plan（全モデル共通）", detail,
                                      status, status != "in_use" and not busy, note))
        return result

    def _build_entry(self, kind: str, folder: Path, busy: bool) -> dict:
        return self._entry(kind, folder, "TensorRT 変換の作業フォルダ（失敗・中断の残り）",
                           _rel(folder), "stale", not busy,
                           "変換中は消せません" if busy else "変換が終わると通常は自動で消えます。残っているものは不要です")

    def _legacy_entries(self) -> list[dict]:
        if not self.legacy_plan.is_file():
            return []
        return [self._entry("legacy", self.legacy_plan, "旧セットアップの TensorRT plan",
                            _rel(self.legacy_plan), "stale", True,
                            "エディタでは使いません（旧 CLI の run.bat などの TensorRT 用）")]

    # ------------------------------------------------------ plan -> model
    def _identify_in_background(self):
        """Hash known checkpoints so plans can name the model they belong to."""
        with self._identify_lock:
            if self._identifying:
                return
            self._identifying = True

        def run():
            try:
                from trt_cache import known_model_digests, model_digest
                seen = set(known_model_digests().values())
                candidates = []
                if self.hub_dir.is_dir():
                    candidates += self.hub_dir.glob("models--*/snapshots/*/**/model.safetensors")
                if self.model_dir.is_dir():
                    candidates += self.model_dir.rglob("*.safetensors")
                for path in candidates:
                    if path.name.endswith(".speaker.safetensors") or str(path.resolve()) in seen:
                        continue
                    # Quantized checkpoints cannot be converted to TensorRT.
                    if _safetensors_config(path).get("_quantization"):
                        continue
                    try:
                        model_digest(path)
                    except OSError:
                        continue
            finally:
                with self._identify_lock:
                    self._identifying = False

        threading.Thread(target=run, name="irodori-plan-identify", daemon=True).start()

    # ------------------------------------------------------------ delete
    def delete(self, ids: list[str]) -> int:
        """Delete entries from the latest scan; returns freed bytes."""
        freed = 0
        errors = []
        for entry_id in ids:
            entry = self._last.get(entry_id)
            if entry is None:
                errors.append(f"{entry_id}: 一覧を更新してから削除してください")
                continue
            if not entry["deletable"]:
                errors.append(f"{entry['label']}: 使用中または必須のため削除できません")
                continue
            path = Path(entry["path"]).resolve()
            if BOX.resolve() not in path.parents:
                errors.append(f"{entry['label']}: アプリのフォルダ外なので削除しません")
                continue
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
                    self._remove_empty_model_folder(path.parent)
                freed += entry["bytes"]
                self._last.pop(entry_id, None)
            except OSError as exc:
                errors.append(f"{entry['label']}: {exc}")
        if errors:
            raise ValueError("\n".join(errors) + (f"\n（{freed} バイトは削除済み）" if freed else ""))
        return freed

    def _remove_empty_model_folder(self, folder: Path):
        """A model subfolder without checkpoints only holds its tokenizer etc."""
        model_dir = self.model_dir.resolve()
        folder = folder.resolve()
        if folder == model_dir or model_dir not in folder.parents:
            return
        if not any(folder.rglob("*.safetensors")):
            shutil.rmtree(folder, ignore_errors=True)
