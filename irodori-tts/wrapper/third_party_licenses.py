"""Use the same model/runtime notices as the editor's Help screen."""
import json
from functools import lru_cache
from pathlib import Path

PUBLIC = Path(__file__).resolve().parents[2] / "voicevox-editor" / "public"


@lru_cache(maxsize=1)
def dependency_licenses():
    entries = []
    # セットアップがこの PC の実行環境から作る runtime-licenses.local.json を優先し、
    # 無ければ（セットアップ前のチェックアウトなど）追跡中の runtime-licenses.json を使う。
    runtime = PUBLIC / "runtime-licenses.local.json"
    if not runtime.is_file():
        runtime = PUBLIC / "runtime-licenses.json"
    for path in (PUBLIC / "licenses.json", runtime):
        entries.extend(json.loads(path.read_text(encoding="utf-8")))
    return entries
