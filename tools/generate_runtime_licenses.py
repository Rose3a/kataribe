"""Collect installed wheel notices for Help; runs offline, without importing packages.

Run with .local/venv/Scripts/python.exe. Pass --site-packages for an optional
backend environment. Missing wheel notices are covered by reviewed snapshots
in licenses/, never by substituting a guessed license.

The result depends on this PC (CUDA / CPU / Radeon and the resolved versions),
so it goes to the untracked runtime-licenses.local.json. The tracked
runtime-licenses.json is only the fallback for a checkout without setup.
"""
import argparse
import importlib.metadata as metadata
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "licenses"
PUBLIC = ROOT / "voicevox-editor" / "public"
OVERRIDES = {
    "dacvae": ("Apache-2.0", "dacvae", "Copyright (c) Meta Platforms, Inc. and affiliates."),
    "descript-audiotools": ("MIT", "descript-audiotools", ""),
    "ffmpy": ("MIT", "ffmpy", ""),
    "flatbuffers": ("Apache-2.0", "flatbuffers", ""),
    "sentencepiece": ("Apache-2.0", "sentencepiece", ""),
    "tokenizers": ("Apache-2.0", "tokenizers", ""),
    "sherpa-onnx-core": ("Apache-2.0", "sherpa-onnx", ""),
    "tensorboard-data-server": ("Apache-2.0", "tensorboard", ""),
}
SNAPSHOT_VERSIONS = {
    "dacvae": "1.0.0", "descript-audiotools": "0.7.2",
    "ffmpy": "1.0.0", "flatbuffers": "25.12.19", "sentencepiece": "0.1.99",
    "tokenizers": "0.23.2", "sherpa-onnx-core": "1.13.8",
    "tensorboard-data-server": "0.7.2",
}


def collect(distributions, sources=SOURCES):
    entries = {}
    missing = []
    urls = json.loads((sources / "sources.json").read_text(encoding="utf-8"))
    for dist in distributions:
        name = dist.metadata["Name"]
        key = re.sub(r"[-_.]+", "-", name).lower()
        texts = []
        for file in sorted(dist.files or [], key=str):
            filename = str(file).replace("\\", "/").rsplit("/", 1)[-1]
            if re.match(r"^(licen[cs]|copying|copyright|notice|third.?party)", filename, re.I):
                path = Path(dist.locate_file(file))
                if path.is_file() and path.suffix.lower() not in {".py", ".pyc", ".so", ".dll", ".json"}:
                    texts.append(f"{str(file).replace(chr(92), '/')}\n\n{path.read_text(encoding='utf-8', errors='replace')}")
        declared = dist.metadata.get("License-Expression") or dist.metadata.get("License") or ""
        classifiers = " / ".join(
            c.split(" :: ")[-1] for c in dist.metadata.get_all("Classifier", [])
            if c.startswith("License ::") and not c.endswith("OSI Approved")
        )
        # The legacy License field may be the entire legal text, starting with
        # a copyright line or a separator. Do not use that as the UI label.
        license_name = dist.metadata.get("License-Expression") or classifiers or (
            declared if len(declared.splitlines()) == 1 and len(declared) < 100
            else "ライセンス本文を参照"
        )
        url = dist.metadata.get("Home-page") or ""
        if key in OVERRIDES and dist.version == SNAPSHOT_VERSIONS[key]:
            license_name, snapshot, credit = OVERRIDES[key]
            if not texts:
                texts.append(credit + "\n\n" + (sources / f"{snapshot}.txt").read_text(encoding="utf-8"))
                url = urls[snapshot]
        if not texts and len(declared.splitlines()) > 5:
            texts.append(declared)
        if not texts:
            missing.append(f"{name}=={dist.version}")
            continue
        entry = {"name": name, "version": dist.version, "license": license_name or "See license text", "text": "\n\n--------------------\n\n".join(texts)}
        if url.startswith(("https://", "http://")):
            entry["url"] = url
        entries[(key, dist.version)] = entry
    if missing:
        raise ValueError("License text missing; review before regenerating: " + ", ".join(sorted(missing)))
    return [entries[k] for k in sorted(entries)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-packages", action="append", default=[])
    parser.add_argument("--output", type=Path, default=PUBLIC / "runtime-licenses.local.json")
    args = parser.parse_args()
    distributions = list(metadata.distributions())
    for folder in args.site_packages:
        distributions.extend(metadata.distributions(path=[folder]))
    entries = collect(distributions)
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if python_license.is_file():
        entries.append({"name": "Python", "version": sys.version.split()[0], "license": "PSF-2.0 and bundled third-party licenses", "url": "https://docs.python.org/3/license.html", "text": python_license.read_text(encoding="utf-8")})
    node_license = ROOT / ".local/node/24.11.1/LICENSE"
    if node_license.is_file():
        entries.append({"name": "Node.js", "version": "24.11.1", "license": "MIT and bundled third-party licenses", "url": "https://github.com/nodejs/node/blob/v24.11.1/LICENSE", "text": node_license.read_text(encoding="utf-8")})
    if (ROOT / ".local/bin/uv.exe").is_file():
        entries.append({"name": "uv（セットアップ用）", "version": "0.9.7", "license": "MIT OR Apache-2.0", "url": "https://github.com/astral-sh/uv/tree/0.9.7", "text": "\n\n".join((SOURCES / filename).read_text(encoding="utf-8") for filename in ("uv-license-mit.txt", "uv-license-apache.txt"))})
    # Windows でも LF で書き、本文の CRLF もそろえる（PC による差を出さない）。
    text = json.dumps(entries, ensure_ascii=False, indent=2).replace("\\r\\n", "\\n")
    args.output.write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"Collected {len(entries)} runtime notices: {args.output}")


if __name__ == "__main__":
    main()
