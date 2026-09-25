"""Build irodori-tts/wrapper/data/cmudict.txt.gz from the upstream CMUdict.

英単語の読み変換（wrapper/english_reading.py）が使う発音辞書を作る。
上流の cmudict.dict（BSD 2-Clause）から、英字だけの見出しの第1発音を残して圧縮する。

    python tools/build_english_dictionary.py            # 固定リビジョンを取得
    python tools/build_english_dictionary.py --source path/to/cmudict.dict
"""
import argparse
import gzip
from pathlib import Path
import re
import urllib.request

REVISION = "74790861f652b15e4ac49015a90074ad62a27690"
URL = f"https://raw.githubusercontent.com/cmusphinx/cmudict/{REVISION}/cmudict.dict"
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "irodori-tts" / "wrapper" / "data" / "cmudict.txt.gz"
ENTRY = re.compile(r"^([a-z]+(?:['.-][a-z]+)*)(\(\d+\))? ([A-Z0-2 ]+?)\s*(?:#.*)?$")


def build(lines):
    words = {}
    for line in lines:
        match = ENTRY.match(line.strip())
        # 別発音 "(2)" は捨て、最初の発音だけを使う。
        if match and not match[2] and match[1] not in words:
            words[match[1]] = match[3]
    return words


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="ローカルの cmudict.dict")
    args = parser.parse_args()
    if args.source:
        text = args.source.read_text(encoding="utf-8")
    else:
        with urllib.request.urlopen(URL, timeout=60) as response:
            text = response.read().decode("utf-8")
    words = build(text.splitlines())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{word} {phones}\n" for word, phones in sorted(words.items()))
    # mtime=0 で毎回同じバイト列にする。
    with open(OUTPUT, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0,
                                                   filename="") as stream:
        stream.write(body.encode("utf-8"))
    print(f"{len(words)} words -> {OUTPUT}")


if __name__ == "__main__":
    main()
