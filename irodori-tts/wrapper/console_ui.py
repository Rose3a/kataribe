"""コンソール表示（rich）。

端末に直接出しているときだけ、色・枠・表・スピナーを使う。ログファイル（bat のリダイレクト）や
パイプに流すときは、従来どおりの 1 行テキストのまま（grep や集計に使えるように）。
rich が入っていなくても動く（requirements-runtime.txt に書いてあり、descript-audiotools 経由でも入る）。
NO_COLOR=1 で色を切る、IRODORI_PLAIN_CONSOLE=1 で常に従来表示にできる。
"""
from __future__ import annotations

import contextlib
import os
import re
import sys
import time
from pathlib import Path

try:
    from rich.console import Console
    from rich.markup import escape
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
except Exception:  # noqa: BLE001 - rich が無い・壊れているときは従来表示
    Console = None  # type: ignore[assignment]

_consoles: dict[bool, "Console"] = {}
_plain = False


def force_plain() -> None:
    """以降は装飾しない（--json のとき。端末でもスピナーや色を出さない）。"""
    global _plain
    _plain = True


def fancy(stderr: bool = False) -> bool:
    """この出力先で装飾してよいか（rich があり、端末に直接出ていて、色を切られていない）。"""
    if Console is None or _plain:
        return False
    if os.environ.get("IRODORI_PLAIN_CONSOLE") == "1" or os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    stream = sys.stderr if stderr else sys.stdout
    try:
        return bool(stream.isatty())
    except Exception:  # noqa: BLE001
        return False


def _console(stderr: bool) -> "Console":
    con = _consoles.get(stderr)
    if con is None:
        con = _consoles[stderr] = Console(stderr=stderr, highlight=False, soft_wrap=True)
    return con


def _utf(stderr: bool = False) -> bool:
    return "utf" in str(getattr(_console(stderr), "encoding", "") or "").lower()


def esc(value) -> str:
    """rich のマークアップとして解釈されないようにする。"""
    return escape(str(value)) if Console is not None else str(value)


def emit(plain: str, markup: str | None = None, *, stderr: bool = False) -> None:
    """装飾できるときは markup、そうでなければ従来の plain をそのまま出す。"""
    if markup is not None and fancy(stderr):
        _console(stderr).print(markup)
        return
    print(plain, file=sys.stderr if stderr else sys.stdout, flush=True)


# ---------------------------------------------------------------- engine log
def stage(message: str) -> None:
    """[irodori] HH:MM:SS stage: ... （エンジンの段階ログ）"""
    stamp = time.strftime("%H:%M:%S")
    plain = f"[irodori] {stamp} stage: {message}"
    if not fancy():
        print(plain, flush=True)
        return
    body = esc(message)
    if message.startswith("[runtime]") and ":" in message:
        head, _, value = message.partition(":")
        body = f"[dim]{esc(head)}:[/] [bold]{esc(value.strip())}[/]"
    elif message.startswith(("TensorRT", "downloading", "downloaded")):
        body = f"[yellow]{body}[/]"
    elif "error" in message.lower() or "failed" in message.lower():
        body = f"[red]{body}[/]"
    _console(False).print(f"[dim]{stamp}[/] [cyan]irodori[/] [dim]›[/] {body}")


def _rtf_style(rtf: float | None) -> str:
    if rtf is None:
        return "white"
    return "green" if rtf < 0.5 else "yellow" if rtf < 1.0 else "red"


def synth_done(plain: str, *, speaker: str, total: float, audio: float | None, rtf: float | None,
               chars: int, steps, backend, model, seed, phases: list[str]) -> None:
    """1 回の合成の結果。plain は従来の 1 行（ログ用）。"""
    if not fancy():
        print(plain, flush=True)
        return
    for line in _synth_lines(speaker, total, audio, rtf, chars, steps, backend, model, seed, phases):
        _console(False).print(line)


def _synth_lines(speaker, total, audio, rtf, chars, steps, backend, model, seed, phases) -> list[str]:
    """合成結果の表示 2 行（rich のマークアップ）。"""
    mark = "✔" if _utf() else "OK"
    audio_text = f"{audio:.2f}s" if audio else "?"
    rtf_text = f"{rtf:.2f}x" if rtf is not None else "-"
    head = (f"[green]{mark}[/] [bold]{esc(speaker)}[/] [dim]{chars}字[/]  "
            f"[bold]{total:.2f}s[/] → 音声 [bold]{audio_text}[/]  "
            f"RTF [bold {_rtf_style(rtf)}]{rtf_text}[/]")
    detail = f"[dim]{esc(backend)} · {esc(model)} · steps={esc(steps)} seed={esc(seed)}"
    if phases:
        detail += "  |  " + " ".join(esc(p) for p in phases)
    return [head, f"  {detail}[/]"]


def http_log(message: str, *, tag: str = "voicevox", stderr: bool = True) -> None:
    """HTTP サーバーのアクセスログ 1 行（'METHOD /path HTTP/1.1' 200 -）。"""
    plain = f"[{tag}] {message}"
    if not fancy(stderr):
        sys.stderr.write(plain + "\n")
        return
    style = "white"
    parts = message.split('" ')
    status = parts[1].split()[0] if len(parts) > 1 and parts[1].split() else ""
    request = message.split('"')[1] if message.count('"') >= 2 else message
    method = request.split(" ", 1)[0]
    if status.isdigit():
        code = int(status)
        style = "green" if code < 300 else "cyan" if code < 400 else "yellow" if code < 500 else "red"
    dim = method in ("OPTIONS", "GET")
    line = f"[dim]{esc(tag)}[/] [{'dim' if dim else 'bold'}]{esc(request)}[/] [{style}]{esc(status)}[/]"
    if not status:
        line = f"[dim]{esc(tag)}[/] {esc(message)}"
    _console(stderr).print(line)


def notice(tag: str, message: str, *, stderr: bool = False, style: str = "") -> None:
    """[tag] message 形式の一言ログ。"""
    plain = f"[{tag}] {message}"
    markup = f"[dim]{esc(tag)}[/] [{style}]{esc(message)}[/]" if style else f"[dim]{esc(tag)}[/] {esc(message)}"
    emit(plain, markup, stderr=stderr)


# ---------------------------------------------------------------- banner / tables
def banner(title: str, rows: list[tuple[str, str]], plain: str, *, stderr: bool = False) -> None:
    """起動時の案内。装飾できないときは plain を 1 行で出す。"""
    if not fancy(stderr):
        print(plain, file=sys.stderr if stderr else sys.stdout, flush=True)
        return
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column()
    for key, value in rows:
        table.add_row(esc(key), esc(value))
    _console(stderr).print(Panel(table, title=f"[bold]{esc(title)}[/]", title_align="left",
                                 border_style="cyan", box=box.ROUNDED, padding=(0, 1), expand=False))


def speaker_table(roots: list, speakers: list[str], default: str | None) -> None:
    """--list-speakers の表示。"""
    if not fancy():
        print("search roots:")
        for d in roots:
            print(f"  - {d}")
        print("available speakers:")
        for name in speakers:
            print(f"  - {name}{' (default)' if name == default else ''}")
        return
    con = _console(False)
    con.print("[dim]search roots[/]")
    for d in roots:
        con.print(f"  [dim]{esc(d)}[/]")
    table = Table(box=box.SIMPLE_HEAD, header_style="bold cyan", title="available speakers",
                  title_justify="left")
    table.add_column("#", justify="right", style="dim")
    table.add_column("speaker")
    table.add_column("", style="green")
    for index, name in enumerate(speakers, 1):
        table.add_row(str(index), esc(name), "default" if name == default else "")
    con.print(table)


def help_page(prog: str, description: str, groups: list, examples: list, environment: list) -> None:
    """--help のページ。groups=[(見出し, [(オプション, 説明)])]、examples=[(説明, コマンド)]、environment=[(名前, 説明)]。"""
    con = _console(False)
    con.print(Panel(f"[bold]{esc(prog)}[/]  [dim]—[/]  {esc(description)}", border_style="cyan",
                    box=box.ROUNDED, padding=(0, 1), expand=False))
    # オプション名の列の幅は全グループでそろえる
    flag_width = max((len(flags) for _, rows in groups for flags, _ in rows), default=10)
    for title, rows in groups:
        table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
        table.add_column(style="bold cyan", no_wrap=True, min_width=flag_width, max_width=flag_width)
        table.add_column(overflow="fold")
        for flags, text in rows:
            table.add_row(esc(flags), esc(text))
        con.print(f"\n[bold]{esc(title)}[/]")
        con.print(table)
    con.print("\n[bold]使用例[/]")
    for text, cmd in examples:
        con.print(f"  [dim]# {esc(text)}[/]\n  [green]{esc(cmd)}[/]")
    con.print("\n[bold]環境変数[/]")
    table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    table.add_column(style="bold cyan", no_wrap=False)
    table.add_column(overflow="fold")
    for name, text in environment:
        table.add_row(esc(name), esc(text))
    con.print(table)


def _file_uri(path) -> str:
    return Path(str(path)).resolve().as_uri()


def play_wav(path, wait: bool = False) -> bool:
    """WAV を鳴らす（Windows 標準の winsound。他の OS では False）。wait=False なら鳴らしたまま戻る。"""
    try:
        import winsound
    except ImportError:
        return False
    flags = winsound.SND_FILENAME | (0 if wait else winsound.SND_ASYNC)
    try:
        winsound.PlaySound(str(path), flags)
    except RuntimeError:
        return False
    return True


def stop_wav() -> None:
    try:
        import winsound
        winsound.PlaySound(None, winsound.SND_PURGE)
    except Exception:  # noqa: BLE001 - 鳴っていなくても、他の OS でも黙って戻る
        pass


MINT, PINK, SKY, DIM = "#8fd9b6", "#ff9ec7", "#8ecbff", "#7f948a"


def _clip(text: str, width: int = 40) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def run_header(backend: str, speaker: str, steps, seed, count: int) -> None:
    """tts_cli の最初の案内（端末のときだけ）。"""
    if not fancy():
        return
    con = _console(False)
    con.print(f"[bold {MINT}]kataribe[/] [{DIM}]tts_cli · {esc(backend)}[/]")
    con.print(f"  [{DIM}]話者[/] [{PINK}]{esc(speaker)}[/]  [{DIM}]steps[/] {esc(steps)}  "
              f"[{DIM}]seed[/] {esc(seed)}  [{DIM}]{count} セリフ[/]")
    con.print()


def line_done(index: int, total: int, text: str, audio: float, wall: float) -> None:
    """1 セリフ分の結果（端末のときだけ）。"""
    if not fancy():
        return
    mark = "✔" if _utf() else "OK"
    rtf = wall / audio if audio else None
    con = _console(False)
    con.print(f"[{MINT}]{mark}[/] [{DIM}]{index}/{total}[/] [bold]{esc(_clip(text))}[/]")
    rtf_text = f" · RTF [{_rtf_style(rtf)}]{rtf:.2f}x[/]" if rtf is not None else ""
    con.print(f"      [{SKY}]{audio:.2f}s[/] [{DIM}]の音声 · 生成[/] [{SKY}]{wall:.2f}s[/][{DIM}]{rtf_text}[/]")


def run_summary(count: int, audio: float, wall: float, out_dir) -> None:
    """全部終わったあとの要約（端末のときだけ）。保存先はリンクになる。"""
    if not fancy():
        return
    mark = "✔" if _utf() else "OK"
    folder = Path(str(out_dir)).resolve()
    con = _console(False)
    con.print()
    con.print(f"[bold {MINT}]{mark}[/] [bold]{count} 件できました[/] [{DIM}]· 合計 {audio:.2f}s の音声 · {wall:.2f}s[/]")
    con.print(f"  [{DIM}]保存先[/] [link={folder.as_uri()}][underline {SKY}]{esc(folder)}[/][/]  "
              f"[{DIM}]Ctrl+クリックで開く[/]")


def error_card(message: str, suggestions: list[str] | None = None, hint: str | None = None) -> None:
    """うまくいかなかったとき。端末では顔文字つきの優しい表示、そうでなければ 1 行。"""
    if not fancy(True):
        print(f"error: {message}", file=sys.stderr, flush=True)
        return
    con = _console(True)
    face = "(；ω；)" if _utf(True) else ""
    con.print()
    con.print(f"[bold {PINK}]{esc(f'{face} {message}'.strip())}[/]")
    if suggestions:
        names = f" [{DIM}]/[/] ".join(f"[bold {MINT}]{esc(name)}[/]" for name in suggestions)
        con.print(f"  [{DIM}]もしかして[/] {names}")
    if hint:
        con.print(f"  [{DIM}]{esc(hint)}[/]")


@contextlib.contextmanager
def working(message: str):
    """時間のかかる処理のあいだ、スピナーを出す（装飾できないときは何も出さない。出力を読むスクリプトを乱さないため）。"""
    if not fancy():
        yield
        return
    with _console(False).status(f"[cyan]{esc(message)}[/]", spinner="dots"):
        yield


# ---------------------------------------------------------------- log viewer
# open_browser.bat のコンソールは、エンジンと UI がファイルに書いた素のログを tools/browser_session.py が
# 流している。エンジン側は何も変えず（負荷なし）、流す側がこの関数で 1 行ずつ見やすくする。
_SYNTH = re.compile(r"^synth done: (?P<kv>.*?)(?: \| phases: (?P<phases>.*))?$")
_STAGE = re.compile(r"^(\d\d:\d\d:\d\d) stage: (.*)$")
_TOKENS = re.compile(r"^tokens \((\d+)\): (.*)$")


def _main_phases(text: str | None) -> list[str]:
    """合成の内訳のうち、時間のかかった段階だけ（0.01 秒以上。total_to_decode は重複なので除く）。"""
    kept = []
    # 段階名には空白を含むもの（start synthesize / done synthesize）がある
    for name, value in re.findall(r"([A-Za-z_][A-Za-z_ ]*?)=(\S+)", text or ""):
        name = name.strip()
        if name in ("total_to_decode", "start synthesize", "done synthesize"):
            continue
        try:
            if float(value.rstrip("s")) >= 0.01:
                kept.append(f"{name}={value}")
        except ValueError:
            kept.append(f"{name}={value}")
    return kept


def log_line_markup(label: str, line: str) -> str:
    """素のログ 1 行を rich のマークアップにする（複数行になることがある）。

    見やすさのため、合成の結果の行にまとめられる細かい行（[runtime] の各段階・info:）と、
    成功したポーリング（GET・OPTIONS・audio_query の 2xx）は空文字にして出さない。
    どれもログファイルには残る。失敗（2xx 以外）は必ず出す。"""
    line = line.rstrip()
    tag = f"[{DIM}]{esc(f'{label:<7}')}[/]"
    low = line.lower()
    if label != "engine":
        style = "red" if "error" in low or "failed" in low else "green" if "ready in" in low else DIM
        return f"{tag}[{style}]{esc(line)}[/]"
    if line.startswith("[irodori] "):
        body = line[len("[irodori] "):]
        m = _STAGE.match(body)
        if m:
            stamp, message = m.groups()
            if message.startswith(("[runtime]", "info:")) and "error" not in message.lower():
                return ""
            if message.startswith(("TensorRT", "downloading", "downloaded")):
                text = f"[yellow]{esc(message)}[/]"
            elif "error" in message.lower() or "failed" in message.lower():
                text = f"[red]{esc(message)}[/]"
            else:
                text = esc(message)
            return f"{tag}[{DIM}]{stamp}[/] {text}"
        m = _SYNTH.match(body)
        if m:
            kv = dict(re.findall(r"(\w+)=(\S+)", m.group("kv")))
            try:
                total = float(kv.get("total", "0").rstrip("s"))
                audio_text = kv.get("audio", "")
                audio = float(audio_text.rstrip("s")) if audio_text not in ("", "unknown") else None
                rtf = float(kv["rtf"].rstrip("x")) if kv.get("rtf", "-") != "-" else None
                chars = int(kv.get("text", "0chars").removesuffix("chars"))
            except ValueError:
                return f"{tag}{esc(body)}"
            phases = _main_phases(m.group("phases"))
            lines = _synth_lines(kv.get("speaker", "?"), total, audio, rtf, chars, kv.get("steps"),
                                 kv.get("backend"), kv.get("model"), kv.get("seed"), phases)
            return "\n".join([f"{tag}{lines[0]}", f"{'':<7}{lines[1]}"])
        m = _TOKENS.match(body)
        if m:
            pieces = f"[{DIM}]|[/]".join(esc(p) for p in m.group(2).split("|"))
            return f"{tag}[{DIM}]読み {m.group(1)} トークン[/]  {pieces}"
        return f"{tag}[{DIM}]{esc(body)}[/]"
    if line.startswith("[voicevox] "):
        body = line[len("[voicevox] "):]
        if body.startswith('"') and '" ' in body:
            request = body.split('"')[1]
            status = body.rsplit('" ', 1)[1].split()[0]
            method, _, target = request.partition(" ")
            path = target.split(" ")[0].split("?")[0]
            style = "green" if status.startswith("2") else "cyan" if status.startswith("3") \
                else "yellow" if status.startswith("4") else "red"
            quiet = method in ("GET", "OPTIONS") or path.startswith("/audio_query")
            if quiet and status.startswith("2"):
                return ""
            if method == "OPTIONS" and status == "204":
                return ""
            name = f"[{DIM}]{esc(method)} {esc(path)}[/]" if quiet else f"[bold]{esc(method)} {esc(path)}[/]"
            return f"{tag}{name} [{style}]{esc(status)}[/]"
        if body.startswith("[blockwise]"):
            return f"{tag}[#b9a7ff]前のセリフから続けて生成[/] [{DIM}]{esc(body[len('[blockwise]'):].strip())}[/]"
        if body.startswith(("request rejected", "request failed", "stream failed")):
            return f"{tag}[bold red]{esc(body)}[/]"
        return f"{tag}[{DIM}]{esc(body)}[/]"
    if low.startswith(("traceback", "error")) or line.startswith(("  File ", "    ")) and "Error" in line:
        return f"{tag}[red]{esc(line)}[/]"
    if "warning" in low or line.startswith("W"):
        return f"{tag}[yellow]{esc(line)}[/]"
    return f"{tag}[{DIM}]{esc(line)}[/]"


def print_markup(markup: str) -> None:
    """マークアップをそのまま 1 回出す（log_line_markup の結果を流す用）。"""
    _console(False).print(markup)
