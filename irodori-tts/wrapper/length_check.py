"""長さの測り直し（2 パス合成）: 時間予測が長すぎたときの、末尾の余計な発話とループを防ぐ。

v4.1 Small（RF）の DiT では、層 0 のヘッド 2・4 が「いま読んでいるテキストのトークン」を指すポインタに
なっている（音声フレームとテキスト位置の相関 0.99）。テキストには終わりの印が無いので、時間予測が
長すぎると、読み終えた後の余ったフレームでモデルが無い言葉を話したり、前に戻って読み直したりする。
モデル自身は普段とほぼ同じ速さで読み終えるので、ポインタが最後から 2 番目のトークンに届いた位置が
「本当の長さ」の見積もりになる。

ポインタは文の途中でもよく文末の「。」を指す（「、」の間など。長い文では何十フレームも）。最初に届いた
位置をそのまま使うと、文の途中で読み終えたとみなして縮めすぎる（早口になる・語尾が欠ける）ので、
ポインタに前へ戻らない読み進みの経路を当てはめ（reading_end）、文末の記号を指すフレームは弱い証拠
（読み終えていない状態に小さな一定の費用）として、経路が最後の語に届いた位置を使う。
検証（work/length_check_trt、JSUT・CV3 の 300 文 × 話速 3 種）は analyze.py・weights.py・judge.py。

経路が届くのは最後の語の読み始めなので、そこから最後の語を言い終えるまでの時間として、時間予測が
その語に出したフレーム数を足す（「ければなりません」のような長い語で、届いた時点＋一定の秒数だと縮めすぎて
早口になる）。最後の語が長いトークンでも、途中で最後の語へ飛ぶスパイクがあっても、文の中身（文頭と文末の
記号を除く）の予測の FLOOR 倍より短くはしない。
検証（work/length_check_trt の guard_*.py、JSUT 500・本番経路の 300 文・余計な声が出やすい 61 候補 × 8 シード
を Blockwise v2 と本家で）: 作り直し JSUT 36→7%・本番経路 26→2%、OFF より悪くなる文 9→0、早口 2→0
（本番経路の短文では 34→0）、余計な声（5 カナ以上）は本家でも 49→0 のまま。

1 回目の合成で最後のステップのポインタを記録し、見積もりが 1 回目の長さより RESYNTH_SHRINK 以上短ければ、
同じシードでその長さで作り直す（長くはしない: 話速を上げて足りないときは何もしない）。
検証（dpo/attn_probe.py、14 文 × 4 本、Haas-ja-v1.0 で採点）: 予測の 1.5 / 2.0 倍の長さでの
「ほぼ正解」41〜93% / 0〜14% → 2 パスで 96〜100%、余計な発話・繰り返しは 0。

ヘッドの位置はモデルの重みで決まるので、構造が v4.1 Small と同じ RF モデル（12 層・20 ヘッド・
幅 1280。その微調整版を含む）にだけ当てる。TensorRT は、plan がポインタを 2 つ目の出力に出すもの
（bf16-fallback/adaln_builder.py の adaLN 版、BF16 と W8A8）だけ。
"""
from __future__ import annotations

import dataclasses
import math
import os
import secrets

import numpy as np
import torch

POINTER_LAYER = 0
POINTER_HEADS = (2, 4)
# ポインタが届いてから、最後の語と句点を言い終えるまでの余裕（時間予測のトークンごとの値が無いとき）
TAIL_SECONDS = 0.5
# 時間予測のトークンごとの値があるとき: 届いた時点 + 最後の語の予測フレーム数 + この秒数
LAST_WORD_TAIL_SECONDS = 0.3
# 見積もりは、文の中身（文頭の BOS と文末の記号を除く）の予測のこの倍より短くしない（約 9 カナ/秒）
FLOOR = 0.8
# 見積もりに掛ける余裕（1.0 だと語尾の削れが少し増えた）
MARGIN = 1.1
# 見積もりが 1 回目よりこの割合以上短いときだけ作り直す。作り直しは読みが少し変わる（良くも悪くも）ので、
# 少し縮むだけの文は 1 回目のまま返す（JSUT 500 で 5% なら作り直し 15%・10% なら 7%、悪くなる文 1→0）
RESYNTH_SHRINK = 0.10
# 作り直しが読み終えず長くするとき、1 回目よりこの割合未満しか短くならないなら 1 回目を返す
MIN_SHRINK = 0.05
# 読み進みの経路で、読み終えた後の状態に 1 フレームいる費用（トークン 1 個ずれと同じ）。
# 文の途中で一瞬最後の語に飛んでも、その後の読み進みと合わないので経路はそこで終わらない
ABSORB_COST = 1.0
# 1 フレームでポインタが経路から外れる費用の上限（トークンの数）。上限が無いと、長い文の途中で最後の語へ
# 飛ぶスパイク（0.4 秒）の費用が「そこで読み終えた」とみなす費用を上回り、半分以下に縮めることがあった
DIST_CAP = 3.0
# 経路がトークンを 1 つ飛ばすごとの費用（1 つ先へ進むのは無料）。スパイクで最後の語へ一気に跳ぶのを防ぐ
SKIP_COST = 2.0
# 文がこれで終わるなら、最後のトークンは記号とみなす。それを指すフレームは、最後の語より前の状態に
# PUNCT_COST（距離によらない一定の費用）、最後の語以降の状態に 0 を足す。文中の「、」の間や、複数の文の
# 途中の「。」と取り違えて文末を指すことが多いので、距離（最大でトークン数）で数えると早く終わったと
# みなしてしまう。0 にすると、2 回目で最後の語をほとんど指さずに「。」へ移ったとき読み終えたと分からない
FINAL_PUNCT = "。．.！!？?…‥、,〜~」』）)"
PUNCT_COST = 0.25
# 作り直しが読み終えなかったとき、長さをこの倍にして作り直す回数（1 回目の長さに届いたらやめて 1 回目を返す）
RETRIES = 1
RETRY_GROWTH = 1.25
# 1 回目をこの回数目の DiT の呼び出しで判断して止める（None なら最後のステップで判断する）。
# 途中のポインタからの見積もりは最後のステップと ±15% ずれる（k=12 でも 1 割が 3% 以上短い）ので使わない
EARLY_CALL = None
# 1 回目を最後のステップ（DiT の最後の呼び出し）で判断し、作り直すならデコードせずに止める
STOP_AT_LAST_CALL = True
# 遅い話速では、1 回目の前に PREVIEW_STEPS ステップだけの下見（デコードなし）で長さを決め、縮めるなら
# 1 回目を省いて作り直す。下見は短く出やすいので見積もりを PREVIEW_GROWTH 倍する。None なら下見しない。
# 試験中（遅い話速 600 本で 4 ステップ: 時間 −27%、長さは今の方式の −12〜+10%）。IRODORI_LENGTH_PREVIEW_STEPS=4 で有効
PREVIEW_STEPS = int(os.environ["IRODORI_LENGTH_PREVIEW_STEPS"]) if os.environ.get("IRODORI_LENGTH_PREVIEW_STEPS") else None
PREVIEW_GROWTH = 1.05
PREVIEW_MIN_SCALE = 1.0
LENGTH_CHECKS = ("on", "off")


def supported(model_cfg) -> bool:
    """ポインタのヘッドを確かめたモデルと同じ構造か。"""
    return (getattr(model_cfg, "flow_parameterization", "rf_velocity") != "meanflow"
            and model_cfg.num_layers == 12 and model_cfg.num_heads == 20
            and model_cfg.model_dim == 1280)


class PointerRecorder:
    """層 0 の注意の入力を横取りし、cond の 1 行目について、各音声フレームがどのテキスト位置を
    いちばん強く見ているか（ヘッドごとの argmax の大きい方）を残す。呼ばれるたびに上書きするので、
    合成が終わったときには最後のステップの値が残る。"""

    def __init__(self, model, on_call=None):
        self.attention = model.blocks[POINTER_LAYER].attention
        self.pointer: torch.Tensor | None = None
        self.text_len = 0
        self.calls = 0
        # 呼び出しごとに on_call(self)。例外を投げると合成がそこで止まる（_Stop）
        self.on_call = on_call
        self.handle = self.attention.register_forward_pre_hook(self._hook, with_kwargs=True)

    def close(self) -> None:
        self.handle.remove()

    @torch.no_grad()
    def _hook(self, mod, args, kwargs):
        x = kwargs["x"][:1]
        seq_len = x.shape[1]
        q = mod.wq(x).reshape(1, seq_len, mod.heads, mod.head_dim)
        q = mod._apply_rotary_half(mod.q_norm(q), kwargs["freqs_cis"][:seq_len])
        context_kv = kwargs.get("context_kv")
        if context_kv is not None:
            k_text = context_kv[0][:1]
        else:
            text = kwargs["text_context"][:1]
            k_text = mod.k_norm(mod.wk_text(text).reshape(1, text.shape[1], mod.heads, mod.head_dim))
        mask = kwargs.get("text_mask")
        valid = (mask[0].bool() if mask is not None
                 else torch.ones(k_text.shape[1], dtype=torch.bool, device=x.device))
        heads = list(POINTER_HEADS)
        # キーのテキスト部分にだけ argmax を取るので、softmax の分母（自分・話者のキー）は要らない
        logits = torch.einsum("bshd,bthd->bhst", q[:, :, heads].float(), k_text[:, :, heads].float())
        logits = logits.masked_fill(~valid[None, None, None, :], float("-inf"))
        self.pointer = logits.argmax(-1).amax(1)[0]  # (S,)
        self.text_len = int(valid.sum())
        self.calls += 1
        if self.on_call is not None:
            self.on_call(self)


class EnginePointer:
    """TensorRT の plan が 2 つ目の出力に出すポインタ（adaln_builder.py の pointer）を読む。

    PointerRecorder と同じ形（pointer・text_len・calls・on_call・close）。plan は呼ばれるたびに上書きする
    ので、合成が終わったときには最後のステップの値が残る（run_engine.Engine.last_pointer）。
    run_engine.Adapter が呼び出しのたびに engine.on_pointer() を呼ぶ。
    """

    def __init__(self, engine, on_call=None):
        self.engine = engine
        self.calls = 0
        self.on_call = on_call
        engine.last_pointer = None
        engine.on_pointer = self._notify

    def _notify(self) -> None:
        self.calls += 1
        if self.on_call is not None:
            self.on_call(self)

    @staticmethod
    def available(engine) -> bool:
        return getattr(engine, "pointer_name", None) is not None

    @property
    def pointer(self):
        return self.engine.last_pointer

    @property
    def text_len(self) -> int:
        mask = self.engine.last_text_mask
        return int(mask[0].sum()) if mask is not None else 0

    def close(self) -> None:
        self.engine.on_pointer = None


def ends_with_punct(text: str | None) -> bool:
    return bool(text) and text.rstrip()[-1:] in FINAL_PUNCT


def reading_end(pointer, text_len: int, final_punct: bool = False) -> int | None:
    """ポインタに読み進みの経路を当てはめ、経路が最後から 2 番目のトークンに届いた最初のステップ。

    経路の状態は読んでいるトークン 0..L-1（前に戻らない）と、読み終えた後（-1）。費用は 1 ステップごとに
    min(|ポインタ - 状態|, DIST_CAP)、読み終えた後は ABSORB_COST（何を指していても同じ）で、読み終えた後へは
    L-2 以上のトークンからだけ入れる。経路が k 個先のトークンへ進むと SKIP_COST × (k - 1)（始まりも 0 から
    数える）。最小の費用の経路を DP で求める（L × ステップ数、長い文で 2〜5 ms）。
    final_punct なら最後のトークンは文末の記号で、それを指すフレームの費用は PUNCT_COST か 0（上の説明）。
    経路が最後まで L-2 に届かなければ None。
    """
    if isinstance(pointer, torch.Tensor):
        pointer = pointer.detach().cpu().numpy()
    last = text_len - 2
    p = np.clip(np.asarray(pointer, dtype=np.float32), 0, text_len - 1)
    steps = len(p)
    if text_len < 3 or steps == 0:
        return None
    pos = np.arange(text_len, dtype=np.float32)
    flat = np.where(pos >= last, 0.0, PUNCT_COST).astype(np.float32)

    def row(v):
        return flat if final_punct and v == text_len - 1 else np.minimum(np.abs(v - pos), DIST_CAP)

    cost = np.empty((steps, text_len), dtype=np.float32)
    done = np.empty(steps, dtype=np.float32)
    cost[0] = row(p[0]) + SKIP_COST * pos
    done[0] = np.inf
    # via[s]: s より 2 つ以上前の状態から飛んで来るときの最小の費用
    via = np.empty(text_len, dtype=np.float32)
    via[0] = np.inf
    for t in range(1, steps):
        prev = cost[t - 1]
        via[1:] = np.minimum.accumulate(prev - SKIP_COST * pos)[:-1] + SKIP_COST * (pos[1:] - 1)
        cost[t] = row(p[t]) + np.minimum(prev, via)
        done[t] = ABSORB_COST + min(done[t - 1], prev[last:].min())
    # 後ろからたどる。経路は戻らないので、L-2 以上（か読み終えた後）の区間は末尾に 1 つだけ
    state = -1 if done[-1] <= cost[-1].min() else int(cost[-1].argmin())
    end = None
    for t in range(steps - 1, -1, -1):
        if state != -1 and state < last:
            break
        end = t
        if t == 0:
            break
        if state == -1:
            tail = cost[t - 1, last:]
            state = -1 if done[t - 1] <= tail.min() else last + int(tail.argmin())
        else:
            back = cost[t - 1, :state + 1].copy()
            if state >= 1:
                back[:state] += SKIP_COST * (state - 1 - pos[:state])
            state = int(back.argmin())
    return end


def duration_tokens(runtime, text_len: int) -> dict | None:
    """ランタイムが残した時間予測のトークンごとの値（inference_runtime._keep_duration_tokens）。
    ポインタのテキストとトークンの数が合わなければ None（見積もりは TAIL_SECONDS に戻る）。"""
    tokens = getattr(runtime, "last_duration_tokens", None)
    if not tokens or len(tokens.get("frames") or ()) != text_len:
        return None
    return tokens


def estimated_frames(pointer: torch.Tensor, text_len: int, frames_per_step: int,
                     frames_per_second: float, text: str | None = None,
                     tokens: dict | None = None, lead: int = 0) -> int | None:
    """ポインタから見積もった、読み終えるまでの潜在フレーム数（余裕込み）。届いていなければ None。
    text（合成する文）が文末の記号で終わるなら、その記号を指すフレームは使わない。
    tokens（duration_tokens）があれば、届いた時点に最後の語の予測フレーム数（記号で終わらない文は最後の 2 つ）
    と LAST_WORD_TAIL_SECONDS を足し、文の中身の予測の FLOOR 倍（＋続きの行の先頭の余白 lead）を下限にする。
    無ければ届いた時点 + TAIL_SECONDS。"""
    punct = ends_with_punct(text)
    end = reading_end(pointer, text_len, punct)
    if end is None:
        return None
    end_frames = (end + 1) * frames_per_step
    if tokens is None:
        return int(math.ceil((end_frames + TAIL_SECONDS * frames_per_second) * MARGIN))
    tf = tokens["frames"]
    remain = tf[text_len - 2] if punct else tf[text_len - 2] + tf[text_len - 1]
    frames = int(math.ceil((end_frames + remain + LAST_WORD_TAIL_SECONDS * frames_per_second) * MARGIN))
    return max(frames, lead + int(math.ceil(FLOOR * tokens["content_frames"])))


def _preview(runtime, request, watch, patch, fps, log_fn, lead=0):
    """遅い話速（duration_scale > PREVIEW_MIN_SCALE）で、PREVIEW_STEPS ステップだけの下見のポインタから長さを
    見積もる。(作り直す長さ, 下見の長さ)。縮めるほどでなければ None（いつもの 1 回目に進む）。"""
    if (not PREVIEW_STEPS or float(getattr(request, "duration_scale", 1.0) or 1.0) <= PREVIEW_MIN_SCALE
            or int(getattr(request, "num_steps", 0) or 0) <= PREVIEW_STEPS):
        return None

    def look(rec):
        if rec.calls == PREVIEW_STEPS:
            raise _Stop(estimated_frames(rec.pointer, rec.text_len, patch, fps, request.text,
                                         duration_tokens(runtime, rec.text_len), lead),
                        int(rec.pointer.shape[0]) * patch)

    rec = watch(look)
    found = None
    try:
        runtime.synthesize(dataclasses.replace(request, num_steps=PREVIEW_STEPS), log_fn=log_fn)
    except _Stop as stop:
        found = (stop.frames, stop.first_frames)
    finally:
        rec.close()
    if found is None or found[0] is None:
        return None
    frames = int(math.ceil(found[0] * PREVIEW_GROWTH))
    return (frames, found[1]) if frames < found[1] * (1 - RESYNTH_SHRINK) else None


class _Stop(Exception):
    """ポインタを見て合成を途中で止める（PointerRecorder / EnginePointer の on_call から投げる）。"""

    def __init__(self, frames=None, first_frames=None):
        super().__init__("length_check stop")
        self.frames, self.first_frames = frames, first_frames


def synthesize(runtime, request, log_fn=None, stream_decoder=None, engine=None):
    """2 パス合成。戻り値は (結果, 情報)。作り直さなかったときは 1 回目の結果をそのまま返す。

    stream_decoder（stream_decode.StreamingDecode）があれば、1 回目は再生側へ流さない
    （作り直すかもしれない音声を先に聞かせないため）。作り直さなかったときは、呼び出し側が
    返した WAV の全体を送る。次のセリフの先読みは、この行の部分だけを切り出すので当てない。
    前のセリフからの続き（prefix）では、ランタイムが長さに先頭の余白（prefix_lead_frames）を足すので、
    ポインタ（余白込み）から出した見積もりから余白を引いた長さで作り直す。
    engine（TensorRT の Engine）を渡すと、PyTorch の注意の代わりに plan のポインタ出力を使う
    （ポインタ付きの plan でないときは何もしない）。

    EARLY_CALL 回目か最後（ステップ数）の DiT の呼び出しで長すぎると分かれば、1 回目をそこで止めて
    （デコードしない）作り直す。
    作り直しの最後の呼び出しでポインタが読み終えていなければ（語尾が切れる）、デコードの前に止めて
    1 回目の結果を返す（ストリーミングで切れた音声を流さないため）。
    """
    def log(message: str) -> None:
        if log_fn is not None:
            log_fn(message)

    info = {"length_check": "skipped"}
    if (request.seconds is not None or not supported(runtime.model_cfg)
            or getattr(request, "lookahead_text", None)):
        return runtime.synthesize(request, log_fn=log_fn), info
    if engine is not None and not EnginePointer.available(engine):
        return runtime.synthesize(request, log_fn=log_fn), info
    sample_rate = int(runtime.codec.sample_rate)
    hop = int(runtime.codec.model.hop_length)
    patch = int(runtime.model_cfg.latent_patch_size)
    fps = sample_rate / hop
    # 続きの行: ランタイムが長さ（予測でも seconds でも）に足す先頭の余白。ポインタはこの余白も含む
    lead = (max(0, int(getattr(request, "prefix_lead_frames", 0) or 0))
            if getattr(request, "prefix_latent", None) is not None
            and getattr(runtime, "supports_prefix", False) else 0)

    def watch(on_call):
        return EnginePointer(engine, on_call) if engine is not None else PointerRecorder(runtime.model, on_call)

    def shrink_to(recorder):
        """(見積もり, 1 回目の長さ)。作り直すほど短くならなければ見積もりは None。"""
        first_frames = int(recorder.pointer.shape[0]) * patch
        frames = estimated_frames(recorder.pointer, recorder.text_len, patch, fps, request.text,
                                  duration_tokens(runtime, recorder.text_len), lead)
        if frames is not None and frames >= first_frames * (1 - RESYNTH_SHRINK):
            return None, first_frames, frames
        return frames, first_frames, frames

    # DiT の呼び出しはステップごとに 1 回。最後の呼び出しで判断できれば、作り直すときに 1 回目のデコードを省ける
    steps = int(request.num_steps) if STOP_AT_LAST_CALL and getattr(request, "num_steps", None) else None

    def early(recorder):
        if recorder.calls == EARLY_CALL or recorder.calls == steps:
            frames, first_frames, _ = shrink_to(recorder)
            if frames is not None:
                raise _Stop(frames, first_frames)

    # シードが無指定（ランダム）でも、1 回目と同じ初期ノイズで作り直す（1 回目を途中で止めても分かるよう先に決める）
    seed = int(request.seed) if request.seed is not None else int(secrets.randbits(63))
    first_request = dataclasses.replace(request, seed=seed)
    sink = getattr(stream_decoder, "sink", None)
    if sink is not None:
        stream_decoder.end()
    first = recorder = None
    try:
        preview = _preview(runtime, first_request, watch, patch, fps, log_fn, lead)
        if preview is not None:
            frames, first_frames = preview
            stopped_at = f"preview{PREVIEW_STEPS}"
        else:
            recorder = watch(early)
            try:
                first = runtime.synthesize(first_request, log_fn=log_fn)
            except _Stop as stop:
                frames, first_frames, stopped_at = stop.frames, stop.first_frames, recorder.calls
            finally:
                recorder.close()
    finally:
        if sink is not None:
            stream_decoder.begin(sink)
    if first is not None:
        if recorder.pointer is None:
            return first, info
        frames, first_frames, estimate = shrink_to(recorder)
        stopped_at = None
        info = {"length_check": "kept", "first_frames": first_frames, "estimated_frames": estimate}
        if frames is None:
            log(f"[length_check] kept {first_frames} frames (estimate {estimate})")
            return first, info
    info = {"length_check": "resynthesized", "first_frames": first_frames, "estimated_frames": frames,
            "first_seconds": first_frames * hop / sample_rate}
    if stopped_at is not None:
        info["early_stop_call"] = stopped_at
    log(f"[length_check] {first_frames} -> {frames} frames "
        f"({first_frames * hop / sample_rate:.2f}s -> {frames * hop / sample_rate:.2f}s)"
        + (f", first pass stopped at call {stopped_at}" if stopped_at is not None else ""))
    # 呼び出しの回数は 1 回目と同じ（途中で止めたときはステップ数）
    last_call = recorder.calls if first is not None else int(request.num_steps or 0)
    punct = ends_with_punct(request.text)

    def verify(rec):
        if rec.calls == last_call and reading_end(rec.pointer, rec.text_len, punct) is None:
            raise _Stop()

    second = None
    for attempt in range(1 + RETRIES):
        second_request = dataclasses.replace(request, seconds=max(1, frames - lead) * hop / sample_rate,
                                             seed=seed)
        checker = watch(verify)
        try:
            second = runtime.synthesize(second_request, log_fn=log_fn)
            break
        except _Stop:
            pass
        finally:
            checker.close()
        # 作り直しでは読み終えなかった（語尾が切れる。短くした分、読み方が遅くなることがある）。少し長くしてもう一度
        longer = int(math.ceil(frames * RETRY_GROWTH))
        log(f"[length_check] {frames} frames did not reach the end"
            + (f"; retrying with {longer}" if attempt < RETRIES and longer < first_frames * (1 - MIN_SHRINK) else ""))
        if longer >= first_frames * (1 - MIN_SHRINK):
            break
        frames = longer
        info["estimated_frames"] = frames
    if second is None:
        # 1 回目の長さのまま返す
        info["length_check"] = "fallback"
        if first is None:
            first = runtime.synthesize(first_request, log_fn=log_fn)
        return first, info
    if attempt:
        info["retries"] = attempt
    if first is not None:
        second.total_to_decode += first.total_to_decode
    return second, info
