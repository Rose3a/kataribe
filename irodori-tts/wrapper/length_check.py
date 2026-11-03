"""長さの測り直し（2 パス合成）: 時間予測が長すぎたときの、末尾の余計な発話とループを防ぐ。

v4.1 Small（RF）の DiT では、層 0 のヘッド 2・4 が「いま読んでいるテキストのトークン」を指すポインタに
なっている（音声フレームとテキスト位置の相関 0.99）。テキストには終わりの印が無いので、時間予測が
長すぎると、読み終えた後の余ったフレームでモデルが無い言葉を話したり、前に戻って読み直したりする。
モデル自身は普段とほぼ同じ速さで読み終えるので、ポインタが最後から 2 番目のトークンに届いた位置が
「本当の長さ」の見積もりになる。

1 回目の合成で最後のステップのポインタを記録し、見積もり × MARGIN が 1 回目の長さより短ければ、
同じシードでその長さで作り直す（長くはしない: 話速を上げて足りないときは何もしない）。
検証（dpo/attn_probe.py、14 文 × 4 本、Haas-ja-v1.0 で採点）: 予測の 1.5 / 2.0 倍の長さでの
「ほぼ正解」41〜93% / 0〜14% → 2 パスで 96〜100%、余計な発話・繰り返しは 0。

ヘッドの位置はモデルの重みで決まるので、構造が v4.1 Small と同じ RF モデル（12 層・20 ヘッド・
幅 1280。その微調整版を含む）にだけ当てる。TensorRT は DiT 全体が 1 つのエンジンで中の注意を
取り出せないので対象外（PyTorch の CUDA / CPU のみ）。
"""
from __future__ import annotations

import dataclasses
import math

import torch

POINTER_LAYER = 0
POINTER_HEADS = (2, 4)
# ポインタが届いてから、最後の語と句点を言い終えるまでの余裕
TAIL_SECONDS = 0.5
# 見積もりに掛ける余裕（1.0 だと語尾の削れが少し増えた）
MARGIN = 1.1
# 1 回目より短くなるのがこの割合未満なら作り直さない（作り直しの時間の割に変わらない）
MIN_SHRINK = 0.05
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

    def __init__(self, model):
        self.attention = model.blocks[POINTER_LAYER].attention
        self.pointer: torch.Tensor | None = None
        self.text_len = 0
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


def estimated_frames(pointer: torch.Tensor, text_len: int, frames_per_step: int,
                     frames_per_second: float) -> int | None:
    """ポインタから見積もった、読み終えるまでの潜在フレーム数（余裕込み）。届いていなければ None。"""
    hit = torch.nonzero(pointer >= text_len - 2)
    if text_len < 3 or hit.numel() == 0:
        return None
    end_steps = int(hit[0, 0]) + 1
    frames = end_steps * frames_per_step + TAIL_SECONDS * frames_per_second
    return int(math.ceil(frames * MARGIN))


def synthesize(runtime, request, log_fn=None, stream_decoder=None):
    """2 パス合成。戻り値は (結果, 情報)。作り直さなかったときは 1 回目の結果をそのまま返す。

    stream_decoder（stream_decode.StreamingDecode）があれば、1 回目は再生側へ流さない
    （作り直すかもしれない音声を先に聞かせないため）。作り直さなかったときは、呼び出し側が
    返した WAV の全体を送る。前のセリフからの続き（prefix）・次のセリフの先読みは、ポインタと
    長さの関係を確かめていないので当てない。
    """
    def log(message: str) -> None:
        if log_fn is not None:
            log_fn(message)

    info = {"length_check": "skipped"}
    if (request.seconds is not None or not supported(runtime.model_cfg)
            or getattr(request, "prefix_latent", None) is not None
            or getattr(request, "lookahead_text", None)):
        return runtime.synthesize(request, log_fn=log_fn), info
    sample_rate = int(runtime.codec.sample_rate)
    hop = int(runtime.codec.model.hop_length)
    patch = int(runtime.model_cfg.latent_patch_size)
    sink = getattr(stream_decoder, "sink", None)
    if sink is not None:
        stream_decoder.end()
    recorder = PointerRecorder(runtime.model)
    try:
        first = runtime.synthesize(request, log_fn=log_fn)
    finally:
        recorder.close()
        if sink is not None:
            stream_decoder.begin(sink)
    if recorder.pointer is None:
        return first, info
    steps = int(recorder.pointer.shape[0])
    first_frames = steps * patch
    frames = estimated_frames(recorder.pointer, recorder.text_len, patch, sample_rate / hop)
    info = {"length_check": "kept", "first_frames": first_frames, "estimated_frames": frames}
    if frames is None or frames >= first_frames * (1 - MIN_SHRINK):
        log(f"[length_check] kept {first_frames} frames (estimate {frames})")
        return first, info
    # シードが無指定（ランダム）でも、1 回目と同じ初期ノイズで作り直す
    second_request = dataclasses.replace(request, seconds=frames * hop / sample_rate,
                                         seed=first.used_seed)
    log(f"[length_check] {first_frames} -> {frames} frames "
        f"({first_frames * hop / sample_rate:.2f}s -> {frames * hop / sample_rate:.2f}s)")
    second = runtime.synthesize(second_request, log_fn=log_fn)
    info.update(length_check="resynthesized", first_seconds=first_frames * hop / sample_rate)
    second.total_to_decode += first.total_to_decode
    return second, info
