"""Transcription with a custom vocabulary, using NeMo's GPU phrase boosting.

Haas's greedy TDT decoder takes one biasing request per audio (per-stream biasing): a
tree of the words' token sequences adds a bonus to each word's tokens while it is being
read, taken back if the word is not completed.

The bonus does not depend on how sure the model is, so on its own it can write a listed
word where the audio says little, and where it forces a token the audio does not support
the decoder can skip the speech after it. Two guards:

- audio with no speech is not boosted (`has_speech`);
- each audio is also decoded without the words, and a boosted change is kept only where
  it includes a whole listed word (`guarded`).

Boosting changes only the decoding, so both decodes share one encoder pass: with the
guard, transcription takes about 1.3x the time of a plain read, not 2x.
"""

from __future__ import annotations

import contextlib
import copy
import difflib
import hashlib
import json

import numpy as np

from haas_asr.audio import SAMPLE_RATE
from haas_asr.vocabulary import Vocabulary

WORD_START = "▁"  # SentencePiece's word-start piece
CONTEXT_SCORE = 1.0  # bonus per token along a word
DEPTH_SCALING = 1.0  # how the bonus grows deeper into a word
MAX_COMPILED = 32  # compiled vocabularies kept for reuse (NeMo's cache is otherwise unbounded)
SPEECH_RMS = 0.012  # a 30 ms frame this loud (about -38 dBFS) is speech, as in the demos
SPEECH_FRAME = int(0.03 * SAMPLE_RATE)


def transcribe(
    model, audios: list[np.ndarray], vocabulary: Vocabulary, *, batch_size: int, guard: bool
) -> list[str]:
    """The texts of `audios` with `vocabulary`'s words boosted and in their written form.

    `guard=False` keeps every boosted change: for comparing, not faster.
    """
    enable(model)
    texts = []
    with _decoded_twice(model):
        for first in range(0, len(audios), batch_size):
            batch = audios[first : first + batch_size]
            # One request per audio, so each call is one of NeMo's batches.
            pairs = model.transcribe(
                batch,
                batch_size=len(batch),
                verbose=False,
                partial_hypothesis=requests(vocabulary, len(batch), model.tokenizer),
            )
            for audio, (plain, boosted) in zip(batch, pairs, strict=True):
                if boosted == plain or not has_speech(audio):
                    text = plain
                elif guard:
                    text = guarded(plain, boosted, vocabulary)
                else:
                    text = boosted
                texts.append(vocabulary.format(text))
    return texts


@contextlib.contextmanager
def _decoded_twice(model):
    """Make `model.transcribe` decode each batch's encoder output twice, without and with
    the batch's biasing requests, and return ``(plain, boosted)`` text pairs.

    NeMo's transcribe hands the encoder output to ``_transcribe_output_processing``;
    replacing it on this instance for the call reuses the rest of transcribe as is. Like
    NeMo's transcribe itself, this is not for concurrent calls on one model.
    """

    def decode_twice(outputs, trcfg):
        encoded, lengths = outputs.pop("encoded"), outputs.pop("encoded_len")
        decode = model.decoding.rnnt_decoder_predictions_tensor
        plain = decode(encoded, lengths)
        boosted = decode(encoded, lengths, partial_hypotheses=trcfg.partial_hypothesis)
        return [(p.text.strip(), b.text.strip()) for p, b in zip(plain, boosted, strict=True)]

    model._transcribe_output_processing = decode_twice
    try:
        yield
    finally:
        del model._transcribe_output_processing


def enable(model) -> None:
    """Let the model's greedy decoding take a vocabulary per call (once per model).

    Transcripts without a vocabulary are unchanged by this.
    """
    from omegaconf import open_dict

    current = model.cfg.decoding
    if current.strategy == "greedy_batch" and current.greedy.get("enable_per_stream_biasing"):
        return
    # A copy: the model's own config changes only if the switch succeeds.
    decoding = copy.deepcopy(current)
    with open_dict(decoding):
        decoding.strategy = "greedy_batch"  # the label-looping decoder boosting runs in
        decoding.greedy.enable_per_stream_biasing = True
    model.change_decoding_strategy(decoding, verbose=False)


def requests(vocabulary: Vocabulary, count: int, tokenizer) -> list:
    """One biasing request per audio of a call, all for one compiled tree (compiled once
    and reused: compiling it per audio made a batch of 32 three times slower)."""
    from nemo.collections.asr.parts.context_biasing.biasing_multi_model import (
        BiasingRequestItemConfig,
    )
    from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

    # One config for the whole batch: NeMo then adds the compiled tree to the batch's
    # boosting models once, not once per audio.
    config = BiasingRequestItemConfig(
        boosting_model_alpha=vocabulary.alpha,
        cache_key=_compiled(vocabulary.spellings, tokenizer),
    )
    return [Hypothesis.empty_with_biasing_cfg(config) for _ in range(count)]


def _compiled(spellings: tuple[str, ...], tokenizer) -> str:
    """Compile the spellings' tree into NeMo's biasing cache once; its cache key.

    NeMo would tokenize each spelling as SentencePiece does, with a leading word-start
    piece, which a Japanese transcript has only where an utterance begins: a word would
    be boosted only as an utterance's first. So each spelling goes in twice, as it starts
    an utterance and as it appears inside one.
    """
    from nemo.collections.asr.parts.context_biasing import biasing_multi_model
    from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import (
        BoostingTreeModelConfig,
        GPUBoostingTreeModel,
    )
    from nemo.collections.asr.parts.context_biasing.context_graph_universal import (
        ContextGraph,
    )

    # Keyed by the tokenizer object too: the token ids are the model's. A tree stays in
    # NeMo's cache after its model is deleted, until MAX_COMPILED newer ones replace it.
    tree = {"words": spellings, "context": CONTEXT_SCORE, "depth": DEPTH_SCALING}
    digest = hashlib.sha1(json.dumps(tree, ensure_ascii=False).encode()).hexdigest()
    key = f"haas-vocabulary-{id(tokenizer)}-{digest}"
    cache = biasing_multi_model._BIASING_MODEL_CACHE
    if key in cache:
        return key
    contexts, phrases = [], []
    for spelling in spellings:
        for tokens in word_forms(tokenizer, spelling):
            contexts.append(tokens)
            phrases.append(spelling)
    graph = ContextGraph(context_score=CONTEXT_SCORE, depth_scaling=DEPTH_SCALING)
    graph.build(token_ids=contexts, phrases=phrases, scores=[0.0] * len(contexts))
    defaults = BoostingTreeModelConfig()
    compiled = [k for k in cache if k.startswith("haas-vocabulary-")]
    for old in compiled[: max(0, len(compiled) - MAX_COMPILED + 1)]:
        del cache[old]  # oldest first
    cache[key] = GPUBoostingTreeModel.from_context_graph(
        context_graph=graph,
        vocab_size=tokenizer.vocab_size,
        unk_score=defaults.unk_score,
        final_eos_score=defaults.final_eos_score,
        use_triton=None,  # auto: Triton when available
        uniform_weights=defaults.uniform_weights,
    )
    return key


def word_forms(tokenizer, word: str) -> list[list[int]]:
    """The token sequences `word` is boosted as: inside an utterance (without the
    word-start marker) and at its start (with it). The marker is a piece of its own
    (▁ ハ ー ス) or merged into the first one (▁大 和); a merged piece is replaced by the
    same text without it, as one piece or one per character."""
    ids = tokenizer.text_to_ids(word)
    first = tokenizer.ids_to_tokens(ids[:1])[0] if ids else ""
    if not first.startswith(WORD_START):
        return [ids]
    rest = first[len(WORD_START) :]
    inside = _pieces(tokenizer, [rest]) or _pieces(tokenizer, list(rest)) if rest else []
    if inside is None or not inside + ids[1:]:
        return [ids]
    return [inside + ids[1:], ids]


def _pieces(tokenizer, pieces: list[str]) -> list[int] | None:
    """The ids of `pieces`, or None when one is not in the vocabulary."""
    ids = tokenizer.tokens_to_ids(pieces)
    return ids if tokenizer.ids_to_tokens(ids) == pieces else None


def has_speech(audio: np.ndarray, rms: float = SPEECH_RMS) -> bool:
    """Whether any 30 ms frame of `audio` is at least `rms` loud."""
    audio = np.asarray(audio, dtype=np.float32).ravel()
    if not audio.size:
        return False
    padded = np.pad(audio, (0, -audio.size % SPEECH_FRAME))  # the last, partial frame too
    frames = padded.reshape(-1, SPEECH_FRAME)
    return bool((np.sqrt((frames**2).mean(axis=1)) >= rms).any())


def guarded(plain: str, boosted: str, vocabulary: Vocabulary) -> str:
    """`boosted` (decoded with the words) where it agrees with `plain` (decoded without
    them) or includes a whole listed word; elsewhere `plain`. So a word the boost started
    without the audio's support goes back to what was read without it, and speech the
    boost skipped after a word comes back."""
    found = [(start, end) for start, end, _ in vocabulary.find(boosted)]
    matcher = difflib.SequenceMatcher(None, plain, boosted, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal" or (
            tag in ("replace", "insert") and any(s < j2 and e > j1 for s, e in found)
        ):
            out.append(boosted[j1:j2])
        elif tag == "delete" and i2 - i1 <= 1 and any(j1 in (s, e) for s, e in found):
            continue  # a word's own spelling may absorb one ("ハスの" -> "ハース")
        else:
            out.append(plain[i1:i2])
    return "".join(out)
