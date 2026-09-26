"""Speed-ups of the resident TRT path must not change what is synthesized.

- speaker embeddings and WAVs stay in memory (identical bytes/tensors),
- the vectorised tail trim picks the same frame as the old per-frame loop,
- CUDA Graph replays are bit-identical to eager calls,
- the TensorRT codec falls back to PyTorch whenever it cannot be trusted,
- the TensorRT codec cache only reuses verified, matching plans.

The real-model A/B check (bit-identical WAVs, timings) is
``tools/bench_trt_pipeline.py``; see docs/TESTING.md.
"""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

import torch

BOX = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BOX / "irodori-tts" / "wrapper"))
sys.path.insert(0, str(BOX / "runtime" / "trt-lab" / "repo"))

import cuda_graphs  # noqa: E402
import trt_codec  # noqa: E402
import tts_cli  # noqa: E402
import voicevox_engine  # noqa: E402
import vram_trim  # noqa: E402
from irodori_tts import inference_runtime  # noqa: E402
from irodori_tts.inference_runtime import (  # noqa: E402
    InferenceRuntime, SamplingRequest, find_flattening_point, save_wav)

CUDA = torch.cuda.is_available()


def loop_flattening_point(latent, target_value=0.0, window_size=20, std_threshold=0.05,
                          mean_threshold=0.1):
    """The original per-frame implementation, kept as the reference."""
    total_steps = int(latent.shape[0])
    if total_steps <= 0 or window_size <= 0:
        return total_steps
    pad = torch.zeros((window_size, latent.shape[1]), device=latent.device, dtype=latent.dtype)
    padded = torch.cat([latent, pad], dim=0)
    for i in range(padded.shape[0] - window_size):
        window = padded[i: i + window_size]
        if (window.std(unbiased=False) < std_threshold
                and torch.abs(window.mean() - target_value) < mean_threshold):
            return int(i)
    return total_steps


class TailTrimTests(unittest.TestCase):
    def cases(self):
        generator = torch.Generator().manual_seed(7)
        for _ in range(120):
            steps = int(torch.randint(1, 300, (1,), generator=generator))
            latent = torch.randn(steps, 32, generator=generator)
            start = int(torch.randint(0, steps + 1, (1,), generator=generator))
            # Tails whose spread sweeps across the 0.05 threshold.
            scale = float(torch.rand(1, generator=generator)) * 0.2
            offset = float(torch.randn(1, generator=generator)) * 0.1
            latent[start:] = latent[start:] * scale + offset
            yield latent
        yield torch.zeros(5, 32)
        yield torch.ones(40, 32)

    def test_matches_loop_on_cpu_in_bf16_and_fp32(self):
        for latent in self.cases():
            for dtype in (torch.bfloat16, torch.float32):
                for window in (1, 5, 20):
                    x = latent.to(dtype)
                    self.assertEqual(find_flattening_point(x, window_size=window),
                                     loop_flattening_point(x, window_size=window))

    @unittest.skipUnless(CUDA, "CUDA is not available")
    def test_matches_loop_on_cuda(self):
        for latent in self.cases():
            x = latent.to("cuda", torch.bfloat16)
            self.assertEqual(find_flattening_point(x), loop_flattening_point(x))


class InMemorySpeakerTests(unittest.TestCase):
    def runtime_stub(self):
        param = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        return types.SimpleNamespace(
            model_cfg=types.SimpleNamespace(use_speaker_condition_resolved=True),
            model=types.SimpleNamespace(parameters=lambda: iter([param])),
            model_device=torch.device("cpu"))

    def test_tensor_ref_embed_equals_safetensors_round_trip(self):
        from safetensors.torch import save_file
        embedding = (torch.randn(16, 768) * 3).to(torch.bfloat16)
        stub = self.runtime_stub()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x.speaker.safetensors"
            save_file({"speaker_embedding": embedding.contiguous()}, str(path))
            from_file = InferenceRuntime._load_speaker_embedding_condition(
                stub, req=SamplingRequest(text="a", ref_embed=str(path)),
                batch_size=1, messages=[])
        from_tensor = InferenceRuntime._load_speaker_embedding_condition(
            stub, req=SamplingRequest(text="a", ref_embed=embedding),
            batch_size=1, messages=[])
        for a, b in zip(from_file, from_tensor):
            self.assertTrue(torch.equal(a, b))

    def test_write_wav_to_buffer_matches_file_bytes(self):
        audio = (torch.rand(1, 4800) * 2 - 1) * 0.7
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "a.wav"
            save_wav(str(path), audio, 48000)
            buffer = io.BytesIO()
            tts_cli.write_wav(buffer, audio, 48000)
            self.assertEqual(buffer.getvalue(), path.read_bytes())

    def test_synthesize_hands_tensor_to_backend_without_files(self):
        captured = {}

        class Backend:
            name = "fake"

            def synthesize(self, request, speaker_tensor, out_wav, log_fn=None):
                captured["request"] = request
                tts_cli.write_wav(out_wav, torch.zeros(1, 480), 48000)
                return {"backend": "fake", "wall_s": 0.0, "audio_s": 0.01}

        speaker = torch.randn(16, 768).to(torch.bfloat16)
        tts = tts_cli.IrodoriTTS.__new__(tts_cli.IrodoriTTS)
        tts.cassette = types.SimpleNamespace(get=lambda name: speaker, has=lambda name: True)
        tts.backend = Backend()
        tts.default_speaker = "fake"
        buffer = io.BytesIO()
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict("os.environ", {"IRODORI_CACHE_DIR": folder}):
            result = tts.synthesize(text="こんにちは", speaker="fake", out_wav=buffer,
                                    speaker_strength=0.5)
            self.assertEqual(list(Path(folder).iterdir()), [])
        request = captured["request"]
        self.assertIsInstance(request.ref_embed, torch.Tensor)
        self.assertTrue(torch.equal(request.ref_embed, speaker * 0.5))
        self.assertFalse(request.no_ref)
        self.assertTrue(buffer.getvalue().startswith(b"RIFF"))
        self.assertIsNone(result["out_wav"])


class VoicevoxTempFolderTests(unittest.TestCase):
    def adapter(self):
        calls = []

        class Tts:
            def synthesize(self, **kwargs):
                calls.append(kwargs)
                kwargs["out_wav"].write(b"RIFFwav")

        adapter = voicevox_engine.VoicevoxAdapter.__new__(voicevox_engine.VoicevoxAdapter)
        adapter.tts = Tts()
        adapter.lock = threading.Lock()
        adapter.progress_callback = None
        adapter.id_to_name = {1: "speaker"}
        return adapter, calls

    def test_no_temporary_folder_without_reference_audio(self):
        adapter, calls = self.adapter()
        with patch.object(voicevox_engine.tempfile, "TemporaryDirectory",
                          side_effect=AssertionError("temp folder created")), \
                patch.object(voicevox_engine, "READING_DICTIONARY") as dictionary:
            dictionary.convert.side_effect = lambda text, **_: text
            self.assertEqual(adapter.synthesize({"irodori_text": "こんにちは"}, 1), b"RIFFwav")
        self.assertIsNone(calls[0]["ref_wav"])

    def test_reference_audio_still_goes_through_a_temporary_file(self):
        adapter, calls = self.adapter()
        query = {"irodori_text": "こんにちは", "irodori_reference_audio": {
            "dataUrl": "data:audio/wav;base64,UklGRg==", "name": "ref.wav"}}
        with patch.object(voicevox_engine, "READING_DICTIONARY") as dictionary:
            dictionary.convert.side_effect = lambda text, **_: text
            self.assertEqual(adapter.synthesize(query, 1), b"RIFFwav")
        self.assertTrue(calls[0]["ref_wav"].endswith(".wav"))
        self.assertFalse(Path(calls[0]["ref_wav"]).exists())  # cleaned up


@unittest.skipUnless(CUDA, "CUDA is not available")
class CudaGraphTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.weight = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)

    def fn(self, x, mask, *, scale=None):
        h = torch.nn.functional.gelu(x @ self.weight) * mask[..., None]
        return (h.sum(dim=1), [h * (1 if scale is None else scale)])

    def test_replay_is_bit_identical_and_outputs_stay_valid(self):
        graphed = cuda_graphs.GraphedCall(self.fn, "test")
        results = []
        with torch.inference_mode():
            for i in range(4):
                x = torch.randn(1, 8, 64, device="cuda", dtype=torch.bfloat16)
                mask = torch.rand(1, 8, device="cuda") > 0.3
                results.append((graphed(x, mask), self.fn(x, mask)))
        self.assertEqual(graphed.captures, 1)
        self.assertEqual(graphed.replays, 4)
        for got, expected in results:  # earlier results were not overwritten
            self.assertTrue(torch.equal(got[0], expected[0]))
            self.assertTrue(torch.equal(got[1][0], expected[1][0]))
            self.assertIsInstance(got[1], list)

    def test_keyword_arguments_and_new_shapes(self):
        graphed = cuda_graphs.GraphedCall(self.fn, "test", max_graphs=2)
        with torch.inference_mode():
            for length in (8, 16, 8, 32):
                x = torch.randn(1, length, 64, device="cuda", dtype=torch.bfloat16)
                mask = torch.ones(1, length, dtype=torch.bool, device="cuda")
                got = graphed(x, mask=mask, scale=None)
                self.assertTrue(torch.equal(got[1][0], self.fn(x, mask)[1][0]))
        self.assertEqual(graphed.captures, 3)
        self.assertEqual(len(graphed.graphs), 2)

    def test_host_sync_falls_back_to_eager(self):
        def syncing(x):
            return x * float(x.sum().item() > 0)

        def noise():
            generator = torch.Generator(device="cuda").manual_seed(9)
            return torch.randn(8, device="cuda", generator=generator)

        expected_noise = noise()
        stream = torch.cuda.current_stream()
        logs = []
        graphed = cuda_graphs.GraphedCall(syncing, "sync", log=logs.append)
        with torch.inference_mode():
            x = torch.ones(4, device="cuda")
            self.assertTrue(torch.equal(graphed(x), syncing(x)))
            self.assertTrue(torch.equal(graphed(x), syncing(x)))
        self.assertTrue(graphed.disabled)
        self.assertEqual(graphed.eager, 2)
        self.assertEqual(len(logs), 1)
        # The failed capture must not leave CUDA, the stream or the RNG broken.
        self.assertEqual(torch.cuda.current_stream(), stream)
        torch.randn(4, device="cuda")
        self.assertTrue(torch.equal(noise(), expected_noise))
        torch.cuda.synchronize()

    def test_cpu_inputs_and_autograd_run_eagerly(self):
        graphed = cuda_graphs.GraphedCall(self.fn, "test")
        x = torch.randn(1, 8, 64, device="cuda", dtype=torch.bfloat16)
        mask = torch.ones(1, 8, dtype=torch.bool, device="cuda")
        graphed(x, mask)  # grad mode enabled
        self.assertEqual((graphed.captures, graphed.eager), (0, 1))
        on_cpu = cuda_graphs.GraphedCall(lambda a: a * 2, "cpu")
        with torch.inference_mode():
            self.assertTrue(torch.equal(on_cpu(torch.ones(2)), torch.full((2,), 2.0)))
        self.assertEqual((on_cpu.captures, on_cpu.eager), (0, 1))

    def test_install_wraps_encoders_and_kv_projection(self):
        model = types.SimpleNamespace(
            text_encoder=torch.nn.Linear(4, 4).cuda(), caption_encoder=None,
            build_context_kv_cache=lambda text_state, speaker_state, caption_state: [(text_state * 2,)])
        wrappers = cuda_graphs.install(model, log=lambda message: None)
        self.assertEqual(set(wrappers), {"text_encoder", "context_kv"})
        with torch.inference_mode():
            x = torch.randn(2, 4, device="cuda")
            expected = torch.nn.functional.linear(x, model.text_encoder.weight, model.text_encoder.bias)
            self.assertTrue(torch.equal(model.text_encoder(x), expected))
            kv = model.build_context_kv_cache(text_state=x, speaker_state=None, caption_state=None)
        self.assertTrue(torch.equal(kv[0][0], x * 2))
        self.assertEqual(wrappers["text_encoder"].captures, 1)


@unittest.skipUnless(CUDA, "CUDA is not available")
class VramTrimTests(unittest.TestCase):
    def model(self):
        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.wq, self.wk_text, self.k_norm = (torch.nn.Linear(4, 4) for _ in range(3))

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attention, self.mlp = Attention(), torch.nn.Linear(4, 4)

        model = torch.nn.Module()
        model.text_backbone = torch.nn.Linear(4, 4)  # first parameter
        model.blocks = torch.nn.ModuleList([Block(), Block()])
        model.cond_module = torch.nn.Linear(4, 4)
        model.out_proj = torch.nn.Linear(4, 4)
        return model.cuda()

    def test_offload_keeps_kv_projection_and_first_parameter(self):
        model = self.model()
        moved = vram_trim.offload_plan_weights(model)
        self.assertGreater(moved, 0)
        self.assertTrue(model.text_backbone.weight.is_cuda)
        for block in model.blocks:
            self.assertTrue(block.attention.wk_text.weight.is_cuda)
            self.assertTrue(block.attention.k_norm.weight.is_cuda)
            self.assertFalse(block.attention.wq.weight.is_cuda)
            self.assertFalse(block.mlp.weight.is_cuda)
        self.assertFalse(model.cond_module.weight.is_cuda)
        self.assertFalse(model.out_proj.weight.is_cuda)

    def test_on_demand_module_runs_on_gpu_and_returns_to_cpu(self):
        module = torch.nn.Linear(4, 4).cuda()
        x = torch.randn(2, 4, device="cuda")
        expected = module(x)
        vram_trim.park_on_demand(module, torch.device("cuda"))
        self.assertFalse(module.weight.is_cuda)
        self.assertTrue(torch.equal(module(x), expected))
        self.assertFalse(module.weight.is_cuda)

    def test_repack_keeps_values_and_ties(self):
        model = self.model()
        model.tied = torch.nn.Linear(4, 4, bias=False).cuda()
        model.tied.weight = model.text_backbone.weight
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        vram_trim.repack(model)
        for name, p in model.named_parameters():
            self.assertTrue(torch.equal(p, before[name]), name)
            self.assertTrue(p.is_cuda)
        self.assertEqual(model.tied.weight.data_ptr(), model.text_backbone.weight.data_ptr())

    def test_idle_release_waits_for_idle_and_never_interrupts_work(self):
        busy = threading.Lock()
        release = vram_trim.IdleRelease(busy, delay=0.2)
        with patch.object(vram_trim.torch.cuda, "empty_cache") as empty:
            release.touch()
            busy.acquire()           # a synthesis is running when the timer fires
            import time
            time.sleep(0.35)
            self.assertEqual(empty.call_count, 0)
            busy.release()
            time.sleep(0.4)
            self.assertEqual(empty.call_count, 1)
            release.touch()
            time.sleep(0.1)
            release.touch()          # used again before going idle: re-armed
            time.sleep(0.15)
            self.assertEqual(empty.call_count, 1)
            time.sleep(0.3)
            self.assertEqual(empty.call_count, 2)


class TrtCodecFallbackTests(unittest.TestCase):
    class Engine:
        torch = torch

        def __init__(self, output, accepts=True):
            self.output, self._accepts = output, accepts

        def accepts(self, latent):
            return self._accepts

        def __call__(self, latent):
            return self.output

    def decoder(self, engine):
        self.fallback_calls = []

        def fallback(latent):
            self.fallback_calls.append(latent)
            return torch.full((1, 1, 4), 7.0)
        return trt_codec.TrtCodecDecoder(engine, fallback, log=lambda message: None)

    def test_uses_engine_output_as_float32(self):
        decoder = self.decoder(self.Engine(torch.ones(1, 1, 4, dtype=torch.float16)))
        audio = decoder(torch.zeros(1, 2, 32))
        self.assertEqual(audio.dtype, torch.float32)
        self.assertTrue(torch.equal(audio, torch.ones(1, 1, 4)))
        self.assertEqual((decoder.calls, decoder.fallbacks, len(self.fallback_calls)), (1, 0, 0))

    def test_non_finite_output_decodes_with_pytorch(self):
        bad = torch.tensor([[[1.0, float("inf"), 0.0, 0.0]]], dtype=torch.float16)
        decoder = self.decoder(self.Engine(bad))
        self.assertTrue(torch.equal(decoder(torch.zeros(1, 2, 32)), torch.full((1, 1, 4), 7.0)))
        self.assertEqual(decoder.fallbacks, 1)

    def test_shape_outside_profile_decodes_with_pytorch(self):
        decoder = self.decoder(self.Engine(None, accepts=False))
        self.assertTrue(torch.equal(decoder(torch.zeros(2, 800, 32)), torch.full((1, 1, 4), 7.0)))
        self.assertEqual(decoder.fallbacks, 1)


class TrtCodecCacheTests(unittest.TestCase):
    def test_requires_verified_undamaged_matching_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            identity = dict(codec="a")
            plan = folder / trt_codec.PLAN_NAME
            plan.write_bytes(b"plan")
            self.assertFalse(trt_codec.valid_cache(folder, identity))
            (folder / "ready.json").write_text(json.dumps(dict(
                identity=identity, plan_sha256=trt_codec.digest(plan))))
            self.assertTrue(trt_codec.valid_cache(folder, identity))
            self.assertFalse(trt_codec.valid_cache(folder, dict(codec="b")))
            plan.write_bytes(b"corrupt")
            self.assertFalse(trt_codec.valid_cache(folder, identity))

    def test_weights_digest_is_remembered_until_the_file_changes(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(trt_codec, "BOX", Path(temp)):
            weights = Path(temp) / "weights.pth"
            weights.write_bytes(b"one")
            first = trt_codec._weights_digest(weights)
            with patch.object(trt_codec, "digest", side_effect=AssertionError("rehashed")):
                self.assertEqual(trt_codec._weights_digest(weights), first)
            weights.write_bytes(b"two!")
            self.assertNotEqual(trt_codec._weights_digest(weights), first)

    def test_failed_build_never_publishes_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            box = Path(temp)
            with patch.object(trt_codec, "BOX", box), \
                    patch.object(trt_codec, "codec_weights", return_value=box / "w.pth"), \
                    patch.object(trt_codec, "cache_identity", return_value={"codec": "a"}), \
                    patch.object(trt_codec.subprocess, "run") as run:
                run.return_value.returncode = 1
                with self.assertRaises(RuntimeError):
                    trt_codec.ensure_codec_plan(lambda *a: None, lambda message: None)
            published = [p for p in (box / ".cache" / "trt-codec").iterdir()
                         if not p.name.startswith("build-")]
            self.assertEqual(published, [])


class EditorCodecSetupTests(unittest.TestCase):
    def run_prepare(self, env, ensure):
        import editor_engine
        logs = []
        stub = types.SimpleNamespace(_set_progress=lambda *a: None, _runtime_log=logs.append)
        with patch.dict("os.environ", env, clear=False), \
                patch.object(trt_codec, "ensure_codec_plan", ensure):
            editor_engine.EditorAdapter._prepare_trt_codec(stub)
            return os.environ.get("IRODORI_TRT_CODEC_PLAN"), logs

    def test_publishes_plan_for_the_backend(self):
        plan, _ = self.run_prepare({"IRODORI_TRT_CODEC": "1"}, lambda *a: Path("x.plan"))
        self.assertEqual(plan, "x.plan")

    def test_failure_keeps_pytorch_codec(self):
        def fail(*_):
            raise RuntimeError("no tensorrt")
        plan, logs = self.run_prepare(
            {"IRODORI_TRT_CODEC": "1", "IRODORI_TRT_CODEC_PLAN": "stale.plan"}, fail)
        self.assertIsNone(plan)
        self.assertIn("no tensorrt", logs[0])

    def test_can_be_switched_off(self):
        plan, _ = self.run_prepare({"IRODORI_TRT_CODEC": "0", "IRODORI_TRT_CODEC_PLAN": "old"},
                                   lambda *a: self.fail("should not build"))
        self.assertIsNone(plan)


if __name__ == "__main__":
    unittest.main()
