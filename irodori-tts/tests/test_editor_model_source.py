"""契約テスト: モデル切り替え（ローカル / Hugging Face）と既定ステップ数。

エンジンはローカルの .safetensors と Hugging Face の repo_id の両方を受け付け、
MeanFlow モデルでは既定ステップ数を4にする。ここでは実モデルを読まずに
解決ロジックと既定値だけを検証する。
"""

import os
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch


IRODORI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IRODORI_ROOT / "wrapper"))
try:
    from editor_engine import EditorAdapter
except ModuleNotFoundError as exc:  # テスト環境に torch が無い場合のスタブ
    if exc.name != "torch":
        raise
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.__version__ = "test"
    torch_stub.version = types.SimpleNamespace(cuda=None)
    with patch.dict(sys.modules, {"torch": torch_stub}):
        from editor_engine import EditorAdapter

EDITOR_GLOBALS = EditorAdapter.__init__.__globals__
MODEL_DIR = EDITOR_GLOBALS["MODEL_DIR"]


def fake_adapter(**attrs) -> EditorAdapter:
    adapter = EditorAdapter.__new__(EditorAdapter)
    adapter.settings = dict(backend="cpu", model="model.safetensors", seed=4763674,
                            sway_coeff=-1.0)
    adapter.state_lock = threading.RLock()
    adapter.operation_lock = threading.RLock()
    adapter._model_info_cache = {}
    adapter._set_progress = Mock()
    adapter._runtime_log = Mock()
    for key, value in attrs.items():
        setattr(adapter, key, value)
    return adapter


class ModelSourceTests(unittest.TestCase):
    def test_failed_trt_is_not_rebuilt_until_settings_applied(self):
        adapter = fake_adapter()
        adapter.settings['backend'] = 'trt'
        adapter._build_delegate_impl = Mock(side_effect=RuntimeError('export failed'))
        for prewarm in (True, False, False):
            with self.assertRaisesRegex(RuntimeError, 'export failed'):
                adapter._build_delegate(prewarm=prewarm)
        adapter._build_delegate_impl.assert_called_once()
        adapter.validate = Mock()
        adapter.status = Mock(return_value={})
        adapter.schedule_prewarm = Mock()
        with tempfile.TemporaryDirectory() as folder:
            adapter.config_path = Path(folder) / 'settings.json'
            adapter.configure(dict(adapter.settings))
        adapter.schedule_prewarm.assert_called_once()
        adapter._build_delegate_impl.side_effect = None
        adapter._build_delegate_impl.return_value = 'ready'
        self.assertEqual(adapter._build_delegate(), 'ready')
        self.assertEqual(adapter._build_delegate_impl.call_count, 2)

    def test_partial_settings_update_keeps_other_values(self):
        adapter = fake_adapter()
        adapter.settings.update(backend="cuda", seed=1001)
        adapter.validate = Mock()
        adapter.status = Mock(return_value={})
        adapter.schedule_prewarm = Mock()
        adapter._close_locked = Mock()
        with tempfile.TemporaryDirectory() as folder:
            adapter.config_path = Path(folder) / "settings.json"
            adapter.configure({"seed": 7})
        self.assertEqual(adapter.settings["backend"], "cuda")
        self.assertEqual(adapter.settings["model"], "model.safetensors")
        self.assertEqual(adapter.settings["seed"], 7)
        adapter.schedule_prewarm.assert_not_called()

    def test_trt_accepts_meanflow_repo(self):
        adapter = fake_adapter()
        adapter.available_backends = lambda: {'trt': True}
        adapter.validate({**adapter.settings, 'backend': 'trt',
                          'model': 'Aratako/Irodori-TTS-v4.1-Small-MF'})

    def test_trt_builds_plan_for_resolved_checkpoint(self):
        adapter = fake_adapter()
        adapter.settings['backend'] = 'trt'
        checkpoint = Path('C:/cache/mf.safetensors')
        adapter._resolve_local_model = lambda source: checkpoint
        adapter.model_info = Mock(return_value={})
        with patch('trt_cache.ensure_plan', return_value=Path('mf.plan')) as build, \
             patch.dict(EDITOR_GLOBALS, {'VoicevoxAdapter': Mock(),
                                         'resolve_embed_dirs': lambda: []}), \
             patch.dict('os.environ'):
            adapter._build_delegate()
            self.assertEqual(build.call_args.args[0], checkpoint)
            self.assertEqual(EDITOR_GLOBALS['VoicevoxAdapter'].call_args.args[3], Path('mf.plan'))

    def test_accepts_hf_repo_id_and_local_filename(self):
        adapter = fake_adapter()
        local = MODEL_DIR / "model.safetensors"
        if local.is_file():
            self.assertEqual(adapter._validate_model_source("model.safetensors"),
                             "model.safetensors")
        self.assertEqual(
            adapter._validate_model_source("Aratako/Irodori-TTS-v4.1-Small-MF"),
            "Aratako/Irodori-TTS-v4.1-Small-MF")

    def test_rejects_paths_outside_models_and_garbage(self):
        adapter = fake_adapter()
        for value in ("C:/tmp/model.safetensors", "/etc/passwd", "model.ckpt", "",
                      "../../etc/passwd", "Aratako/..", "Aratako/repo/../x"):
            with self.assertRaises(ValueError):
                adapter._validate_model_source(value)

    def test_resolves_cached_hf_checkpoint_without_downloading(self):
        adapter = fake_adapter()
        cached = Path("C:/cache/model.safetensors")
        adapter._hf_checkpoint_from_cache = lambda source: cached
        adapter._download_hf_model = Mock(side_effect=AssertionError("must not download"))
        self.assertEqual(
            adapter._resolve_model("Aratako/Irodori-TTS-v4.1-Small-MF"), cached)

    def test_uncached_hf_source_raises_until_applied(self):
        adapter = fake_adapter()
        adapter._hf_checkpoint_from_cache = lambda source: None
        with self.assertRaises(ValueError):
            adapter._resolve_model("Aratako/Irodori-TTS-v4.1-Small-MF")

    def test_build_delegate_downloads_uncached_hf_model(self):
        adapter = fake_adapter()
        adapter.settings = dict(backend="cpu", model="Aratako/Irodori-TTS-v4.1-Small-MF")
        adapter._resolve_local_model = lambda source: None
        adapter._hf_checkpoint_from_cache = lambda source: None
        downloaded = Path("C:/cache/hub/snapshots/abc/model.safetensors")
        adapter._download_hf_model = Mock(return_value=downloaded)
        adapter._plan_path = lambda: Path("fallback.plan")
        adapter.model_info = Mock(return_value={})
        delegate = types.SimpleNamespace(synthesize=Mock())

        with patch.dict(EDITOR_GLOBALS, {
            "VoicevoxAdapter": Mock(return_value=delegate),
            "resolve_embed_dirs": lambda: [Path("speakers")],
        }), patch.dict("os.environ", {}, clear=False):
            result = adapter._build_delegate()
            self.assertIs(result, delegate)
            adapter._download_hf_model.assert_called_once_with(
                "Aratako/Irodori-TTS-v4.1-Small-MF")
            self.assertEqual(
                EDITOR_GLOBALS["os"].environ["IRODORI_CHECKPOINT"], str(downloaded))


class ModelInfoTests(unittest.TestCase):
    def test_meanflow_model_defaults_to_four_steps(self):
        adapter = fake_adapter()
        adapter._resolve_local_model = lambda source: None
        adapter._hf_checkpoint_from_cache = lambda source: Path("C:/cache/model.safetensors")
        adapter._read_safetensors_config = lambda path: {"flow_parameterization": "meanflow"}
        adapter.settings["model"] = "Aratako/Irodori-TTS-v4.1-Small-MF"
        info = adapter.model_info()
        self.assertTrue(info["meanflow"])
        self.assertEqual(info["defaultSteps"], 4)
        self.assertEqual(info["flowParameterization"], "meanflow")

    def test_rf_model_keeps_eight_steps(self):
        adapter = fake_adapter()
        adapter._resolve_local_model = lambda source: Path("C:/models/model.safetensors")
        adapter._read_safetensors_config = lambda path: {"flow_parameterization": "rf_velocity"}
        info = adapter.model_info()
        self.assertFalse(info["meanflow"])
        self.assertEqual(info["defaultSteps"], 8)

    def test_large_model_reports_gemma_license(self):
        adapter = fake_adapter()
        adapter._resolve_local_model = lambda source: None
        adapter._hf_checkpoint_from_cache = lambda source: Path("C:/cache/model.safetensors")
        adapter._read_safetensors_config = lambda path: {"flow_parameterization": "rf_velocity"}
        info = adapter.model_info("Aratako/Irodori-TTS-v4-Large")
        self.assertEqual(info["license"], "Gemma Terms of Use")
        self.assertIn("Irodori-TTS-v4-Large", info["licenseUrl"])
        self.assertEqual(info["defaultSteps"], 8)

    def test_quantized_large_is_detected_and_rejected_for_tensorrt(self):
        source = "Aratako/Irodori-TTS-v4-Large-Quantized/int8-weight-only"
        adapter = fake_adapter()
        adapter._resolve_local_model = lambda source: None
        adapter._hf_checkpoint_from_cache = lambda source: None
        adapter._read_hf_config = lambda source: EditorAdapter._config_from_metadata({
            "config_json": '{"flow_parameterization": "rf_velocity", "speaker_dim": 1280}',
            "irodori_quantization_json": '{"quantization_type": "int8_weight_only"}',
        })
        info = adapter.model_info(source)
        self.assertEqual(info["quantization"], "int8_weight_only")
        self.assertEqual(info["speakerDim"], 1280)
        # サブフォルダ付きでもリポジトリ単位でライセンスを引く。
        self.assertEqual(info["license"], "Gemma Terms of Use")
        adapter.available_backends = lambda: {"cpu": True, "cuda": True, "trt": True, "radeon": False}
        settings = dict(backend="trt", model=source, seed=1, sway_coeff=-1.0)
        with self.assertRaisesRegex(ValueError, "TensorRT"):
            adapter.validate(settings)
        adapter.validate({**settings, "backend": "cuda"})

    def test_int4_model_is_accepted_for_tensorrt_but_other_quantizations_are_not(self):
        adapter = fake_adapter()
        adapter.available_backends = lambda: {"cpu": True, "cuda": True, "trt": True, "radeon": False}
        settings = dict(backend="trt", model="Aratako/Irodori-TTS-v4-Large-Quantized/int4-weight-only",
                        seed=1, sway_coeff=-1.0)
        adapter.model_info = lambda model: {"quantization": "int4_weight_only"}
        adapter.validate(settings)
        for other in ("int8_weight_only", "float8_weight_only", "unknown"):
            adapter.model_info = lambda model, other=other: {"quantization": other}
            with self.assertRaisesRegex(ValueError, "INT4"):
                adapter.validate(settings)
            adapter.validate({**settings, "backend": "cuda"})

    def test_token_split_threshold_is_validated_and_defaults_to_minus_13(self):
        adapter = fake_adapter()
        adapter.available_backends = lambda: {"cpu": True, "cuda": False, "trt": False, "radeon": False}
        self.assertEqual(EditorAdapter.DEFAULT_SETTINGS["token_split_threshold"], "-13")
        settings = dict(backend="cpu", model="model.safetensors", seed=1, sway_coeff=-1.0)
        adapter.validate(settings)                                   # 未指定は既定（-13）
        for value in ("-10", "-11", "-12", "-13", "none"):
            adapter.validate({**settings, "token_split_threshold": value})
        for bad in ("-14", -13, "small"):
            with self.assertRaisesRegex(ValueError, "token_split_threshold"):
                adapter.validate({**settings, "token_split_threshold": bad})

    def test_token_split_state_follows_the_model_tokenizer(self):
        adapter = fake_adapter(delegate=None)
        small = {"textTokenizerRepo": "sbintuitions/modernbert-ja-310m"}
        large = {"textTokenizerRepo": "google/t5gemma-2-1b-1b"}
        self.assertEqual(adapter.token_split_state({"token_split_threshold": "-12"}, small),
                         {"threshold": "-12", "applies": True,
                          "modelTokenizer": "sbintuitions/modernbert-ja-310m"})
        self.assertFalse(adapter.token_split_state({}, large)["applies"])
        self.assertEqual(adapter.token_split_state({}, large)["threshold"], "-13")
        # メタデータが読めないモデルは当てる側に倒す
        self.assertTrue(adapter.token_split_state({}, {})["applies"])
        # 読み込み済みのモデルは、読み込んだトークナイザで判定する
        adapter.delegate = Mock(text_tokenizer_repo="google/t5gemma-2-1b-1b")
        self.assertFalse(adapter.token_split_state({}, small)["applies"])

    def test_int4_checkpoint_gets_the_int4_plan_and_others_the_bf16_plan(self):
        import trt_cache
        import trt_int4
        local = Path("C:/models/large.safetensors")
        for is_int4, expected in ((True, "int4.plan"), (False, "bf16.plan")):
            adapter = fake_adapter(model_info=Mock(), _prepare_trt_codec=Mock(),
                                   _resolve_local_model=lambda source: local)
            adapter.settings.update(backend="trt", model=str(local))
            with patch.object(trt_int4, "is_int4_checkpoint", return_value=is_int4),                  patch.object(trt_int4, "ensure_plan", return_value=Path("int4.plan")),                  patch.object(trt_cache, "ensure_plan", return_value=Path("bf16.plan")),                  patch.dict(EDITOR_GLOBALS, {"VoicevoxAdapter": Mock()}),                  patch.dict(os.environ):
                adapter._build_delegate_impl()
            self.assertEqual(adapter._active_plan, Path(expected))

    def test_metadata_failure_falls_back_to_rf_default(self):
        adapter = fake_adapter()
        adapter._resolve_local_model = lambda source: Path("C:/models/model.safetensors")

        def boom(path):
            raise OSError("unreadable")

        adapter._read_safetensors_config = boom
        info = adapter.model_info()
        self.assertFalse(info["metadataAvailable"])
        self.assertEqual(info["defaultSteps"], 8)

    def test_line_steps_follows_model_default_when_query_omits_it(self):
        adapter = fake_adapter()
        adapter._default_steps = lambda: 4
        self.assertEqual(adapter._line_steps({}), 4)
        self.assertEqual(adapter._line_steps({"irodori_steps": 6}), 6)
        with self.assertRaises(ValueError):
            adapter._line_steps({"irodori_steps": 0})


if __name__ == "__main__":
    unittest.main()
