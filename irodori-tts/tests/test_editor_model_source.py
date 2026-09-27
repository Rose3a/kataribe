"""契約テスト: モデル切り替え（ローカル / Hugging Face）と既定ステップ数。

エンジンはローカルの .safetensors と Hugging Face の repo_id の両方を受け付け、
MeanFlow モデルでは既定ステップ数を4にする。ここでは実モデルを読まずに
解決ロジックと既定値だけを検証する。
"""

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
        for value in ("C:/tmp/model.safetensors", "/etc/passwd", "model.ckpt", ""):
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
