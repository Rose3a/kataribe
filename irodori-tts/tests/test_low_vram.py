"""省VRAM（設定 low_vram）の契約テスト。実モデルは読まない。

- 既定はオフ（速さ優先）、真偽値以外は弾く
- TensorRT・CUDA で切り替えたときだけエンジンを読み込み直す（CUDA Graph の有無などは作るときに決まる）
（TrtBackend 側の中身は GPU が要るので、work/blockwise_quant/vram_probe.py で実機確認）
"""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))
import editor_engine  # noqa: E402

BACKENDS = {"cpu": True, "cuda": True, "trt": True, "radeon": False}


class LowVramSettingTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(editor_engine.EditorAdapter, "available_backends", return_value=BACKENDS)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.adapter = editor_engine.EditorAdapter()
        self.adapter.config_path = Path(tempfile.mkdtemp()) / "editor-settings.json"
        self.adapter.settings = dict(self.adapter.DEFAULT_SETTINGS, backend="trt",
                                     model="Aratako/Irodori-TTS-v4.1-Small")
        self.adapter.schedule_prewarm = lambda: True

    def test_default_is_off(self):
        self.assertIs(self.adapter.DEFAULT_SETTINGS["low_vram"], False)

    def test_non_bool_is_rejected(self):
        for bad in ("true", 1, None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.adapter.configure({"low_vram": bad})

    def test_toggling_on_tensorrt_reloads_the_engine(self):
        with patch.object(self.adapter, "_close_locked") as close:
            self.adapter.configure({"low_vram": True})
            close.assert_called_once()
            self.assertIs(self.adapter.settings["low_vram"], True)
            self.adapter.configure({"seed": 5})  # 変えていなければ読み込み直さない
            close.assert_called_once()

    def test_toggling_on_cuda_reloads_the_engine(self):
        self.adapter.settings["backend"] = "cuda"
        with patch.object(self.adapter, "_close_locked") as close:
            self.adapter.configure({"low_vram": True})
            close.assert_called_once()

    def test_toggling_on_cpu_does_not_reload(self):
        self.adapter.settings["backend"] = "cpu"
        with patch.object(self.adapter, "_close_locked") as close:
            self.adapter.configure({"low_vram": True})
            close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
