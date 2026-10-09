"""全行共通の既定ステップ数（設定 default_steps）の契約テスト。実モデルは読まない。

未指定のセリフは、MeanFlow を含めて default_steps（既定8）を使い、セリフ側の指定が優先する。
"""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))
import editor_engine  # noqa: E402

MF_MODEL = "Aratako/Irodori-TTS-v4.1-Small-MF"


class DefaultStepsTests(unittest.TestCase):
    def setUp(self):
        self.adapter = editor_engine.EditorAdapter()
        self.adapter.config_path = Path(tempfile.mkdtemp()) / "editor-settings.json"
        self.adapter.settings = dict(self.adapter.DEFAULT_SETTINGS, backend="cpu", model=MF_MODEL)

    def test_default_is_eight_even_for_meanflow(self):
        self.assertEqual(self.adapter.DEFAULT_SETTINGS["default_steps"], 8)
        self.assertEqual(self.adapter._line_steps({}), 8)
        status = self.adapter.status()
        self.assertEqual(status["modelInfo"]["defaultSteps"], 8)

    def test_setting_is_used_and_line_value_wins(self):
        self.adapter.configure({"default_steps": 12})
        self.assertEqual(self.adapter._line_steps({}), 12)
        self.assertEqual(self.adapter._line_steps({"irodori_steps": 4}), 4)
        self.assertEqual(self.adapter.status()["modelInfo"]["defaultSteps"], 12)

    def test_invalid_values_are_rejected(self):
        for bad in (0, 81, "8", True, 2.5):
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.adapter.configure({"default_steps": bad})


if __name__ == "__main__":
    unittest.main()
