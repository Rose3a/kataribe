import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))
from third_party_licenses import dependency_licenses


class LicenseTests(unittest.TestCase):
    def test_engine_help_includes_asr_and_runtime_notices(self):
        entries = dependency_licenses()
        self.assertTrue(any("口パク ASR" in item["name"] and item["license"] == "CC BY 4.0" for item in entries))
        names = {item["name"] for item in entries}
        self.assertTrue({"sherpa-onnx", "dacvae", "soundfile"} <= names)
        self.assertTrue(all(item["text"].strip() for item in entries))

    def test_prefers_local_runtime_notices_and_falls_back_to_tracked(self):
        import json
        import tempfile
        from unittest.mock import patch
        import third_party_licenses

        def entry(name):
            return [{"name": name, "version": "1", "license": "MIT", "text": "x"}]

        with tempfile.TemporaryDirectory() as folder:
            public = Path(folder)
            (public / "licenses.json").write_text(json.dumps(entry("model")), encoding="utf-8")
            (public / "runtime-licenses.json").write_text(json.dumps(entry("tracked")), encoding="utf-8")
            with patch.object(third_party_licenses, "PUBLIC", public):
                try:
                    third_party_licenses.dependency_licenses.cache_clear()
                    names = [item["name"] for item in third_party_licenses.dependency_licenses()]
                    self.assertEqual(names, ["model", "tracked"])

                    (public / "runtime-licenses.local.json").write_text(
                        json.dumps(entry("local")), encoding="utf-8")
                    third_party_licenses.dependency_licenses.cache_clear()
                    names = [item["name"] for item in third_party_licenses.dependency_licenses()]
                    self.assertEqual(names, ["model", "local"])
                finally:
                    third_party_licenses.dependency_licenses.cache_clear()


if __name__ == "__main__":
    unittest.main()
