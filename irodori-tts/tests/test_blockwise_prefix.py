"""Blockwise v2 の見分け方（tts_cli.blockwise_prefix_seconds）。

HF の rose3/Irodori-TTS-v4.1-Small-Blockwise は v1 を v2 で上書きしたので、名前では見分けられない。
v2 のチェックポイントはメタデータ blockwise_prefix_seconds（前の行から見せる長さの上限、15 秒）を持つ。
"""
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))
import tts_cli  # noqa: E402


class BlockwisePrefixSecondsTest(unittest.TestCase):
    def setUp(self):
        tts_cli._prefix_seconds_cache.clear()
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def checkpoint(self, folder, metadata=None):
        path = self.root / folder / "model.safetensors"
        path.parent.mkdir(parents=True)
        save_file({"x": torch.zeros(1)}, str(path), metadata=metadata)
        return str(path)

    def test_metadata_wins_under_the_shared_repository_name(self):
        path = self.checkpoint("models--rose3--Irodori-TTS-v4.1-Small-Blockwise/snapshots/abc/bf16",
                               {"blockwise_version": "2", "blockwise_prefix_seconds": "15"})
        self.assertEqual(tts_cli.blockwise_prefix_seconds(path), 15.0)

    def test_v1_without_metadata_uses_the_runtime_default(self):
        path = self.checkpoint("models--rose3--Irodori-TTS-v4.1-Small-Blockwise/snapshots/old/bf16")
        self.assertIsNone(tts_cli.blockwise_prefix_seconds(path))

    def test_local_v2_folder_without_metadata(self):
        path = self.checkpoint("Irodori-TTS-v4.1-Small-Blockwise-v2")
        self.assertEqual(tts_cli.blockwise_prefix_seconds(path), tts_cli.BLOCKWISE_V2_PREFIX_SECONDS)

    def test_missing_file_falls_back_to_the_name(self):
        self.assertIsNone(tts_cli.blockwise_prefix_seconds(str(self.root / "none.safetensors")))
        self.assertEqual(tts_cli.blockwise_prefix_seconds("D:\\models\\Irodori-TTS-v4.1-Small-Blockwise-v2\\model.safetensors"),
                         tts_cli.BLOCKWISE_V2_PREFIX_SECONDS)


if __name__ == "__main__":
    unittest.main()
