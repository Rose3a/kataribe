import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'wrapper'))
import model_storage
import trt_cache


def _write(path, data=b'x'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


class ModelStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.box = Path(self.temp.name)
        patches = [patch.object(model_storage, 'BOX', self.box),
                   patch.object(trt_cache, 'BOX', self.box)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.temp.cleanup)
        hub = self.box / '.cache/huggingface/hub'
        _write(hub / 'models--Aratako--Irodori-TTS-v4.1-Small/snapshots/r/model.safetensors')
        _write(hub / 'models--Aratako--Irodori-TTS-v4-Large/snapshots/r/model.safetensors')
        _write(hub / 'models--Aratako--Semantic-DACVAE-Japanese-32dim/snapshots/r/weights.pth')
        self.quant = _write(self.box / 'models/quant/a.safetensors')
        _write(self.box / 'models/quant/tokenizer/tokenizer.json')
        self.env = dict(gpu='RTX', trt='10')
        for name, env in (('fresh', self.env), ('old', dict(self.env, trt='9'))):
            folder = self.box / '.cache/trt' / name
            _write(folder / 'fallback_bf16.plan', b'plan')
            (folder / 'ready.json').write_text(json.dumps(dict(identity=dict(env, model='m'))))
        _write(self.box / '.cache/trt/build-abc/model.onnx')
        self.storage = model_storage.ModelStorage(
            self.box / 'models', self.box / '.cache/huggingface', self.box / 'models/asr')

    def scan(self, **kwargs):
        args = dict(selected_model='Aratako/Irodori-TTS-v4.1-Small', active_checkpoint=None,
                    active_plan=None, active_codec_plan=None, asr_loaded=False, busy=False)
        args.update(kwargs)
        with patch.object(trt_cache, 'env_identity', return_value=self.env), \
             patch.object(trt_cache, 'known_model_digests', return_value={}), \
             patch.object(self.storage, '_identify_in_background'):
            return {e['id']: e for e in self.storage.scan(**args)['entries']}

    def test_classifies_models_and_plans(self):
        entries = self.scan()
        small = entries['hf:.cache/huggingface/hub/models--Aratako--Irodori-TTS-v4.1-Small']
        self.assertEqual((small['status'], small['deletable']), ('in_use', False))
        large = entries['hf:.cache/huggingface/hub/models--Aratako--Irodori-TTS-v4-Large']
        self.assertEqual((large['status'], large['deletable']), ('unused', True))
        codec = entries['hf:.cache/huggingface/hub/models--Aratako--Semantic-DACVAE-Japanese-32dim']
        self.assertEqual((codec['status'], codec['deletable']), ('required', False))
        self.assertEqual(entries['trt:.cache/trt/fresh']['status'], 'unused')
        self.assertEqual(entries['trt:.cache/trt/old']['status'], 'stale')
        self.assertEqual(entries['trt-build:.cache/trt/build-abc']['status'], 'stale')

    def test_active_plan_and_busy_block_deletion(self):
        entries = self.scan(active_plan=self.box / '.cache/trt/fresh/fallback_bf16.plan')
        self.assertEqual(entries['trt:.cache/trt/fresh']['status'], 'in_use')
        self.assertFalse(entries['trt:.cache/trt/fresh']['deletable'])
        entries = self.scan(busy=True)
        self.assertFalse(entries['trt-build:.cache/trt/build-abc']['deletable'])

    def test_delete_only_known_deletable_entries(self):
        entries = self.scan()
        expected = entries['trt:.cache/trt/old']['bytes'] + 1
        with self.assertRaisesRegex(ValueError, '必須'):
            self.storage.delete(['hf:.cache/huggingface/hub/models--Aratako--Semantic-DACVAE-Japanese-32dim'])
        with self.assertRaisesRegex(ValueError, '一覧を更新'):
            self.storage.delete(['local:../outside.safetensors'])
        freed = self.storage.delete(['trt:.cache/trt/old', 'trt-build:.cache/trt/build-abc'])
        self.assertEqual(freed, expected)
        self.assertFalse((self.box / '.cache/trt/old').exists())
        self.assertTrue((self.box / '.cache/trt/fresh').exists())

    def test_deleting_last_checkpoint_removes_its_model_folder(self):
        self.scan()
        self.storage.delete(['local:models/quant/a.safetensors'])
        self.assertFalse((self.box / 'models/quant').exists())
        self.assertTrue((self.box / 'models').exists())

    def test_digest_memo_prefers_hf_path_and_reuses_hash(self):
        checkpoint = _write(self.box / 'models/model.safetensors', b'weights')
        hf_copy = _write(self.box / '.cache/huggingface/hub/models--A--B/snapshots/r/model.safetensors',
                         b'weights')
        first = trt_cache.model_digest(checkpoint)
        self.assertEqual(trt_cache.model_digest(hf_copy), first)
        with patch.object(trt_cache, 'digest', side_effect=AssertionError('rehashed')):
            self.assertEqual(trt_cache.model_digest(checkpoint), first)
        self.assertIn('models--A--B', trt_cache.known_model_digests()[first])
        self.assertEqual(model_storage.checkpoint_label(trt_cache.known_model_digests()[first]), 'A/B')


if __name__ == '__main__':
    unittest.main()
