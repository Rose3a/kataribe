"""契約テスト: VOICEVOX互換エンジンの MeanFlow サンプリング。

MeanFlow 蒸留モデルでは既定ステップ数を4にし、RF 用の CFG スケールと Sway
Sampling を 0 / linear として送る（CFG は蒸留時に教師の軌跡へ融合済みで、
推論時には効かないため）。RF モデルは従来どおり8ステップ・CFGクエリ・sway の
ままであることを、実モデルを読まずに検証する。
"""

from json import dumps
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock, patch


IRODORI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IRODORI_ROOT / "wrapper"))
try:
    import voicevox_engine
    from voicevox_engine import (VoicevoxAdapter, _manifest_sampling_fields,
                                 _resolve_flow_parameterization, _sampling_settings)
except ModuleNotFoundError as exc:  # テスト環境に torch が無い場合
    if exc.name != "torch":
        raise
    voicevox_engine = None


@unittest.skipIf(voicevox_engine is None, "torch is not installed in this interpreter")
class SamplingSettingsTests(unittest.TestCase):
    def test_meanflow_uses_four_steps_and_no_rf_controls(self):
        settings = _sampling_settings("meanflow", 4, {})
        self.assertEqual(settings["num_steps"], 4)
        self.assertEqual(settings["cfg_scale_text"], 0.0)
        self.assertEqual(settings["cfg_scale_caption"], 0.0)
        self.assertEqual(settings["cfg_scale_speaker"], 0.0)
        self.assertEqual(settings["t_schedule_mode"], "linear")

    def test_meanflow_honours_explicit_steps_but_keeps_guidance_fixed(self):
        settings = _sampling_settings("meanflow", 4, {
            "irodori_steps": 6,
            "irodori_cfg_text": 9.0,
            "irodori_schedule": "sway",
            "irodori_sway_coeff": -0.5,
        })
        self.assertEqual(settings["num_steps"], 6)
        self.assertEqual(settings["cfg_scale_text"], 0.0)
        self.assertEqual(settings["t_schedule_mode"], "linear")

    def test_rf_keeps_eight_steps_cfg_and_sway(self):
        settings = _sampling_settings("rf_velocity", 8, {})
        self.assertEqual(settings["num_steps"], 8)
        self.assertEqual(settings["cfg_scale_text"], 3.0)
        self.assertEqual(settings["cfg_scale_caption"], 3.0)
        self.assertEqual(settings["cfg_scale_speaker"], 5.0)
        self.assertEqual(settings["t_schedule_mode"], "sway")
        self.assertEqual(settings["sway_coeff"], -1.0)

    def test_rf_query_values_win(self):
        settings = _sampling_settings("rf_velocity", 8, {
            "irodori_steps": 12,
            "irodori_cfg_speaker": 4.0,
            "irodori_schedule": "linear",
            "irodori_sway_coeff": -2.0,
        })
        self.assertEqual(settings["num_steps"], 12)
        self.assertEqual(settings["cfg_scale_speaker"], 4.0)
        self.assertEqual(settings["t_schedule_mode"], "linear")
        self.assertEqual(settings["sway_coeff"], -2.0)

    def test_invalid_steps_are_rejected(self):
        for value in (0, -1, "abc"):
            with self.assertRaises(ValueError):
                _sampling_settings("meanflow", 4, {"irodori_steps": value})

    def test_blank_steps_fall_back_to_model_default(self):
        for value in (None, ""):
            settings = _sampling_settings("meanflow", 4, {"irodori_steps": value})
            self.assertEqual(settings["num_steps"], 4)


@unittest.skipIf(voicevox_engine is None, "torch is not installed in this interpreter")
class FlowParameterizationTests(unittest.TestCase):
    def test_editor_adapter_without_delegate_uses_rf_manifest_defaults(self):
        # EditorAdapter does not create its delegate until first synthesis.
        fields = _manifest_sampling_fields(object())
        self.assertEqual(fields["irodori_flow_parameterization"], "rf_velocity")
        self.assertEqual(fields["irodori_flow_parameterization_source"], "default")
        self.assertEqual(fields["irodori_default_steps"], 8)

    def test_env_override_wins(self):
        with patch.dict("os.environ", {"IRODORI_FLOW_PARAMETERIZATION": "meanflow"}):
            self.assertEqual(_resolve_flow_parameterization("missing.safetensors"),
                             ("meanflow", "env"))

    def test_metadata_selects_meanflow(self):
        with patch.object(voicevox_engine, "_read_safetensors_config",
                          return_value={"flow_parameterization": "meanflow"}):
            found = _resolve_flow_parameterization(Path(__file__))
        self.assertEqual(found, ("meanflow", "metadata"))

    def test_missing_metadata_falls_back_to_rf(self):
        self.assertEqual(_resolve_flow_parameterization("missing.safetensors"),
                         ("rf_velocity", "default"))
        with patch.object(voicevox_engine, "_read_safetensors_config",
                          side_effect=OSError("unreadable")):
            self.assertEqual(_resolve_flow_parameterization(Path(__file__)),
                             ("rf_velocity", "default"))

    def test_rf_metadata_key_is_absent_in_shipped_box_models(self):
        """同梱モデルは鍵そのものが無い = RF とみなす。"""
        with patch.object(voicevox_engine, "_read_safetensors_config",
                          return_value={"latent_dim": 32}):
            self.assertEqual(_resolve_flow_parameterization(Path(__file__)),
                             ("rf_velocity", "default"))


@unittest.skipIf(voicevox_engine is None, "torch is not installed in this interpreter")
class AdapterRequestTests(unittest.TestCase):
    def build_adapter(self, flow_parameterization: str) -> tuple[VoicevoxAdapter, dict]:
        captured: dict = {}

        class FakeTts:
            def synthesize(self, **kwargs):
                captured.update(kwargs)
                kwargs["out_wav"].write(b"RIFFtest")

        adapter = VoicevoxAdapter.__new__(VoicevoxAdapter)
        adapter.tts = FakeTts()
        adapter.lock = threading.Lock()
        adapter.progress_callback = None
        adapter.id_to_name = {1: "test-speaker"}
        adapter.flow_parameterization = flow_parameterization
        adapter.default_steps = 4 if flow_parameterization == "meanflow" else 8
        return adapter, captured

    def test_meanflow_request_uses_four_steps_and_zero_cfg(self):
        adapter, captured = self.build_adapter("meanflow")
        with patch.object(voicevox_engine, "READING_DICTIONARY") as dictionary:
            dictionary.convert.side_effect = lambda text, **_: text
            data = adapter.synthesize({"irodori_text": "こんにちは"}, 1)
        self.assertEqual(data, b"RIFFtest")
        self.assertEqual(captured["num_steps"], 4)
        self.assertEqual(captured["cfg_scale_text"], 0.0)
        self.assertEqual(captured["cfg_scale_speaker"], 0.0)
        self.assertEqual(captured["cfg_scale_caption"], 0.0)
        self.assertEqual(captured["t_schedule_mode"], "linear")

    def test_rf_request_keeps_eight_steps_and_cfg(self):
        adapter, captured = self.build_adapter("rf_velocity")
        with patch.object(voicevox_engine, "READING_DICTIONARY") as dictionary:
            dictionary.convert.side_effect = lambda text, **_: text
            adapter.synthesize({"irodori_text": "こんにちは"}, 1)
        self.assertEqual(captured["num_steps"], 8)
        self.assertEqual(captured["cfg_scale_text"], 3.0)
        self.assertEqual(captured["cfg_scale_speaker"], 5.0)
        self.assertEqual(captured["t_schedule_mode"], "sway")

    def test_explicit_steps_still_reach_the_runtime(self):
        adapter, captured = self.build_adapter("meanflow")
        with patch.object(voicevox_engine, "READING_DICTIONARY") as dictionary:
            dictionary.convert.side_effect = lambda text, **_: text
            adapter.synthesize({"irodori_text": "こんにちは", "irodori_steps": 8}, 1)
        self.assertEqual(captured["num_steps"], 8)


if __name__ == "__main__":
    unittest.main()
