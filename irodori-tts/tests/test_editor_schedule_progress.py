"""Contract tests for editor scheduling and synthesis progress support."""

from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch


IRODORI_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = IRODORI_ROOT.parent
sys.path.insert(0, str(IRODORI_ROOT / "wrapper"))
try:
    from editor_engine import EditorAdapter
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.__version__ = "test"
    torch_stub.version = types.SimpleNamespace(cuda=None)
    with patch.dict(sys.modules, {"torch": torch_stub}):
        from editor_engine import EditorAdapter
EDITOR_GLOBALS = EditorAdapter.__init__.__globals__


EDITOR_ENGINE = (IRODORI_ROOT / "wrapper" / "editor_engine.py").read_text(
    encoding="utf-8"
)
VOICEVOX_ENGINE = (IRODORI_ROOT / "wrapper" / "voicevox_engine.py").read_text(
    encoding="utf-8"
)
TTS_CLI = (IRODORI_ROOT / "wrapper" / "tts_cli.py").read_text(encoding="utf-8")
IRODORI_SETTINGS = (
    PROJECT_ROOT / "voicevox-editor" / "src" / "components" / "Talk" / "IrodoriSettings.vue"
).read_text(encoding="utf-8")
PROGRESS_VIEW = (
    PROJECT_ROOT / "voicevox-editor" / "src" / "components" / "ProgressView.vue"
).read_text(encoding="utf-8")
AUDIO_ITEM_TYPE = (
    PROJECT_ROOT / "voicevox-editor" / "src" / "store" / "type.ts"
).read_text(encoding="utf-8")
AUDIO_ITEM_SCHEMA = (
    PROJECT_ROOT / "voicevox-editor" / "src" / "domain" / "project" / "schema.ts"
).read_text(encoding="utf-8")
AUDIO_GENERATE = (
    PROJECT_ROOT / "voicevox-editor" / "src" / "store" / "audioGenerate.ts"
).read_text(encoding="utf-8")
IRODORI_CONSTANTS = (
    PROJECT_ROOT / "voicevox-editor" / "src" / "domain" / "irodori.ts"
).read_text(encoding="utf-8")
AUDIO_INFO = (
    PROJECT_ROOT / "voicevox-editor" / "src" / "components" / "Talk" / "AudioInfo.vue"
).read_text(encoding="utf-8")


class EditorScheduleProgressTests(unittest.TestCase):
    def test_editor_runtime_log_is_bounded_and_maps_known_phases_monotonically(self):
        adapter = EditorAdapter.__new__(EditorAdapter)
        adapter.progress_lock = threading.Lock()
        adapter.progress = dict(active=True, percent=0, stage="generating audio")

        adapter._runtime_log("[runtime] tokenize_text: 1.0 ms")
        adapter._runtime_log("[runtime] sample_rf: 2.0 ms")
        adapter._runtime_log("[runtime] tokenize_text: 3.0 ms")

        self.assertEqual(list(adapter.logs)[-3:], [
            "[runtime] tokenize_text: 1.0 ms",
            "[runtime] sample_rf: 2.0 ms",
            "[runtime] tokenize_text: 3.0 ms",
        ])
        self.assertGreaterEqual(adapter.progress["percent"], 70)
        self.assertEqual(adapter.progress["percent"], adapter._progress_percent)

    def test_editor_status_exposes_recent_console_lines_with_progress(self):
        self.assertRegex(
            EDITOR_ENGINE,
            r"logs\s*=\s*(?:deque|collections\.deque)",
            "editor must keep a bounded in-memory console buffer",
        )
        status_start = EDITOR_ENGINE.index("def status")
        status = EDITOR_ENGINE[status_start:]
        self.assertRegex(status, r"logs|console")

    def test_runtime_log_callback_is_forwarded_to_runtime_synthesize(self):
        voicevox_synthesize = VOICEVOX_ENGINE[VOICEVOX_ENGINE.index("def synthesize(self, query: dict, speaker_id: int)") :]
        # 文字数で切るとメソッドが伸びただけで外れるので、次のクラス定義までを見る。
        voicevox_synthesize = voicevox_synthesize[: voicevox_synthesize.index("\nclass ")]
        self.assertIn(
            "log_fn=self.progress_callback",
            voicevox_synthesize,
            "VoicevoxAdapter must pass its progress callback into IrodoriTTS",
        )
        self.assertRegex(
            TTS_CLI,
            r"def synthesize\([\s\S]{0,900}log_fn",
            "IrodoriTTS.synthesize must accept the runtime log callback",
        )
        radeon = TTS_CLI[TTS_CLI.index("class RadeonBackend") :]
        self.assertRegex(radeon, r"runtime\.synthesize\(request,\s*log_fn=log_fn\)")

    def test_progress_view_keeps_stage_without_console(self):
        self.assertNotIn("コンソール", PROGRESS_VIEW)
        self.assertNotIn("backendProgress.logs", PROGRESS_VIEW)
        self.assertIn("backendProgress.stage", PROGRESS_VIEW)

    def test_editor_adapter_has_bounded_synthesis_slots(self):
        init = EDITOR_ENGINE[EDITOR_ENGINE.index("def __init__(self):") :]
        self.assertRegex(
            init[:700],
            r"self\.synthesis_slots\s*=\s*threading\.BoundedSemaphore\(2\)",
            "editor adapter must expose the bounded synthesis semaphore used by Handler",
        )

    def test_set_progress_resets_percent_when_new_work_starts(self):
        adapter = EditorAdapter.__new__(EditorAdapter)
        adapter.progress_lock = threading.Lock()
        adapter.operation_lock = threading.Lock()
        adapter.progress = dict(active=False, percent=100, stage="complete")

        adapter._set_progress("new", 0)

        self.assertEqual(adapter.progress["percent"], 0)

    def test_synthesize_resets_completed_progress_before_work_starts(self):
        adapter = EditorAdapter.__new__(EditorAdapter)
        adapter.progress_lock = threading.Lock()
        adapter.operation_lock = threading.Lock()
        adapter.progress = dict(active=False, percent=100, stage="complete")
        observed = []

        def fake_synthesize(query, speaker_id):
            observed.append(dict(adapter.progress))
            return b"audio"

        with patch.object(adapter, "_synthesize", side_effect=fake_synthesize):
            result = adapter.synthesize({}, 0)

        self.assertEqual(result, b"audio")
        self.assertEqual(observed, [dict(active=True, percent=0, stage="validating synthesis")])

    def test_editor_defaults_to_sway_schedule(self):
        self.assertRegex(
            EDITOR_ENGINE,
            r"_line_schedule\([\s\S]*?query\.get\([\"']irodori_schedule[\"']\s*,\s*[\"']sway[\"']\)",
            "editor must default each line to the sway schedule",
        )
        self.assertRegex(
            EDITOR_ENGINE,
            r"settings\s*=[\s\S]*?sway_coeff\s*(?:=|:)\s*-1\.0",
            "editor defaults must use sway_coeff=-1.0",
        )
        self.assertRegex(EDITOR_ENGINE, r"seed\s*=\s*4763674")

    def test_editor_validates_and_forwards_schedule_settings(self):
        self.assertRegex(
            EDITOR_ENGINE,
            r"_line_schedule[\s\S]*?\(\"linear\", \"sway\"\)",
            "only linear and sway schedules may be accepted per line",
        )
        self.assertIn("irodori_schedule", EDITOR_ENGINE)
        self.assertIn("irodori_sway_coeff", EDITOR_ENGINE)

    def test_voicevox_forwards_schedule_arguments_to_sampling_request(self):
        synthesize = VOICEVOX_ENGINE[VOICEVOX_ENGINE.index("def synthesize"):]
        self.assertRegex(synthesize, r"t_schedule_mode")
        self.assertRegex(synthesize, r"sway_coeff")

        sampling_request = TTS_CLI[TTS_CLI.index("request = SamplingRequest"):]
        self.assertRegex(sampling_request, r"t_schedule_mode\s*=")
        self.assertRegex(sampling_request, r"sway_coeff\s*=")

    def test_editor_exposes_progress_object_in_status(self):
        self.assertRegex(
            EDITOR_ENGINE,
            r"progress\s*=\s*(?:\{[^}]*|dict\([^)]*)\bactive\b",
            "editor must maintain active, percent, and stage progress fields",
        )
        progress_start = EDITOR_ENGINE.index("progress")
        progress_declaration = EDITOR_ENGINE[progress_start : progress_start + 400]
        self.assertIn("percent", progress_declaration)
        self.assertIn("stage", progress_declaration)
        status_start = EDITOR_ENGINE.index("def status")
        status_end = EDITOR_ENGINE.find("\n    def ", status_start + 1)
        status = EDITOR_ENGINE[status_start:] if status_end < 0 else EDITOR_ENGINE[status_start:status_end]
        self.assertRegex(status, r"\bprogress\b")

    def test_each_synthesis_resets_progress_before_monotonic_updates(self):
        synthesize = EDITOR_ENGINE[EDITOR_ENGINE.index("def synthesize"):]
        self.assertRegex(
            synthesize,
            r"def synthesize[\s\S]*?_set_progress\([^\n]*,\s*0\s*\)",
            "each synthesis must reset progress to 0 before reporting progress",
        )

    def test_settings_ui_has_schedule_select_and_determinate_percentage(self):
        global_settings = (PROJECT_ROOT / 'voicevox-editor/src/components/Talk/IrodoriGlobalSettings.vue').read_text(encoding='utf-8')
        self.assertRegex(
            IRODORI_SETTINGS,
            r"QSelect[\s\S]*?(?:scheduleValue|settings\.t_schedule_mode)",
            "IrodoriSettings must expose a per-line schedule QSelect",
        )
        self.assertRegex(IRODORI_SETTINGS, r"sway")
        self.assertRegex(IRODORI_SETTINGS, r"linear")
        self.assertRegex(
            global_settings,
            r"Q(?:Linear|Circular)Progress[\s\S]*?:value=",
            "progress must be determinate rather than indeterminate",
        )
        self.assertRegex(global_settings, r"percent|%")

    def test_progress_view_polls_active_engine_irodori_settings_endpoint(self):
        self.assertRegex(
            PROGRESS_VIEW,
            r"createEngineUrl|engineInfos|altPortInfos",
            "ProgressView must resolve the active engine URL",
        )
        self.assertRegex(
            PROGRESS_VIEW,
            r"import \{ fetchIrodoriStatus \} from \"@/helpers/irodoriEngine\"",
            "ProgressView must poll through the shared Irodori client (it attaches"
            " the session token)",
        )
        self.assertRegex(
            PROGRESS_VIEW,
            r"fetchIrodoriStatus\([\s\S]{0,120}\)",
            "ProgressView must poll the active engine's Irodori settings endpoint",
        )
        self.assertRegex(
            PROGRESS_VIEW,
            r"setInterval\([\s\S]{0,240}(?:pollBackendProgress|fetchIrodoriStatus)",
            "Irodori progress polling must repeat while the overlay is mounted",
        )

    def test_progress_view_displays_backend_percent_or_stage_in_generation_overlay(self):
        overlay = PROGRESS_VIEW[PROGRESS_VIEW.index("<template>") :]
        self.assertRegex(
            overlay,
            r"progress\.(?:percent|stage)|(?:percent|stage)[\s\S]{0,100}progress",
            "the generation overlay must display backend percent or stage",
        )

    def test_audio_item_and_project_schema_preserve_per_line_irodori_settings(self):
        audio_item = AUDIO_ITEM_TYPE[
            AUDIO_ITEM_TYPE.index("export type AudioItem") :
        ]
        self.assertRegex(audio_item, r"irodori\??\s*:\s*\{")
        self.assertRegex(audio_item, r"caption\??\s*:\s*string")
        self.assertRegex(audio_item, r"referenceAudio\??\s*:\s*\{")

        schema = AUDIO_ITEM_SCHEMA[
            AUDIO_ITEM_SCHEMA.index("export const audioItemSchema") :
        ]
        self.assertRegex(schema, r"irodori\s*:\s*z\.object\(")
        self.assertRegex(schema, r"caption\s*:\s*z\.string\(\)\.max\(2000\)\.optional\(\)")
        self.assertRegex(schema, r"referenceAudio\s*:\s*z\.object\(")

    def test_audio_item_and_project_schema_preserve_optional_per_line_seed(self):
        audio_item = AUDIO_ITEM_TYPE[
            AUDIO_ITEM_TYPE.index("export type AudioItem") :
        ]
        self.assertRegex(audio_item, r"seed\??\s*:\s*number\s*\|\s*null")

        schema = AUDIO_ITEM_SCHEMA[
            AUDIO_ITEM_SCHEMA.index("export const audioItemSchema") :
        ]
        self.assertRegex(
            schema,
            r"seed\s*:\s*z\.number\(\)\.nullable\(\)\.optional\(\)",
        )

    def test_audio_generate_forwards_selected_audio_item_irodori_settings(self):
        synthesis = AUDIO_GENERATE[AUDIO_GENERATE.index('invoke("synthesis")') :]
        self.assertRegex(synthesis, r"irodoriCaption\s*:\s*audioItem\.irodori\?\.caption")
        self.assertRegex(synthesis, r"irodoriReferenceAudio\s*:\s*audioItem\.irodori\?\.referenceAudio")

    def test_audio_generate_forwards_seed_and_randomizes_blank_seed_cache_nonce(self):
        synthesis = AUDIO_GENERATE[AUDIO_GENERATE.index('invoke("synthesis")') :]
        self.assertRegex(synthesis, r"irodori_seed\s*:\s*effectiveSeed")
        self.assertIn("IRODORI_DEFAULT_SEED = 4763674", IRODORI_CONSTANTS)
        self.assertRegex(
            AUDIO_GENERATE,
            r"(?:irodori_seed|seed)[\s\S]{0,220}(?:Math\.random|crypto\.randomUUID|randomUUID)",
            "blank per-line seeds must receive a random cache nonce",
        )
        self.assertRegex(
            AUDIO_GENERATE,
            r"(?:cacheNonce|CacheNonce|nonce|Nonce)[\s\S]{0,220}(?:generateTempUniqueId|cache|Cache)|(?:cache|Cache)[\s\S]{0,220}(?:cacheNonce|CacheNonce|nonce|Nonce)",
            "the random value must participate in the synthesis cache key",
        )

    def test_audio_info_passes_active_key_and_exposes_reference_controls(self):
        self.assertRegex(
            AUDIO_INFO,
            r"<IrodoriSettings[\s\S]*:activeAudioKey=\"activeAudioKey\"",
        )
        self.assertRegex(IRODORI_SETTINGS, r"referenceFile|accept=\"audio")

    def test_speed_slider_is_in_irodori_settings_before_seed_and_not_audio_info(self):
        settings_template = IRODORI_SETTINGS[: IRODORI_SETTINGS.index("<script")]
        audio_info_template = AUDIO_INFO[: AUDIO_INFO.index("<script")]
        speed_start = settings_template.index("話速")
        seed_start = settings_template.index("seedText")

        self.assertLess(speed_start, seed_start)
        self.assertLess(
            settings_template.index("<QSlider", speed_start),
            seed_start,
        )
        self.assertNotIn("話速", audio_info_template)

    def test_irodori_settings_keeps_seed_input_with_caption_and_reference_audio(self):
        self.assertRegex(IRODORI_SETTINGS, r"captionText|キャプション")
        self.assertRegex(IRODORI_SETTINGS, r"referenceFile|accept=\"audio")
        self.assertRegex(
            IRODORI_SETTINGS,
            r"seedText[\s\S]{0,180}(?:blank|Blank|random|Random|空欄|ランダム)",
        )

    def test_irodori_settings_exposes_caption_textarea(self):
        self.assertRegex(IRODORI_SETTINGS, r"captionText[\s\S]{0,250}type=\"textarea\"")

    def test_irodori_settings_exposes_reference_audio_controls(self):
        self.assertRegex(IRODORI_SETTINGS, r"referenceFile|handleReferenceFileChange|accept=\"audio")
        self.assertRegex(IRODORI_SETTINGS, r"clearReferenceAudio|解除")

    def test_background_prewarm_coalesces_duplicate_requests_and_uses_daemon_worker(self):
        adapter = EditorAdapter.__new__(EditorAdapter)
        adapter._prewarm_lock = threading.Lock()
        adapter._prewarm_scheduled = False
        adapter.progress_lock = threading.Lock()
        adapter.progress = dict(active=False, percent=0, stage="idle")

        with patch.object(EDITOR_GLOBALS["threading"], "Thread") as thread_factory:
            first = adapter.schedule_prewarm()
            second = adapter.schedule_prewarm()

        self.assertTrue(first)
        self.assertFalse(second)
        thread_factory.assert_called_once()
        self.assertTrue(thread_factory.call_args.kwargs["daemon"])
        thread_factory.return_value.start.assert_called_once_with()

    def test_background_prewarm_resets_completed_progress_before_loading(self):
        adapter = EditorAdapter.__new__(EditorAdapter)
        adapter._prewarm_lock = threading.Lock()
        adapter._prewarm_scheduled = False
        adapter.progress_lock = threading.Lock()
        adapter.progress = dict(active=False, percent=100, stage="complete")

        with patch.object(EDITOR_GLOBALS["threading"], "Thread"):
            self.assertTrue(adapter.schedule_prewarm())

        self.assertEqual(
            adapter.progress,
            dict(active=True, percent=0, stage="preloading model/backend"),
        )

    def test_prewarm_builds_resident_delegate_without_synthesis(self):
        adapter = EditorAdapter.__new__(EditorAdapter)
        adapter.settings = dict(backend="cpu", model="model.safetensors")
        adapter.state_lock = threading.RLock()
        adapter._model_info_cache = {}
        adapter._resolve_local_model = lambda source: Path("model.safetensors")
        adapter._plan_path = lambda: Path("fallback.plan")
        adapter._set_progress = Mock()
        adapter._runtime_log = Mock()
        fake_delegate = types.SimpleNamespace(synthesize=Mock())

        constructor = Mock(return_value=fake_delegate)
        with patch.dict(EDITOR_GLOBALS, {
            "VoicevoxAdapter": constructor,
            "resolve_embed_dirs": lambda: [Path("speakers")],
        }):
            result = adapter._build_delegate()

        self.assertIs(result, fake_delegate)
        constructor.assert_called_once()
        fake_delegate.synthesize.assert_not_called()

    def test_voicevox_materializes_data_url_reference_audio_and_passes_ref_wav(self):
        self.assertIn(";base64,", VOICEVOX_ENGINE)
        self.assertRegex(VOICEVOX_ENGINE, r"base64\.(?:b64decode|urlsafe_b64decode)")
        self.assertRegex(VOICEVOX_ENGINE, r"ref_wav\s*=")
        self.assertRegex(VOICEVOX_ENGINE, r"ref_wav[\s\S]*?self\.tts\.synthesize")

    def test_voicevox_decodes_and_materializes_reference_audio_before_synthesis(self):
        synthesize = VOICEVOX_ENGINE[VOICEVOX_ENGINE.index("def synthesize") :]
        self.assertRegex(synthesize, r"irodori_reference_audio")
        self.assertRegex(synthesize, r"(?:tempfile|NamedTemporaryFile|\.write_bytes\(|open\([^)]*wb)")
        self.assertRegex(synthesize, r"ref_wav\s*=\s*[^\n]+")
        self.assertRegex(synthesize, r"ref_wav[\s\S]{0,500}self\.tts\.synthesize")

    def test_tts_cli_accepts_ref_wav_without_forcing_speaker_embedding(self):
        irodori_tts = TTS_CLI[TTS_CLI.index("class IrodoriTTS") :]
        synthesize = irodori_tts[irodori_tts.index("def synthesize(") :]
        self.assertRegex(synthesize, r"ref_wav\s*:\s*Optional\[str\]")
        self.assertRegex(synthesize, r"ref_wav\s*=\s*ref_wav|ref_wav=ref_wav")
        self.assertRegex(
            synthesize,
            r"ref_wav[\s\S]{0,120}self\.cassette\.get|self\.cassette\.get[\s\S]{0,80}ref_wav",
        )

    def test_tts_cli_allows_blank_seed_and_reference_audio_bypasses_cassette(self):
        irodori_tts = TTS_CLI[TTS_CLI.index("class IrodoriTTS") :]
        synthesize = irodori_tts[irodori_tts.index("def synthesize(") :]
        self.assertRegex(synthesize, r"seed\s*:\s*Optional\[int\]\s*=\s*1001")
        self.assertRegex(synthesize, r"ref_wav\s*:\s*Optional\[str\]")
        self.assertRegex(synthesize, r"ref_wav\s*=\s*ref_wav|ref_wav=ref_wav")

    def test_editor_forwards_per_line_text_and_independent_cfg_scales(self):
        self.assertIn("irodoriCfgText", AUDIO_GENERATE)
        self.assertIn("irodoriCfgCaption", AUDIO_GENERATE)
        self.assertIn("irodoriCfgSpeaker", AUDIO_GENERATE)
        self.assertIn("irodori_cfg_text", VOICEVOX_ENGINE)
        self.assertIn("cfg_scale_text", TTS_CLI)
        self.assertIn("cfg_scale_caption", TTS_CLI)
        self.assertIn("cfg_scale_speaker", TTS_CLI)
        self.assertRegex(
            EDITOR_ENGINE,
            r"irodori_cfg_text[\s\S]{0,500}irodori_cfg_caption[\s\S]{0,500}irodori_schedule",
        )

    def test_per_line_project_schema_and_settings_ui_expose_cfg_values(self):
        self.assertRegex(AUDIO_ITEM_TYPE, r"cfgText\??\s*:\s*number")
        self.assertRegex(AUDIO_ITEM_TYPE, r"cfgCaption\??\s*:\s*number")
        self.assertRegex(AUDIO_ITEM_TYPE, r"cfgSpeaker\??\s*:\s*number")
        self.assertIn("cfgText", AUDIO_ITEM_SCHEMA)
        self.assertIn("cfgCaption", AUDIO_ITEM_SCHEMA)
        self.assertIn("cfgSpeaker", AUDIO_ITEM_SCHEMA)
        self.assertRegex(IRODORI_SETTINGS, r"テキストCFG")
        self.assertRegex(IRODORI_SETTINGS, r"キャプションCFG")
        self.assertRegex(IRODORI_SETTINGS, r"スピーカーCFG")
        self.assertNotRegex(IRODORI_SETTINGS, r"label=\"テキスト\"|textText|saveText")
        self.assertRegex(IRODORI_SETTINGS, r"既定値: 3")
        self.assertRegex(IRODORI_SETTINGS, r"既定値: 5")
        self.assertNotRegex(IRODORI_SETTINGS, r"全行共通の生成設定[\s\S]{0,200}label=\"CFG\"")

    def test_editor_rejects_non_finite_per_line_cfg_values(self):
        self.assertRegex(EDITOR_ENGINE, r"irodori_cfg_text")
        self.assertRegex(EDITOR_ENGINE, r"math\.isfinite")

    def test_continuous_player_filters_whitespace_only_lines(self):
        player = (PROJECT_ROOT / "voicevox-editor" / "src" / "store" / "audioContinuousPlayer.ts").read_text(
            encoding="utf-8"
        )
        self.assertIn("filterNonEmptyAudioKeys", player)
        self.assertRegex(player, r"text\.trim\(\)\.length")


if __name__ == "__main__":
    unittest.main()
