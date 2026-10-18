"""コンソール表示（console_ui.py）と tts_cli の出力（--json・終了コード）のテスト。

端末に直接出すときだけ装飾し、ログファイルやパイプへは従来の 1 行テキスト（結果は JSON）のままであること、
装飾のときもマークアップ記号（[...]）を含む文字列で壊れないこと、
open_browser.bat のコンソールに流すログ整形（log_line_markup）が各行の種類を取り違えないことを確かめる。
"""
import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))
import console_ui  # noqa: E402


def _capture(fn, *, fancy: bool):
    """fn を実行して (stdout, stderr) を返す。fancy=True のときは rich を端末扱いで動かす。"""
    out, err = io.StringIO(), io.StringIO()
    console_ui._consoles.clear()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        with mock.patch.object(console_ui, "fancy", lambda stderr=False: fancy):
            # rich の Console は作った時点の sys.stdout/stderr に書く（リダイレクト後に作られる）
            fn()
    console_ui._consoles.clear()
    return out.getvalue(), err.getvalue()


class PlainModeTest(unittest.TestCase):
    def test_stage_keeps_the_log_line(self):
        out, _ = _capture(lambda: console_ui.stage("[runtime] sample_rf: 338.4 ms"), fancy=False)
        self.assertRegex(out, r"^\[irodori\] \d\d:\d\d:\d\d stage: \[runtime\] sample_rf: 338\.4 ms\n$")

    def test_http_log_keeps_the_log_line(self):
        _, err = _capture(lambda: console_ui.http_log('"GET /version HTTP/1.1" 200 -'), fancy=False)
        self.assertEqual(err, '[voicevox] "GET /version HTTP/1.1" 200 -\n')

    def test_working_prints_nothing(self):
        def run():
            with console_ui.working("読み込み中"):
                pass
        out, err = _capture(run, fancy=False)
        self.assertEqual((out, err), ("", ""))

    def test_synth_done_keeps_the_log_line(self):
        out, _ = _capture(lambda: console_ui.synth_done(
            "[irodori] synth done: total=1.00s", speaker="s", total=1.0, audio=2.0, rtf=0.5, chars=3,
            steps=16, backend="trt", model="m", seed=1, phases=[]), fancy=False)
        self.assertEqual(out, "[irodori] synth done: total=1.00s\n")

    def test_run_output_helpers_print_nothing_when_not_on_a_terminal(self):
        def run():
            console_ui.run_header("trt", "tsukuyomi", 8, 1001, 3)
            console_ui.line_done(1, 3, "こんにちは", 2.0, 0.5)
            console_ui.run_summary(3, 6.0, 1.5, "outputs")
        self.assertEqual(_capture(run, fancy=False), ("", ""))

    def test_error_card_is_one_plain_line_on_stderr(self):
        out, err = _capture(lambda: console_ui.error_card("話者が見つかりません", ["a"], "hint"), fancy=False)
        self.assertEqual(out, "")
        self.assertEqual(err, "error: 話者が見つかりません\n")

    def test_force_plain_turns_decoration_off_even_on_a_terminal(self):
        was = console_ui._plain
        try:
            console_ui.force_plain()
            with mock.patch("sys.stdout.isatty", create=True, return_value=True):
                self.assertFalse(console_ui.fancy())
        finally:
            console_ui._plain = was


class FancyModeTest(unittest.TestCase):
    def test_square_brackets_in_text_do_not_break_markup(self):
        out, _ = _capture(lambda: console_ui.stage("[runtime] decode: [/bold] [red]x"), fancy=True)
        self.assertIn("[/bold]", out)
        self.assertIn("[red]x", out)

    def test_http_log_shows_status(self):
        _, err = _capture(lambda: console_ui.http_log('"POST /synthesis?speaker=1 HTTP/1.1" 500 -'),
                          fancy=True)
        self.assertIn("POST /synthesis?speaker=1 HTTP/1.1", err)
        self.assertIn("500", err)

    def test_http_log_without_status_does_not_fail(self):
        _, err = _capture(lambda: console_ui.http_log("stream client went away"), fancy=True)
        self.assertIn("stream client went away", err)

    def test_synth_done_and_banner_render(self):
        def run():
            console_ui.synth_done("x", speaker="つくよみ", total=1.5, audio=3.0, rtf=0.5, chars=12,
                                  steps=16, backend="trt", model="m [x]", seed=1, phases=["a=0.1s"])
            console_ui.banner("Engine", [("url", "http://127.0.0.1:1")], "plain")
        out, _ = _capture(run, fancy=True)
        self.assertIn("0.50x", out)
        self.assertIn("http://127.0.0.1:1", out)
        self.assertIn("m [x]", out)

    def test_speaker_table_marks_default(self):
        out, _ = _capture(lambda: console_ui.speaker_table(["root"], ["a", "b"], "b"), fancy=True)
        self.assertIn("default", out)

    def test_run_output_shows_each_line_and_the_summary(self):
        def run():
            console_ui.run_header("trt", "tsukuyomi", 8, 1001, 2)
            console_ui.line_done(1, 2, "今日はいい天気ですね。", 2.04, 0.82)
            console_ui.run_summary(2, 4.0, 1.6, "outputs")
        out, _ = _capture(run, fancy=True)
        self.assertIn("1/2", out)
        self.assertIn("今日はいい天気ですね。", out)
        self.assertIn("RTF", out)
        self.assertIn("2 件できました", out)
        self.assertNotIn("♪", out)
        self.assertNotIn("✿", out)

    def test_long_text_is_clipped_on_one_line(self):
        out, _ = _capture(lambda: console_ui.line_done(1, 1, "あ" * 100, 1.0, 0.5), fancy=True)
        self.assertIn("…", out)
        self.assertNotIn("あ" * 60, out)

    def test_error_card_has_a_face_and_suggestions(self):
        out, err = _capture(lambda: console_ui.error_card("話者「x」が見つかりません", ["tsukuyomi"], "一覧を見る"),
                            fancy=True)
        self.assertIn("(；ω；)", err)
        self.assertIn("tsukuyomi", err)
        self.assertEqual(out, "")


class PlaybackTest(unittest.TestCase):
    def test_stop_without_playing_is_silent(self):
        console_ui.stop_wav()


class LogViewerTest(unittest.TestCase):
    """open_browser.bat のコンソールに流す素のログの整形。実際のエンジンのログ行で確かめる。"""

    def _render(self, label, line):
        out, _ = _capture(lambda: console_ui.print_markup(console_ui.log_line_markup(label, line)), fancy=True)
        return out

    def test_stage_line_keeps_time_and_message(self):
        out = self._render("engine", "[irodori] 06:08:17 stage: TensorRT: 対応する plan を再利用")
        self.assertIn("06:08:17", out)
        self.assertIn("対応する plan を再利用", out)

    def test_fine_runtime_lines_are_hidden_because_the_summary_covers_them(self):
        for line in ("[irodori] 07:04:43 stage: [runtime] sample_rf: 178.8 ms",
                     "[irodori] 07:04:43 stage: info: predicted duration frames=53.6"):
            self.assertEqual(console_ui.log_line_markup("engine", line), "")

    def test_runtime_error_is_not_hidden(self):
        line = "[irodori] 07:04:43 stage: [runtime] decode failed: error in codec"
        self.assertIn("error in codec", self._render("engine", line))

    def test_synth_done_line_becomes_a_summary(self):
        line = ("[irodori] synth done: total=0.32s audio=2.44s rtf=0.13x wav=234284B speaker=tsukuyomi "
                "text=13chars steps=16 schedule=sway seconds=None seed=4763674 cfg=3.0/3.0/5.0 backend=trt "
                "model=rose3/Irodori-TTS-v4.1-Small-Blockwise | phases: predict_duration=0.01s sample_rf=0.22s")
        out = self._render("engine", line)
        self.assertIn("tsukuyomi", out)
        self.assertIn("0.13x", out)
        self.assertIn("13字", out)
        self.assertIn("sample_rf=0.22s", out)
        self.assertNotIn("wav=234284B", out)

    def test_malformed_synth_line_is_still_shown(self):
        out = self._render("engine", "[irodori] synth done: total=oops audio=2.44s")
        self.assertIn("synth done", out)

    def test_successful_polling_is_hidden_but_failures_and_posts_are_shown(self):
        for line in ('[voicevox] "GET /version HTTP/1.1" 200 -',
                     '[voicevox] "OPTIONS /synthesis HTTP/1.1" 204 -',
                     '[voicevox] "POST /audio_query?text=a HTTP/1.1" 200 -'):
            self.assertEqual(console_ui.log_line_markup("engine", line), "")
        failed = self._render("engine", '[voicevox] "GET /irodori/voices HTTP/1.1" 500 -')
        post = self._render("engine", '[voicevox] "POST /synthesis?speaker=1 HTTP/1.1" 200 -')
        self.assertIn("500", failed)
        self.assertIn("POST /synthesis", post)
        self.assertNotIn("?speaker", post)

    def test_only_slow_phases_are_listed(self):
        line = ("[irodori] synth done: total=0.24s audio=2.40s rtf=0.10x wav=1B speaker=s text=3chars steps=16 "
                "schedule=sway seconds=None seed=1 cfg=3.0/3.0/5.0 backend=trt model=m | phases: "
                "tokenize_text=0.00s sample_rf=0.18s total_to_decode=0.00s decode_latent=0.02s")
        out = self._render("engine", line)
        self.assertIn("sample_rf=0.18s", out)
        self.assertIn("decode_latent=0.02s", out)
        self.assertNotIn("tokenize_text", out)
        self.assertNotIn("total_to_decode", out)

    def test_label_and_message_are_separated(self):
        out = self._render("engine", "[irodori] 06:08:17 stage: initializing model/backend")
        self.assertRegex(out, r"engine\s+06:08:17")

    def test_blockwise_line_is_called_out(self):
        out = self._render("engine", "[voicevox] [blockwise] continuing from the remembered latent of the previous line")
        self.assertIn("前のセリフから続けて生成", out)

    def test_tokens_line_shows_the_split(self):
        out = self._render("engine", "[irodori] tokens (4): 今日は|いい天気|ですね|。")
        self.assertIn("4 トークン", out)
        self.assertIn("いい天気", out)

    def test_markup_characters_in_a_log_line_do_not_break_it(self):
        out = self._render("ui", "[plugin] ready in 300 ms [/x] [red]")
        self.assertIn("[/x]", out)
        self.assertIn("[red]", out)

    def test_ui_error_is_not_swallowed(self):
        out = self._render("ui", "Error: Failed to resolve import")
        self.assertIn("Failed to resolve import", out)


class CliOutputTest(unittest.TestCase):
    """tts_cli の話者の確認・JSON 出力・終了コード（モデルは読み込まずに確かめられる範囲）。"""

    @classmethod
    def setUpClass(cls):
        import tts_cli
        cls.cli = tts_cli

    def _run(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = self.cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def _with_speakers(self, names):
        class FakeCassette:
            dirs = [Path("speakers")]

            def __init__(self, _dirs):
                self.speakers = names

            def has(self, name):
                return name in names
        return mock.patch.object(self.cli, "SpeakerCassette", FakeCassette)

    def test_list_speakers_json_has_default_and_dirs(self):
        with self._with_speakers(["vivian", "tsukuyomi"]):
            code, out, _ = self._run("--list-speakers", "--json")
        data = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual(data["speakers"], ["vivian", "tsukuyomi"])
        self.assertEqual(data["default"], "tsukuyomi")
        self.assertIn("search_dirs", data)

    def test_unknown_speaker_exits_3_with_suggestions_as_json(self):
        with self._with_speakers(["tsukuyomi", "vivian"]):
            code, out, _ = self._run("--json", "--speaker", "tsukumyomi", "--text", "x")
        error = json.loads(out)["error"]
        self.assertEqual(code, self.cli.EXIT_SPEAKER)
        self.assertEqual(error["code"], "speaker_not_found")
        self.assertEqual(error["suggestions"], ["tsukuyomi"])
        self.assertFalse(json.loads(out)["ok"])

    def test_missing_speaker_and_text_exit_2(self):
        with self._with_speakers(["tsukuyomi"]):
            code, out, _ = self._run("--json", "--text", "x")
            self.assertEqual((code, json.loads(out)["error"]["code"]), (2, "missing_speaker"))
            code, out, _ = self._run("--json", "--speaker", "tsukuyomi")
            self.assertEqual((code, json.loads(out)["error"]["code"]), (2, "missing_text"))

    def test_unreadable_or_empty_text_file_exits_2(self):
        import tempfile
        with self._with_speakers(["tsukuyomi"]):
            code, out, _ = self._run("--json", "--speaker", "tsukuyomi", "--text-file", "no-such-file.txt")
            self.assertEqual((code, json.loads(out)["error"]["code"]), (2, "text_file_unreadable"))
            with tempfile.TemporaryDirectory() as tmp:
                empty = Path(tmp) / "empty.txt"
                empty.write_text("\n  \n", encoding="utf-8")
                code, out, _ = self._run("--json", "--speaker", "tsukuyomi", "--text-file", str(empty))
            self.assertEqual((code, json.loads(out)["error"]["code"]), (2, "text_file_empty"))

    def test_model_load_failure_exits_4(self):
        with self._with_speakers(["tsukuyomi"]), \
                mock.patch.object(self.cli, "_build_tts", side_effect=FileNotFoundError("model.safetensors")):
            code, out, _ = self._run("--json", "--speaker", "tsukuyomi", "--text", "x")
        self.assertEqual((code, json.loads(out)["error"]["code"]), (4, "environment"))

    def test_generation_json_line_has_no_tensor_and_exit_0(self):
        class FakeTTS:
            class backend:
                name = "torch"

            def synthesize(self, **kw):
                import torch
                return {"backend": "torch", "audio_s": 2.0, "wall_s": 0.5, "latent": torch.zeros(3)}

            def close(self):
                pass

        with self._with_speakers(["tsukuyomi"]), mock.patch.object(self.cli, "_build_tts", return_value=FakeTTS()):
            code, out, _ = self._run("--json", "--speaker", "tsukuyomi", "--text", "x",
                                     "--out-dir", "out-test")
        row = json.loads(out)
        self.assertEqual(code, 0)
        self.assertTrue(row["ok"])
        self.assertEqual(row["rtf"], 0.25)
        self.assertNotIn("latent", row)
        self.assertTrue(row["wav"].endswith("tsukuyomi_1001.wav"))

    def test_generation_failure_exits_5_and_names_the_line(self):
        class FakeTTS:
            class backend:
                name = "torch"

            def synthesize(self, **kw):
                raise RuntimeError("out of memory")

            def close(self):
                pass

        with self._with_speakers(["tsukuyomi"]), mock.patch.object(self.cli, "_build_tts", return_value=FakeTTS()):
            code, out, _ = self._run("--json", "--speaker", "tsukuyomi", "--text", "x")
        error = json.loads(out)["error"]
        self.assertEqual((code, error["code"], error["index"]), (5, "synthesis_failed", 1))
        self.assertIn("out of memory", error["message"])

    def test_model_is_loaded_only_once_for_generation(self):
        class FakeTTS:
            class backend:
                name = "torch"

            def synthesize(self, **kw):
                return {"backend": "torch", "audio_s": 1.0, "wall_s": 0.1}

            def close(self):
                pass

        build = mock.Mock(return_value=FakeTTS())
        with self._with_speakers(["tsukuyomi"]), mock.patch.object(self.cli, "_build_tts", build):
            self._run("--json", "--speaker", "tsukuyomi", "--text", "x")
        self.assertEqual(build.call_count, 1)


if __name__ == "__main__":
    unittest.main()
