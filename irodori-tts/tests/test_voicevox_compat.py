"""VOICEVOX 互換 API の契約テスト。

一般の VOICEVOX クライアント（Origin もトークンも付けない）がそのまま使えること、
ブラウザ経由の要求は従来どおり Origin とトークンで守られること、
間違った入力が既定値で黙って合成されず 422 になることを確かめる。
"""
import json
import sys
import threading
import types
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))
try:
    import editor_engine
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.__version__ = "test"
    torch_stub.version = types.SimpleNamespace(cuda=None)
    sys.modules["torch"] = torch_stub
    import editor_engine
import voicevox_engine as engine

EDITOR_ORIGIN = "http://localhost:5173"


class _FakeAdapter:
    backend_name = "test"
    speakers_json = []

    def __init__(self):
        self.synthesis_slots = threading.BoundedSemaphore(2)
        self.queries = []

    def synthesize(self, query, speaker):
        self.queries.append((query, speaker))
        return b"RIFFtest"


class _Server:
    def __init__(self, handler):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_port

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, method, path, body=None, headers=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        all_headers = {"Host": f"127.0.0.1:{self.port}", **(headers or {})}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            all_headers["Content-Type"] = "application/json"
        conn.request(method, path, data, all_headers)
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        ctype = response.getheader("Content-Type", "")
        payload = json.loads(raw) if raw and "json" in ctype else raw
        return response.status, payload, response


class VoicevoxCompatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = _FakeAdapter()
        engine.Handler.adapter = cls.adapter
        cls.srv = _Server(engine.Handler)

    @classmethod
    def tearDownClass(cls):
        cls.srv.close()

    def setUp(self):
        self.adapter.queries.clear()

    def post(self, path, body=None, headers=None):
        return self.srv.request("POST", path, body, headers)

    def synth(self, body, path="/synthesis?speaker=0"):
        return self.post(path, body)

    def assert_422(self, result, loc):
        status, payload, _ = result
        self.assertEqual(status, 422, payload)
        self.assertIn(loc, [item["loc"] for item in payload["detail"]])
        return payload["detail"]

    # --- 認証 -----------------------------------------------------------

    def test_stock_voicevox_client_works_without_token(self):
        status, query, _ = self.post("/audio_query?text=hello&speaker=0")
        self.assertEqual(status, 200)
        status, wav, _ = self.synth(query)
        self.assertEqual((status, wav), (200, b"RIFFtest"))

    def test_browser_requests_still_need_allowed_origin_and_token(self):
        foreign = {"Origin": "https://attacker.example"}
        self.assertEqual(self.post("/audio_query?text=a&speaker=0", headers=foreign)[0], 403)
        self.assertEqual(self.post("/audio_query?text=a&speaker=0", headers={
            **foreign, "X-Irodori-Session": engine.SESSION_TOKEN})[0], 403)
        self.assertEqual(self.post("/audio_query?text=a&speaker=0",
                                   headers={"Origin": EDITOR_ORIGIN})[0], 403)
        self.assertEqual(self.post("/audio_query?text=a&speaker=0", headers={
            "Origin": EDITOR_ORIGIN, "X-Irodori-Session": engine.SESSION_TOKEN})[0], 200)

    def test_cross_site_browser_request_without_origin_is_rejected(self):
        # img タグなどの no-cors GET は Origin を付けないが Sec-Fetch-Site は付く。
        status, _, _ = self.srv.request(
            "GET", "/refresh", headers={"Sec-Fetch-Site": "cross-site"})
        self.assertEqual(status, 403)

    def test_dns_rebinding_host_is_rejected(self):
        status, _, _ = self.post("/audio_query?text=a&speaker=0",
                                 headers={"Host": "attacker.example"})
        self.assertEqual(status, 403)

    # --- VOICEVOX と同じ必須パラメータ ------------------------------------

    def test_audio_query_requires_text_and_speaker(self):
        self.assert_422(self.post("/audio_query?text=a"), ["query", "speaker"])
        self.assert_422(self.post("/audio_query?speaker=0"), ["query", "text"])
        self.assert_422(self.post("/audio_query?text=a&speaker=abc"), ["query", "speaker"])
        status, query, _ = self.post("/audio_query?text=&speaker=0")
        self.assertEqual((status, query["irodori_text"]), (200, ""))

    def test_synthesis_requires_integer_speaker(self):
        body = {"irodori_text": "a"}
        self.assert_422(self.synth(body, "/synthesis"), ["query", "speaker"])
        self.assert_422(self.synth(body, "/synthesis?speaker=abc"), ["query", "speaker"])
        self.assertEqual(self.adapter.queries, [])

    # --- 間違った入力を黙って既定値で合成しない ----------------------------

    def test_camel_case_field_is_rejected_with_suggestion(self):
        detail = self.assert_422(self.synth({"irodori_text": "a", "irodoriSteps": 999}),
                                 ["body", "irodoriSteps"])
        self.assertIn("irodori_steps", detail[0]["msg"])
        self.assert_422(self.synth({"irodori_text": "a", "irodori_step": 4}),
                        ["body", "irodori_step"])
        self.assertEqual(self.adapter.queries, [])

    def test_field_ranges_and_types(self):
        cases = {
            "irodori_steps": 999,
            "irodori_schedule": "bogus",
            "irodori_seconds": 0,
            "irodori_seed": 1.7,
            "irodori_cfg_text": 50,
            "irodori_speaker_strength": 5,
            "speedScale": 0,
        }
        for key, value in cases.items():
            with self.subTest(key=key):
                self.assert_422(self.synth({"irodori_text": "a", key: value}), ["body", key])
        self.assert_422(self.synth({"irodori_text": "a", "irodori_additional_speakers": [
            {"style_id": "x"}]}), ["body", "irodori_additional_speakers", 0, "style_id"])
        self.assert_422(self.synth({"irodori_text": "a", "irodori_reference_audio": {}}),
                        ["body", "irodori_reference_audio", "dataUrl"])
        self.assertEqual(self.adapter.queries, [])

    def test_editor_style_query_is_accepted(self):
        # エディタが実際に送る形。ここが 422 になるとエディタの合成が止まる。
        body = {
            "accent_phrases": [], "speedScale": 1.0, "pitchScale": 0.0,
            "intonationScale": 1.0, "volumeScale": 1.0, "prePhonemeLength": 0.1,
            "postPhonemeLength": 0.1, "pauseLengthScale": 1, "outputSamplingRate": 48000,
            "outputStereo": False, "kana": "こんにちは",
            "irodori_seed": None, "irodori_steps": 8, "irodori_schedule": "sway",
            "irodori_seconds": None, "irodori_caption": "落ち着いて",
            "irodori_caption_strength": 1, "irodori_cfg_text": 3, "irodori_cfg_caption": 3,
            "irodori_cfg_speaker": 5, "irodori_reference_strength": 1,
            "irodori_speaker_strength": 0.8, "irodori_secondary_speaker_strength": 0.5,
            "irodori_additional_speakers": [{"style_id": 1, "strength": 0.5}],
            "irodori_reference_audio": {"dataUrl": "data:audio/wav;base64,AA==",
                                        "mime": "audio/wav", "name": "a.wav"},
        }
        status, wav, _ = self.synth(body)
        self.assertEqual((status, wav), (200, b"RIFFtest"))

    def test_queue_full_tells_client_when_to_retry(self):
        slots = self.adapter.synthesis_slots
        self.adapter.synthesis_slots = threading.BoundedSemaphore(1)
        self.adapter.synthesis_slots.acquire()
        try:
            status, _, response = self.synth({"irodori_text": "a"})
        finally:
            self.adapter.synthesis_slots = slots
        self.assertEqual(status, 429)
        self.assertEqual(response.getheader("Retry-After"), "1")

    # --- 仕様の取得 -------------------------------------------------------

    def test_core_versions_and_openapi(self):
        status, versions, _ = self.srv.request("GET", "/core_versions")
        self.assertEqual((status, versions), (200, [engine.ENGINE_VERSION]))
        status, doc, _ = self.srv.request("GET", "/openapi.json")
        self.assertEqual(status, 200)
        properties = doc["components"]["schemas"]["AudioQuery"]["properties"]
        for name in engine.IRODORI_QUERY_FIELDS:
            self.assertIn(name, properties)
        self.assertIn("/synthesis", doc["paths"])


class EditorHandlerTests(unittest.TestCase):
    def test_unauthorized_timeline_get_is_403_not_404(self):
        editor_engine.EditorHandler.adapter = _FakeAdapter()
        srv = _Server(editor_engine.EditorHandler)
        self.addCleanup(srv.close)
        status, _, _ = srv.request("GET", "/irodori/timeline?text=a",
                                   headers={"Origin": "https://attacker.example"})
        self.assertEqual(status, 403)


if __name__ == "__main__":
    unittest.main()
