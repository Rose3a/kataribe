import json
import sys
import types
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))
try:
    from editor_engine import EditorAdapter
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.__version__ = "test"
    torch_stub.version = types.SimpleNamespace(cuda=None)
    sys.modules["torch"] = torch_stub
    from editor_engine import EditorAdapter
from voicevox_engine import Handler, SESSION_TOKEN, MAX_BODY_BYTES


class _FakeVoicevoxAdapter:
    backend_name = "test"
    speakers_json = []
    synthesis_slots = threading.BoundedSemaphore(2)

    def synthesize(self, query, speaker):
        return b"RIFFtest"


class RouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Handler.adapter = _FakeVoicevoxAdapter()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, path, method="GET", body=None, token=None, origin="http://localhost:5173"):
        headers = {"Host": f"127.0.0.1:{self.server.server_port}", "Origin": origin}
        if token is not None:
            headers["X-Irodori-Session"] = token
        if body is not None:
            encoded = body if isinstance(body, bytes) else json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        else:
            encoded = None
        return urlopen(Request(self.url + path, data=encoded, headers=headers, method=method), timeout=3)

    def test_session_bootstrap_rejects_attacker_origin(self):
        with self.assertRaises(HTTPError) as error:
            self.request("/irodori/session", origin="https://attacker.example")
        self.assertEqual(error.exception.code, 403)

    def test_post_requires_token_and_legitimate_generated_requests_work(self):
        with self.assertRaises(HTTPError) as error:
            self.request("/audio_query?text=hello", method="POST")
        self.assertEqual(error.exception.code, 403)
        response = self.request("/irodori/session")
        self.assertEqual(response.status, 200)
        token = json.loads(response.read())["token"]
        response = self.request("/audio_query?text=hello&speaker=0", method="POST", token=token)
        self.assertEqual(response.status, 200)
        response = self.request("/synthesis?speaker=0", method="POST", body={"irodori_text": "hello", "irodori_seed": 123}, token=token)
        self.assertEqual(response.read(), b"RIFFtest")

    def test_generated_audio_query_json_preserves_seed_field(self):
        audio_query = Path(__file__).resolve().parents[2] / "voicevox-editor" / "src" / "openapi" / "models" / "AudioQuery.ts"
        source = audio_query.read_text(encoding="utf-8")
        self.assertIn("'irodori_seed': value.irodoriSeed", source)
        self.assertIn("irodoriSeed: effectiveSeed", (audio_query.parent.parent.parent / "store" / "audioGenerate.ts").read_text(encoding="utf-8"))

    def test_body_errors_are_clear_and_bounded(self):
        response = self.request("/irodori/session")
        token = json.loads(response.read())["token"]
        with self.assertRaises(HTTPError) as error:
            self.request("/synthesis?speaker=0", method="POST", token=token)
        self.assertEqual(error.exception.code, 400)
        request = Request(self.url + "/synthesis?speaker=0", data=b"{}", headers={
            "Host": f"127.0.0.1:{self.server.server_port}",
            "Origin": "http://localhost:5173", "X-Irodori-Session": token,
            "Content-Length": str(MAX_BODY_BYTES + 1)}, method="POST")
        with self.assertRaises(HTTPError) as error:
            urlopen(request, timeout=3)
        self.assertEqual(error.exception.code, 413)


class LifecycleLockTests(unittest.TestCase):
    def test_configure_waits_for_synthesis_without_deadlock(self):
        adapter = EditorAdapter.__new__(EditorAdapter)
        adapter.operation_lock = threading.RLock()
        adapter.state_lock = threading.RLock()
        adapter.progress_lock = threading.Lock()
        adapter.progress = {"active": False, "percent": 0, "stage": "idle"}
        adapter.DEFAULT_SETTINGS = dict(backend="cpu", model="model.safetensors", steps=8, seed=1,
                                        seconds=None, caption="", cfg=5.0, t_schedule_mode="sway", sway_coeff=-1.0)
        adapter.settings = dict(adapter.DEFAULT_SETTINGS)
        adapter.config_path = Path(__file__).with_name("_audit-settings.json")
        adapter.id_to_name = {0: ""}
        adapter.delegate = None
        adapter.validate = lambda value: None
        adapter.available_backends = lambda: {"cpu": True}
        adapter.models = lambda: []
        adapter._synthesize = lambda query, speaker: (entered.set(), release.wait(2), b"ok")[2]
        adapter._set_progress = lambda *args, **kwargs: None
        adapter._close_locked = lambda: None
        entered, release = threading.Event(), threading.Event()
        synthesis = threading.Thread(target=adapter.synthesize, args=({}, 0))
        synthesis.start()
        self.assertTrue(entered.wait(1))
        configure = threading.Thread(target=adapter.configure, args=({"seed": 2},))
        configure.start()
        time.sleep(0.1)
        self.assertTrue(configure.is_alive())
        release.set()
        synthesis.join(2)
        configure.join(2)
        self.assertFalse(synthesis.is_alive())
        self.assertFalse(configure.is_alive())
        adapter.config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
