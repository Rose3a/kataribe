import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

WRAPPER = Path(__file__).resolve().parents[1] / "wrapper"
sys.path.insert(0, str(WRAPPER))

import speaker_train as st  # noqa: E402
import speaker_train_job as job_mod  # noqa: E402


def settings(**overrides):
    base = dict(steps=500, lr=0.01, warmup=0, batch=0, dur_weight=1.0, val_frac=0.05, val_every=0, seed=0,
                lr_schedule="constant", keep="last")
    base.update(overrides)
    return SimpleNamespace(**base)


class LrScheduleTest(unittest.TestCase):
    def test_constant_is_flat(self):
        self.assertEqual([st.lr_factor(s, 100) for s in (1, 50, 100)], [1.0, 1.0, 1.0])

    def test_warmup_rises_linearly(self):
        self.assertAlmostEqual(st.lr_factor(1, 100, "constant", 10), 0.1)
        self.assertAlmostEqual(st.lr_factor(10, 100, "constant", 10), 1.0)
        self.assertEqual(st.lr_factor(11, 100, "constant", 10), 1.0)

    def test_cosine_starts_full_and_ends_near_zero(self):
        self.assertAlmostEqual(st.lr_factor(1, 100, "cosine"), 1.0)
        self.assertLess(st.lr_factor(100, 100, "cosine"), 0.01)
        self.assertGreater(st.lr_factor(100, 100, "cosine"), 0.0)  # 最後の 1 歩も 0 にはしない
        values = [st.lr_factor(s, 100, "cosine", 10) for s in range(11, 101)]
        self.assertEqual(values, sorted(values, reverse=True))


class ValSplitTest(unittest.TestCase):
    def test_small_sets_are_not_split(self):
        self.assertEqual(st.val_count(st.VAL_MIN_TOTAL - 1, 0.05), 0)
        self.assertEqual(st.val_count(200, 0.0), 0)

    def test_count_is_clamped(self):
        self.assertEqual(st.val_count(30, 0.05), st.VAL_MIN_CLIPS)
        self.assertEqual(st.val_count(200, 0.05), 10)
        self.assertEqual(st.val_count(5000, 0.3), st.VAL_MAX_CLIPS)

    def test_split_is_stable_and_disjoint(self):
        train, val = st.split_rows(100, 0.1, seed=3)
        self.assertEqual((train, val), st.split_rows(100, 0.1, seed=3))
        self.assertEqual(len(val), 10)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(sorted(train + val), list(range(100)))

    def test_auto_val_every(self):
        self.assertEqual(st.auto_val_every(500, cuda=True), 25)
        self.assertEqual(st.auto_val_every(500, cuda=False), 50)
        self.assertEqual(st.auto_val_every(50, cuda=True), 5)


class CheckSettingsTest(unittest.TestCase):
    def test_defaults_pass(self):
        st.check_settings(settings())

    def test_out_of_range(self):
        for bad in (dict(lr=0.0), dict(lr=2.0), dict(steps=10), dict(batch=65), dict(val_frac=0.5),
                    dict(warmup=500), dict(lr_schedule="step"), dict(keep="first"), dict(steps=500.5),
                    dict(lr=float("nan")), dict(seed=True)):
            with self.subTest(bad=bad), self.assertRaises(st.TrainError):
                st.check_settings(settings(**bad))

    def test_job_reads_payload_numbers(self):
        args = job_mod.SpeakerTrainJob._settings({"steps": 300.0, "lr": 1, "lrSchedule": "cosine", "warmup": 20})
        self.assertEqual((args.steps, args.lr, args.lr_schedule, args.warmup), (300, 1.0, "cosine", 20))
        self.assertIsInstance(args.steps, int)
        self.assertIsInstance(args.lr, float)
        with self.assertRaises(ValueError):
            job_mod.SpeakerTrainJob._settings({"lr": 5})


class FakeProc:
    def __init__(self, lines):
        self.stdout = io.StringIO("".join(line + "\n" for line in lines))

    def wait(self):
        return 0

    def poll(self):
        return 0


class JobHistoryTest(unittest.TestCase):
    def test_metrics_events_build_history(self):
        lines = [
            json.dumps({"event": "config", "batch": 16, "val_clips": 4}),
            json.dumps({"event": "metrics", "kind": "val", "step": 0, "loss": 0.6, "dur_mae_sec": 0.3}),
            json.dumps({"event": "metrics", "kind": "train", "step": 2, "loss": 0.7, "lr": 0.01}),
            "some log line",
            json.dumps({"event": "progress", "stage": "train", "percent": 50, "message": "学習 2/4"}),
            json.dumps({"event": "done", "result": {"name": "x", "metrics": {"stopped_at": 2}}}),
        ]
        job = job_mod.SpeakerTrainJob()
        job.state.update(state="running")
        job._read(FakeProc(lines))
        status = job.status()
        self.assertEqual(status["config"], {"batch": 16, "val_clips": 4})
        self.assertEqual(status["history"]["val"], [{"step": 0, "loss": 0.6, "dur_mae_sec": 0.3}])
        self.assertEqual(status["history"]["train"], [{"step": 2, "loss": 0.7, "lr": 0.01}])
        self.assertEqual(status["state"], "done")
        self.assertIn("ステップ 2", status["message"])
        self.assertEqual(status["log"], ["some log line"])


if __name__ == "__main__":
    unittest.main()
