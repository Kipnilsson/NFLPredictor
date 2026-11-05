"""Offline verification: no real credentials, network, downloads or test-set data."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import config
from src.jev import client, template
from src.jev.experiment import execute, plan
from src.jev.freeze import require_frozen
from src.jev.pilot import jobs_for, sample_plays

F = dict(down=3, ydstogo=7, yardline_100=35, qtr=4, half_seconds_remaining=120,
         score_differential=-3, posteam_timeouts_remaining=2, defteam_timeouts_remaining=3)
MODEL = "jev-1.13.0"


def response():
    return {"model": MODEL, "answers": {"next_play": {"type": "choice",
            "choice": "dropback", "probabilities": {"dropback": .7, "designed_run": .3}}},
            "usage": {"input_tokens": 100, "output_tokens": 10}}


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patches = [patch.object(config, "CACHE_DIR", self.root / "cache"),
                        patch.object(config, "ROOT", self.root),
                        patch.object(client, "urlopen", side_effect=AssertionError("network forbidden"))]
        for p in self.patches:
            p.start(); self.addCleanup(p.stop)
        self.prompt = template.render(F)
        self.labels = template.labels()
        self.jobs = [{"prompt": self.prompt, "labels": self.labels}]

    def test_feature_allowlist(self):
        self.assertEqual(template.render(F | {"desc": "secret result", "qb_dropback": 1}), self.prompt)
        with self.assertRaises(ValueError): template.render(F | {"ydstogo": float("nan")})
        with self.assertRaises(ValueError): template.render(F | {"ydstogo": 1.5})

    def test_request_shape_and_precise_label(self):
        p = client.make_payload(MODEL, self.prompt, self.labels)
        self.assertEqual(set(p), {"model", "state", "questions"})
        self.assertEqual(list(p["questions"]["next_play"]["criteria"]), self.labels)
        self.assertIn("no quarterback dropback", p["questions"]["next_play"]["criteria"][template.RUN])

    def test_order_and_instructions_change_cache_identity(self):
        key = client.cache_key(MODEL, self.prompt, self.labels)
        self.assertNotEqual(key, client.cache_key(MODEL, self.prompt, template.labels(True)))
        with patch.object(client, "INSTRUCTIONS", "new instructions"):
            self.assertNotEqual(key, client.cache_key(MODEL, self.prompt, self.labels))

    def test_model_aliases_rejected(self):
        for model in [None, "jev-latest", "jev-preview", "jev-1", "other-1.0.0"]:
            with self.assertRaises(ValueError): client.validate_model_id(model)

    def test_invalid_responses_rejected(self):
        changes = [lambda r: r.update(model="jev-1.14.0"),
                   lambda r: r["answers"]["next_play"].update(choice="designed_run"),
                   lambda r: r["answers"]["next_play"].update(type="score")]
        for v in [float("nan"), float("inf"), -.1, 1.1, True, "0.7", .5]:
            changes.append(lambda r, v=v: r["answers"]["next_play"]["probabilities"].update(dropback=v))
        for change in changes:
            r = copy.deepcopy(response()); change(r)
            with self.assertRaises(ValueError): client.validate_answer(r, MODEL)

    def test_default_cannot_create_paid_backend(self):
        with self.assertRaises(ValueError): client.HTTPBackend()
        with self.assertRaises(RuntimeError): execute(self.jobs, MODEL)
        self.assertEqual(plan(self.jobs * 2, MODEL)["new_requests"], 1)

    def test_over_budget_fails_before_backend_or_key_read(self):
        jobs = self.jobs + [{"prompt": self.prompt + "other", "labels": self.labels}]
        with patch.object(client, "HTTPBackend") as backend:
            with self.assertRaises(ValueError): execute(jobs, MODEL, live=True, max_requests=1)
            backend.assert_not_called()

    def test_cache_resume_and_replay_do_not_call_network(self):
        with patch.object(client, "HTTPBackend") as backend:
            backend.return_value.choose.return_value = response()
            backend.return_value.calls = 1
            records, summary = execute(self.jobs * 2, MODEL, live=True, max_requests=1)
            self.assertEqual(len(records), 2)
            backend.return_value.choose.assert_called_once()
        with patch.object(client, "HTTPBackend") as backend:
            _, summary = execute(self.jobs * 2, MODEL)
            self.assertEqual(summary["requests_sent"], 0)
            backend.assert_not_called()

    def test_invalid_response_is_not_cached(self):
        with patch.object(client, "HTTPBackend") as backend:
            backend.return_value.choose.return_value = {}
            with self.assertRaises(ValueError): execute(self.jobs, MODEL, live=True, max_requests=1)
        self.assertIsNone(client.cached(MODEL, self.prompt, self.labels))

    def test_attempt_cap_and_timeout_no_retry(self):
        with patch.object(client, "load_api_key", return_value="fake-test-key"):
            backend = client.HTTPBackend(allow_paid=True, max_requests=1)
        with patch.object(client, "urlopen", side_effect=TimeoutError) as http:
            with self.assertRaises(RuntimeError): backend.choose(client.make_payload(MODEL, self.prompt, self.labels))
            with self.assertRaises(RuntimeError): backend.choose(client.make_payload(MODEL, self.prompt, self.labels))
            http.assert_called_once()

    def test_dotenv_is_data_not_shell(self):
        (self.root / ".env").write_text('TYPESAFE_API_KEY="fake-key"\nUNRELATED=$(never_execute)\n')
        with patch.dict("os.environ", {}, clear=True): self.assertEqual(client.load_api_key(), "fake-key")

    def test_partial_cache_rejected_without_partial_live_calls(self):
        with patch.object(client, "HTTPBackend") as backend:
            backend.return_value.choose.return_value = response()
            client.query(backend.return_value, MODEL, self.prompt, self.labels)
        jobs = self.jobs + [{"prompt": "uncached", "labels": self.labels}]
        with self.assertRaises(RuntimeError): execute(jobs, MODEL)

    def test_sensitivity_variants_keep_same_play_ids(self):
        plays = [{"game_id": "g", "play_id": "1", "true_label": 1, "features": F}]
        jobs = jobs_for(plays, True)
        self.assertEqual(len(jobs), 3)
        self.assertEqual(jobs[0]["prompt"], jobs[2]["prompt"])
        self.assertEqual(jobs[0]["labels"], jobs[2]["labels"][::-1])
        self.assertEqual(jobs[1]["labels"], jobs[0]["labels"])

    def test_final_test_requires_prompt_freeze(self):
        with patch("src.jev.freeze.FREEZE_PATH", self.root / "absent.json"):
            with self.assertRaises(RuntimeError): require_frozen(MODEL)


if __name__ == "__main__":
    unittest.main()
