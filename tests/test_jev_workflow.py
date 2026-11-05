"""Verify runner defaults, output isolation and freeze evidence with synthetic data."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl

from src import config
from src.jev import client, template
from src.jev import run, freeze, pilot
from src.jev.pilot import jobs_for
from test_jev import F, MODEL, response


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.frame = pl.DataFrame([F | {"game_id": "2024_game", "play_id": 1.,
                                       "season": 2024, "qb_dropback": 1}])
        self.output = io.StringIO()
        for p in [patch.object(config, "ROOT", self.root),
                  patch.object(config, "CACHE_DIR", self.root / "cache"),
                  patch.object(config, "RESULTS_DIR", self.root / "results"),
                  patch.object(config, "PRED_DIR", self.root / "predictions"),
                  patch.object(config, "JEV_MODEL_ID", MODEL),
                  patch.object(freeze, "FREEZE_PATH", self.root / "freeze.json"),
                  patch.object(run, "load_split", return_value=self.frame),
                  patch.object(client, "urlopen", side_effect=AssertionError("No network"))]:
            p.start(); self.addCleanup(p.stop)

    def test_default_runner_only_previews(self):
        with patch.object(client, "HTTPBackend") as backend, contextlib.redirect_stdout(self.output):
            run.main([])
            backend.assert_not_called()
        self.assertFalse(config.PRED_DIR.exists())
        self.assertFalse(config.RESULTS_DIR.exists())
        self.assertIn("Preview only", self.output.getvalue())

    def test_variant_outputs_are_separate(self):
        backend = Mock()
        backend.choose.return_value = response()
        for variant, reverse in [("numeric", False), ("verbal", False), ("numeric", True)]:
            client.query(backend, MODEL, template.render(F, variant), template.labels(reverse))
        with contextlib.redirect_stdout(self.output):
            run.main(["--cache-only"])
            run.main(["--variant", "verbal", "--cache-only"])
            run.main(["--reverse", "--cache-only"])
        self.assertEqual([p.name for p in config.PRED_DIR.glob("*.parquet")], ["jev_dev.parquet"])
        self.assertEqual(len(list((config.RESULTS_DIR / "jev/variants").glob("*.parquet"))), 2)

    def test_pilot_report_and_exact_replay(self):
        import csv
        source = self.root / "sample.csv"
        row = F | {"game_id": "g", "play_id": "1", "season": "2024",
                   "season_type": "REG", "play_type": "pass", "qb_dropback": "1"}
        with source.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader(); writer.writerow(row)
        args = ["--source", str(source), "--plays", "1", "--checks"]
        with patch.object(client, "HTTPBackend") as backend, contextlib.redirect_stdout(self.output):
            backend.return_value.choose.return_value = response()
            backend.return_value.calls = 3
            pilot.main(args + ["--live", "--max-requests", "3"])
            self.assertEqual(backend.return_value.choose.call_count, 3)
        report_path = config.RESULTS_DIR / "jev/pilots/pilot_1_checks.json"
        report = json.loads(report_path.read_text())
        self.assertEqual(report["summary"]["requests_sent"], 3)
        self.assertAlmostEqual(report["variant_summary"]["numeric"]["brier"], .09)
        self.assertEqual(report["sensitivity"]["numeric-reversed"]["max_absolute_probability_change"], 0)
        with patch.object(client, "HTTPBackend") as backend, contextlib.redirect_stdout(self.output):
            pilot.main(args + ["--cache-only"])
            backend.assert_not_called()
        replay = json.loads(report_path.read_text())
        self.assertEqual(replay["summary"]["requests_sent"], 0)
        self.assertEqual(replay["rows"], report["rows"])
        self.assertFalse(config.PRED_DIR.exists())

    def test_freeze_needs_valid_evidence_and_blocks_later_changes(self):
        plays = [{"game_id": "g", "play_id": "1", "true_label": 1, "features": F}]
        jobs = jobs_for(plays, True)
        backend = Mock(); backend.choose.return_value = response()
        rows = []
        for job in jobs:
            client.query(backend, MODEL, job["prompt"], job["labels"])
            rows.append({k: job[k] for k in ("game_id", "play_id", "variant", "reverse")} |
                        {"cache_key": client.cache_key(MODEL, job["prompt"], job["labels"])})
        report = {"status": "completed", "checks_included": True, "model_id": MODEL,
                  "template_version": template.TEMPLATE_VERSION,
                  "fingerprint": freeze.fingerprint(MODEL), "rows": rows}
        path = self.root / "report.json"; path.write_text(json.dumps(report))
        with contextlib.redirect_stdout(self.output):
            freeze.main(["--pilot-report", str(path), "--confirm-freeze"])
        freeze.require_frozen(MODEL)
        with patch.object(client, "INSTRUCTIONS", "changed"):
            with self.assertRaises(RuntimeError): freeze.require_frozen(MODEL)
        with self.assertRaises(SystemExit):
            freeze.main(["--pilot-report", str(path), "--confirm-freeze"])


if __name__ == "__main__": unittest.main()
