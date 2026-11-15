import contextlib
import io
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import polars as pl

from src import benchmark, config
from src.jev import client


class BenchmarkTests(unittest.TestCase):
    def test_saved_models_and_cache_only_comparison(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            rows = [
                {key: value for key, value in zip(config.FEATURES, (1, 10, 75, 1, 1800, 0, 3, 3))}
                | {"game_id": game, "play_id": play, "season": 2024,
                   config.LABEL: play - 1, "half_seconds_remaining": 1800 - play}
                for game in ("game_a", "game_b") for play in (1, 2)
            ]
            pl.DataFrame(rows).write_parquet(root / "plays_2024.parquet")
            with patch.multiple(config, PROCESSED_DIR=root, RESULTS_DIR=root,
                                CACHE_DIR=root / "cache", JEV_MODEL_ID="jev-1.13.0"), \
                    patch.object(client, "urlopen", side_effect=AssertionError("HTTP disabled")):
                benchmark.main(["2024", "--preview", "--games", "1"])
                benchmark.main(["2024", "--lock", "--games", "1"])
                benchmark.main(["2024", "--specialists"])
                out = root / "benchmarks/2024"
                manifest = json.loads((out / "cohort.json").read_text())
                self.assertEqual(len(manifest["game_ids"]), 1)
                cohort = pl.DataFrame(rows).filter(pl.col("game_id").is_in(manifest["game_ids"]))

                class FakeBackend:
                    def choose(self, payload):
                        return {"model": "jev-1.13.0", "answers": {"next_play": {
                            "type": "choice", "choice": "dropback",
                            "probabilities": {"dropback": .6, "designed_run": .4}}}}

                for job in benchmark.make_jobs(cohort):
                    client.query(FakeBackend(), config.JEV_MODEL_ID, job["prompt"], job["labels"])
                benchmark.main(["2024", "--cache-only"])
                benchmark.main(["2024", "--report"])
                metrics = json.loads((out / "head_to_head/metrics.json").read_text())
                self.assertEqual(metrics["common_set"], {"n_plays": 2, "n_games": 1})
                self.assertEqual(metrics["coverage"], {"baseline": 4, "logreg": 4, "gbt": 4, "jev": 2})
                self.assertEqual(json.loads((out / "jev_run.json").read_text())["requests_sent"], 0)
                self.assertTrue((out / "head_to_head/comparison.csv").exists())

    def test_holdout_requires_gate(self):
        with patch.object(config, "final_eval_enabled", return_value=False), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            benchmark.main(["2025", "--preview", "--all-games"])

    def test_prospective_requires_complete_games_inside_registered_window(self):
        schedule = pl.DataFrame({
            "game_id": ["before", "inside", "after"],
            "game_type": ["REG", "REG", "REG"],
            "gameday": ["2026-11-15", "2026-11-16", "2026-11-19"],
            "result": [1, 3, None],
        })
        protocol = {"start": "2026-11-16", "through": "2026-11-18"}
        today = date(2026, 11, 19)
        with patch("nflreadpy.load_schedules", return_value=schedule):
            data = pl.DataFrame({"game_id": ["before", "inside", "after"]})
            self.assertEqual(benchmark.prospective_plays(data, protocol, today)["game_id"].to_list(),
                             ["inside"])
            with self.assertRaisesRegex(ValueError, "Play-by-play has not arrived"):
                benchmark.prospective_plays(data.filter(pl.col("game_id") != "inside"),
                                            protocol, today)
