import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

from src import config
from src.data import TestDataLocked, load_split
from src.eval.run import gather


class EvaluationTests(unittest.TestCase):
    def test_2025_locked_before_read(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(TestDataLocked): load_split("test")

    def test_variants_not_auto_included_and_bad_predictions_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plays = pl.DataFrame({"game_id": ["g"], "play_id": [1.0], "qb_dropback": [1],
                                  **{k: [1] for k in config.FEATURES}})
            pred = plays.select(["game_id", "play_id", "qb_dropback"]).with_columns(pl.lit(.7).alias("p_dropback"))
            for name in ["baseline", "logreg", "gbt", "jev-verbal"]:
                pred.write_parquet(root / f"{name}_dev.parquet")
            with patch.object(config, "PRED_DIR", root), patch("src.eval.run.load_split", return_value=plays):
                _, names, _ = gather("dev")
                self.assertNotIn("jev-verbal", names)
                pl.concat([pred, pred]).write_parquet(root / "logreg_dev.parquet")
                with self.assertRaises(AssertionError): gather("dev", ["logreg"])
                pred.with_columns(pl.lit(float("nan")).alias("p_dropback")).write_parquet(root / "logreg_dev.parquet")
                with self.assertRaises(AssertionError): gather("dev", ["logreg"])


if __name__ == "__main__": unittest.main()
