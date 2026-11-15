"""Evaluate explicitly selected main approaches for a split.

Usage:
    python -m src.eval.run --split dev
    NFL_FINAL_EVAL=1 python -m src.eval.run --split test   # final evaluation only

For --split test, specialist predictions are generated from the saved (dev-selected)
models first. JEV predictions come from src.jev.run.

Outputs go to results/eval_{split}/. Every metric is computed on the COMMON set of plays
that all approaches predicted, so comparisons use identical plays. Each approach is also
reported on its own full coverage for reference.
"""
from __future__ import annotations

import argparse
import itertools
import json

import polars as pl
import numpy as np

from src import config
from src.data import load_split, xy
from src.eval.bootstrap import B_DEFAULT, paired_bootstrap
from src.eval.metrics import all_metrics
from src.eval.reliability import plot_reliability, reliability_table
from src.eval.situational import breakdown
from src.models.common import load_model, save_predictions

SPECIALISTS = ["baseline", "logreg", "gbt"]


def predict_specialists(split: str) -> None:
    df = load_split(split)  # raises unless final eval is enabled for test
    X, _ = xy(df)
    for name in SPECIALISTS:
        save_predictions(df, load_model(name).predict_proba(X)[:, 1], name, split)


def gather(split: str, models: list[str] | None = None) -> tuple[pl.DataFrame, list[str], dict[str, int]]:
    plays = load_split(split).select(["game_id", "play_id", config.LABEL] + config.FEATURES)
    if models is None:
        models = SPECIALISTS + (["jev"] if (config.PRED_DIR / f"jev_{split}.parquet").exists() else [])
    if not models or len(models) != len(set(models)) or any(n not in SPECIALISTS + ["jev"] for n in models):
        raise ValueError("Choose distinct main models: baseline, logreg, gbt, jev.")
    return join_predictions(plays, models, split, config.PRED_DIR)


def join_predictions(plays, models, split, prediction_dir):
    files = [prediction_dir / f"{name}_{split}.parquet" for name in models]
    names, coverage, joined = [], {}, plays
    for f in files:
        name = f.name.removesuffix(f"_{split}.parquet")
        pred = pl.read_parquet(f).select(["game_id", "play_id", config.LABEL, "p_dropback"])
        assert pred.select(["game_id", "play_id"]).is_unique().all(), f"{name}: duplicate play IDs"
        probabilities = pred["p_dropback"].to_numpy()
        assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all(), f"{name}: invalid probabilities"
        assert pred.select(["game_id", "play_id"]).join(plays.select(["game_id", "play_id"]), on=["game_id", "play_id"], how="anti").height == 0, f"{name}: unknown plays"
        check = plays.join(pred, on=["game_id", "play_id"], how="inner")
        assert (check[config.LABEL] == check[f"{config.LABEL}_right"]).all(), name
        coverage[name] = pred.height
        joined = joined.join(pred.select(["game_id", "play_id",
                                          pl.col("p_dropback").alias(name)]),
                             on=["game_id", "play_id"], how="left")
        names.append(name)
    return joined, names, coverage


def write_evaluation(df, names, coverage, split, out_dir, figure, *, B=B_DEFAULT, metadata=None):
    """Write the same comparison artifacts for legacy and season benchmarks."""
    out_dir.mkdir(parents=True, exist_ok=True)

    common = df.drop_nulls(names)
    if not names or common.height == 0:
        raise ValueError("No common predictions; evaluation stopped.")
    report = {"split": split, "approaches": names, "coverage": coverage,
              "common_set": {"n_plays": common.height,
                             "n_games": common["game_id"].n_unique()},
              "on_common_set": {}, "on_own_coverage": {}, "reliability_common": {}}
    if metadata:
        report.update(metadata)
    y_common = common[config.LABEL].to_numpy()
    for n in names:
        report["on_common_set"][n] = all_metrics(y_common, common[n].to_numpy())
        own = df.drop_nulls([n])
        report["on_own_coverage"][n] = all_metrics(own[config.LABEL].to_numpy(),
                                                   own[n].to_numpy())
        report["reliability_common"][n] = reliability_table(y_common, common[n].to_numpy())

    paired = {}
    for a, b in itertools.combinations(names, 2):
        paired[f"{a} - {b}"] = paired_bootstrap(common, a, b, B=B)

    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    (out_dir / "paired_bootstrap.json").write_text(json.dumps(paired, indent=2))
    breakdown(common, names).write_csv(out_dir / "situational.csv")

    figure.parent.mkdir(parents=True, exist_ok=True)
    plot_reliability({n: (y_common, common[n].to_numpy()) for n in names if n != "baseline"},
                     figure,
                     f"Reliability, {split} common set (n={common.height:,} plays)")

    print(f"common set: {common.height} plays / {common['game_id'].n_unique()} games")
    for n in names:
        m = report["on_common_set"][n]
        print(f"{n:>12}  acc={m['accuracy']:.4f}  bal_acc={m['balanced_accuracy']:.4f}  "
              f"brier={m['brier']:.4f}")
    for k, v in paired.items():
        a = v["accuracy"]
        print(f"{k:>24}  acc diff={a['diff']:+.4f}  95% CI [{a['ci95'][0]:+.4f}, "
              f"{a['ci95'][1]:+.4f}]  brier diff={v['brier']['diff']:+.4f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--models", nargs="+", choices=SPECIALISTS + ["jev"])
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args(argv)

    if args.split == "test":
        predict_specialists("test")

    df, names, coverage = gather(args.split, args.models)
    out_dir = config.RESULTS_DIR / f"eval_{args.split}"
    write_evaluation(df, names, coverage, args.split, out_dir,
                     config.FIG_DIR / f"reliability_{args.split}.png", B=args.B)


if __name__ == "__main__":
    main()
