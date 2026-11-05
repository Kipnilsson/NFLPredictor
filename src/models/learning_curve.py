"""Learning curves: retrain specialists on growing subsets of training GAMES, score on dev.

Usage: python -m src.models.learning_curve

Uses the dev-selected hyperparameters from results/model_selection/selected.json.
Each fraction is repeated with several seeds; subsets are nested within a seed.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src import config
from src.data import load_split, xy
from src.models.common import SELECTION_DIR
from src.models.specialists import FACTORIES
from src.models.train import score

FRACTIONS = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
REPEATS = 3


def main() -> None:
    selected = json.loads((SELECTION_DIR / "selected.json").read_text())
    train, dev = load_split("train"), load_split("dev")
    Xdv, ydv = xy(dev)
    games = np.array(sorted(train["game_id"].unique().to_list()))
    rows = []
    for rep in range(REPEATS):
        rng = np.random.default_rng(config.SEED + rep)
        order = rng.permutation(games)
        for frac in FRACTIONS:
            n_games = max(1, int(round(frac * len(games))))
            sub = train.filter(pl.col("game_id").is_in(order[:n_games].tolist()))
            Xs, ys = xy(sub)
            for name in ["logreg", "gbt"]:
                model = FACTORIES[name](**selected[name]["params"]).fit(Xs, ys)
                s = score(ydv, model.predict_proba(Xdv)[:, 1])
                rows.append({"model": name, "repeat": rep, "fraction": frac,
                             "n_games": n_games, "n_plays": sub.height, **s})
                print(rows[-1])
            # Baseline reference: subset class rate
            p = np.full(len(ydv), ys.mean())
            rows.append({"model": "baseline", "repeat": rep, "fraction": frac,
                         "n_games": n_games, "n_plays": sub.height, **score(ydv, p)})

    df = pl.DataFrame(rows)
    df.write_csv(SELECTION_DIR / "learning_curve.csv")

    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    agg = df.group_by(["model", "n_games"]).agg(
        [pl.col(m).mean().alias(m) for m in ["log_loss", "brier", "accuracy"]]
        + [pl.col(m).std().alias(f"{m}_sd") for m in ["log_loss", "brier", "accuracy"]]
    ).sort("n_games")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ["accuracy", "brier", "log_loss"]):
        for name in ["baseline", "logreg", "gbt"]:
            a = agg.filter(pl.col("model") == name)
            ax.errorbar(a["n_games"], a[metric], yerr=a[f"{metric}_sd"], marker="o",
                        capsize=3, label=name)
        ax.set_xscale("log")
        ticks = sorted(agg["n_games"].unique().to_list())
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("training games (log scale)")
        ax.set_title(f"dev {metric}")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(f"Learning curves (mean ± sd over {REPEATS} game-subset seeds), dev = 2024")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "learning_curve_dev.png", dpi=150)


if __name__ == "__main__":
    main()
