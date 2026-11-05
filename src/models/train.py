"""Fit baseline, logistic regression, and gradient boosting; select hyperparameters on dev.

Usage: python -m src.models.train

Selection criterion: dev log loss (a proper scoring rule, so it rewards calibration as
well as discrimination). Every grid point is written to results/model_selection/.
"""
from __future__ import annotations

import json

import polars as pl
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from src import config
from src.data import load_split, xy
from src.models.common import SELECTION_DIR, save_model, save_predictions
from src.models.specialists import FACTORIES, GRIDS, MajorityBaseline


def score(y, p) -> dict:
    return {"log_loss": log_loss(y, p, labels=[0, 1]), "brier": brier_score_loss(y, p),
            "accuracy": accuracy_score(y, p >= 0.5)}


def main() -> None:
    train, dev = load_split("train"), load_split("dev")
    Xtr, ytr = xy(train)
    Xdv, ydv = xy(dev)
    SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    base = MajorityBaseline().fit(Xtr, ytr)
    p = base.predict_proba(Xdv)[:, 1]
    save_predictions(dev, p, "baseline", "dev")
    save_model(base, "baseline", {"train_rate": base.rate_, "dev": score(ydv, p)})
    summary["baseline"] = {"params": {"train_rate": base.rate_}, **score(ydv, p)}
    print("baseline", summary["baseline"])

    for name, grid in GRIDS.items():
        rows = []
        best = None
        for params in grid:
            model = FACTORIES[name](**params).fit(Xtr, ytr)
            p = model.predict_proba(Xdv)[:, 1]
            s = score(ydv, p)
            extra = {"n_iter": model.n_iter_} if hasattr(model, "n_iter_") else {}
            rows.append({**params, **extra, **s})
            print(name, params, extra, {k: round(v, 5) for k, v in s.items()})
            if best is None or s["log_loss"] < best[2]["log_loss"]:
                best = (params, model, s, p)
        pl.DataFrame(rows).write_csv(SELECTION_DIR / f"{name}_grid.csv")
        params, model, s, p = best
        save_predictions(dev, p, name, "dev")
        save_model(model, name, {"params": params, "selected_on": "dev log_loss",
                                 "seed": config.SEED, "dev": s})
        summary[name] = {"params": params, **s}

    (SELECTION_DIR / "selected.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
