"""Game-level (cluster) bootstrap for paired metric differences on identical plays.

Plays within a game are correlated, so we resample whole games with replacement.
Metrics are computed from per-game sufficient statistics, which makes B=2000 fast.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from src import config

B_DEFAULT = 2000


def _game_stats(game_idx: np.ndarray, n_games: int, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-game sums: [n, correct, sq_err, pos, tp, neg, tn]."""
    yhat = (p >= 0.5).astype(int)
    cols = [np.ones_like(y), (yhat == y), (p - y) ** 2, y, (yhat == 1) & (y == 1),
            1 - y, (yhat == 0) & (y == 0)]
    return np.column_stack([np.bincount(game_idx, weights=c.astype(float), minlength=n_games)
                            for c in cols])


def _metrics_from_sums(s: np.ndarray) -> dict[str, np.ndarray]:
    n, correct, sq, pos, tp, neg, tn = (s[..., i] for i in range(7))
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"accuracy": correct / n, "brier": sq / n,
                "balanced_accuracy": 0.5 * (tp / pos + tn / neg)}


def paired_bootstrap(df: pl.DataFrame, col_a: str, col_b: str, B: int = B_DEFAULT,
                     seed: int = config.SEED) -> dict:
    """Bootstrap metric(a) - metric(b) on the plays where both have predictions.

    df needs columns game_id, qb_dropback, col_a, col_b.
    """
    games, game_idx = np.unique(df["game_id"].to_numpy(), return_inverse=True)
    G = len(games)
    y = df[config.LABEL].to_numpy().astype(float)
    sa = _game_stats(game_idx, G, y, df[col_a].to_numpy())
    sb = _game_stats(game_idx, G, y, df[col_b].to_numpy())

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, G, size=(B, G))
    W = np.stack([np.bincount(d, minlength=G) for d in draws]).astype(float)  # B x G
    ma, mb = _metrics_from_sums(W @ sa), _metrics_from_sums(W @ sb)
    pa, pb = _metrics_from_sums(sa.sum(0)), _metrics_from_sums(sb.sum(0))

    out = {"n_plays": int(df.height), "n_games": int(G), "B": B}
    for k in ma:
        diff = ma[k] - mb[k]
        lo, hi = np.nanpercentile(diff, [2.5, 97.5])
        out[k] = {"a": float(pa[k]), "b": float(pb[k]), "diff": float(pa[k] - pb[k]),
                  "ci95": [float(lo), float(hi)],
                  # share of bootstrap draws where a is NOT better (lower brier is better)
                  "p_a_not_better": float(np.nanmean(diff >= 0) if k == "brier"
                                          else np.nanmean(diff <= 0))}
    return out
