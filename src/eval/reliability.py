"""Reliability (calibration) plots."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append({"bin": b, "lo": edges[b], "hi": edges[b + 1], "n": int(m.sum()),
                         "mean_pred": float(p[m].mean()), "obs_rate": float(y[m].mean())})
    return rows


def plot_reliability(curves: dict[str, tuple[np.ndarray, np.ndarray]], path, title: str,
                     n_bins: int = 10) -> None:
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(6, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    bins = np.linspace(0, 1, n_bins + 1)
    for name, (y, p) in curves.items():
        t = reliability_table(y, p, n_bins)
        ax.plot([r["mean_pred"] for r in t], [r["obs_rate"] for r in t], marker="o", label=name)
        axh.hist(p, bins=bins, histtype="step", label=name)
    ax.set_ylabel("observed dropback rate")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    axh.set_xlabel("predicted P(dropback)")
    axh.set_ylabel("plays")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
