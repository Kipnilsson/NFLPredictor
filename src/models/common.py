"""Shared helpers for specialist models: prediction files and model persistence."""
from __future__ import annotations

import json

import joblib
import numpy as np
import polars as pl

from src import config

MODEL_DIR = config.RESULTS_DIR / "models"
SELECTION_DIR = config.RESULTS_DIR / "model_selection"


def save_predictions(df: pl.DataFrame, p: np.ndarray, name: str, split: str) -> None:
    """Write per-play P(dropback) next to the identifying columns and the label."""
    config.PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = df.select(["game_id", "play_id", "season", config.LABEL]).with_columns(
        pl.Series("p_dropback", np.asarray(p, dtype=float)))
    out.write_parquet(config.PRED_DIR / f"{name}_{split}.parquet")


def save_model(model, name: str, meta: dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / f"{name}.joblib")
    (MODEL_DIR / f"{name}.json").write_text(json.dumps(meta, indent=2, default=str))


def load_model(name: str):
    return joblib.load(MODEL_DIR / f"{name}.joblib")
