"""Load processed splits. Refuses to touch 2025 unless final evaluation is enabled."""
from __future__ import annotations

import polars as pl

from src import config


class TestDataLocked(RuntimeError):
    pass


def _check_seasons(seasons: list[int]) -> None:
    locked = set(seasons) & set(config.TEST_SEASONS)
    if locked and not config.final_eval_enabled():
        raise TestDataLocked(
            f"Seasons {sorted(locked)} are the held-out test set. Set "
            f"{config.FINAL_EVAL_ENV}=1 only when the team starts final evaluation."
        )


def load_seasons(seasons: list[int]) -> pl.DataFrame:
    _check_seasons(seasons)
    frames = []
    for s in seasons:
        path = config.PROCESSED_DIR / f"plays_{s}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run `python -m src.prep` first.")
        frames.append(pl.read_parquet(path))
    return pl.concat(frames)


def load_split(split: str) -> pl.DataFrame:
    seasons = {"train": config.TRAIN_SEASONS, "dev": config.DEV_SEASONS,
               "test": config.TEST_SEASONS}[split]
    return load_seasons(seasons)


def xy(df: pl.DataFrame):
    """Return (X as pandas DataFrame of the 8 allowlisted features, y as numpy int array)."""
    X = df.select(config.FEATURES).to_pandas()
    for c in config.CATEGORICAL:
        X[c] = X[c].astype(int)
    y = df[config.LABEL].cast(pl.Int8).to_numpy()
    return X, y
