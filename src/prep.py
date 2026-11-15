"""Download nflverse play-by-play, apply eligibility filters, write processed parquet.

Usage (from repo root):
    python -m src.prep                 # 2021-2024 (train + dev)
    python -m src.prep --include-test  # also 2025; requires NFL_FINAL_EVAL=1
    python -m src.prep --refresh       # re-download raw files

Data: nflverse play-by-play (CC BY 4.0), via nflreadpy.
"""
from __future__ import annotations

import argparse
import json
import sys

import nflreadpy as nfl
import polars as pl

from src import config

FLAG_FILTERS = ["qb_kneel", "qb_spike", "two_point_attempt", "aborted_play", "play_deleted"]


def raw_path(season: int):
    return config.RAW_DIR / f"pbp_{season}.parquet"


def load_raw(season: int, refresh: bool) -> pl.DataFrame:
    path = raw_path(season)
    if refresh or not path.exists():
        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        nfl.load_pbp([season]).write_parquet(path)
    return pl.read_parquet(path)


def filter_season(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    counts: dict[str, int] = {"raw_rows": df.height}
    df = df.filter(pl.col("season_type") == "REG")
    counts["after_reg_season"] = df.height
    df = df.filter(pl.col("play_type").is_in(["run", "pass"]))
    counts["after_run_pass"] = df.height
    for col in FLAG_FILTERS:
        if col not in df.columns:
            counts[f"dropped_{col}"] = None  # column absent in this season's data
            continue
        mask = (pl.col(col) == 1).fill_null(False)
        counts[f"dropped_{col}"] = df.filter(mask).height
        df = df.filter(~mask)
    before = df.height
    df = df.drop_nulls(config.FEATURES)
    counts["dropped_missing_feature"] = before - df.height
    before = df.height
    df = df.drop_nulls([config.LABEL])
    counts["dropped_missing_label"] = before - df.height
    counts["eligible"] = df.height
    int_cols = ["down", "qtr", "posteam_timeouts_remaining", "defteam_timeouts_remaining",
                config.LABEL]
    for c in int_cols:
        assert (df[c] == df[c].round()).all(), f"{c} has non-integer values"
    df = df.with_columns([pl.col(c).cast(pl.Int64) for c in int_cols])
    return df, counts


def sanity_checks(df: pl.DataFrame, season: int) -> None:
    assert df[config.LABEL].is_in([0, 1]).all(), "label must be 0/1"
    assert df["down"].is_in([1, 2, 3, 4]).all(), "down outside 1-4"
    assert df["qtr"].is_in([1, 2, 3, 4, 5]).all(), "qtr outside 1-5"
    assert (df["yardline_100"].is_between(1, 99)).all(), "yardline_100 outside 1-99"
    assert (df["half_seconds_remaining"].is_between(0, 1800)).all()
    assert df.select(["game_id", "play_id"]).is_unique().all(), "duplicate (game_id, play_id)"
    # Scrambles are play_type == "run" but must be labeled dropback.
    scr = df.filter((pl.col("play_type") == "run") & (pl.col("qb_scramble") == 1))
    assert (scr[config.LABEL] == 1).all(), "scramble labeled as designed run"
    assert (df["season"] == season).all()


def prepare_season(season: int, refresh: bool = False, *, output_path=None, game_ids=None):
    raw = load_raw(season, refresh)
    if game_ids is not None:
        raw = raw.filter(pl.col("game_id").is_in(game_ids))
    df, counts = filter_season(raw)
    sanity_checks(df, season)
    if df.is_empty():
        raise ValueError(f"No eligible plays for {season}.")
    if season == 2024 and df.height != config.CHECKPOINT_2024:
        raise ValueError(f"2024 checkpoint expected {config.CHECKPOINT_2024}, got {df.height}.")
    counts["games"] = df["game_id"].n_unique()
    counts["dropbacks"] = int(df[config.LABEL].sum())
    counts["runs"] = df.height - counts["dropbacks"]
    out = df.select(config.BOOKKEEPING + config.FEATURES).sort(["game_id", "play_id"])
    path = output_path or config.PROCESSED_DIR / f"plays_{season}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    out.write_parquet(tmp)
    tmp.replace(path)
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-test", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)

    seasons = list(config.DEFAULT_PREP_SEASONS)
    if args.include_test:
        if not config.final_eval_enabled():
            print(f"--include-test requires {config.FINAL_EVAL_ENV}=1 (final evaluation).")
            return 2
        seasons += config.TEST_SEASONS

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_counts = {}
    for season in seasons:
        counts = prepare_season(season, args.refresh)
        all_counts[season] = counts
        print(season, json.dumps(counts))

    name = "filter_counts_with_test.json" if args.include_test else "filter_counts.json"
    (config.RESULTS_DIR / name).write_text(json.dumps(all_counts, indent=2))
    print("2024 checkpoint OK" if 2024 in all_counts else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
