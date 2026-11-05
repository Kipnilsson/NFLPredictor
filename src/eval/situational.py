"""Situational breakdowns: by down, distance bucket, and late-game score situation."""
from __future__ import annotations

import polars as pl

from src import config
from src.eval.metrics import all_metrics

LATE_SECONDS = 300  # final 5:00 of the 4th quarter, or any overtime


def add_buckets(df: pl.DataFrame) -> pl.DataFrame:
    dist = (pl.when(pl.col("ydstogo") <= 2).then(pl.lit("1-2 short"))
            .when(pl.col("ydstogo") <= 6).then(pl.lit("3-6 medium"))
            .when(pl.col("ydstogo") <= 10).then(pl.lit("7-10 long"))
            .otherwise(pl.lit("11+ very long")))
    late = (pl.col("qtr") == 5) | ((pl.col("qtr") == 4)
                                   & (pl.col("half_seconds_remaining") <= LATE_SECONDS))
    sd = pl.col("score_differential")
    score = (pl.when(sd <= -9).then(pl.lit("late: trailing 9+"))
             .when(sd < 0).then(pl.lit("late: trailing 1-8"))
             .when(sd == 0).then(pl.lit("late: tied"))
             .when(sd <= 8).then(pl.lit("late: leading 1-8"))
             .otherwise(pl.lit("late: leading 9+")))
    return df.with_columns(
        pl.col("down").cast(pl.Utf8).alias("by_down"),
        dist.alias("by_distance"),
        pl.when(late).then(score).otherwise(pl.lit("not late")).alias("by_late_score"),
    )


def breakdown(df: pl.DataFrame, pred_cols: list[str]) -> pl.DataFrame:
    """Long table: one row per (grouping, bucket, model) with n always reported."""
    df = add_buckets(df)
    rows = []
    for grouping in ["by_down", "by_distance", "by_late_score"]:
        for bucket in sorted(df[grouping].unique().to_list()):
            sub = df.filter(pl.col(grouping) == bucket)
            y = sub[config.LABEL].to_numpy()
            for col in pred_cols:
                m = all_metrics(y, sub[col].to_numpy())
                rows.append({"grouping": grouping, "bucket": bucket, "model": col,
                             "n": m["n"], "dropback_rate": m["dropback_rate"],
                             "accuracy": m["accuracy"],
                             "balanced_accuracy": m["balanced_accuracy"],
                             "brier": m["brier"]})
    return pl.DataFrame(rows)
