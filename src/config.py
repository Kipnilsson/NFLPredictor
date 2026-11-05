"""Project-wide constants: paths, seasons, features, seeds, JEV settings."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
PRED_DIR = RESULTS_DIR / "predictions"
FIG_DIR = RESULTS_DIR / "figures"

SEED = 74

TRAIN_SEASONS = [2021, 2022, 2023]
DEV_SEASONS = [2024]
TEST_SEASONS = [2025]
DEFAULT_PREP_SEASONS = TRAIN_SEASONS + DEV_SEASONS  # 2025 only with --include-test

CHECKPOINT_2024 = 33_224

# Feature allowlist (exactly 8). Changing this requires a team decision in docs/decisions.md.
CATEGORICAL = ["down", "qtr"]
NUMERIC = [
    "ydstogo",
    "yardline_100",
    "half_seconds_remaining",
    "score_differential",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
]
FEATURES = ["down", "ydstogo", "yardline_100", "qtr", "half_seconds_remaining",
            "score_differential", "posteam_timeouts_remaining", "defteam_timeouts_remaining"]
assert sorted(FEATURES) == sorted(CATEGORICAL + NUMERIC) and len(FEATURES) == 8

LABEL = "qb_dropback"
BOOKKEEPING = ["game_id", "play_id", "season", "week", "posteam", "defteam", "desc",
               "pass_oe", "qb_dropback"]

# Final (test) evaluation gate. Set NFL_FINAL_EVAL=1 only once the team starts final eval.
FINAL_EVAL_ENV = "NFL_FINAL_EVAL"


def final_eval_enabled() -> bool:
    return os.environ.get(FINAL_EVAL_ENV) == "1"


# JEV settings. The model ID must be a pinned, versioned ID (never an alias).
JEV_MODEL_ID: str | None = os.environ.get("JEV_MODEL_ID", "jev-1.13.0")
JEV_API_KEY_ENV = "TYPESAFE_API_KEY"
JEV_DEV_SUBSET_GAMES = 60  # complete 2024 games used for prompt checks
