"""Fixed, readable prompt template for JEV: the same 8 facts for every play.

Freeze status: results/jev/prompt_freeze.json (absent until dev checks are reviewed).
Do not change wording after the freeze is recorded.

Variants (secondary dev-only checks):
  numeric  - main template, facts stated as labeled numbers
  verbal   - same facts in football phrasing ("3rd & 7 at the opponent's 35")
Option order is controlled separately via `labels(reverse=...)`.
"""
from __future__ import annotations

import math

DROPBACK = "dropback"
RUN = "designed_run"

QUESTION = ("Based only on the game situation before the snap, which kind of play did the "
            "offense run?")

TEMPLATE_VERSION = "v2-http"


def labels(reverse: bool = False) -> list[str]:
    return [RUN, DROPBACK] if reverse else [DROPBACK, RUN]


def _clock(seconds: int) -> str:
    return f"{seconds // 60}:{seconds % 60:02d}"


def _ordinal(n: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}[n]


def render_numeric(f: dict) -> str:
    qtr = "overtime" if f["qtr"] == 5 else str(f["qtr"])
    clock_period = "overtime period" if f["qtr"] == 5 else "half"
    return (
        "NFL regular-season play, pre-snap situation (offense's perspective):\n"
        f"- Down: {f['down']}\n"
        f"- Yards to go for a first down: {f['ydstogo']}\n"
        f"- Yards from the opponent's goal line: {f['yardline_100']}\n"
        f"- Quarter: {qtr}\n"
        f"- Seconds remaining in the {clock_period}: {f['half_seconds_remaining']}\n"
        f"- Score differential (offense minus defense): {f['score_differential']:+d}\n"
        f"- Offense timeouts remaining: {f['posteam_timeouts_remaining']}\n"
        f"- Defense timeouts remaining: {f['defteam_timeouts_remaining']}\n\n"
        f"{QUESTION}"
    )


def render_verbal(f: dict) -> str:
    y = f["yardline_100"]
    field = ("midfield" if y == 50 else f"the opponent's {y}" if y < 50
             else f"their own {100 - y}")
    goal = f["ydstogo"] == y
    dist = f"goal ({f['ydstogo']} yards)" if goal else str(f["ydstogo"])
    qtr = "overtime" if f["qtr"] == 5 else f"the {_ordinal(f['qtr'])} quarter"
    period = "the period" if f["qtr"] == 5 else "the half"
    sd = f["score_differential"]
    score = ("the game is tied" if sd == 0 else f"the offense leads by {sd}" if sd > 0
             else f"the offense trails by {-sd}")
    return (
        f"NFL regular season. It's {_ordinal(f['down'])} & {dist} at {field}, in {qtr} with "
        f"{_clock(f['half_seconds_remaining'])} left in {period}, and {score}. The offense "
        f"has {f['posteam_timeouts_remaining']} timeout(s) left; the defense has "
        f"{f['defteam_timeouts_remaining']}.\n\n{QUESTION}"
    )


RENDERERS = {"numeric": render_numeric, "verbal": render_verbal}


def render(features: dict, variant: str = "numeric") -> str:
    from src import config
    f = {}
    for k in config.FEATURES:
        value = float(features[k])
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"Invalid integer feature: {k}")
        f[k] = int(value)
    return RENDERERS[variant](f)
