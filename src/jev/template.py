"""Fixed, readable prompt template for JEV: the same 8 facts for every play.

FROZEN: no (see docs/decisions.md). Do not change wording after the freeze is recorded.

Variants (secondary dev-only checks):
  numeric  - main template, facts stated as labeled numbers
  verbal   - same facts in football phrasing ("3rd & 7 at the opponent's 35")
Option order is controlled separately via `labels(reverse=...)`.
"""
from __future__ import annotations

DROPBACK = "Dropback (the quarterback drops back to pass; includes sacks and scrambles)"
RUN = "Designed run (a planned handoff or quarterback run)"

QUESTION = ("Based only on the game situation before the snap, which kind of play did the "
            "offense run?")

TEMPLATE_VERSION = "v1"


def labels(reverse: bool = False) -> list[str]:
    return [RUN, DROPBACK] if reverse else [DROPBACK, RUN]


def _clock(seconds: int) -> str:
    return f"{seconds // 60}:{seconds % 60:02d}"


def _ordinal(n: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}[n]


def render_numeric(f: dict) -> str:
    qtr = "overtime" if f["qtr"] == 5 else str(f["qtr"])
    return (
        "NFL regular-season play, pre-snap situation (offense's perspective):\n"
        f"- Down: {f['down']}\n"
        f"- Yards to go for a first down: {f['ydstogo']}\n"
        f"- Yards from the opponent's goal line: {f['yardline_100']}\n"
        f"- Quarter: {qtr}\n"
        f"- Seconds remaining in the half: {f['half_seconds_remaining']}\n"
        f"- Score differential (offense minus defense): {f['score_differential']:+d}\n"
        f"- Offense timeouts remaining: {f['posteam_timeouts_remaining']}\n"
        f"- Defense timeouts remaining: {f['defteam_timeouts_remaining']}\n\n"
        f"{QUESTION}"
    )


def render_verbal(f: dict) -> str:
    y = f["yardline_100"]
    field = ("midfield" if y == 50 else f"the opponent's {y}" if y < 50
             else f"their own {100 - y}")
    goal = f["ydstogo"] >= y
    dist = "goal" if goal else str(f["ydstogo"])
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
    f = {k: int(v) for k, v in features.items()}  # all 8 allowlisted facts are integers
    return RENDERERS[variant](f)
