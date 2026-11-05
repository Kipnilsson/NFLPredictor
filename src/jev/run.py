"""Run JEV zero-shot on a reproducible subset of complete games.

Usage:
    python -m src.jev.run --dry-run                    # build prompts, show sample, no API
    python -m src.jev.run                              # main: numeric template, dev subset
    python -m src.jev.run --variant verbal             # secondary check (dev only)
    python -m src.jev.run --reverse                    # secondary check (dev only)
    NFL_FINAL_EVAL=1 python -m src.jev.run --split test --games N   # final eval only

Game subsets are chosen from game IDs only (fixed seed), never from labels. The test
subset is chosen from the 2025 schedule and written once to results/jev_test_games.json.
"""
from __future__ import annotations

import argparse
import json

import nflreadpy as nfl
import numpy as np
import polars as pl

from src import config
from src.data import load_split
from src.jev import client, template
from src.models.common import save_predictions


def choose_games(game_ids: list[str], n: int, seed: int = config.SEED) -> list[str]:
    ids = sorted(set(game_ids))
    rng = np.random.default_rng(seed)
    return sorted(rng.permutation(ids)[: min(n, len(ids))].tolist())


def test_games(n: int) -> list[str]:
    """Pick the 2025 subset from the schedule (no play-by-play, no labels); freeze it."""
    path = config.RESULTS_DIR / "jev_test_games.json"
    if path.exists():
        return json.loads(path.read_text())["game_ids"]
    sched = nfl.load_schedules(config.TEST_SEASONS)
    reg = sched.filter((pl.col("game_type") == "REG") & pl.col("result").is_not_null())
    games = choose_games(reg["game_id"].to_list(), n)
    path.write_text(json.dumps({"seed": config.SEED, "n": len(games), "game_ids": games},
                               indent=2))
    return games


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--variant", choices=list(template.RENDERERS), default="numeric")
    ap.add_argument("--reverse", action="store_true", help="reverse option order")
    ap.add_argument("--games", type=int, default=config.JEV_DEV_SUBSET_GAMES)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.split == "test" and (args.variant != "numeric" or args.reverse):
        raise SystemExit("Secondary prompt checks are dev-only.")

    df = load_split(args.split)  # test raises unless NFL_FINAL_EVAL=1
    games = (test_games(args.games) if args.split == "test"
             else choose_games(df["game_id"].to_list(), args.games))
    df = df.filter(pl.col("game_id").is_in(games)).sort(["game_id", "play_id"])

    labels = template.labels(reverse=args.reverse)
    prompts = [template.render(row, args.variant)
               for row in df.select(config.FEATURES).iter_rows(named=True)]

    if args.dry_run:
        model_id = config.JEV_MODEL_ID or "<unset>"
        hits = (sum(client.cached(model_id, p, labels) is not None for p in prompts)
                if config.JEV_MODEL_ID else 0)
        print(f"{len(games)} games, {len(prompts)} prompts, {hits} cached, "
              f"{len(set(prompts))} unique prompts. Labels: {labels}\n")
        print(prompts[0])
        return

    model_id = client.validate_model_id(config.JEV_MODEL_ID)
    backend = client.TypeSafeBackend()
    # Identical situations produce identical prompts; the cache ensures one call each.
    p = []
    for i, prompt in enumerate(prompts):
        p.append(client.query(backend, model_id, prompt, labels)["p_dropback"])
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(prompts)}")

    name = "jev" + ("" if args.variant == "numeric" else f"-{args.variant}") + \
           ("-reversed" if args.reverse else "")
    save_predictions(df, np.array(p), name, args.split)
    print(f"wrote {name}_{args.split}: {len(p)} plays from {len(games)} games")


if __name__ == "__main__":
    main()
