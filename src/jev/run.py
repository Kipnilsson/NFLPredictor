"""Run JEV zero-shot on a reproducible subset of complete games.

Usage:
    python -m src.jev.run --dry-run                    # build prompts, show sample, no API
    python -m src.jev.run                              # preview only; no API calls
    python -m src.jev.run --variant verbal             # secondary check (dev only)
    python -m src.jev.run --reverse                    # secondary check (dev only)
    NFL_FINAL_EVAL=1 python -m src.jev.run --split test --games N   # frozen final preview

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
from src.jev.experiment import execute, plan
from src.jev.freeze import require_frozen
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": config.SEED, "n": len(games), "game_ids": games},
                               indent=2))
    return games


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--variant", choices=list(template.RENDERERS), default="numeric")
    ap.add_argument("--reverse", action="store_true", help="reverse option order")
    ap.add_argument("--games", type=int, default=config.JEV_DEV_SUBSET_GAMES)
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="preview (also the default)")
    modes.add_argument("--cache-only", action="store_true")
    modes.add_argument("--live", action="store_true", help="explicitly enable paid requests")
    ap.add_argument("--max-requests", type=int, default=0)
    args = ap.parse_args(argv)

    if args.games <= 0:
        raise SystemExit("--games must be positive")
    if args.split == "test":
        require_frozen(client.validate_model_id(config.JEV_MODEL_ID))
    if args.split == "test" and (args.variant != "numeric" or args.reverse):
        raise SystemExit("Secondary prompt checks are dev-only.")

    df = load_split(args.split)  # test raises unless NFL_FINAL_EVAL=1
    games = (test_games(args.games) if args.split == "test"
             else choose_games(df["game_id"].to_list(), args.games))
    df = df.filter(pl.col("game_id").is_in(games)).sort(["game_id", "play_id"])

    labels = template.labels(reverse=args.reverse)
    prompts = [template.render(row, args.variant)
               for row in df.select(config.FEATURES).iter_rows(named=True)]

    model_id = client.validate_model_id(config.JEV_MODEL_ID)
    jobs = [{"prompt": prompt, "labels": labels} for prompt in prompts]
    if not jobs:
        raise SystemExit("No eligible plays; no API calls made.")
    print(json.dumps(plan(jobs, model_id), indent=2))
    print(prompts[0])
    if not (args.live or args.cache_only):
        print("Preview only. No API calls made.")
        return
    records, summary = execute(jobs, model_id, live=args.live, max_requests=args.max_requests)
    name = "jev" + ("" if args.variant == "numeric" else f"-{args.variant}") + \
           ("-reversed" if args.reverse else "")
    secondary = args.variant != "numeric" or args.reverse
    output_dir = config.RESULTS_DIR / "jev" / "variants" if secondary else config.PRED_DIR
    save_predictions(df, np.array([r["p_dropback"] for r in records]), name,
                     args.split, output_dir=output_dir)
    config.RESULTS_DIR.joinpath("jev").mkdir(parents=True, exist_ok=True)
    from src.jev.freeze import fingerprint
    manifest = {"model_id": model_id, "fingerprint": fingerprint(model_id),
                "variant": args.variant, "reverse": args.reverse, "game_ids": games,
                "summary": summary, "template_version": template.TEMPLATE_VERSION}
    (output_dir / f"{name}_{args.split}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {name}: {len(records)} plays; {summary['requests_sent']} requests sent.")


if __name__ == "__main__":
    main()
