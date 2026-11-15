"""Prepare, plan, score and report one season using frozen specialist models."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from src import config
from src.data import load_seasons, xy
from src.eval.run import SPECIALISTS, join_predictions, write_evaluation
from src.jev import client, freeze, template
from src.jev.experiment import execute, execution_lock, plan as request_plan
from src.jev.run import choose_games
from src.models.common import MODEL_DIR, load_model, save_predictions
from src.prep import prepare_season

SEASONS = (2024, 2025, 2026)
KEYS = ["game_id", "play_id"]
ET = ZoneInfo("America/New_York")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def model_hashes():
    return {name: sha256(MODEL_DIR / f"{name}.joblib") for name in SPECIALISTS}


def register_prospective(path, through, model_id, today):
    freeze.require_frozen(model_id)
    if path.exists():
        raise ValueError("A prospective window is already registered.")
    if through <= today:
        raise ValueError("--through must be a future Eastern date.")
    write_json(path, {
        "registered_at": datetime.now(ET).isoformat(),
        "start": (today + timedelta(days=1)).isoformat(), "through": through.isoformat(),
        "model_hashes": model_hashes(), "jev_fingerprint": freeze.fingerprint(model_id),
    })
    print(f"Saved {path}")


def prospective_plays(df, protocol, today):
    import nflreadpy as nfl

    if today <= date.fromisoformat(protocol["through"]):
        raise ValueError("Wait until the registered 2026 window has closed in Eastern time.")
    schedule = nfl.load_schedules([2026])
    if schedule.schema["gameday"] == pl.String:
        schedule = schedule.with_columns(pl.col("gameday").str.to_date())
    games = schedule.filter(
        (pl.col("game_type") == "REG") &
        pl.col("gameday").is_between(date.fromisoformat(protocol["start"]),
                                     date.fromisoformat(protocol["through"]))
    )
    if games.is_empty() or games["result"].null_count():
        raise ValueError("The window needs completed regular-season games in the schedule.")
    ids = games["game_id"].to_list()
    cohort = df.filter(pl.col("game_id").is_in(ids))
    if set(cohort["game_id"]) != set(ids):
        raise ValueError("Play-by-play has not arrived for every registered game. Refresh data.")
    return cohort


def season_to_date_schedule(model_id):
    import nflreadpy as nfl

    freeze.require_frozen(model_id)
    freeze_day = datetime.fromisoformat(read_json(freeze.FREEZE_PATH)["timestamp"]).astimezone(ET).date()
    schedule = nfl.load_schedules([2026])
    if schedule.schema["gameday"] == pl.String:
        schedule = schedule.with_columns(pl.col("gameday").str.to_date())
    return schedule.filter(
        (pl.col("game_type") == "REG") & pl.col("result").is_not_null()
        & (pl.col("gameday") < freeze_day)
    ).select("game_id", "gameday", "week")



def make_jobs(df):
    labels = template.labels()
    return [{"prompt": template.render(row), "labels": labels}
            for row in df.select(config.FEATURES).iter_rows(named=True)]


def save_report(df, cohort, year, directory, manifest, include_jev):
    pred_dir = directory / "predictions"
    models = SPECIALISTS + (["jev"] if include_jev else [])
    name = "head_to_head" if include_jev else "specialists"
    out = directory / name
    joined, names, coverage = join_predictions(df, models, year, pred_dir)
    if any(coverage[k] != df.height for k in SPECIALISTS):
        raise ValueError("A specialist is missing plays. Restore the season prediction files.")
    if include_jev:
        expected = cohort.select(KEYS).sort(KEYS)
        actual = joined.filter(pl.col("jev").is_not_null()).select(KEYS).sort(KEYS)
        if not actual.equals(expected):
            raise ValueError("JEV prediction coverage differs from the locked cohort.")
    if joined.drop_nulls(names)[config.LABEL].n_unique() < 2:
        raise ValueError("The comparison needs observed dropbacks and runs.")
    metadata = {"season": int(year),
                "evaluation_role": manifest.get("evaluation_role",
                                                {"2024": "development", "2025": "holdout",
                                                 "2026": "prospective"}[year]),
                "season_eligible_plays": df.height,
                "season_dropbacks": int(df[config.LABEL].sum()),
                "season_runs": df.height - int(df[config.LABEL].sum()),
                "planned_jev_games": len(manifest["game_ids"]),
                "common_dropbacks": int(joined.drop_nulls(names)[config.LABEL].sum()),
                "model_hashes": manifest["model_hashes"],
                "jev_model_id": manifest["jev_model_id"] if include_jev else None}
    write_evaluation(joined, names, coverage, year, out, out / "reliability.png",
                     metadata=metadata)
    metrics = read_json(out / "metrics.json")
    columns = ["n", "accuracy", "balanced_accuracy", "brier", "log_loss"]
    pl.DataFrame([{"season": int(year), "model": model,
                   **{key: metrics["on_common_set"][model][key] for key in columns}}
                  for model in names]).write_csv(out / "comparison.csv")
    print(f"Saved {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("season", type=int, choices=SEASONS)
    actions = ap.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--preview", action="store_true")
    actions.add_argument("--lock", action="store_true")
    actions.add_argument("--specialists", action="store_true")
    actions.add_argument("--live", action="store_true")
    actions.add_argument("--cache-only", action="store_true")
    actions.add_argument("--report", action="store_true")
    actions.add_argument("--register-prospective", type=date.fromisoformat, metavar="THROUGH")
    size = ap.add_mutually_exclusive_group()
    size.add_argument("--games", type=int)
    size.add_argument("--all-games", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--season-to-date", action="store_true",
                    help="2026 retrospective benchmark on completed pre-freeze games")
    ap.add_argument("--max-requests", type=int)
    args = ap.parse_args(argv)
    if args.live and (args.max_requests is None or args.max_requests <= 0):
        ap.error("--live requires a positive --max-requests")
    if args.max_requests is not None and not args.live:
        ap.error("--max-requests is only for --live")
    if args.refresh and not args.prepare:
        ap.error("--refresh is only for --prepare")
    if args.games is not None and args.games <= 0:
        ap.error("--games must be positive")
    if (args.games is not None or args.all_games) and not (args.preview or args.lock):
        ap.error("Choose the game count during preview or lock")
    if args.season_to_date and (args.season != 2026 or args.register_prospective):
        ap.error("--season-to-date is for retrospective 2026 benchmarking")
    if args.season != 2026 and args.register_prospective:
        ap.error("Prospective registration is for 2026")
    if args.season != 2024 and not config.final_eval_enabled():
        ap.error(f"Set {config.FINAL_EVAL_ENV}=1 for held-out seasons")

    year = str(args.season)
    directory = config.RESULTS_DIR / "benchmarks" / ("2026_to_date" if args.season_to_date else year)
    data_name = "plays_2026_to_date.parquet" if args.season_to_date else f"plays_{year}.parquet"
    data_path = config.PROCESSED_DIR / data_name
    manifest_path = directory / "cohort.json"
    protocol_path = directory / "prospective.json"
    today = datetime.now(ET).date()
    model_id = client.validate_model_id(config.JEV_MODEL_ID)
    if args.register_prospective:
        register_prospective(protocol_path, args.register_prospective, model_id, today)
        return
    protocol = read_json(protocol_path) if args.season == 2026 and not args.season_to_date else None
    if protocol and today <= date.fromisoformat(protocol["through"]):
        raise ValueError("Wait until the registered 2026 window closes.")
    if args.prepare:
        if manifest_path.exists():
            raise ValueError("This season already has a locked cohort and data snapshot.")
        if args.season_to_date:
            schedule = season_to_date_schedule(model_id)
            counts = prepare_season(2026, args.refresh, output_path=data_path,
                                    game_ids=schedule["game_id"].to_list())
            ids = pl.read_parquet(data_path, columns=["game_id"])["game_id"].unique()
            covered = schedule.filter(pl.col("game_id").is_in(ids.to_list()))
            counts.update({"first_game_date": covered["gameday"].min().isoformat(),
                           "last_game_date": covered["gameday"].max().isoformat(),
                           "latest_week_with_data": int(covered["week"].max()),
                           "completed_games_without_play_data": schedule.height - covered.height})
        else:
            counts = prepare_season(args.season, args.refresh)
        write_json(directory / "data_counts.json", counts)
        print(f"Prepared {year}: {counts['eligible']:,} eligible plays.")
        return

    df = (pl.read_parquet(data_path) if args.season_to_date else load_seasons([args.season])).sort(KEYS)
    hashes = model_hashes()
    prompt_hash = freeze.fingerprint(model_id)
    if protocol:
        if hashes != protocol["model_hashes"] or prompt_hash != protocol["jev_fingerprint"]:
            raise ValueError("A model or prompt changed after prospective registration.")
        if not manifest_path.exists():
            df = prospective_plays(df, protocol, today)

    signature = {"data_sha256": sha256(data_path), "model_hashes": hashes,
                 "jev_model_id": model_id, "jev_fingerprint": prompt_hash}
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if any(manifest.get(k) != v for k, v in signature.items()):
            raise ValueError("Inputs changed since the cohort was locked. Restore the pinned inputs.")
        if args.lock or args.games is not None or args.all_games:
            raise ValueError("This season already has a locked cohort.")
        if protocol:
            df = df.filter(pl.col("game_id").is_in(manifest["eligible_game_ids"]))
    else:
        if not (args.preview or args.lock):
            raise ValueError("Preview and lock the cohort before scoring.")
        ids = sorted(df["game_id"].unique().to_list())
        if not ids or (args.games is not None and args.games > len(ids)):
            raise ValueError(f"Only {len(ids)} eligible games are available.")
        count = len(ids) if args.all_games else args.games
        if count is None:
            raise ValueError("Choose --games N or --all-games explicitly.")
        manifest = {"season": args.season, "seed": config.SEED, **signature,
                    "eligible_game_ids": ids, "game_ids": choose_games(ids, count)}
        if args.season_to_date:
            manifest["evaluation_role"] = "retrospective_season_to_date"

    cohort = df.filter(pl.col("game_id").is_in(manifest["game_ids"]))
    if cohort.is_empty():
        raise ValueError("The selected cohort has no eligible plays.")
    if args.preview or args.lock:
        print(json.dumps({"season": args.season, "eligible_plays": df.height,
                          "selected_games": len(manifest["game_ids"]),
                          **request_plan(make_jobs(cohort), model_id)}, indent=2))
        if args.lock:
            if args.season != 2024:
                freeze.require_frozen(model_id)
            with execution_lock():
                if manifest_path.exists():
                    raise ValueError("This season already has a locked cohort.")
                write_json(manifest_path, manifest)
            print(f"Locked {manifest_path}")
        return

    pred_dir = directory / "predictions"
    if args.specialists:
        with execution_lock():
            X = None
            for name in SPECIALISTS:
                if not (pred_dir / f"{name}_{year}.parquet").exists():
                    if X is None:
                        X, _ = xy(df)
                    save_predictions(df, load_model(name).predict_proba(X)[:, 1], name, year,
                                     output_dir=pred_dir)
            save_report(df, cohort, year, directory, manifest, include_jev=False)
        return

    if args.season != 2024:
        freeze.require_frozen(model_id)
    # Validate specialist outputs before requesting any paid inference.
    if not (pred_dir / f"jev_{year}.parquet").exists() and not args.report:
        _, _, coverage = join_predictions(df, SPECIALISTS, year, pred_dir)
        if any(n != df.height for n in coverage.values()):
            raise ValueError("Complete the specialist predictions first.")
        if cohort[config.LABEL].n_unique() < 2:
            raise ValueError("The selected cohort has only one observed class.")
        records, summary = execute(make_jobs(cohort), model_id, live=args.live,
                                   max_requests=args.max_requests or 0)
        save_predictions(cohort, [record["p_dropback"] for record in records], "jev", year,
                         output_dir=pred_dir)
        write_json(directory / "jev_run.json", summary)
        print(f"JEV requests sent: {summary['requests_sent']}")
    save_report(df, cohort, year, directory, manifest, include_jev=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(str(exc)) from None
