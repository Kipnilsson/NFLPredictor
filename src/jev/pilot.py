"""Small, reproducible 2024 pilot, independent of the model-training dependencies.

Default: preview only. --checks adds verbal and reversed variants on the same plays.
Uses the saved CSV from the original demo, or an explicit --source CSV/CSV.gz.
Never writes benchmark prediction files or unlocks 2025.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.jev import client, template
from src.jev.experiment import execute, plan

DEFAULT_SOURCE = config.ROOT.parent / "data" / "play_by_play_2024.csv.gz"


def sample_plays(source: Path, n: int) -> list[dict]:
    if n <= 0 or n > 100:
        raise ValueError("Pilot sample must contain 1-100 plays.")
    rng, sample, seen = random.Random(config.SEED), [], 0
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt") as stream:
        for row in csv.DictReader(stream):
            if row.get("season") != "2024" or row.get("season_type") != "REG":
                continue
            if row.get("play_type") not in ("run", "pass"):
                continue
            if any(row.get(k) in ("1", "1.0") for k in
                   ("qb_kneel", "qb_spike", "two_point_attempt", "aborted_play", "play_deleted")):
                continue
            if row.get(config.LABEL) not in ("0", "0.0", "1", "1.0"):
                continue
            try:
                f = {k: float(row[k]) for k in config.FEATURES}
                template.render(f)  # finite integer inputs; no outcomes in prompt
            except (ValueError, KeyError, TypeError):
                continue
            item = {"game_id": row["game_id"], "play_id": row["play_id"],
                    "features": f, "true_label": int(float(row[config.LABEL]))}
            seen += 1
            if len(sample) < n:
                sample.append(item)
            else:
                idx = rng.randrange(seen)
                if idx < n:
                    sample[idx] = item
    if len(sample) < n:
        raise ValueError("Not enough eligible 2024 plays in source.")
    return sorted(sample, key=lambda r: (r["game_id"], float(r["play_id"])))


def jobs_for(plays: list[dict], checks: bool) -> list[dict]:
    variants = [("numeric", False)]
    if checks:
        variants += [("verbal", False), ("numeric", True)]
    return [{"game_id": p["game_id"], "play_id": p["play_id"],
             "true_label": p["true_label"], "variant": variant, "reverse": reverse,
             "prompt": template.render(p["features"], variant),
             "labels": template.labels(reverse)}
            for variant, reverse in variants for p in plays]


def diagnostics(rows: list[dict], records: list[dict]) -> dict:
    """Descriptive pilot checks, not reliable estimates of comparative performance."""
    groups = {}
    for row in rows:
        name = row["variant"] + ("-reversed" if row["reverse"] else "")
        groups.setdefault(name, []).append(row)
    summaries = {}
    for name, group in groups.items():
        n = len(group)
        summaries[name] = {
            "n": n,
            "accuracy": sum((r["p_dropback"] >= .5) == r["true_label"] for r in group) / n,
            "brier": sum((r["p_dropback"] - r["true_label"]) ** 2 for r in group) / n,
            "mean_p_dropback": sum(r["p_dropback"] for r in group) / n,
        }
    baseline = {(r["game_id"], r["play_id"]): r["p_dropback"] for r in groups["numeric"]}
    sensitivity = {}
    for name, group in groups.items():
        if name == "numeric":
            continue
        differences = [r["p_dropback"] - baseline[(r["game_id"], r["play_id"])] for r in group]
        sensitivity[name] = {"n_paired": len(group),
                             "mean_absolute_probability_change": sum(map(abs, differences)) / len(group),
                             "max_absolute_probability_change": max(map(abs, differences))}
    unique_records = {row["cache_key"]: rec for row, rec in zip(rows, records)}
    usage = {token: sum(rec["usage"].get(token, 0) for rec in unique_records.values())
             for token in ("input_tokens", "output_tokens")}
    return {"variant_summary": summaries, "sensitivity": sensitivity,
            "unique_response_usage_including_cache": usage,
            "mean_original_latency_s": sum(r["latency_s"] for r in unique_records.values()) / len(unique_records)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--plays", type=int, default=20)
    ap.add_argument("--checks", action="store_true")
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--live", action="store_true")
    modes.add_argument("--cache-only", action="store_true")
    ap.add_argument("--max-requests", type=int, default=0)
    args = ap.parse_args(argv)
    model = client.validate_model_id(config.JEV_MODEL_ID)
    plays = sample_plays(args.source, args.plays)
    jobs = jobs_for(plays, args.checks)
    summary = plan(jobs, model)
    print(json.dumps(summary, indent=2))
    print(json.dumps(client.make_payload(model, jobs[0]["prompt"], jobs[0]["labels"]), indent=2))
    if not (args.live or args.cache_only):
        print("Preview only. No inference or output artifacts written.")
        return
    records, summary = execute(jobs, model, live=args.live, max_requests=args.max_requests)
    rows = []
    for job, record in zip(jobs, records):
        rows.append({k: job[k] for k in ("game_id", "play_id", "true_label", "variant", "reverse")} |
                    {"cache_key": client.cache_key(model, job["prompt"], job["labels"]),
                     "p_dropback": record["p_dropback"], "usage": record["usage"],
                     "original_latency_s": record["latency_s"]})
    from src.jev.freeze import fingerprint
    report = {"fingerprint": fingerprint(model), "status": "completed", "timestamp": datetime.now(timezone.utc).isoformat(),
              "template_version": template.TEMPLATE_VERSION, "model_id": model,
              "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
              "n_distinct_plays": len(plays), "checks_included": args.checks,
              "summary": summary, "rows": rows,
              "interpretation": "Development pilot only; not final performance evidence."}
    report.update(diagnostics(rows, records))
    out = config.RESULTS_DIR / "jev" / "pilots"
    out.mkdir(parents=True, exist_ok=True)
    suffix = "checks" if args.checks else "main"
    path = out / f"pilot_{args.plays}_{suffix}.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"Saved {path.relative_to(config.ROOT)}; {summary['requests_sent']} requests sent.")


if __name__ == "__main__":
    main()
