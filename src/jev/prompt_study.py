"""Paired prompt experiment on the locked 2024 development cohort."""
from __future__ import annotations

import argparse
import json

import polars as pl
from sklearn.metrics import roc_auc_score

from src import config
from src.benchmark import KEYS, read_json, sha256, write_json
from src.data import load_seasons
from src.eval.bootstrap import paired_bootstrap
from src.eval.metrics import all_metrics
from src.jev import client, freeze, template
from src.jev.experiment import execute_payloads, plan_payloads
from src.models.common import save_predictions

VARIANTS = ("numeric", "frequency", "clear_labels")


def make_payload(model_id, state, variant):
    payload = client.make_payload(model_id, state, template.labels())
    question = payload["questions"]["next_play"]
    if variant == "frequency":
        question["instructions"] += (
            " Imagine 100 eligible NFL snaps with these same pre-snap facts. "
            "Estimate how many would be dropbacks and how many would be runs. "
            "Set the two Choice probabilities to those expected frequencies divided "
            "by 100. Use probabilities near 0.5 when the two outcomes are similarly likely."
        )
    if variant == "clear_labels":
        question["criteria"] = {
            template.DROPBACK: ("The quarterback attempts a pass, is sacked on a pass play, "
                                "or scrambles after dropping back to pass."),
            template.RUN: ("A runner takes a handoff or pitch, or the quarterback carries "
                           "the ball on a designed run. Exclude scrambles after a dropback."),
        }
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--live", action="store_true")
    modes.add_argument("--cache-only", action="store_true")
    parser.add_argument("--max-requests", type=int, default=0)
    args = parser.parse_args(argv)
    if args.live != (args.max_requests > 0):
        parser.error("Use --live and a positive --max-requests together.")

    benchmark = config.RESULTS_DIR / "benchmarks" / "2024"
    manifest = read_json(benchmark / "cohort.json")
    model_id = manifest["jev_model_id"]
    if (manifest["jev_fingerprint"] != freeze.fingerprint(model_id)
            or manifest["data_sha256"] != sha256(config.PROCESSED_DIR / "plays_2024.parquet")):
        raise ValueError("Restore the original 2024 benchmark inputs.")
    data = load_seasons([2024]).filter(pl.col("game_id").is_in(manifest["game_ids"])).sort(KEYS)
    original = pl.read_parquet(benchmark / "predictions/jev_2024.parquet").sort(KEYS)
    if not original.select([*KEYS, config.LABEL]).equals(data.select([*KEYS, config.LABEL])):
        raise ValueError("Original JEV predictions do not match the locked plays.")
    states = [template.render(row) for row in data.select(config.FEATURES).iter_rows(named=True)]
    alternatives = VARIANTS[1:]
    payloads = [make_payload(model_id, state, variant) for variant in alternatives for state in states]
    print(json.dumps(plan_payloads(payloads, model_id), indent=2), flush=True)
    if not (args.live or args.cache_only):
        return

    records, usage = execute_payloads(payloads, model_id, live=args.live, max_requests=args.max_requests)
    out = config.RESULTS_DIR / "jev" / "prompt_study_2024"
    prediction_dir = out / "predictions"
    paired = data.select([*KEYS, config.LABEL]).with_columns(original["p_dropback"].alias("numeric"))
    probabilities = [record["p_dropback"] for record in records]
    for i, variant in enumerate(alternatives):
        scores = probabilities[i * data.height:(i + 1) * data.height]
        save_predictions(data, scores, variant, "2024", output_dir=prediction_dir)
        paired = paired.with_columns(pl.Series(variant, scores))
    y = paired[config.LABEL].to_numpy()
    comparison = []
    for variant in VARIANTS:
        p = paired[variant].to_numpy()
        metrics = all_metrics(y, p)
        comparison.append({"prompt": variant, **{k: metrics[k] for k in (
            "n", "accuracy", "balanced_accuracy", "brier", "recall_run", "recall_dropback")},
            "auroc": roc_auc_score(y, p), "mean_p_dropback": float(p.mean()),
            "predicted_dropback_rate": float((p >= 0.5).mean())})
    write_json(out / "study.json", {
        "season": 2024, "role": "exploratory development", "game_ids": manifest["game_ids"],
        "model_id": model_id, "selected_by_brier": min(comparison, key=lambda r: r["brier"])["prompt"],
        "variants": {variant: make_payload(model_id, states[0], variant)["questions"]["next_play"]
                     for variant in VARIANTS},
        "comparison": comparison,
        "paired_vs_numeric": {variant: paired_bootstrap(paired, variant, "numeric")
                              for variant in alternatives},
        "requests": usage,
    })
    pl.DataFrame(comparison).write_csv(out / "comparison.csv")
    print(pl.DataFrame(comparison))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
