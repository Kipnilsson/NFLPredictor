"""Record a prompt freeze only after main/verbal/reversed live checks have cached evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.jev import client, template

FREEZE_PATH = config.RESULTS_DIR / "jev" / "prompt_freeze.json"


def fingerprint(model_id: str) -> str:
    body = {"model_id": model_id, "template_version": template.TEMPLATE_VERSION,
            "template_source": Path(template.__file__).read_text(),
            "instructions": client.INSTRUCTIONS, "criteria": client.CRITERIA,
            "labels": template.labels(), "endpoint": client.ENDPOINT}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def require_frozen(model_id: str):
    if not FREEZE_PATH.exists():
        raise RuntimeError("No prompt freeze. Complete dev checks and run src.jev.freeze first.")
    record = json.loads(FREEZE_PATH.read_text())
    if record.get("fingerprint") != fingerprint(model_id):
        raise RuntimeError("Prompt/model changed since freeze; final evaluation blocked.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot-report", type=Path, required=True)
    ap.add_argument("--confirm-freeze", action="store_true")
    args = ap.parse_args(argv)
    if not args.confirm_freeze:
        raise SystemExit("Review the dev checks, then explicitly pass --confirm-freeze.")
    report = json.loads(args.pilot_report.read_text())
    model = client.validate_model_id(config.JEV_MODEL_ID)
    if (report.get("status") != "completed" or not report.get("checks_included") or
            report.get("fingerprint") != fingerprint(model) or
            report.get("model_id") != model or report.get("template_version") != template.TEMPLATE_VERSION):
        raise SystemExit("Need a completed pilot report with numeric, verbal and reversed checks.")
    rows = report.get("rows", [])
    groups = {}
    for row in rows:
        key = (row["variant"], row["reverse"])
        groups.setdefault(key, set()).add((row["game_id"], row["play_id"]))
        path = client.cache_path(row["cache_key"])
        cached = json.loads(path.read_text())
        payload = cached["request"]
        request = client.make_payload(model, payload["state"], list(payload["questions"]["next_play"]["criteria"]))
        if request != payload or client.cache_key(model, payload["state"], list(payload["questions"]["next_play"]["criteria"])) != row["cache_key"]:
            raise SystemExit("Pilot evidence does not match current request definitions.")
        client.validate_answer(cached["response"], model)
    expected = {("numeric", False), ("verbal", False), ("numeric", True)}
    if set(groups) != expected or not all(v == groups[("numeric", False)] for v in groups.values()) or not rows:
        raise SystemExit("Prompt checks must cover the same nonempty set of plays.")
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "model_id": model,
              "template_version": template.TEMPLATE_VERSION, "fingerprint": fingerprint(model),
              "pilot_report": str(args.pilot_report),
              "pilot_report_sha256": hashlib.sha256(args.pilot_report.read_bytes()).hexdigest(),
              "main_variant": "numeric", "main_labels": template.labels(),
              "selection_reason": "Prespecified numeric template and operational labels; no test tuning."}
    if FREEZE_PATH.exists():
        raise SystemExit("A freeze already exists; do not overwrite historical experiment records.")
    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_PATH.write_text(json.dumps(record, indent=2))
    print(f"Freeze saved to {FREEZE_PATH.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
