"""Cache-first JEV client.

Every call is cached in cache/ as one JSON file keyed by (model_id, prompt, label order).
A prompt with a cached response is never re-queried.

The SDK call itself lives in `TypeSafeBackend.choose`, the only place that touches the
TypeSafe SDK. It is a stub until the team confirms the SDK package and the `Choice` API
(see docs/decisions.md, "JEV SDK adapter").
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from src import config
from src.jev import template


@dataclass
class ChoiceResult:
    probs: dict[str, float]  # label -> probability
    raw: object               # JSON-serializable raw response
    usage: dict


class Backend(Protocol):
    def choose(self, prompt: str, labels: list[str], model_id: str) -> ChoiceResult: ...


class TypeSafeBackend:
    """Adapter around the TypeSafe SDK `Choice` primitive (two explicit labels)."""

    def __init__(self) -> None:
        key = os.environ.get(config.JEV_API_KEY_ENV)
        if not key:
            raise RuntimeError(f"Set {config.JEV_API_KEY_ENV} in the environment (never commit it).")
        self.api_key = key

    def choose(self, prompt: str, labels: list[str], model_id: str) -> ChoiceResult:
        # TODO(team): implement with the course's TypeSafe SDK, e.g.
        #   result = client.run(Choice(labels=labels), prompt=prompt, model=model_id)
        # and return ChoiceResult(probs={label: p, ...}, raw=<serializable>, usage={...}).
        raise NotImplementedError(
            "TypeSafe SDK adapter not implemented yet: confirm the SDK package and Choice "
            "API, then fill in src/jev/client.py:TypeSafeBackend.choose.")


_ALIAS = re.compile(r"(latest|default|stable)$", re.IGNORECASE)


def validate_model_id(model_id: str | None) -> str:
    if not model_id or _ALIAS.search(model_id) or not re.search(r"\d", model_id):
        raise ValueError(f"JEV model ID must be pinned and versioned, got {model_id!r}. "
                         "Set JEV_MODEL_ID.")
    return model_id


def cache_key(model_id: str, prompt: str, labels: list[str]) -> str:
    blob = json.dumps({"model_id": model_id, "prompt": prompt, "labels": labels},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def cache_path(key: str):
    return config.CACHE_DIR / key[:2] / f"{key}.json"


def cached(model_id: str, prompt: str, labels: list[str]) -> dict | None:
    path = cache_path(cache_key(model_id, prompt, labels))
    return json.loads(path.read_text()) if path.exists() else None


def query(backend: Backend | None, model_id: str, prompt: str, labels: list[str]) -> dict:
    """Return the cached record, or call the backend once and cache the full record."""
    hit = cached(model_id, prompt, labels)
    if hit is not None:
        return hit
    if backend is None:
        raise RuntimeError("cache miss and no backend (dry run)")
    t0 = time.perf_counter()
    res = backend.choose(prompt, labels, model_id)
    latency = time.perf_counter() - t0
    record = {
        "prompt": prompt, "labels": labels, "model_id": model_id,
        "template_version": template.TEMPLATE_VERSION,
        "response": res.raw,
        "p_dropback": float(res.probs[template.DROPBACK]),
        "p_run": float(res.probs[template.RUN]),
        "usage": res.usage, "latency_s": latency,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = cache_path(cache_key(model_id, prompt, labels))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2, default=str))
    tmp.rename(path)
    return record
