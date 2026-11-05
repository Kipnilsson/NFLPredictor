"""Direct HTTP Choice client. No network on import; paid calls require explicit opt-in."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src import config
from src.jev import template

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
INSTRUCTIONS = (
    "Predict the observed category of the next NFL offensive snap from the supplied "
    "pre-snap situation. The snap is known to be an eligible run or pass, excluding "
    "kneels, spikes, two-point attempts, aborted and deleted plays. Predict what "
    "the offense will do, rather than which play would be optimal. Only the "
    "provided situation is available; no team tendencies or play outcomes are given."
)
CRITERIA = {
    template.DROPBACK: "A quarterback dropback, including pass attempts, sacks and QB scrambles.",
    template.RUN: "An eligible run with no quarterback dropback; the designed-run proxy.",
}


def validate_model_id(model_id: str | None) -> str:
    if not model_id or not re.fullmatch(r"jev-\d+\.\d+\.\d+", model_id):
        raise ValueError("Use an exact versioned model ID, e.g. jev-1.13.0.")
    return model_id


def make_payload(model_id: str, prompt: str, labels: list[str]) -> dict:
    validate_model_id(model_id)
    if len(labels) != 2 or set(labels) != set(CRITERIA):
        raise ValueError("Expected the two explicit play labels exactly once.")
    return {"model": model_id, "state": prompt, "questions": {"next_play": {
        "type": "choice", "instructions": INSTRUCTIONS,
        "criteria": {label: CRITERIA[label] for label in labels},
    }}}


def cache_key(model_id: str, prompt: str, labels: list[str]) -> str:
    # Do NOT sort keys: the criteria insertion order is an experimental variable.
    payload = make_payload(model_id, prompt, labels)
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def cache_path(key: str):
    return config.CACHE_DIR / "jev-http-v2" / key[:2] / f"{key}.json"


def validate_answer(response: dict, model_id: str) -> dict:
    try:
        answer = response["answers"]["next_play"]
        probabilities = answer["probabilities"]
        if response["model"] != model_id or answer["type"] != "choice":
            raise ValueError()
        if set(probabilities) != set(CRITERIA):
            raise ValueError()
        values = list(probabilities.values())
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
               not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError()
        if not math.isclose(sum(values), 1, abs_tol=0.001):
            raise ValueError()
        if answer["choice"] not in probabilities:
            raise ValueError()
        if probabilities[answer["choice"]] < max(values):
            raise ValueError()
        return answer
    except (KeyError, TypeError, ValueError, AttributeError):
        raise ValueError("Invalid JEV response: model, labels, choice or probabilities.") from None


def cached(model_id: str, prompt: str, labels: list[str]) -> dict | None:
    path = cache_path(cache_key(model_id, prompt, labels))
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if (record.get("request") != make_payload(model_id, prompt, labels) or
            list(record["request"]["questions"]["next_play"]["criteria"]) != labels):
        raise ValueError("Cached request does not match the requested payload.")
    answer = validate_answer(record["response"], model_id)
    if (record["p_dropback"] != answer["probabilities"][template.DROPBACK] or
            record["p_run"] != answer["probabilities"][template.RUN]):
        raise ValueError("Cached dropback probability disagrees with raw response.")
    return record


def load_api_key() -> str:
    """Read only the key from environment or repo .env. Never execute dotenv contents."""
    for name in (config.JEV_API_KEY_ENV, "JEV_API_KEY"):
        if os.environ.get(name, "").strip():
            return os.environ[name].strip()
    path = config.ROOT / ".env"
    if path.exists():
        values = {}
        for line in path.read_text().splitlines():
            name, sep, value = line.strip().removeprefix("export ").partition("=")
            if sep and name.strip() in (config.JEV_API_KEY_ENV, "JEV_API_KEY"):
                values[name.strip()] = value.strip().strip("\"'")
        for name in (config.JEV_API_KEY_ENV, "JEV_API_KEY"):
            if values.get(name):
                return values[name]
    raise RuntimeError("Set TYPESAFE_API_KEY in your environment or ignored repo .env.")


class HTTPBackend:
    """Sequential, bounded requests. No automatic retries, including after timeouts."""

    def __init__(self, *, allow_paid: bool = False, max_requests: int = 0):
        if not allow_paid or max_requests <= 0:
            raise ValueError("Paid calls require allow_paid=True and a positive request cap.")
        self.api_key = load_api_key()
        self.max_requests = max_requests
        self.calls = 0

    def choose(self, payload: dict) -> dict:
        if self.calls >= self.max_requests:
            raise RuntimeError("Request cap reached; no request sent.")
        self.calls += 1  # Attempts count even when a response is lost or invalid.
        request = Request(ENDPOINT, data=json.dumps(payload).encode(), headers={
            "Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=45) as result:
                return json.load(result)
        except HTTPError as error:
            raise RuntimeError(f"TypeSafe HTTP {error.code}; stopped without retry.") from None
        except (URLError, TimeoutError):
            raise RuntimeError("TypeSafe connection failed; stopped without retry. "
                               "The server may have processed the request.") from None


def query(backend: HTTPBackend | None, model_id: str, prompt: str, labels: list[str]) -> dict:
    """Validate and replay cache; cache misses cannot call the API without a backend."""
    hit = cached(model_id, prompt, labels)
    if hit is not None:
        return hit
    if backend is None:
        raise RuntimeError("Cache miss; live inference is disabled.")
    payload = make_payload(model_id, prompt, labels)
    t0 = time.perf_counter()
    response = backend.choose(payload)
    latency = time.perf_counter() - t0
    answer = validate_answer(response, model_id)
    record = {"request": payload, "response": response,
              "template_version": template.TEMPLATE_VERSION,
              "p_dropback": float(answer["probabilities"][template.DROPBACK]),
              "p_run": float(answer["probabilities"][template.RUN]),
              "usage": response.get("usage", {}), "latency_s": latency,
              "timestamp": datetime.now(timezone.utc).isoformat()}
    path = cache_path(cache_key(model_id, prompt, labels))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2, allow_nan=False))
    tmp.replace(path)
    return record
