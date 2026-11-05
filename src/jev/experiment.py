"""Shared execution controls for pilots and benchmarks; standard library only."""
from __future__ import annotations

import fcntl
from contextlib import contextmanager

from src import config
from src.jev import client


def plan(jobs: list[dict], model_id: str) -> dict:
    unique = {client.cache_key(model_id, j["prompt"], j["labels"]): j for j in jobs}
    hits = sum(client.cached(model_id, j["prompt"], j["labels"]) is not None
               for j in unique.values())
    return {"rows": len(jobs), "unique_requests": len(unique), "cached_requests": hits,
            "new_requests": len(unique) - hits, "model_id": model_id}


@contextmanager
def execution_lock():
    """One JEV writer at a time per clone; avoids simultaneous duplicate charges."""
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with (config.CACHE_DIR / "jev.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another JEV run holds this clone's cache lock.") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def execute(jobs: list[dict], model_id: str, *, live: bool = False,
            max_requests: int = 0) -> tuple[list[dict], dict]:
    with execution_lock():
        summary = plan(jobs, model_id)
        if live and max_requests <= 0:
            raise ValueError("--live requires a positive --max-requests.")
        if summary["new_requests"] and not live:
            raise RuntimeError("Cache is incomplete; no API calls made. Preview or explicitly opt in.")
        if live and summary["new_requests"] > max_requests:
            raise ValueError(f"Needs {summary['new_requests']} new requests, exceeding cap "
                             f"{max_requests}. No API calls made; reduce the sample.")
        backend = (client.HTTPBackend(allow_paid=True, max_requests=max_requests)
                   if live and summary["new_requests"] else None)
        records = [client.query(backend, model_id, j["prompt"], j["labels"]) for j in jobs]
        summary["requests_sent"] = backend.calls if backend else 0
        return records, summary
