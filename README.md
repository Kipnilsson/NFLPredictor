# NFL play prediction

Compare a constant baseline, logistic regression, gradient boosting, and zero-shot JEV
using the same eight pre-snap facts. Train: 2021–2023; development: 2024; final test:
2025. A prospective 2026 test still needs dates and a prespecified game list.

## Start here

Open `project_walkthrough.ipynb`. **Run All only reads local files and builds an example
request.** It never calls an API, downloads data, trains models, or evaluates 2025.
Reusable implementations live in `src/`; the notebook explains and displays their outputs.
The committed specialist results are development results, not an independent rerun.

Python 3.11+ (project developed on Python 3.12):

```sh
make setup
.venv/bin/python -m pip install -r requirements-notebook.txt
.venv/bin/python -m unittest discover -s tests -v
```

Select `.venv/bin/python` as the notebook kernel in your editor. For a standalone notebook
server, install JupyterLab in that environment. Do not put credentials in notebook cells.

## JEV: explicit execution only

JEV uses Python's standard-library HTTP client, matching the earlier demo's
`POST https://api.typesafe.ai/v1/systemone` Choice request. Default model:
`jev-1.13.0`; override only with an exact version through the **environment** variable
`JEV_MODEL_ID`. A different model is a new experimental choice.

Put `TYPESAFE_API_KEY=your_key` in the repo's ignored `.env`, or set that environment
variable. The local parser reads only the key; it does not execute shell statements or
load model settings. `JEV_API_KEY` is accepted as a legacy key name. Never commit keys.
Merely creating `.env` cannot trigger requests.

### Small pilot

The pilot reads a saved nflverse 2024 CSV/CSV.gz, selects eligible plays with seed 74,
and writes pilot artifacts separately from the benchmark. On Charlie's checkout, it
finds the original demo's CSV at `../data/play_by_play_2024.csv.gz`. On another checkout,
pass `--source /path/to/play_by_play_2024.csv.gz` to each command below. Raw data are
attributed to nflverse, CC BY 4.0.

Safe preview (no key, no requests, no output artifacts):

```sh
.venv/bin/python -m src.jev.pilot --plays 20
```

**Paid opt-in, only when ready** (at most 20 new HTTP requests):

```sh
.venv/bin/python -m src.jev.pilot --plays 20 --live --max-requests 20
```

Replay the exact sample with no API access:

```sh
.venv/bin/python -m src.jev.pilot --plays 20 --cache-only
```

Prompt sensitivity checks use the same five plays for numeric, verbal and reversed
option order (at most 15 new requests). Inspect the resulting probabilities; these
are diagnostics, not enough data to make performance claims.

```sh
.venv/bin/python -m src.jev.pilot --plays 5 --checks
# Paid opt-in; run separately only when ready:
.venv/bin/python -m src.jev.pilot --plays 5 --checks --live --max-requests 15
```

A live request cap counts **attempts**, not only successful responses. Preflight refuses
an incomplete batch above the cap before sending anything. No automatic retries are
made. A timeout may still have been processed by the server; don't blindly rerun it.
One writer per checkout holds a lock. Locks don't coordinate separate clones, so assign
one person to run JEV. Back up/share response caches securely outside Git.

Cache identity includes the complete request, instructions, model, criteria descriptions,
and option order. Old demo responses are not relabeled as new-template predictions.
Successful responses are validated and cached immediately; interrupted batches can resume.
Cache-only execution requires all requested responses to exist and be valid.

### Freeze after checks

No live pilot has been run in this implementation session, at Charlie's request.
The prompt is a **reviewed candidate**, not a completed experiment freeze.
After reviewing successful main/verbal/reversed checks:

```sh
.venv/bin/python -m src.jev.freeze \
  --pilot-report results/jev/pilots/pilot_5_checks.json --confirm-freeze
```

This records the model, timestamp, evidence hash and prompt fingerprint. Test JEV runs
require that freeze to exist and match the current prompt/model. Don't tune against test
outcomes. Record any later protocol amendment instead of overwriting a freeze.

### Benchmark

Prepare data explicitly with `make prep`. Preview the 2024 benchmark with:

```sh
.venv/bin/python -m src.jev.run --games 1
```

The default is always preview; `--dry-run` is an alias for it. Use `--cache-only` to replay,
or explicitly add `--live --max-requests N` with a deliberate positive cap. Choose the
number of games from measured pilot usage and runtime; there is no automatic full run.

Main predictions: `results/predictions/jev_dev.parquet`.
Secondary variants: `results/jev/variants/`.
Pilot reports: `results/jev/pilots/`.
All raw JEV responses: ignored `cache/jev-http-v2/`.

Main evaluation defaults to the three specialists plus JEV when its main file exists.
It does not sweep experimental files into the comparison. For an explicit comparison:

```sh
.venv/bin/python -m src.eval.run --split dev --models baseline logreg gbt jev
```

Every comparison uses the common plays of the selected approaches. Missing files,
duplicate IDs, unknown plays and invalid probabilities cause evaluation to stop.

## Offline specialist pipeline

`make prep`, `make train`, `make curves`, and `make eval-dev` are separate, deliberate
steps. `make dev` runs them all and may download data and take time, but cannot call JEV.
Data, model binaries, per-play predictions and API caches are ignored by Git; they must
be regenerated or obtained as documented artifacts. Training and evaluation do not load
`.env`. 2025 remains gated behind `NFL_FINAL_EVAL=1`; do not enable it for development.

## Still outstanding

- Live JEV pilot, prompt checks/freeze, and full comparison.
- TA-requested per-season class balances and feature distributions; 2025 loading audit.
- Dated team milestones and prospective 2026 experiment protocol.
- Situational confidence intervals, JEV learning-curve reference, final demo and report.
- Team review of proposed decisions in `docs/decisions.md`.

## Sources

- [TypeSafe HTTP API](https://docs.typesafe.ai/api/system-one)
- [Choice request/response format](https://docs.typesafe.ai/primitives/choice)
- [Versioned models](https://docs.typesafe.ai/models)
- [nflverse play-by-play](https://github.com/nflverse/nflverse-data/releases/tag/pbp)
