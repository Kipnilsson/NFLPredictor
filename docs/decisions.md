# Decisions log

Every feature, preprocessing, prompt, or evaluation decision goes here, with date and who
agreed. Entries marked **PROPOSED** were defaults chosen while scaffolding the repo and
need team sign-off (Charlie, Kip, Will); change the status to **AGREED** once accepted.
Do not delete superseded entries; mark them **SUPERSEDED** and link the replacement.

---

## D1. Dataframe library: Polars — PROPOSED (2026-11-05)
`nflreadpy.load_pbp` returns Polars natively. All `src/` code uses Polars; data is
converted to pandas only at the scikit-learn boundary (`src/data.py:xy`).

## D2. Overtime handling for `qtr` — PROPOSED (2026-11-05)
Keep overtime as its own category, `qtr = 5` (172 eligible plays in 2024). Logistic
regression one-hot encodes `qtr` (Q1 as the reference level); the GBT model treats it as
categorical. `half_seconds_remaining` is used as given by nflverse (in OT it counts down
the OT period, max 600 s in the regular season). Alternative considered: merging OT into
Q4. Rejected because OT clock and rules differ.

## D3. Deleted-play filter column — PROPOSED (2026-11-05)
The nflverse column is `play_deleted` (present in 2021–2024 data). It drops 0 plays in
every season after the run/pass filter. Likewise `qb_kneel` and `qb_spike` drop 0,
because nflverse gives those plays their own `play_type` values, which the run/pass
filter already excludes.

## D4. 2025 isolation mechanism — PROPOSED (2026-11-05)
- `src/prep.py` processes 2021–2024 by default. `--include-test` also requires
  `NFL_FINAL_EVAL=1`.
- `src/data.py:load_seasons` raises `TestDataLocked` for 2025 unless `NFL_FINAL_EVAL=1`.
- A 2025 JEV subset is chosen from the 2025 **schedule** (game IDs of completed REG games,
  seed 74), never from play-by-play, and frozen in `results/jev_test_games.json` on first
  use.

## D5. Hyperparameter selection criterion — PROPOSED (2026-11-05)
Select on dev **log loss** (a proper scoring rule, so it rewards calibration as well as
accuracy). Grids:
- Logistic regression (L2): `C ∈ {0.001, 0.01, 0.1, 1, 10}`. Selected C = 10. C = 1 and
  C = 10 tie to 4 decimals; the penalty barely matters with ~100k rows and ~12 columns.
- HistGradientBoosting: `learning_rate ∈ {0.03, 0.1}`, `max_leaf_nodes ∈ {15, 31, 63}`,
  `min_samples_leaf ∈ {20, 100}`, `max_iter = 1000` with early stopping on a 10% holdout
  carved from **training** data. Selected: lr 0.03, 31 leaves, min leaf 100 (507 iters).
Full grids: `results/model_selection/*_grid.csv`.

## D6. Classification threshold — PROPOSED (2026-11-05)
Accuracy, balanced accuracy, precision/recall and confusion matrices use threshold 0.5
on P(dropback) for every approach, including JEV.

## D7. Situational buckets — PROPOSED (2026-11-05)
- Distance: 1–2, 3–6, 7–10, 11+ yards to go.
- Late game: Q4 with ≤ 300 s left in the half, or any OT. Split by score differential
  (trailing 9+, trailing 1–8, tied, leading 1–8, leading 9+). All other plays are
  "not late".

## D8. Paired bootstrap — PROPOSED (2026-11-05)
Resample whole games with replacement, B = 2000, seed 74. For each pair, use only the plays
both approaches predicted. Report the point difference, the 95% percentile CI, and the
share of draws where the first approach is not better.

## D9. Learning curve — PROPOSED (2026-11-05)
Training-game fractions {2, 5, 10, 25, 50, 100}% × 3 seeds (74, 75, 76). Subsets are nested
within a seed, and the dev-selected hyperparameters are held fixed.

## D10. JEV dev subset — PROPOSED (2026-11-05)
60 complete 2024 games chosen by seed 74 from sorted game IDs (7,258 plays, 7,204 unique
prompts). Used for prompt-wording checks and the numeric-vs-verbal and option-order checks.

## D11. JEV SDK adapter — OPEN
Two PyPI packages match the name (`typesafe` 0.9.1, `typesafe-sdk` 0.7.5). Neither is
installed until the team confirms which one is the course SDK. Pending:
- confirm package + version, and add it to `requirements.txt`
- implement `src/jev/client.py:TypeSafeBackend.choose` with the `Choice` primitive
- record the pinned versioned model ID (env var `JEV_MODEL_ID`)

## D12. JEV prompt freeze — OPEN
Template `v1` is in `src/jev/template.py`. Record the freeze here (date, template version,
dev checks run) before any test-set JEV call.
