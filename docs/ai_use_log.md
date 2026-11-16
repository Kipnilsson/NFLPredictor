# AI Use Log

## Summary

We used two AI coding agents: **Claude Code (Claude Opus 5.5)** and **OpenAI Codex**. Both helped write the implementation. The idea, the research question, the proposal and the experiment design were made without AI. The design covers the 8 features, the season splits, the 2025 holdout, which models to compare, the metrics, and resampling by game. We wrote those rules into a project instructions file before any code existed, and then worked from it with agents. We read through all the generated code before trusting any results.

Claude:

**What it did**
- Cleaned up our project instructions file. We wrote the original; we asked it to flag gaps (like enforcing the 2025 lock in code), and we approved the fixes.
- Data pipeline: `src/prep.py`, `src/data.py`, `src/config.py`.
- Training code for the baseline, logistic regression, gradient boosting and learning curves: `src/models/`. Ideas were from class: L2 logistic regression, gradient boosting, small hyperparameter grids tuned on a dev set, and learning curves over training size.
- Evaluation code: `src/eval/` (metrics, game-level bootstrap, reliability plots, situational breakdowns).
- First version of the JEV template and cache in `src/jev/`. Most of the client was later replaced by Charlie.
- `Makefile`, `requirements.txt`, `.gitignore`, and the first draft of `docs/decisions.md`.
- Later: summarized the benchmark results, did a leakage and reproducibility check on them, and drafted this log.

**How we checked it**
- The 2024 play count had to match our proposal exactly (33,224).
- prep has built-in checks: scrambles must be labeled dropback, no duplicate play IDs, valid value ranges.
- We reran the whole pipeline from scratch and got identical numbers.
- Its defaults (overtime handling, tuning on log loss, bucket cutoffs, etc.) went into decisions.md as PROPOSED for team review.
- We didn't install an unverified "typesafe" package from PyPI and used Charlie's direct HTTP approach instead.
- The final check confirmed the 2025 run used the exact committed models, the prompt was frozen before 2025 was scored, and the JEV game samples come from the seed alone.

## Codex

**What it did**
- Rewrote the JEV client to call TypeSafe's HTTP API directly, based on Charlie's earlier demo: `src/jev/client.py`, `src/jev/experiment.py`.
- Safeguards for paid calls: an explicit `--live` flag, request caps, response validation, cache-only replay, a run lock.
- Pilot runner, prompt-freeze command and the 2024 prompt study: `src/jev/pilot.py`, `freeze.py`, `prompt_study.py`.
- Season benchmark runner for 2024, 2025 and 2026 to date: `src/benchmark.py`.
- Stricter evaluation checks in `src/eval/run.py`.
- Tests (`tests/`), `README.md`, and the walkthrough notebook (`project_walkthrough.ipynb`).

**How we checked it**
- Offline tests with mocked JEV responses, all passing before any paid call.
- A 20-play pilot plus wording and option-order checks before the full runs.
- The prompt was frozen before 2025 was scored, and the final runs used that frozen prompt with no tuning on test data. The prompt study was exploratory, on 2024 only.
- The notebook was run with network access blocked.
- Charlie decided when to make live (paid) runs and how many requests to allow; the agent didn't.

## Where AI-generated material appears
- **Code:** most of `src/`, `tests/`, the `Makefile` and the notebook. The agents wrote the implementation; the design came from us.
- **Docs:** `README.md` (Codex), and some of this log.
- **Results:** every number in `results/` comes from running the code.
- **Report / slides / demo:** _[fill in: e.g. "used Claude to tighten two paragraphs in the results section," or "none"]_
- **Will:** _[confirm whether AI was used for the interactive demo]_

We wrote
- The idea, the research question and the proposal.
- The experiment design and decisions that we had to make through the process. Some of these an agent brought up to us as a problem, and we decided the course of action. 
- Instructing the agents on what to build, and reviewing all the code.
- The conclusions and framing in the report. Claude summarized results; the interpretation is ours.

Thank you!
