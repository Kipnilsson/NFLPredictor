# AI use log

One entry per session in which an AI assistant wrote or changed project code, data
handling, or documentation. Record: date, tool/model, who ran it, what it did, and what a
human reviewed.

---

## 2026-11-05: Claude Code (Claude Opus 5.5), run by Kip Nilsson
- Reviewed and revised `CLAUDE.md`: renamed from `claude.md`; clarified 2025 isolation,
  the subset-selection rules, the deleted-play column and the docs requirements.
- Scaffolded the repo: `src/prep.py`, `src/data.py`, `src/models/`, `src/eval/`,
  `src/jev/`, `Makefile`, `requirements.txt`, `.gitignore`, `docs/decisions.md`.
- Ran the dev pipeline on 2021–2024 only. 2024 checkpoint matched (33,224). Trained the
  baseline, logistic regression and GBT; ran learning curves and the dev evaluation.
- Did not load 2025 data. Did not implement the TypeSafe SDK call (package unconfirmed).
- Defaults chosen are logged as PROPOSED in `docs/decisions.md`.
- **Human review needed:** decisions D1–D10, and the JEV template wording.


## 2026-11-14: Claude Code (Claude Opus 5.5), run by Will Mellis
- Drafted the 5-slide presentation deck and speaker notes from the proposal, the course
  rubric and the committed dev results (`results/eval_dev/`, `results/model_selection/`);
  copied the outline and notes to `docs/presentation.md`. JEV numbers left as `[__]`.
- Set up a local environment, reran `src.prep` (2021–2024 counts matched the committed
  `results/filter_counts.json`), ran the 20 offline tests (all passed), and previewed
  the JEV pilot, checks and benchmark request counts. No API requests were made; no code
  was changed.
- **Human review needed:** check every slide number against `results/`, the wording of
  the "changes from the proposal" table, and the speaker-note timing in rehearsal.


## 2026-11-05: OpenAI Codex, run by Charlie Martin
- Adapted the earlier demo's direct HTTP JEV integration; removed the unimplemented
  API client placeholder and package-selection blocker.
- Added strict response/cache validation, explicit live opt-in and request caps,
  cache-only replay, a per-clone execution lock, and separate pilot/variant outputs.
- Added a 2024 CSV pilot and evidence-backed prompt-freeze command. Live pilot and
  final freeze are deliberately deferred at Charlie's request; no API requests made.
- Added a read-only project walkthrough notebook, README and offline verification.
- Tightened main evaluation model selection, play-ID/probability checks and common-set
  paired comparisons. Specialist training code and saved findings were not changed.
- Human review needed: candidate prompt, pilot usage and sensitivity checks once run,
  final freeze and any upcoming 2026 prospective-test protocol.
- Verification: 20 offline tests passed; all 8 notebook code cells executed with HTTP
  and API-key access blocked, with two saved figures and no errors. Previewed a
  deterministic 20-play main pilot and five-play/three-variant sensitivity plan.
