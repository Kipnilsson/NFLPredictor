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
