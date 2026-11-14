# Presentation (4 minutes, 5 slides)

Deck: https://claude.ai/artifact/KAqVDKyPXa5Hs2QpH2YfeR (Will shares access from the deck
page). The deck is the source of truth; this file is a copy of the outline and speaker
notes so everyone can rehearse. Every number comes from `results/` (2024 dev season).

**Still to fill in:** JEV numbers on slide 3 (`[__]` in the table, the note under it and
the speaker notes). These need the JEV runs below.

## Slides and speaker notes

### 1. Can a general model call the play? (~35 s)
Hi, we're Charlie, Kip and Will. Our question: can JEV, TypeSafe's pretrained decision
model, predict whether an NFL offense drops back to pass or runs a designed run as
accurately and as reliably as models we train on past plays?

Every approach gets the same eight facts about the situation before the snap: down,
distance, field position, quarter, clock, score, and both teams' timeouts. No team names,
no formation, nothing about the result.

The label is nflverse's qb_dropback flag, so sacks and scrambles count as dropbacks. We
predict the play that actually happened, not the best call.

### 2. Pipeline and changes from the proposal (~55 s)
We use nflverse play-by-play, regular season only. We keep run and pass plays and drop
two-point tries and aborted snaps. That leaves about 33,500 plays a season; 2024 gives
33,224, the same count as in our proposal.

We split by season: train on 2021 to 2023, tune on 2024, and 2025 is locked in code until
the final evaluation.

There are four approaches: a majority-class baseline, L2 logistic regression, histogram
gradient-boosted trees, and JEV zero-shot through its Choice endpoint, pinned to version
1.13.0. To prevent leakage, inputs come from an explicit allowlist of the eight features,
and the encoders and scalers are fit on training data only. Outcomes, play descriptions
and nflverse's own pass-over-expected column are never inputs.

Changes from the proposal, on the right: we call JEV over HTTP instead of the SDK. Because
of API cost, JEV runs on a fixed set of whole games, which was our proposal's fallback. We
pick hyperparameters by log loss so calibration counts. Overtime is its own category. And
we added spending caps and a prompt freeze so we can't tune on test outcomes.

### 3. Results on the 2024 development season (~50 s)
| Approach | Accuracy | Balanced accuracy | Brier score |
|---|---|---|---|
| Majority baseline | 60.5% | 0.500 | 0.239 |
| Logistic regression | 67.5% | 0.643 | 0.202 |
| Gradient-boosted trees | 71.3% | 0.703 | 0.179 |
| JEV, zero-shot | [__] | [__] | [__] |

On the 2024 dev season, always guessing "dropback" gets 60.5%, because about 61% of
training plays are dropbacks.

Logistic regression gets 67.5% and gradient-boosted trees 71.3%. Balanced accuracy shows
the gap more clearly, 0.64 against 0.70, because logistic regression misses about half of
the runs.

To compare models we resample whole games 2,000 times, since plays in the same game aren't
independent. Trees beat logistic regression by 3.8 points, with a 95% interval from 3.4 to
4.2. The trees are also well calibrated: in every probability bin, predicted and actual
dropback rates are within 2 points.

JEV: [fill in once the pilot and dev run are done: accuracy, Brier score, and whether it
beats the baseline and logistic regression].

One caveat: we also tuned on this season, so these numbers are a little optimistic. 2025
is the honest test.

### 4. Accuracy by down and by amount of training data (~55 s)
Breaking results down by down shows where the trees earn their lead.

On first down, everyone struggles: it's close to a coin flip, and the trees reach 63%. On
third down, logistic regression barely beats always-dropback, 80.8 against 80.2, while the
trees reach 87. On fourth down, the trees reach 87 against 73.

Our reading: the call depends on distance, clock and score together, and a linear model
can't represent those interactions without hand-built features. Fourth down has only 760
plays, so read that group with care.

On the right is the learning curve. Trees trained on just 16 games, 2% of the training
data, already beat logistic regression trained on all 815 games. About 50 times more data
adds only 3 more points. That's the bar JEV has to clear with no task data at all.

### 5. Limitations and next steps (~45 s)
Limitations. Our label is what happened, not what the coach called; scrambles and
run-pass options blur that line. We give the models only eight situation facts, so
there's a ceiling; formation and personnel would help, but they would change the
question. JEV may have seen these games during pretraining, so we treat this as a
comparison of approaches, not proof of football knowledge. And the dev results guided our
tuning, so they're a little optimistic.

Next steps: run the small JEV pilot with the wording and option-order checks, then freeze
the prompt. Score all four approaches on the same games, then run the 2025 test once.
After that, add team tendencies built only from earlier games, and test the frozen models
on 2026 games played after we finish.

Our code and data instructions are on GitHub. Thanks, and we're happy to take questions.

## JEV status (2026-11-14)

No live JEV requests have been made yet. Previews (no requests) on Will's checkout give
these counts of new paid requests:

| Step | New requests |
|---|---|
| Pilot, 20 plays | 20 |
| Prompt checks, 5 plays × 3 versions | 15 |
| Dev benchmark, 10 / 20 / 30 / 60 games | 1,188 / 2,391 / 3,555 / 7,204 |

Requests run one at a time (45 s timeout), so the 60-game run may take hours.

**Before anyone runs live:** agree on ONE person to run JEV (responses are cached per
checkout and don't sync). If someone already has `cache/jev-http-v2/`, copy it over
instead of paying twice.

The pilot needs the 2024 CSV. Without the original demo's `../data` folder:

```sh
mkdir -p data/source
curl -sSfL -o data/source/play_by_play_2024.csv.gz \
  https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2024.csv.gz
.venv/bin/python -m src.jev.pilot --source data/source/play_by_play_2024.csv.gz --plays 20
```

Then follow the README: the live pilot and checks (35 requests), a review of the
results, `src.jev.freeze`, and the dev benchmark with a chosen game count.
