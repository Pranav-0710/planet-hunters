# Planet Hunters (Team Cyclops)

ML Challenge Hack4Hack 2026, Round 1. Classify 7,000 test stars into 8 classes from TESS-like light curves. Metric: macro-F1. Round 1 closes 23 Oct 2026, 10:00 AM IST.

Problem statement: `docs/PROBLEM_STATEMENT.md`.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt
```

Put the competition files in `data/` (not committed). Download link is in the team workspace announcements.

## Layout

- `src/ph/` shared code: data loading, detrending, feature functions
- `scripts/` runnable steps (build features, train, blend, make a submission)
- `notebooks/` exploration and Kaggle notebooks (outputs stripped before commit)
- `oof/` out-of-fold and test probabilities from each model (`<model>_oof.csv`, `<model>_test.csv`; small CSVs are committed)
- `submissions/` every CSV we upload, named `NN_<short-name>.csv`, logged in `docs/SUBMISSIONS.md`

## Shared contract

- Folds: `folds.csv` (`id,fold`), 5 stratified folds, fixed seed 42. Every model uses it.
- Model outputs: `oof/<model>_oof.csv` and `oof/<model>_test.csv`, columns `id` plus one probability column per label in sorted label order.
- Every upload is tied to a git commit and logged in `docs/SUBMISSIONS.md`.
