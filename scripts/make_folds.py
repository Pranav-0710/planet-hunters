"""Write folds.csv: 5 stratified folds over the training stars, seed 42. Run once; commit the output."""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ph.data import DATA, ROOT, SEED  # noqa: E402

stars = pd.read_csv(DATA / "train_stars.csv")
folds = pd.Series(-1, index=stars.index)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for k, (_, va) in enumerate(skf.split(stars, stars.label)):
    folds.iloc[va] = k
pd.DataFrame({"id": stars.id, "fold": folds}).to_csv(ROOT / "folds.csv", index=False)
print(pd.crosstab(stars.label, folds))
