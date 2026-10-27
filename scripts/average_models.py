"""Average the probabilities of several models into one oof/<out>_{oof,test}.csv pair.

Used for the CNN seed ensemble in upload 07:
    python scripts/average_models.py --models cnn cnn_s1 cnn_s2 --out cnn3
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ph.data import LABELS, ROOT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    for part in ("oof", "test"):
        frames = [pd.read_csv(ROOT / f"oof/{m}_{part}.csv") for m in args.models]
        mean = sum(f[LABELS].to_numpy() for f in frames) / len(frames)
        out = pd.concat([frames[0][["id"]], pd.DataFrame(mean, columns=LABELS)], axis=1)
        out.to_csv(ROOT / f"oof/{args.out}_{part}.csv", index=False)
        print(f"wrote oof/{args.out}_{part}.csv from {', '.join(args.models)}")


if __name__ == "__main__":
    main()
