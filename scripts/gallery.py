"""Class gallery (issue #8): docs/gallery/<label>.png with 8 training stars per label.

Each star shows relative flux (ppt) and the centroid column offset (millipixels), with momentum dumps marked.
Usage: python scripts/gallery.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ph.data import LABELS, ROOT, SEED, load_split, relative_flux  # noqa: E402

PER_LABEL = 8
OUT = ROOT / "docs" / "gallery"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s = load_split("train")
    t, dumps = s.time, s.time[s.dump_mask]
    rng = np.random.default_rng(SEED)
    for label in LABELS:
        idx = rng.choice(s.stars.index[s.stars.label == label], PER_LABEL, replace=False)
        fig, axes = plt.subplots(PER_LABEL, 2, figsize=(16, 1.7 * PER_LABEL), sharex=True)
        for row, i in enumerate(idx):
            f = relative_flux(s.flux[i]) * 1e3
            c = np.asarray(s.centroid[i, 0], np.float32)
            axes[row, 0].plot(t, f, ".", ms=1.5, c="k")
            axes[row, 1].plot(t, c - np.nanmedian(c), ".", ms=1.5, c="tab:blue")
            for ax in axes[row]:
                for d in dumps:
                    ax.axvline(d, c="orange", lw=0.5, alpha=0.6)
            star = s.stars.loc[i]
            axes[row, 0].set_title(f"{star.id}  Tmag {star.tmag:.1f}  Teff {star.teff}  contam {star.contamination:.2f}",
                                   fontsize=8, loc="left")
            axes[row, 0].set_ylabel("ppt", fontsize=7)
            axes[row, 1].set_ylabel("mpix", fontsize=7)
        axes[0, 1].set_title("centroid column offset", fontsize=8, loc="left")
        fig.suptitle(f"{label}: relative flux (left), centroid (right); orange = momentum dumps", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / f"{label}.png", dpi=60)
        plt.close(fig)
        print(f"wrote {OUT / f'{label}.png'}")


if __name__ == "__main__":
    main()
