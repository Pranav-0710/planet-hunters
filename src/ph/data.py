"""Shared data loading. Shared contract: flag before changing (see AGENTS.md)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

LABELS = sorted([
    "planet", "eclipsing_binary", "background_binary", "pulsating_star",
    "flare_star", "starspots", "instrument_glitch", "quiet_star",
])
SEED = 42


@dataclass
class Split:
    """One sector: flux (n, 1315), centroid (n, 2, 1315), star table, cadence table."""

    flux: np.ndarray
    centroid: np.ndarray
    stars: pd.DataFrame
    cadences: pd.DataFrame

    @property
    def time(self) -> np.ndarray:
        return self.cadences.time_btjd.to_numpy()

    @property
    def dump_mask(self) -> np.ndarray:
        """True at momentum-dump cadences (quality & 1)."""
        return (self.cadences.quality.to_numpy() & 1) > 0


def load_split(name: str, mmap: bool = True) -> Split:
    """Load 'train' or 'test'. mmap=True reads the .npy lazily (8 GB laptops)."""
    mode = "r" if mmap else None
    return Split(
        flux=np.load(DATA / f"{name}_flux.npy", mmap_mode=mode),
        centroid=np.load(DATA / f"{name}_centroid.npy", mmap_mode=mode),
        stars=pd.read_csv(DATA / f"{name}_stars.csv"),
        cadences=pd.read_csv(DATA / f"{name}_cadences.csv"),
    )


def load_folds() -> pd.Series:
    """Fold number (0-4) per train star, aligned to train_stars.csv row order."""
    folds = pd.read_csv(ROOT / "folds.csv")
    train_ids = pd.read_csv(DATA / "train_stars.csv", usecols=["id"]).id
    return folds.set_index("id").fold.loc[train_ids].reset_index(drop=True)


def relative_flux(flux: np.ndarray) -> np.ndarray:
    """Divide each row by its median and subtract 1. Accepts (n, t) or (t,)."""
    f = np.asarray(flux, np.float64)
    return f / np.nanmedian(f, axis=-1, keepdims=True) - 1
