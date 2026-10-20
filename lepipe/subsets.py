"""Nested, reproducible label subsets of the training split."""
import json
from pathlib import Path

import numpy as np

FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]


def n_chips(frac, total):
    """Number of chips for a label fraction (round half up, at least 1)."""
    return max(1, int(np.floor(frac * total + 0.5)))


def make_permutations(names, seeds):
    """One shuffled ordering of the chip names per seed; subsets are prefixes."""
    names = sorted(names)
    return {str(s): [names[i] for i in np.random.default_rng(s).permutation(len(names))] for s in seeds}


def save(path, perms, n_train):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    meta = {"n_train": n_train, "seeds": sorted(int(s) for s in perms), "fractions": FRACTIONS,
            "rule": "subset = first n_chips(frac, n_train) names of the seed's permutation"}
    with open(path, "w") as f:
        json.dump({"meta": meta, "permutations": perms}, f, indent=1)


def get_subset(path, seed, frac):
    """Chip names (S2 file names) for a given seed and label fraction."""
    order = json.load(open(path))["permutations"][str(seed)]
    return order[: n_chips(frac, len(order))]
