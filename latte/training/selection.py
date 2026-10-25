import csv

import numpy as np

RULES = ("auto", "smoothed", "best_val", "last")


def read_curves(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def has_validation(curves):
    return not np.allclose(curves["val_accs"], curves["test_accs"])


def select(curves, rule="auto", window=5):
    val, test = curves["val_accs"], curves["test_accs"]
    if rule == "auto":
        rule = "smoothed" if has_validation(curves) else "best_val"
    if rule == "smoothed":
        if len(val) < window:
            return float(test[int(np.argmax(val))])
        ma = np.convolve(val, np.ones(window) / window, mode="valid")
        return float(test[int(np.argmax(ma)) + window // 2])
    if rule == "best_val":
        return float(test[int(np.argmax(val))])
    if rule == "last":
        return float(test[-1])
    raise ValueError(f"unknown rule {rule!r}; choose from {RULES}")
