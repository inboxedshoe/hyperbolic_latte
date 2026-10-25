import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latte.training.selection import RULES, read_curves, select  # noqa: E402


def iter_runs(root):
    for dirpath, _, files in os.walk(root):
        if "results.csv" in files and "params.json" in files:
            with open(os.path.join(dirpath, "params.json")) as f:
                yield dirpath, json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--rule", default="auto", choices=RULES)
    ap.add_argument("--filter", action="append", default=[], help="key=value on params.json")
    ap.add_argument("--group_by", action="append", default=[], help="extra params.json keys")
    args = ap.parse_args()

    filters = [f.split("=", 1) for f in args.filter]
    per_seed = defaultdict(lambda: defaultdict(list))   # group -> seed -> [scores]
    for root in args.roots:
        for run_dir, p in iter_runs(root):
            if any(str(p.get(k)) != v for k, v in filters):
                continue
            setting = p.get("setting") or ("sc" if str(p.get("sub")) == "all" else "ss")
            group = (p.get("model"), p.get("dataset"), setting) + tuple(str(p.get(k)) for k in args.group_by)
            per_seed[group][p.get("seed")].append(select(read_curves(os.path.join(run_dir, "results.csv")), args.rule))

    keys = ["model", "dataset", "setting"] + args.group_by
    print(" | ".join(keys) + " | seeds | runs/seed | mean | std")
    for group, seeds in sorted(per_seed.items(), key=lambda kv: tuple(map(str, kv[0]))):
        means = [100 * np.mean(v) for v in seeds.values()]
        std = np.std(means, ddof=1) if len(means) > 1 else 0.0
        n_runs = sorted({len(v) for v in seeds.values()})
        print(" | ".join(map(str, group)) + f" | {len(means)} | {n_runs} | {np.mean(means):.2f} | {std:.2f}")


if __name__ == "__main__":
    main()
