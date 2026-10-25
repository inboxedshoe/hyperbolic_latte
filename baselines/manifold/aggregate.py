import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

MODEL_LABEL = {
    "tsmnet": "TSMNet+SPDDSMBN",
    "heegnet": "HEEGNet (DSMDBN+EA)",
}
ARM_LABEL = {
    "default": "default",
    "inductive": "domain-specific BN collapsed to a single shared domain",
    "sfuda": "target refit forced in every setting",
}
SETTING_ORDER = ["ss", "sc", "loso"]
# Per-subject settings have one fold per subject, and the count is per dataset.
N_SUBJECTS = {"bcic2a": 9, "bcic2b": 9, "bcicha": 16}


def load(directory):
    runs = []
    for p in sorted(Path(directory).glob("*.json")):
        try:
            with open(p) as f:
                runs.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            print(f"  ! unreadable, skipped: {p}")
    return runs


def target_access(rs):
    """What a cell used of the test trials, without labels.

    Two mechanisms use test inputs:

      BN refit    `domainadapt_finetune`, gated on an unseen domain existing
                  (LOSO), or forced by `sfuda`. Recorded in the run JSON's
                  `unlabeled_target` field and in `meta.target_refit`.
      input align HEEGNet's Euclidean Alignment. `train.py::run_heegnet` applies
                  it per domain over concat(train, val, test) before any CV
                  split, so each domain's reference is fitted on its test trials too.
    """
    refit = any(x.get("unlabeled_target") or x["meta"].get("target_refit") for x in rs)
    ea = any(x["meta"].get("input_align") for x in rs)
    parts = []
    if refit:
        parts.append("unlabeled target trials (BN refit)")
    if ea:
        parts.append("test inputs (Euclidean Alignment reference)")
    return " + ".join(parts) if parts else "none"


def key_of(r):
    a = r["args"]
    hp = sorted(a.get("hp") or [])
    if a["model"] == "heegnet" and a.get("swd_weight") is not None:
        hp.append(f"swd_weight={a['swd_weight']}")
    return (a["model"], a["dataset"], a["setting"], a["arm"], " ".join(hp), a.get("epochs"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/baselines")
    ap.add_argument("--rule", default="best_val",
                    choices=["best_val", "ma_best_val", "both"],
                    help="best_val = epoch of lowest validation loss; "
                         "ma_best_val = peak of the 5-point moving average of the validation score")
    ap.add_argument("--metric", default="test_bacc", choices=["test_bacc", "test_acc"])
    ap.add_argument("--per_subject", action="store_true")
    ap.add_argument("--dataset", nargs="*", default=None)
    args = ap.parse_args()

    runs = load(args.dir)
    if not runs:
        print(f"No runs in {args.dir}.")
        return
    if args.dataset:
        runs = [r for r in runs if r["args"]["dataset"] in args.dataset]

    rules = ["best_val", "ma_best_val"] if args.rule == "both" else [args.rule]
    buckets = defaultdict(list)
    folds = defaultdict(lambda: defaultdict(list))
    for r in runs:
        k = key_of(r)
        buckets[k].append(r)
        folds[k][r["args"]["heldout"]].append(r)

    datasets = sorted({k[1] for k in buckets})
    for ds in datasets:
        print(f"\n## {ds} — external baselines\n")
        for setting in SETTING_ORDER:
            keys = [k for k in buckets if k[1] == ds and k[2] == setting]
            if not keys:
                continue
            print(f"### {setting.upper()}\n")
            cols = " | ".join(f"{ru} {args.metric}" for ru in rules)
            print(f"| Model | Arm | overrides | {cols} | Std (seeds) | Runs | Folds | Target access |")
            print("|---|---|---|" + "---|" * (len(rules) + 4))
            for k in sorted(keys, key=lambda k: (k[0], k[3], k[4], str(k[5]))):
                rs = buckets[k]
                cells = []
                for ru in rules:
                    v = [x[ru][args.metric] for x in rs]
                    cells.append(f"{statistics.mean(v):.3f}")
                by_seed = defaultdict(list)
                for x in rs:
                    by_seed[x["args"]["seed"]].append(x[rules[0]][args.metric])
                sm = [statistics.mean(v) for v in by_seed.values()]
                sd = statistics.stdev(sm) if len(sm) > 1 else 0.0
                access = target_access(rs)
                ov = " ".join(filter(None, [k[4], f"epochs={k[5]}" if k[5] else ""])) or "-"
                print(f"| {MODEL_LABEL[k[0]]} | {ARM_LABEL[k[3]]} | {ov} | "
                      + " | ".join(cells)
                      + f" | {sd:.3f} | {len(rs)} | {len(folds[k])} | {access} |")
            print()

            if args.per_subject:
                for k in sorted(keys, key=lambda k: (k[0], k[3], k[4], str(k[5]))):
                    fold_ids = sorted(folds[k])
                    print(f"  {MODEL_LABEL[k[0]]} / {ARM_LABEL[k[3]]} / {k[4] or '-'}:")
                    for f in fold_ids:
                        v = [x[rules[0]][args.metric] for x in folds[k][f]]
                        print(f"    s{f}: {statistics.mean(v):.3f} "
                              f"({len(v)} seed{'s' if len(v) != 1 else ''})")
                print()

    print("---\n")
    if any(r["data"]["val_is_test"] for r in runs):
        print("- SS/SC select on the test split.")
    if any(r["args"]["setting"] == "ss" for r in runs):
        print("- SS has one domain, so domain-specific batch norm is inert.")
    if any(r["meta"].get("input_align") for r in runs):
        print("- Rows marked \"test inputs (Euclidean Alignment reference)\" fit "
              "per-domain EA whitening on all of that domain's trials, test split "
              "included, in SS and SC as well as LOSO. No labels are used; "
              "z-scaling is fit on train alone.")
    if any(r.get("unlabeled_target") for r in runs):
        print("- Rows marked \"unlabeled target trials\": the method is given the "
              "held-out subject's unlabeled trials. Labels are never used.")

    incomplete = [k for k in buckets
                  if len(folds[k]) < (1 if k[2] == "sc" else N_SUBJECTS.get(k[1], 9))]
    if incomplete:
        print("\n_Incomplete cells (fewer than the expected folds):_")
        for k in sorted(incomplete):
            print(f"  - {k[0]} {k[1]} {k[2]} {k[3]} ({len(folds[k])} folds)")


if __name__ == "__main__":
    main()
