import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

LABEL = {
    ("tsmnet", "default"): "TSMNet+SPDDSMBN",
    ("tsmnet", "inductive"): "TSMNet, BN collapsed to one shared domain",
    ("heegnet", "default"): "HEEGNet+EA",
    ("heegnet", "inductive"): "HEEGNet, BN collapsed to one shared domain",
    ("latte", "base"): "LAtte (base)",
    ("latte", "adaptmom"): "LAtte + adaptive momentum",
    ("latte", "domain"): "LAtte + domain stats",
    ("latte", "domain_subj"): "LAtte + both conditionings",
    # GyroAtt has no domain-specific statistics, so `--arm` does not apply.
    ("gyroatt", "plain"): "GyroAtt-SPD-LEM",
    ("gyroatt", "ea"): "GyroAtt-SPD-LEM + EA",
}
ORDER = [("tsmnet", "default"), ("tsmnet", "inductive"),
         ("heegnet", "default"), ("heegnet", "inductive"),
         ("gyroatt", "plain"), ("gyroatt", "ea")]

UNLABELED_TARGET = {("tsmnet", "default"), ("heegnet", "default"),
                ("gyroatt", "ea")}

LATTE_ROWS = ["base", "adaptmom", "domain", "domain_subj"]

# Optional arms encoded in the run tag, in the order they appear.
FLAG_SUFFIX = {
    "z":  " + z-scored input",
    "ea": " + EA on input",
    "bnrefit": " + BN refit on target",
}
UNLABELED_TARGET_FLAGS = {"ea", "bnrefit"}


def _latte_flags(r):
    """Optional arms encoded in the run name.

    The tag sits between `_{strategy}_` and `_{cell}_seed` and reads
    `latte[-z][-ea][-bnrefit]-{row}`. Neither the model name nor any row name
    contains a hyphen, so splitting on '-' is unambiguous.

    An EA or BN-refit run records the same `row` as the plain one, so the
    flags are read from the run name.
    """
    tag = r["run"].split(f"_{r['strategy']}_", 1)[-1]
    tag = tag.rsplit(f"_{r['cell']}_seed", 1)[0]
    return [p for p in tag.split("-")[1:-1] if p in FLAG_SUFFIX]


def variant_of(r):
    """Which row of the table a run belongs to.

    GyroAtt records carry `arm: "default"` from the argparse default, so its
    EA variant is read from the run name.
    """
    if r["model"] == "latte":
        lbl = [f"@{r['label']}"] if r.get("label") else []
        return "+".join([r.get("row")] + _latte_flags(r) + lbl)
    if r["model"] == "gyroatt":
        return "ea" if "gyroatt-ea" in r["run"] else "plain"
    return r.get("arm")


def label_for(model, variant):
    if model != "latte":
        return LABEL[(model, variant)]
    row, *rest = variant.split("+")
    flags = [f for f in rest if not f.startswith("@")]
    sweep = [f[1:] for f in rest if f.startswith("@")]
    return (LABEL.get(("latte", row), f"LAtte ({row})")
            + "".join(FLAG_SUFFIX[f] for f in flags)
            + (f"  [{sweep[0]}]" if sweep else ""))


def is_unlabeled_target(model, variant):
    if model == "latte":
        return bool(UNLABELED_TARGET_FLAGS.intersection(variant.split("+")[1:]))
    return (model, variant) in UNLABELED_TARGET


def ordered_variants(keys):
    """(model, variant) pairs present in `keys`, in table order.

    Each LAtte row is followed by its flagged variants, fewest flags first.
    """
    present = {(m, s) for _, _, m, s in keys}
    out = []
    for row in LATTE_ROWS:
        variants = [s for m, s in present
                    if m == "latte" and s.split("+")[0] == row]
        out += [("latte", s) for s in
                sorted(variants, key=lambda s: (len(s.split("+")), s))]
    out += sorted((m, s) for m, s in present
                  if m == "latte" and ("latte", s) not in out)
    out += [(m, s) for m, s in ORDER if (m, s) in present]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/moabb_protocol")
    ap.add_argument("--metric", default="test_bacc", choices=["test_bacc", "test_acc"])
    ap.add_argument("--per_cell", action="store_true")
    args = ap.parse_args()

    runs = []
    for p in sorted(Path(args.dir).glob("*.json")):
        try:
            runs.append(json.load(open(p)))
        except (json.JSONDecodeError, OSError):
            print(f"  ! unreadable, skipped: {p}")
    if not runs:
        print(f"No runs in {args.dir}.")
        return

    buckets = defaultdict(list)
    for r in runs:
        buckets[(r["dataset"], r["strategy"], r["model"], variant_of(r))].append(r)

    for ds in sorted({k[0] for k in buckets}):
        for strat in sorted({k[1] for k in buckets if k[0] == ds}):
            print(f"\n## {ds} — {strat}\n")
            print(f"| Row | best_val ({args.metric}) | ma_best_val (ref.) | Std (folds) | Cells "
                  f"| Reference | Params | Target access |")
            print("|---|---|---|---|---|---|---|---|")
            keys = [k for k in buckets if k[0] == ds and k[1] == strat]
            for m, s in ordered_variants(keys):
                k = (ds, strat, m, s)
                if k not in buckets:
                    continue
                rs = buckets[k]
                pv = [x["best_val"][args.metric] for x in rs]
                hv = [x["ma_best_val"][args.metric] for x in rs]
                sd = statistics.stdev(pv) if len(pv) > 1 else 0.0
                tgt = rs[0].get("reference_target")
                tgt = f"{tgt:.3f}" if tgt else "—"
                npar = rs[0].get("meta", {}).get("n_params", "—")
                unseen = any(x.get("n_unseen_domains", 0) for x in rs)
                access = ("unlabeled target trials"
                          if is_unlabeled_target(m, s) and unseen else "none")
                print(f"| {label_for(m, s)} | {statistics.mean(pv):.3f} | "
                      f"{statistics.mean(hv):.3f} | {sd:.3f} | {len(rs)} | "
                      f"{tgt} | {npar} | {access} |")
                if args.per_cell:
                    for x in sorted(rs, key=lambda x: x["cell"]):
                        print(f"|   {x['cell']} | {x['best_val'][args.metric]:.3f} "
                              f"| {x['ma_best_val'][args.metric]:.3f} | | | | | |")

    print("\n---\n")
    print("- Every row is scored at the epoch of lowest validation loss; the "
          "peak of the 5-point-smoothed validation score is shown for reference.")


if __name__ == "__main__":
    main()
