import argparse
import json
import os
import time
import warnings

import numpy as np
import torch

TARGETS = {
    ("tsmnet", "inter-session"): 0.6903935185185186,
    ("heegnet", "inter-subject"): 0.541,
}

REFERENCE_TARGETS = {
    ("tsmnet", "BNCI2014001", "inter-session"): 0.690394,
    ("tsmnet", "BNCI2014001", "inter-subject"): 0.515625,
    ("tsmnet", "BNCI2015001", "inter-session"): 0.858472,
    ("tsmnet", "BNCI2015001", "inter-subject"): 0.769722,
    ("heegnet", "BNCI2014001", "inter-subject"): 0.541,
    ("heegnet", "BNCI2015001", "inter-subject"): 0.758,
}

FMIN, FMAX = 4.0, 36.0
DATASETS = {
    "BNCI2014001": dict(classes=["left_hand", "right_hand", "feet", "tongue"],
                        tmin=0.5, tmax=3.496, resample=None),
    "BNCI2015001": dict(classes=["right_hand", "feet"],
                        tmin=1.0, tmax=4.0, resample=256),
}
CLASSES = DATASETS["BNCI2014001"]["classes"]


def load_bnci(dataset, cache_dir):
    """X, y, subject, session for `dataset` under the preprocessing above."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset}; have {list(DATASETS)}")
    spec = DATASETS[dataset]
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(
        cache_dir,
        f"{dataset}_bb{FMIN:g}-{FMAX:g}Hz_{spec['tmin']:g}-{spec['tmax']:g}.npz")
    if os.path.isfile(cache):
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["y"], z["subject"], z["session"]

    import moabb.datasets as mds
    from moabb.paradigms import MotorImagery

    ds = {"BNCI2014001": mds.BNCI2014_001,
          "BNCI2015001": mds.BNCI2015_001}[dataset]()
    paradigm = MotorImagery(events=spec["classes"], n_classes=len(spec["classes"]),
                            fmin=FMIN, fmax=FMAX,
                            tmin=spec["tmin"], tmax=spec["tmax"],
                            resample=spec["resample"])
    X, labels, meta = paradigm.get_data(ds, ds.subject_list)

    from sklearn.preprocessing import LabelEncoder
    y = LabelEncoder().fit_transform(labels).astype(np.int64)
    subject = meta["subject"].to_numpy().astype(np.int64)
    session = LabelEncoder().fit_transform(meta["session"].to_numpy()).astype(np.int64)
    X = np.ascontiguousarray(X, dtype=np.float32)
    np.savez_compressed(cache, X=X, y=y, subject=subject, session=session)
    return X, y, subject, session


def make_cell(X, y, domain, train_mask, test_mask, seed, val_frac=0.2, n_classes=4):
    """Build the dict `baselines.manifold.train`'s runners consume, with a val split.

    The split is `StratifiedShuffleSplit(n_splits=1, test_size=0.2)` over
    `labels + domains * n_classes`, i.e. stratified on class *and* domain, seeded
    per fold from the target domain id.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    tr = np.flatnonzero(train_mask)
    te = np.flatnonzero(test_mask)
    strat = y[tr] + domain[tr] * n_classes
    # Classes with a single member cannot be stratified; fold them into a
    # catch-all bucket.
    vals, counts = np.unique(strat, return_counts=True)
    rare = set(vals[counts < 2].tolist())
    if rare:
        strat = np.array([-1 if s in rare else s for s in strat])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    sub_tr, sub_va = next(sss.split(strat, strat))
    tr_idx, va_idx = tr[sub_tr], tr[sub_va]

    train_domains = np.unique(domain[tr_idx])
    test_domains = np.unique(domain[te])
    return dict(
        X_train=X[tr_idx], y_train=y[tr_idx], d_train=domain[tr_idx],
        X_val=X[va_idx], y_val=y[va_idx], d_val=domain[va_idx],
        X_test=X[te], y_test=y[te], d_test=domain[te],
        n_channels=X.shape[1], n_samples=X.shape[2], n_classes=n_classes,
        n_domains=int(len(np.unique(domain))),
        train_domains=train_domains, test_domains=test_domains,
        unseen_domains=np.setdiff1d(test_domains, train_domains),
        val_is_test=False, prep="paper", z_scale=False,
        # Row indices into the full arrays, so a caller can attach any other
        # per-trial variable to the same split.
        idx_train=tr_idx, idx_val=va_idx, idx_test=te,
    )


def build_cells(strategy, subject, session, subjects):
    """The two evaluation strategies.

    inter-session   domain = session; one model per subject;
                    leave-one-session-out.
    inter-subject   domain = session + subject * 1000; groups are subjects, one
                    held out per fold.
    """
    cells = []
    if strategy == "inter-session":
        for s in subjects:
            m = subject == s
            for held in sorted(np.unique(session[m]).tolist()):
                cells.append(dict(name=f"sub{s}_sess{held}", seed=42 + held,
                                  train=m & (session != held),
                                  test=m & (session == held), domain=session))
    elif strategy == "inter-subject":
        dom = session + subject * 1000
        _, dom = np.unique(dom, return_inverse=True)
        for s in subjects:
            m = subject == s
            cells.append(dict(name=f"sub{s}", seed=42 + int(dom[m][0]),
                              train=~m, test=m, domain=dom))
    else:
        raise ValueError(strategy)
    return cells


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["tsmnet", "heegnet"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--subjects", type=int, nargs="*", default=None,
                    help="Restrict the check to a few subjects; the result is "
                         "marked partial.")
    ap.add_argument("--cache", default=os.path.expanduser("~/mne_data/baseline_anchor"))
    ap.add_argument("--out", default="runs/baseline_anchors")
    args = ap.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    from baselines.manifold.train import run_tsmnet, run_heegnet, fix_seed

    strategy = "inter-session" if args.model == "tsmnet" else "inter-subject"
    target = TARGETS[(args.model, strategy)]

    X, y, subject, session = load_bnci("BNCI2014001", args.cache)
    print(f"BNCI2014001: X={X.shape} classes={len(np.unique(y))} "
          f"subjects={len(np.unique(subject))} sessions={sorted(set(session.tolist()))}",
          flush=True)

    subjects = args.subjects or sorted(np.unique(subject).tolist())
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")

    cells = build_cells(strategy, subject, session, subjects)

    os.makedirs(args.out, exist_ok=True)
    runner = run_tsmnet if args.model == "tsmnet" else run_heegnet
    records, t0 = [], time.time()

    for cell in cells:
        fix_seed(cell["seed"])
        data = make_cell(X, y, cell["domain"], cell["train"], cell["test"],
                         seed=cell["seed"], n_classes=int(len(np.unique(y))))
        run_args = _Args(arm="default", epochs=args.epochs, swd_weight=None,
                         setting=strategy, model=args.model, hp=[])
        rows, tracker, evaluate, best_val_state, best_val_epoch, meta = \
            runner(run_args, data, device)
        acc, bacc = evaluate(best_val_state)
        ma_best_val_epoch, ma_best_val_state = tracker.resolve()
        ma_best_val_acc, ma_best_val_bacc = evaluate(ma_best_val_state)
        records.append(dict(cell=cell["name"], test_bacc=bacc, test_acc=acc,
                            ma_best_val_bacc=ma_best_val_bacc, best_val_epoch=best_val_epoch,
                            n_test=int(cell["test"].sum())))
        run = np.mean([r["test_bacc"] for r in records])
        print(f"  {cell['name']:>12}  bacc={bacc:.4f}  (ma_best_val {ma_best_val_bacc:.4f})  "
              f"running mean={run:.4f}  target={target:.4f}", flush=True)

    mean = float(np.mean([r["test_bacc"] for r in records]))
    std = float(np.std([r["test_bacc"] for r in records], ddof=1)) if len(records) > 1 else 0.0
    partial = args.subjects is not None or args.epochs is not None
    summary = dict(model=args.model, strategy=strategy, target=target,
                   mean_test_bacc=mean, std_test_bacc=std, n_cells=len(records),
                   delta=mean - target, partial=partial,
                   epochs_override=args.epochs, subjects=args.subjects,
                   wall_seconds=round(time.time() - t0, 1), cells=records)
    path = os.path.join(args.out, f"anchor_{args.model}_{strategy}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    verdict = "PARTIAL" if partial else (
        "MATCH" if abs(mean - target) <= 0.03 else "MISMATCH")
    print(f"\n{args.model} {strategy}: {mean:.4f} +- {std:.4f} over {len(records)} "
          f"cells | target {target:.4f} | delta {mean - target:+.4f} -> {verdict}")
    print(f"written to {path}")


if __name__ == "__main__":
    main()
