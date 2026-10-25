import argparse
import copy
import csv
import json
import os
import random
import time
import warnings

import numpy as np
import torch

from baselines.manifold.data import load_setting, summarise
from baselines.manifold.selection import SmoothedBest


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scores(y_true, y_pred, n_classes):
    """Accuracy and balanced accuracy."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float((y_true == y_pred).mean())
    per_class = []
    for c in range(n_classes):
        m = y_true == c
        if m.sum():
            per_class.append(float((y_pred[m] == c).mean()))
    return acc, float(np.mean(per_class)) if per_class else float("nan")


def cpu_state(model):
    return {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}


def hp_overrides(args, defaults):
    """`--hp key=value` overrides, validated against the method's default hyperparameters.

    Unknown keys are an error. Values are coerced to the type of the default.
    """
    over = {}
    for item in args.hp:
        if "=" not in item:
            raise SystemExit(f"--hp expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        if k not in defaults:
            raise SystemExit(f"--hp {k}: not a knob of {args.model}. "
                             f"Known: {', '.join(sorted(defaults))}")
        cur = defaults[k]
        try:
            over[k] = (v.lower() in ("1", "true", "yes") if isinstance(cur, bool)
                       else type(cur)(v))
        except (TypeError, ValueError):
            raise SystemExit(f"--hp {k}={v!r}: expected {type(cur).__name__}")
    return over


# ----------------------------------------------------------------------------
# TSMNet
# ----------------------------------------------------------------------------
def run_tsmnet(args, data, device):
    from baselines.manifold import tsmnet_arm as T

    dtype = T.DTYPE
    ds_train = T.build_dataset(data["X_train"], data["y_train"], data["d_train"], dtype)
    ds_val = T.build_dataset(data["X_val"], data["y_val"], data["d_val"], dtype)
    ds_test = T.build_dataset(data["X_test"], data["y_test"], data["d_test"], dtype)

    base_arm = "default" if args.arm == "sfuda" else args.arm
    over = hp_overrides(args, T.DEFAULTS)
    if args.epochs:
        over["epochs"] = args.epochs
    model, cfg = T.build(data, base_arm, device, cfg=over or None)
    opt = T.make_optimizer(model, cfg["lr"], cfg["weight_decay"])
    loader, bs, dpb = T.train_loader(ds_train, np.arange(len(data["y_train"])),
                                     cfg, len(data["train_domains"]))
    sched = T.make_scheduler(base_arm, cfg["epochs"], cfg["batch_size_train"],
                             cfg["domains_per_batch"])
    sched.on_train_begin(T._NetShim(model))

    tracker = SmoothedBest()
    best_val_loss, best_val_state, best_val_epoch = float("inf"), None, -1
    rows = []
    n_cls = data["n_classes"]

    for epoch in range(cfg["epochs"]):
        sched.on_epoch_begin(None)
        model.train()
        tr_loss = tr_n = 0
        for inp, y in loader:
            out = model(inp["x"], inp["d"])
            loss = model.calculate_objective(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * y.shape[0]
            tr_n += y.shape[0]

        yhat, logits = T.predict(model, ds_val)
        val_acc, val_bacc = scores(data["y_val"], yhat, n_cls)
        val_loss = float(torch.nn.functional.cross_entropy(
            torch.from_numpy(logits), torch.from_numpy(data["y_val"]).long()).item())

        rows.append(dict(epoch=epoch, train_loss=tr_loss / max(tr_n, 1),
                         val_loss=val_loss, val_accs=val_acc, val_bacc=val_bacc))
        tracker.update(epoch, val_bacc, lambda: cpu_state(model))
        if val_loss < best_val_loss:
            best_val_loss, best_val_epoch = val_loss, epoch
            best_val_state = cpu_state(model)

        if epoch % 10 == 0 or epoch == cfg["epochs"] - 1:
            print(f"  ep{epoch:03d} train_loss {tr_loss / max(tr_n, 1):.4f} "
                  f"val_bacc {val_bacc:.4f} val_loss {val_loss:.4f}", flush=True)

    def evaluate(state):
        model.load_state_dict(state)
        if base_arm == "default":
            if args.arm == "sfuda" or data["unseen_domains"].size:
                T.domain_adapt(model, ds_test, device)
        else:
            T.source_finetune(model, ds_train, device)
        yh, _ = T.predict(model, ds_test)
        return scores(data["y_test"], yh, n_cls)

    return rows, tracker, evaluate, best_val_state, best_val_epoch, dict(
        batch_size=bs, domains_per_batch=dpb, epochs=cfg["epochs"],
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        n_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        bnorm="spddsbn" if base_arm == "default" else "spdbn",
        target_refit=bool(args.arm == "sfuda" or data["unseen_domains"].size),
    )


# ----------------------------------------------------------------------------
# HEEGNet
# ----------------------------------------------------------------------------
def run_heegnet(args, data, device):
    from baselines.manifold import heegnet_arm as H

    dtype = H.DTYPE
    cfg = {**H.DEFAULTS}
    if args.epochs:
        cfg["epochs"] = args.epochs
        cfg["min_epochs"] = min(cfg["min_epochs"], max(1, args.epochs - 10))
    if args.swd_weight is not None:
        cfg["swd_weight"] = args.swd_weight
    # Applied last, so `--hp swd_weight=` wins over `--swd_weight` if both appear.
    cfg.update(hp_overrides(args, H.DEFAULTS))

    base_arm = "default" if args.arm == "sfuda" else args.arm
    X = dict(train=data["X_train"], val=data["X_val"], test=data["X_test"])
    if base_arm == "default":
        allX = np.concatenate([X["train"], X["val"], X["test"]])
        alld = np.concatenate([data["d_train"], data["d_val"], data["d_test"]])
        allX = H.euclidean_align(allX, alld, np.unique(alld))
        n1, n2 = len(X["train"]), len(X["train"]) + len(X["val"])
        X = dict(train=allX[:n1], val=allX[n1:n2], test=allX[n2:])

    ds_train = H.build_dataset(X["train"], data["y_train"], data["d_train"], dtype)
    ds_val = H.build_dataset(X["val"], data["y_val"], data["d_val"], dtype)
    ds_test = H.build_dataset(X["test"], data["y_test"], data["d_test"], dtype)

    model, cfg = H.build(data, base_arm, device, cfg=cfg, dtype=dtype)
    loader, dpb = H.train_loader(ds_train, cfg, len(data["train_domains"]))
    val_loader = H.eval_loader(ds_val)
    test_loader = H.eval_loader(ds_test)

    snap = H.SnapshotEval()
    trainer, es = H.make_trainer(cfg, device, dtype, extra_callbacks=[snap])
    trainer.fit(model, train_dataloader=loader, val_dataloader=val_loader)
    best_val_state = cpu_state(model)
    best_val_epoch = int(es.best_epoch)

    tracker = SmoothedBest()
    rows = []
    for i, (sc, ls, st) in enumerate(zip(snap.scores, snap.losses, snap.states)):
        rows.append(dict(epoch=i, train_loss=float("nan"), val_loss=ls,
                         val_accs=sc, val_bacc=sc))
        tracker.update(i, sc, lambda st=st: st)

    def evaluate(state):
        model.load_state_dict(state)
        if base_arm == "default" and (args.arm == "sfuda"
                                        or data["unseen_domains"].size):
            H.domain_adapt(model, ds_test, device, dtype)
        res = trainer.test(model, test_loader)
        yt, yh = [], []
        model.eval()
        with torch.no_grad():
            for feats, y in test_loader:
                feats["inputs"] = feats["inputs"].to(dtype=dtype, device=device)
                pred, _ = model(**feats)
                yt.append(y.numpy())
                yh.append(pred.argmax(1).cpu().numpy())
        acc, bacc = scores(np.concatenate(yt), np.concatenate(yh), data["n_classes"])
        assert abs(bacc - res["score"]) < 1e-6, (bacc, res["score"])
        return acc, bacc

    return rows, tracker, evaluate, best_val_state, best_val_epoch, dict(
        batch_size=cfg["batch_size_train"], domains_per_batch=dpb,
        epochs=len(rows), max_epochs=cfg["epochs"], min_epochs=cfg["min_epochs"],
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        swd_weight=cfg["swd_weight"], input_align=(base_arm == "default"),
        n_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        bnorm="dsmdbn" if base_arm == "default" else "shared",
        target_refit=bool(args.arm == "sfuda" or data["unseen_domains"].size),
    )


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["tsmnet", "heegnet"])
    ap.add_argument("--dataset", required=True,
                    choices=["bcic2a", "bcic2b", "bcicha"])
    ap.add_argument("--setting", required=True, choices=["ss", "sc", "loso"])
    ap.add_argument("--arm", default="default",
                    choices=["default", "inductive", "sfuda"],
                    help="'inductive' collapses domain-specific BN to a single shared domain; "
                         "LOSO only. 'sfuda' is the default model with the "
                         "label-free refit on the test trials forced on even where "
                         "the test domain was seen in training.")
    ap.add_argument("--heldout", type=int, default=1,
                    help="SS subject, or LOSO held-out subject. Ignored for SC.")
    ap.add_argument("--seed", type=int, default=3330)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override the default epoch budget. The momentum schedules "
                         "are defined relative to it.")
    ap.add_argument("--swd_weight", type=float, default=None,
                    help="HEEGNet SWD weight. Default 0.01.")
    ap.add_argument("--hp", action="append", default=[], metavar="KEY=VALUE",
                    help="Override a default hyperparameter, repeatable. Keys "
                         "are validated against the method's DEFAULTS dict "
                         "(baselines/manifold/{tsmnet,heegnet}_arm.py); an unknown "
                         "key is an error.")
    ap.add_argument("--skip_test", action="store_true",
                    help="Train and write the per-epoch validation CSV, but never "
                         "evaluate on test.")
    ap.add_argument("--config", default="configs/datasets.yaml")
    ap.add_argument("--data_path", default=None,
                    help="directory of the BCI Challenge .mat files")
    ap.add_argument("--out", default="runs/baselines")
    ap.add_argument("--tag", default="bl")
    ap.add_argument("--run_name", default=None)
    args = ap.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    if args.setting != "loso" and args.arm == "inductive":
        raise SystemExit(
            "--arm inductive is only defined for LOSO.")

    fix_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data = load_setting(args.dataset, args.setting, args.heldout,
                        config=args.config, data_path=args.data_path)

    run = args.run_name or (f"{args.tag}_{args.model}_{args.setting}_{args.arm}_"
                            f"{args.dataset}_s{args.heldout}_seed{args.seed}")
    os.makedirs(args.out, exist_ok=True)
    print(f"[{run}] {summarise(data)}", flush=True)
    if args.setting == "ss":
        print("  note: one domain -> domain-specific BN is inert.", flush=True)
    if args.arm == "sfuda" and args.setting != "loso":
        print("  note: sfuda refits the BN statistics, label-free, on the test "
              "domain's own trials, which was seen in training.", flush=True)
    if args.setting == "loso" and args.arm == "default":
        print("  note: BN statistics refitted, label-free, on the held-out "
              "subject's own trials.", flush=True)

    t0 = time.time()
    runner = run_tsmnet if args.model == "tsmnet" else run_heegnet
    rows, tracker, evaluate, best_val_state, best_val_epoch, meta = runner(args, data, device)

    ma_best_val_epoch, ma_best_val_state = tracker.resolve()
    if args.skip_test:
        nan = float("nan")
        ma_best_val_acc = ma_best_val_bacc = best_val_acc = best_val_bacc = nan
    else:
        ma_best_val_acc, ma_best_val_bacc = evaluate(copy.deepcopy(ma_best_val_state))
        best_val_acc, best_val_bacc = evaluate(copy.deepcopy(best_val_state))

    csv_path = os.path.join(args.out, run + ".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "run": run,
        "args": vars(args),
        "meta": meta,
        "data": {k: (int(v) if isinstance(v, (int, np.integer)) else v)
                 for k, v in dict(
                     n_train=len(data["y_train"]), n_val=len(data["y_val"]),
                     n_test=len(data["y_test"]), n_classes=data["n_classes"],
                     n_channels=data["n_channels"], n_samples=data["n_samples"],
                     n_train_domains=len(data["train_domains"]),
                     n_unseen_domains=len(data["unseen_domains"]),
                     val_is_test=data["val_is_test"],
                     z_scale=data["z_scale"]).items()},
        "epochs_run": len(rows),
        "wall_seconds": round(time.time() - t0, 1),
        "hp": dict(args.hp and [i.split("=", 1) for i in args.hp] or []),
        "skip_test": bool(args.skip_test),
        "best_val": {"selected_epoch": best_val_epoch, "test_acc": best_val_acc,
                       "test_bacc": best_val_bacc},
        "ma_best_val": {"selected_epoch": ma_best_val_epoch, "test_acc": ma_best_val_acc,
                       "test_bacc": ma_best_val_bacc},
        "unlabeled_target": bool(args.arm == "sfuda"
                             or (args.arm == "default"
                                 and data["unseen_domains"].size > 0)),
    }
    with open(os.path.join(args.out, run + ".json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[{run}] best_val test_bacc {best_val_bacc:.4f} (ep {best_val_epoch}) | "
          f"ma_best_val test_bacc {ma_best_val_bacc:.4f} (ep {ma_best_val_epoch}) | "
          f"{summary['wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
