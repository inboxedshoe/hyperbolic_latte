import argparse
import copy
import csv
import json
import os
import time
import warnings

import numpy as np
import torch

from baselines.manifold.anchor import load_bnci, make_cell, build_cells, REFERENCE_TARGETS
from baselines.manifold.train import scores, cpu_state, fix_seed
from baselines.manifold.selection import SmoothedBest


def subject_slots(data, subject):
    """Raw subject ids per split, for LAtte's slot-indexed LoRA banks.

    `make_cell` returns the row indices it used, so the subject id for each
    trial is a direct lookup.

    The banks have `num_subjects + 1` slots indexed by the raw id, so ids pass
    through unmapped. Under inter-subject
    the held-out subject's slot is never trained and stays at its
    initialisation.
    """
    return dict(
        s_train=subject[data["idx_train"]].astype(np.int64),
        s_val=subject[data["idx_val"]].astype(np.int64),
        s_test=subject[data["idx_test"]].astype(np.int64),
    )


def parse_overrides(pairs, base):
    """`--set key=value` overrides for the model config, typed off `base`.

    Unknown keys are an error.
    """
    import ast
    out = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        if k not in base:
            raise SystemExit(f"--set: unknown config key {k!r}. "
                             f"Known keys: {sorted(base)}")
        cur = base[k]
        if isinstance(cur, bool):                      # before int: bool is an int
            out[k] = v.lower() in ("1", "true", "yes", "t")
        elif isinstance(cur, int):
            out[k] = int(v)
        elif isinstance(cur, float):
            out[k] = float(v)
        else:
            try:
                out[k] = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                out[k] = v
    return out


def run_latte(args, data, device):
    """LAtte, with its own recipe (see latte_arm.py)."""
    from baselines.manifold import latte_arm as L

    cfg = {**L.DEFAULTS}
    cfg.update(parse_overrides(getattr(args, "set", None), L.DEFAULTS))
    if args.epochs:
        cfg["epochs"] = args.epochs

    net, cfg, n_adaptive = L.build(data, args.row, device, cfg=cfg,
                                   dataset_key=args.dataset_key)
    tr, va, te, sampler = L.loaders(data, cfg)

    if args.subject_dropout_p > 0 or args.mask_test_subject:
        from latte.training.subject_mask import SubjectMaskedLoader
        if args.subject_dropout_p > 0:
            tr = SubjectMaskedLoader(tr, mask_prob=args.subject_dropout_p)
        if args.mask_test_subject:
            te = SubjectMaskedLoader(te, force_mask=True)
        print(f"  slot-0 routing: train dropout p={args.subject_dropout_p} "
              f"test forced to slot 0={args.mask_test_subject}", flush=True)
    opt, sched = L.make_optimizer(net, cfg, cfg["epochs"])
    CE = torch.nn.CrossEntropyLoss()
    print(f"  LAtte row={args.row} adaptiveBN={n_adaptive} "
          f"batches={len(tr)} bs={sampler.batch_size} "
          f"({sampler.domains_per_batch}dom x {sampler.obs_per_domain}obs)",
          flush=True)

    tracker = SmoothedBest()
    best_val_loss, best_val_state, best_val_epoch = float("inf"), None, -1
    rows = []
    L.bn_epoch(net, 0, cfg["epochs"])

    for epoch in range(cfg["epochs"]):
        L.bn_epoch(net, epoch + 1, cfg["epochs"])   # schedule is 1-indexed
        sampler.set_epoch(epoch)                    # reshuffle domains per epoch
        tr_loss = L.epoch_step(net, tr, opt, sched, CE, device, cfg,
                               data["n_classes"])
        L.bn_finalize(net)                          # refresh the pooled fallback

        yt, yh = L.predict(net, va, device)
        val_acc, val_bacc = scores(yt, yh, data["n_classes"])
        val_loss = _val_loss(net, va, CE, device)

        rows.append(dict(epoch=epoch, train_loss=tr_loss, val_loss=val_loss,
                         val_accs=val_acc, val_bacc=val_bacc))
        tracker.update(epoch, val_bacc, lambda: cpu_state(net))
        if val_loss < best_val_loss:
            best_val_loss, best_val_epoch = val_loss, epoch
            best_val_state = cpu_state(net)
        if epoch % 20 == 0 or epoch == cfg["epochs"] - 1:
            print(f"  ep{epoch:03d} train_loss {tr_loss:.4f} "
                  f"val_bacc {val_bacc:.4f} val_loss {val_loss:.4f}", flush=True)

    def evaluate(state):
        net.load_state_dict(state)
        L.bn_finalize(net)
        if args.bn_refit:
            # Scoped to this call: `state` carries buffers, so the other
            # selection rule re-enters with the trained statistics.
            L.bn_refit(net, te, device)
        yt, yh = L.predict(net, te, device)
        return scores(yt, yh, data["n_classes"])

    return rows, tracker, evaluate, best_val_state, best_val_epoch, dict(
        row=args.row, epochs=cfg["epochs"], lr=cfg["lr"],
        weight_decay=cfg["weight_decay"], batch_size=sampler.batch_size,
        domains_per_batch=sampler.domains_per_batch,
        obs_per_domain=sampler.obs_per_domain,
        sub_rank=cfg.get("sub_rank"), decoder_rank=cfg.get("decoder_rank"),
        proc_lora_lr=cfg.get("proc_lora_lr"),
        bn_refit=bool(args.bn_refit),
        subject_dropout_p=args.subject_dropout_p,
        mask_test_subject=bool(args.mask_test_subject),
        interaug=False, adaptive_bn_layers=n_adaptive,
        n_params=sum(p.numel() for p in net.parameters() if p.requires_grad),
        **{k: v for k, v in L.ROWS[args.row].items()},
    )


def run_gyroatt(args, data, device):
    """GyroAtt (see baselines/manifold/gyroatt_arm.py).

    GyroAtt has no domain-specific statistics and no test-time adaptation, so
    `--arm` has no effect on it.
    """
    from baselines.manifold import gyroatt_arm as G

    cfg = dict(G.DEFAULTS)
    if args.epochs:
        cfg["epochs"] = args.epochs
    if getattr(args, "lr", None):
        cfg["lr"] = args.lr

    net, cfg = G.build(data, device, cfg=cfg)
    tr, va, te = G.loaders(data, cfg, device)
    opt = torch.optim.RAdam([p for p in net.parameters() if p.requires_grad],
                            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    CE = torch.nn.CrossEntropyLoss()
    n_cls = data["n_classes"]
    print(f"  GyroAtt-{cfg['manifold']}-{cfg['spd_metric']} batches={len(tr)} "
          f"bs={cfg['batch_size']} lr={cfg['lr']} epochs={cfg['epochs']}", flush=True)

    tracker = SmoothedBest()
    best_val_loss, best_val_state, best_val_epoch = float("inf"), None, -1
    rows = []

    for epoch in range(cfg["epochs"]):
        net.train()
        tr_loss = tr_n = 0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            if x.dim() == 3:
                x = x.unsqueeze(1)
            loss = CE(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * y.shape[0]
            tr_n += y.shape[0]

        yt, logits = G.predict(net, va, device)
        val_acc, val_bacc = scores(yt, logits.argmax(1).numpy(), n_cls)
        val_loss = float(CE(logits, torch.from_numpy(yt).long()).item())

        rows.append(dict(epoch=epoch, train_loss=tr_loss / max(tr_n, 1),
                         val_loss=val_loss, val_accs=val_acc, val_bacc=val_bacc))
        tracker.update(epoch, val_bacc, lambda: cpu_state(net))
        if val_loss < best_val_loss:
            best_val_loss, best_val_epoch = val_loss, epoch
            best_val_state = cpu_state(net)
        if epoch % 20 == 0 or epoch == cfg["epochs"] - 1:
            print(f"  ep{epoch:03d} train_loss {tr_loss / max(tr_n, 1):.4f} "
                  f"val_bacc {val_bacc:.4f} val_loss {val_loss:.4f}", flush=True)

    def evaluate(state):
        net.load_state_dict(state)
        yt, logits = G.predict(net, te, device)
        return scores(yt, logits.argmax(1).numpy(), n_cls)

    return rows, tracker, evaluate, best_val_state, best_val_epoch, dict(
        epochs=cfg["epochs"], lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        batch_size=cfg["batch_size"], in_size=cfg["in_size"],
        patches=cfg["patches"], spd_metric=cfg["spd_metric"],
        manifold=cfg["manifold"],
        n_params=sum(p.numel() for p in net.parameters() if p.requires_grad),
    )


@torch.no_grad()
def _val_loss(net, loader, CE, device):
    net.eval()
    tot = n = 0.0
    for x, s, y in loader:
        x, s, y = x.to(device), s.to(device), y.to(device)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        out = net(x, s)
        out = out[0] if isinstance(out, (list, tuple)) else out
        tot += float(CE(out, y).item()) * y.shape[0]
        n += y.shape[0]
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["tsmnet", "heegnet", "latte", "gyroatt"])
    ap.add_argument("--row", default="base", choices=["base", "adaptmom",
                                                      "domain", "domain_subj"],
                    help="LAtte only: which BN-knob setting.")
    ap.add_argument("--dataset", default="BNCI2014001",
                    choices=["BNCI2014001", "BNCI2015001"])
    ap.add_argument("--strategy", default="inter-subject",
                    choices=["inter-session", "inter-subject"])
    ap.add_argument("--cell", default=None,
                    help="Run a single cell, e.g. sub3 or sub3_sess1. Omit for all.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Overrides the per-fold seed. Use for multi-seed runs; "
                         "omit to use the per-fold seed.")
    ap.add_argument("--arm", default="default", choices=["default", "inductive"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--swd_weight", type=float, default=None)
    ap.add_argument("--subject_dropout_p", type=float, default=0.15,
                    help="LAtte only: probability of routing a training sample through "
                         "the shared slot 0. Set 0 to disable.")
    ap.add_argument("--mask_test_subject", type=lambda v: str(v).lower() in ("1", "true", "yes"),
                    default=True,
                    help="LAtte only: route the held-out subject's test trials through "
                         "the trained slot 0 instead of their never-trained slot.")
    ap.add_argument("--bn_refit", action="store_true",
                    help="LAtte only: re-estimate the BN statistics from the held-out "
                         "subject's unlabeled test trials before testing.")
    ap.add_argument("--zscore", action="store_true",
                    help="per-channel z-scoring fitted on the training pool.")
    ap.add_argument("--input_align", action="store_true",
                    help="per-domain Euclidean Alignment on the input, fitted on all "
                         "trials of each domain (the held-out subject's test trials "
                         "included).")
    ap.add_argument("--lr", type=float, default=None,
                    help="gyroatt only; defaults to 1e-3.")
    ap.add_argument("--hp", action="append", default=[], metavar="KEY=VALUE",
                    help="tsmnet/heegnet only: override one default "
                         "hyperparameter, repeatable. Validated against the "
                         "method's DEFAULTS dict; an unknown key is an "
                         "error. (LAtte's equivalent is --set.)")
    ap.add_argument("--set", action="append", metavar="KEY=VAL",
                    help="LAtte only: override one key of latte_arm.DEFAULTS, "
                         "repeatable (e.g. --set dropout=0 --set bias=true). "
                         "Value is typed off the existing entry; an unknown key "
                         "is an error.")
    ap.add_argument("--label", default=None,
                    help="suffix appended to the run tag.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache", default=os.path.expanduser("~/mne_data/baseline_anchor"))
    ap.add_argument("--out", default="runs/moabb_protocol")
    args = ap.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    from baselines.manifold.train import run_tsmnet, run_heegnet
    runner = {"tsmnet": run_tsmnet, "heegnet": run_heegnet,
              "latte": run_latte, "gyroatt": run_gyroatt}[args.model]

    args.dataset_key = {"BNCI2014001": "bcic2a",
                        "BNCI2015001": "bnci2015001"}[args.dataset]
    args.setting = args.strategy

    X, y, subject, session = load_bnci(args.dataset, args.cache)
    subjects = sorted(np.unique(subject).tolist())
    n_classes = int(len(np.unique(y)))
    print(f"{args.dataset}: X={X.shape} {n_classes}cl subjects={len(subjects)} "
          f"sessions={sorted(set(session.tolist()))}", flush=True)

    if args.input_align:
        try:
            from baselines.manifold.heegnet_arm import euclidean_align
        except ImportError as e:
            raise SystemExit(f"--input_align uses HEEGNet's euler_align: {e}")
        dom_all = session + subject * 1000
        _, dom_all = np.unique(dom_all, return_inverse=True)
        X = euclidean_align(X, dom_all, np.unique(dom_all))
        print(f"input_align: per-domain EA applied over {len(np.unique(dom_all))} domains",
              flush=True)

    cells = build_cells(args.strategy, subject, session, subjects)
    if args.cell:
        cells = [c for c in cells if c["name"] == args.cell]
        if not cells:
            raise SystemExit(f"no cell named {args.cell}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    tag = (f"{args.model}" + ("-z" if args.zscore else "")
           + ("-ea" if args.input_align else "") + ("-bnrefit" if args.bn_refit else "")
           + ("" if args.model == "gyroatt" else
              (f"-{args.row}" if args.model == "latte" else f"-{args.arm}")))
    if args.label:
        tag = f"{tag}-{args.label}"
    for cell in cells:
        seed = args.seed if args.seed is not None else cell["seed"]
        run = f"tp_{args.dataset}_{args.strategy}_{tag}_{cell['name']}_seed{seed}"
        jpath = os.path.join(args.out, run + ".json")
        if os.path.isfile(jpath):
            print(f"  skip (done): {run}", flush=True)
            continue

        fix_seed(seed)
        data = make_cell(X, y, cell["domain"], cell["train"], cell["test"],
                         seed=seed, n_classes=n_classes)

        if args.zscore:
            from latte.data.base import BaseDataModule
            data["X_train"], data["X_val"], data["X_test"] = \
                BaseDataModule._z_scale_tvt(data["X_train"], data["X_val"],
                                            data["X_test"])
            data["z_scale"] = True
            print("  z-scored inputs (fit on train, applied to val/test)", flush=True)
        data.update(subject_slots(data, subject))

        t0 = time.time()
        rows, tracker, evaluate, best_val_state, best_val_epoch, meta = \
            runner(args, data, device)
        ma_best_val_epoch, ma_best_val_state = tracker.resolve()
        ma_best_val_acc, ma_best_val_bacc = evaluate(copy.deepcopy(ma_best_val_state))
        best_val_acc, best_val_bacc = evaluate(copy.deepcopy(best_val_state))

        with open(os.path.join(args.out, run + ".csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

        summary = dict(
            run=run, model=args.model, dataset=args.dataset,
            strategy=args.strategy, cell=cell["name"], seed=seed,
            row=(args.row if args.model == "latte" else None),
            arm=(args.arm if args.model != "latte" else None),
            label=args.label,
            cfg_overrides=(parse_overrides(args.set, __import__(
                "baselines.manifold.latte_arm", fromlist=["DEFAULTS"]).DEFAULTS)
                if (args.set and args.model == "latte") else None),
            meta=meta, epochs_run=len(rows),
            wall_seconds=round(time.time() - t0, 1),
            n_train=int(len(data["y_train"])), n_val=int(len(data["y_val"])),
            n_test=int(len(data["y_test"])),
            n_train_domains=int(len(data["train_domains"])),
            n_unseen_domains=int(len(data["unseen_domains"])),
            best_val=dict(selected_epoch=best_val_epoch, test_acc=best_val_acc,
                            test_bacc=best_val_bacc),
            ma_best_val=dict(selected_epoch=ma_best_val_epoch, test_acc=ma_best_val_acc,
                            test_bacc=ma_best_val_bacc),
            reference_target=REFERENCE_TARGETS.get(
                (args.model, args.dataset, args.strategy)),
        )
        with open(jpath, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"  {cell['name']:>12}  best_val {best_val_bacc:.4f}  ma_best_val {ma_best_val_bacc:.4f}  "
              f"({summary['wall_seconds']}s)", flush=True)

    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
