import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from latte.training.cli import add_common_args, parse_args, load_config, prepare_model, init_wandb, flag  
from latte.training.run_dir import make_run_dir  
from latte.training.subject_mask import SubjectMaskedLoader  
from latte.training.subject_routing import (find_nearest_training_subject,  
                                            find_nearest_training_subject_spd)
from latte.training.trainer import trainNetworkMultiple  
from latte.training.utils import build_datamodule, build_model, fix_seed, is_hyperbolic  

SUBJECTS = {"bcic2a": 9, "bcic2b": 9, "bcicha": 16}


def apply_input_alignment(dm, ea_all_trials, fallback="identity"):
    from torch.utils.data import DataLoader, TensorDataset
    from latte.data.euclidean_align import align_splits, align_splits_ea_all_trials

    names, datasets = [], []
    for name in ("train", "val", "test"):
        ds = getattr(dm, f"{name}_dataset", None)
        # val falls back to test on datasets without a validation split; align once
        if ds is None or any(ds is seen for seen in datasets):
            continue
        names.append(name)
        datasets.append(ds)

    def materialise(ds):
        if hasattr(ds, "tensors") and len(ds.tensors) == 3:
            return ds.tensors
        chunks = list(DataLoader(ds, batch_size=512, shuffle=False,
                                 num_workers=min(8, os.cpu_count() or 1)))
        return tuple(torch.cat([c[i] for c in chunks]) for i in range(3))

    materialised = [materialise(ds) for ds in datasets]
    packed = [(X.numpy(), s.numpy().reshape(-1)) for X, s, _ in materialised]
    if ea_all_trials:
        aligned, info = align_splits_ea_all_trials(packed)
    else:
        (X_train, s_train), *others = packed
        X_train, rest, info = align_splits(X_train, s_train, others, fallback=fallback)
        aligned = [X_train, *rest]
    for name, (_, s, y), Xa in zip(names, materialised, aligned):
        setattr(dm, f"{name}_dataset", TensorDataset(torch.from_numpy(Xa).float(), s, y))
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.set_defaults(head_center="batch")
    g = ap.add_argument_group("LOSO")
    g.add_argument("--folds", default="all", help="comma-separated held-out subjects, or 'all'")
    g.add_argument("--subject_dropout_p", type=float, default=0.2,
                   help="probability of routing a training sample through the shared slot 0")
    flag(g, "--mask_test_subject", True, "route every test sample of the held-out subject through slot 0")
    flag(g, "--route_to_nearest_subject", False, "route the held-out subject through the closest training subject")
    g.add_argument("--route_metric", default="embedding", choices=["embedding", "spd_cov"])
    g.add_argument("--route_recompute_every", type=int, default=1, help="[embedding] epochs between updates")
    g.add_argument("--route_max_samples", type=int, default=512)
    flag(g, "--input_align", False, "Euclidean Alignment fitted on training trials only")
    flag(g, "--input_align_ea_all_trials", False,
         "Euclidean Alignment fitted per subject on all trials")
    g.add_argument("--input_align_fallback", default="identity", choices=["identity", "grand"])
    args = parse_args(ap)
    args["setting"] = "loso"

    if args["mask_test_subject"] and args["route_to_nearest_subject"]:
        raise ValueError("--mask_test_subject and --route_to_nearest_subject both rewrite the "
                         "test subject ids; choose one")

    cfg = load_config(args)
    all_sids = list(range(1, SUBJECTS[args["dataset"]] + 1))
    folds = all_sids if args["folds"] == "all" else [int(s) for s in args["folds"].split(",")]

    run_name = args["run_name"] or f"{args['model']}_{args['dataset']}_loso"
    top_dir, _ = make_run_dir(base_dir=args["runs_dir"], run_name=run_name, label=args["label"], params=args)
    print(f"Top-level run directory: {top_dir}")

    for heldout in folds:
        print(f"\n=================== LOSO fold: held-out subject {heldout} ===================\n")
        fix_seed(args["seed"])
        dm, meta = build_datamodule(cfg, heldout, args["bs"], interaug=args["interaug"],
                                    z_scale=cfg.get("z_scale", True), loso=True)
        if args["input_align"] or args["input_align_ea_all_trials"]:
            info = apply_input_alignment(dm, ea_all_trials=args["input_align_ea_all_trials"],
                                         fallback=args["input_align_fallback"])
            print(f"[loso EA] {info.get('mode')} alignment, {info.get('n_subjects_fitted')} subjects fitted")

        raw_train = trainloader = dm.train_dataloader()
        validloader = dm.val_dataloader()
        raw_test = testloader = dm.test_dataloader()
        if args["subject_dropout_p"] > 0:
            trainloader = SubjectMaskedLoader(trainloader, mask_prob=args["subject_dropout_p"])
        if args["mask_test_subject"] or args["route_to_nearest_subject"]:
            # for routing, the slot is set below / by the per-epoch callback
            testloader = SubjectMaskedLoader(testloader, force_mask=True, mask_value=0)

        for repeat in range(1, args["repeat"] + 1):
            fold_args = {**args, "sub": heldout, "seed": args["seed"] + repeat - 1}
            fix_seed(fold_args["seed"])
            net = build_model(meta["classes"], **fold_args)
            prepare_model(net, trainloader, fold_args)

            run_dir = os.path.join(top_dir, f"s{heldout}" + (f"_r{repeat}" if args["repeat"] > 1 else ""))
            checkpoints_dir = os.path.join(run_dir, "checkpoints")
            os.makedirs(checkpoints_dir, exist_ok=True)
            with open(os.path.join(run_dir, "params.json"), "w") as f:
                json.dump(fold_args, f, indent=2, default=str)

            callback = None
            if args["route_to_nearest_subject"]:
                if args["route_metric"] == "spd_cov":
                    # model-free: computed once per fold from the raw signals
                    nearest, dists = find_nearest_training_subject_spd(
                        raw_train, raw_test, heldout_id=heldout, device=args["device"],
                        max_trials_per_subject=args["route_max_samples"])
                    testloader.set_mask_value(nearest)
                    print(f"[loso route spd_cov] s{heldout} -> s{nearest}")
                else:
                    every = max(1, args["route_recompute_every"])

                    def callback(net, epoch, testloader, **_):
                        if epoch % every:
                            return
                        nearest, _ = find_nearest_training_subject(
                            net, raw_train, raw_test, heldout_id=heldout, device=args["device"],
                            max_samples_per_subject=args["route_max_samples"])
                        testloader.set_mask_value(nearest)
                        print(f"[loso route embedding] epoch {epoch}: s{heldout} -> s{nearest}")

            wandb_run = init_wandb(args, f"{run_name}_s{heldout}", run_dir, group=run_name)
            try:
                trainNetworkMultiple(net, trainloader, validloader, testloader,
                                     hyperbolic=is_hyperbolic(args["model"]), num_classes=meta["classes"],
                                     epoch_end_callback=callback, run_dir=run_dir,
                                     checkpoints_dir=checkpoints_dir,
                                     **{k: v for k, v in fold_args.items()
                                        if k not in ("run_dir", "checkpoints_dir")})
            finally:
                if wandb_run is not None:
                    wandb_run.finish()
            del net
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
