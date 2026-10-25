import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch 

from latte.training.cli import add_common_args, parse_args, load_config, init_wandb  # noqa: E402
from latte.training.run_dir import make_run_dir  # noqa: E402
from latte.training.trainer import trainNetworkMultiple  # noqa: E402
from latte.training.utils import fix_seed, get_dataloaders_from_config, is_hyperbolic  # noqa: E402

N_SUBJECTS = {"bcic2a": 9, "bcic2b": 9, "bcicha": 16}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--checkpoint", required=True, help="full-model checkpoint from scripts/train.py --sub all")
    ap.add_argument("--subjects", default="all", help="comma-separated subject ids, or 'all'")
    ap.add_argument("--learn_predecoder_lora", type=lambda v: v.lower() in ("1", "true", "t", "yes"),
                    default=True, help="train the head adapter during fine-tuning")
    args = parse_args(ap)
    args["setting"] = "sc_ft"
    cfg = load_config(args)

    subjects = (range(1, N_SUBJECTS[args["dataset"]] + 1) if args["subjects"] == "all"
                else [int(s) for s in args["subjects"].split(",")])
    run_name = args["run_name"] or f"finetune_{args['dataset']}"
    top_dir, _ = make_run_dir(base_dir=args["runs_dir"], run_name=run_name, label=args["label"], params=args)
    print(f"Run directory: {top_dir}")

    for sub in subjects:
        trainloader, validloader, testloader, _, num_classes, _ = get_dataloaders_from_config(
            cfg, subject=sub, batch_size=args["bs"], interaug=args["interaug"],
            z_scale=cfg.get("z_scale", True), include_subject_ids=True,
            val_split=args["val_split"], cross_subject_normalize=True)

        for repeat in range(1, args["repeat"] + 1):
            fix_seed(args["seed"] + repeat - 1)
            net = torch.load(args["checkpoint"], map_location="cpu", weights_only=False).to(args["device"])
            net.set_trainable(learn_decoder=args["learn_decoder"],
                              learn_predecoder=args["learn_predecoder"],
                              learn_lora=args["learn_lora"])
            if not args["learn_predecoder_lora"]:
                for p in net.pre_decoder.l2.parameters():
                    p.requires_grad = False

            run_dir = os.path.join(top_dir, f"sub{sub}" + (f"_r{repeat}" if args["repeat"] > 1 else ""))
            checkpoints_dir = os.path.join(run_dir, "checkpoints")
            os.makedirs(checkpoints_dir, exist_ok=True)
            fold_args = {**args, "sub": sub, "seed": args["seed"] + repeat - 1}
            with open(os.path.join(run_dir, "params.json"), "w") as f:
                import json
                json.dump(fold_args, f, indent=2, default=str)

            wandb_run = init_wandb(args, f"{run_name}_sub{sub}_r{repeat}", run_dir, group=run_name)
            try:
                trainNetworkMultiple(net, trainloader, validloader, testloader,
                                     hyperbolic=is_hyperbolic(args["model"]), num_classes=num_classes,
                                     run_dir=run_dir, checkpoints_dir=checkpoints_dir,
                                     **{k: v for k, v in fold_args.items()
                                        if k not in ("run_dir", "checkpoints_dir")})
            finally:
                if wandb_run is not None:
                    wandb_run.finish()
            del net
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
