import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from latte.training.cli import add_common_args, parse_args, load_config, prepare_model, init_wandb  # noqa: E402
from latte.training.run_dir import make_run_dir  # noqa: E402
from latte.training.trainer import trainNetworkMultiple  # noqa: E402
from latte.training.utils import build_model, fix_seed, get_dataloaders_from_config, is_hyperbolic  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--sub", default="all", help="'all' for SC, or a subject id for SS")
    args = parse_args(ap)
    args["setting"] = "sc" if args["sub"] == "all" else "ss"

    cfg = load_config(args)
    trainloader, validloader, testloader, in_channels, num_classes, _ = get_dataloaders_from_config(
        cfg, subject=args["sub"], batch_size=args["bs"], interaug=args["interaug"],
        z_scale=cfg.get("z_scale", True), include_subject_ids=True, val_split=args["val_split"])

    for repeat in range(1, args["repeat"] + 1):
        fix_seed(args["seed"] + repeat - 1)
        net = build_model(num_classes, **args)
        prepare_model(net, trainloader, args)

        run_name = args["run_name"] or f"{args['model']}_{args['dataset']}_sub{args['sub']}"
        run_dir, checkpoints_dir = make_run_dir(base_dir=args["runs_dir"], run_name=run_name,
                                                label=args["label"], params=args)
        print(f"Run directory: {run_dir}")
        wandb_run = init_wandb(args, f"{run_name}_r{repeat}", run_dir)
        try:
            trainNetworkMultiple(net, trainloader, validloader, testloader,
                                 hyperbolic=is_hyperbolic(args["model"]), num_classes=num_classes,
                                 run_dir=run_dir, checkpoints_dir=checkpoints_dir,
                                 **{k: v for k, v in args.items() if k not in ("run_dir", "checkpoints_dir")})
        finally:
            if wandb_run is not None:
                wandb_run.finish()
        del net
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
