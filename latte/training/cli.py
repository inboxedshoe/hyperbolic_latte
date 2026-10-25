import argparse
import os
import sys

import yaml

from latte.geometry.manifold import set_limits_mode
from latte.models.configs import DEFAULT_MODEL_CONFIGS
from latte.training.utils import MODELS


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def flag(ap, name, default, help):
    """Boolean option: `--name`, `--name true` and `--name false` all work."""
    ap.add_argument(name, type=str2bool, nargs="?", const=True, default=default, help=help)


def add_common_args(ap):
    g = ap.add_argument_group("run")
    g.add_argument("--cfg", default=None,
                   help="hyperparameter file, configs/<task>/<dataset>.cfg; flags given on the "
                        "command line override it")
    g.add_argument("--device", default="cuda:0")
    g.add_argument("--seed", type=int, default=3330)
    g.add_argument("--dataset", default="bcic2a", choices=["bcic2a", "bcic2b", "bcicha"])
    g.add_argument("--config", default="configs/datasets.yaml", help="dataset preprocessing yaml")
    g.add_argument("--data_path", default=None,
                   help="directory with the .mat files (bcicha); overrides the yaml")
    g.add_argument("--iterations", type=int, default=200, help="training epochs")
    g.add_argument("--repeat", type=int, default=1)
    g.add_argument("--run_name", default=None)
    g.add_argument("--label", default=None)
    g.add_argument("--tag", default="")
    g.add_argument("--runs_dir", default="./runs")
    flag(g, "--use_wandb", False, "log to Weights & Biases")
    g.add_argument("--wandb_project", default="latte")
    g.add_argument("--wandb_entity", default=None)
    g.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    g.add_argument("--wandb_group", default=None)

    g = ap.add_argument_group("model")
    g.add_argument("--model", default="TCFormerDeviationHeadLora", choices=MODELS)
    g.add_argument("--conv_type", default="base", choices=["base"])
    g.add_argument("--batch_type", default="gyro", choices=["gyro"])
    g.add_argument("--linear_type", default="plfc_scaled",
                   choices=["chen", "plfc", "plfc_scaled"],
                   help="Lorentz linear layer inside the convolutions and attention")
    g.add_argument("--pool_type", default="dirty",
                   choices=["dirty", "clean", "clean_centroid", "soft", "tangent"],
                   help="task-branch max pool. dirty: max over each space coordinate, then lift; "
                        "clean: the point farthest from the origin in each window; "
                        "clean_centroid: the point farthest from the window's centroid; "
                        "soft: window centroid weighted by softmax(radius / tau); "
                        "tangent: logmap0, Euclidean max pool, expmap0")
    g.add_argument("--pool_tau", type=float, default=1.0,
                   help="initial temperature of the soft pool, learned per layer")
    g.add_argument("--learnable_k", type=int, default=1, help="learn the curvature")
    flag(g, "--bias", True, "bias in the Lorentz linear layers")
    g.add_argument("--dropout", type=float, default=0)
    g.add_argument("--windows", type=int, default=2, help="number of patches")
    g.add_argument("--lora_type", default="boost", choices=["boost", "linear", "geodesic", "hyplora"],
                   help="subject adapter after the projection")
    g.add_argument("--sub_rank", type=int, default=None, help="processor LoRA rank")
    g.add_argument("--decoder_rank", type=int, default=None, help="number of boosts (LB-Adapter rank)")
    g.add_argument("--lora_lr", type=float, default=1e-1, help="adapter scaling")
    g.add_argument("--proc_lora_lr", type=float, default=1.0, help="processor LoRA scaling")
    flag(g, "--dw_lora", False, "LoRA on the depthwise spatial conv as well")
    flag(g, "--preprocessor_adaptive", False, "per-subject BN affine in the processor")
    flag(g, "--encoder_adaptive", False, "per-subject GyroBN affine in the encoder")
    g.add_argument("--reference", default="block", choices=["block", "none", "none_matched"],
                   help="reference branch. block: reference block and contrast; none: task "
                        "branch only; none_matched: task branch only with its bottleneck doubled")
    g.add_argument("--block_norm", default="encoder", choices=["encoder", "block"],
                   help="encoder: GyroBN on each branch output; block: HBN + HAct "
                        "inside each inception block")
    g.add_argument("--decoder_proj", default="learned", choices=["learned", "frozen", "none"],
                   help="learned: trainable projection; frozen: unit-norm random "
                        "projection; none: identity (no projection)")
    g.add_argument("--proto_init", default="lift", choices=["lift", "wrapped_normal"],
                   help="prototype initialisation. lift: Gaussian space coordinates lifted "
                        "with add_time; wrapped_normal: expmap0 of a Gaussian tangent vector")
    g.add_argument("--proto_std", type=float, default=None,
                   help="std of the Gaussian the prototypes are drawn from; default: 1 for lift, "
                        "radius-matched for wrapped_normal")
    g.add_argument("--head_center", default="none", choices=["none", "batch"],
                   help="batch: centre the head embeddings of each batch on their mean before the "
                        "prototypes (train and test alike). none: no centring")
    g.add_argument("--manifold_limits", default="constant", choices=["constant", "curvature"],
                   help="radial bound of rescale_to_max: constant or "
                        "curvature-dependent")

    g = ap.add_argument_group("optimisation")
    g.add_argument("--bs", type=int, default=64)
    g.add_argument("--lr", type=float, default=5e-5)
    g.add_argument("--wd", type=float, default=1e-2)
    flag(g, "--scheduler", False, "linear warm-up + cosine decay")
    g.add_argument("--clip_grad", type=float, default=0)
    flag(g, "--interaug", True, "segment-recombination augmentation")
    flag(g, "--cutfill", False, "cut-and-fill augmentation")
    flag(g, "--learn_decoder", True, "train the class prototypes")
    flag(g, "--learn_predecoder", True, "train the projection and head adapter")
    flag(g, "--learn_lora", True, "train the subject adapters; false keeps them at the identity")
    g.add_argument("--val_split", type=float, default=None,
                   help="hold out this fraction of the bcic2a training session for validation")
    return ap


def read_cfg(path):
    """Read a configs/<task>/<dataset>.cfg file: one `key = value` per line, `#` comments."""
    values = {}
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            key, sep, value = line.partition("=")
            if not sep:
                raise ValueError(f"{path}:{n}: expected `key = value`, got {line!r}")
            values[key.strip()] = value.strip()
    return values


def parse_args(ap, argv=None):
    """Parse the command line, with --cfg read first so that explicit flags override it."""
    argv = sys.argv[1:] if argv is None else list(argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--cfg", default=None)
    path = pre.parse_known_args(argv)[0].cfg
    if path:
        argv = [tok for k, v in read_cfg(path).items() for tok in (f"--{k}", v)] + argv
    return resolve(vars(ap.parse_args(argv)))


def resolve(args):
    """Apply process-wide settings implied by the parsed arguments."""
    set_limits_mode(args["manifold_limits"])
    cfg_entry = DEFAULT_MODEL_CONFIGS[args["dataset"]][0]
    if args["sub_rank"] is not None:
        cfg_entry["sub_rank"] = args["sub_rank"]
    if args["decoder_rank"] is not None:
        cfg_entry["decoder_rank"] = args["decoder_rank"]
    return args


def load_config(args):
    with open(args["config"]) as f:
        cfg = yaml.safe_load(f)
    cfg["dataset_name"] = args["dataset"]
    if args.get("data_path"):
        cfg["data_path"] = args["data_path"]
    return cfg


def prepare_model(net, trainloader, args):
    """Build the lazily-sized head from one batch, then apply the --learn_* flags."""
    if hasattr(net, "get_decoder"):
        x, s, _ = next(iter(trainloader))
        net.get_decoder(x.to(args["device"]), s.to(args["device"]))
    if hasattr(net, "set_trainable"):
        net.set_trainable(learn_decoder=args["learn_decoder"],
                          learn_predecoder=args["learn_predecoder"],
                          learn_lora=args["learn_lora"])
    if hasattr(net, "manifold") and net.manifold is not None:
        print("Curvature:", net.manifold.k.item())
    return net


def init_wandb(args, run_name, run_dir, group=None):
    if not args.get("use_wandb"):
        return None
    import wandb
    return wandb.init(project=args["wandb_project"], entity=args["wandb_entity"],
                      mode=args["wandb_mode"], name=run_name, group=group or args.get("wandb_group"),
                      tags=[t for t in [args.get("tag"), args["model"], args["dataset"]] if t],
                      dir=run_dir, config={k: v for k, v in args.items()
                                           if k not in ("run_dir", "checkpoints_dir")},
                      reinit="finish_previous")


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
