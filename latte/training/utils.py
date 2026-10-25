import os
import random

import geoopt
import numpy as np
import torch

# Encoder width. Must stay a multiple of 4: the inception block emits
# 4 * int(features / 4) + 1 channels and GyroBN sizes its buffers as
# features + 1. 4 * round(450 / 16) = 112.
FEATURES = 4 * max(1, round(450 / 4 / 4))

MODELS = ("TCFormerDeviationHeadLora",)


def fix_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_model(num_pred_classes, manifold=None, **kwargs):
    """Construct the model named by ``kwargs['model']`` (TCFormerDeviationHeadLora: LAtte)."""
    device = kwargs["device"]
    model = kwargs["model"]
    if model == "TCFormerDeviationHeadLora":
        from latte.models.latte import TCFormerDeviationHeadLora
        net = TCFormerDeviationHeadLora(manifold=manifold,
                                        n_classes=num_pred_classes,
                                        features=FEATURES,
                                        dataset=kwargs["dataset"],
                                        learn_k=kwargs["learnable_k"] != 0,
                                        pool_type=kwargs["pool_type"],
                                        pool_tau=kwargs.get("pool_tau", 1.0),
                                        conv_type=kwargs["conv_type"],
                                        batch_type=kwargs["batch_type"],
                                        fc_type=kwargs["linear_type"],
                                        dropout=kwargs["dropout"],
                                        windows=kwargs["windows"],
                                        lora_lr=kwargs["lora_lr"],
                                        lora_type=kwargs["lora_type"],
                                        bias=kwargs["bias"],
                                        proc_lora_lr=kwargs["proc_lora_lr"],
                                        dw_lora=kwargs.get("dw_lora", False),
                                        preprocessor_adaptive=kwargs.get("preprocessor_adaptive", False),
                                        encoder_adaptive=kwargs.get("encoder_adaptive", False),
                                        reference=kwargs.get("reference", "block"),
                                        block_norm=kwargs.get("block_norm", "encoder"),
                                        decoder_proj=kwargs.get("decoder_proj", "learned"),
                                        proto_init=kwargs.get("proto_init", "lift"),
                                        proto_std=kwargs.get("proto_std", None),
                                        head_center=kwargs.get("head_center", "none"))
    else:
        raise ValueError(f"unknown model {model!r}; choose from {MODELS}")
    return net.to(device)


def is_hyperbolic(model_name):
    """Whether the model is trained with the Riemannian optimiser."""
    return model_name == "TCFormerDeviationHeadLora"


# ---------------------------------------------------------------------- data
_DATASET_META = {
    "bcic2a": {"channels": 22, "classes": 4, "subjects": 9},
    "bcic2b": {"channels": 3,  "classes": 2, "subjects": 9},
    "bcicha": {"channels": 56, "classes": 2, "subjects": 16},
}


def build_datamodule(config, subject, batch_size, interaug=False, z_scale=True,
                     include_subject_ids=True, loso=False, val_split=None,
                     cross_subject_normalize=False):
    """Build dataloaders from the preprocessing block of configs/datasets.yaml.

    subject: int for a single subject (SS / fine-tuning), 'all' for joint
             training (SC), or the held-out subject when loso=True.
    Returns the set-up datamodule and the dataset metadata dict.
    """
    from latte.data.bcic4_2a import BCICIV2a, BCICIV2aLOSO, BCICIV2aAll
    from latte.data.bcic4_2b import BCICIV2b, BCICIV2bLOSO, BCICIV2bAll
    from latte.data.bcic4_cha import BCIcha, BCIchaLOSO, BCIchaAll

    dataset_name = config["dataset_name"].lower()
    if dataset_name not in _DATASET_META:
        raise ValueError(f"dataset '{dataset_name}' not supported; choose from {list(_DATASET_META)}")
    meta = _DATASET_META[dataset_name]
    preproc_raw = config.get("preprocessing", {}).get(dataset_name, {})

    preprocessing_dict = {
        "sfreq":      preproc_raw.get("sfreq", 250),
        "low_cut":    preproc_raw.get("low_cut", None),
        "high_cut":   preproc_raw.get("high_cut", None),
        "start":      preproc_raw.get("start", 0.0),
        "stop":       preproc_raw.get("stop", 0.0),
        "batch_size": batch_size,
        "z_scale":    z_scale,
        "interaug":   interaug,
        "num_workers": preproc_raw.get("num_workers", 8),
    }
    if preproc_raw.get("channel_selection"):
        preprocessing_dict["channel_selection"] = True
    if preproc_raw.get("remove_artifacts"):
        preprocessing_dict["remove_artifacts"] = True
    data_path = config.get("data_path") or preproc_raw.get("data_path")
    if data_path:
        preprocessing_dict["data_path"] = data_path

    table = {
        "bcic2a": (BCICIV2a, BCICIV2aAll, BCICIV2aLOSO),
        "bcic2b": (BCICIV2b, BCICIV2bAll, BCICIV2bLOSO),
        "bcicha": (BCIcha, BCIchaAll, BCIchaLOSO),
    }[dataset_name]
    if loso:
        if subject == "all":
            raise ValueError("loso=True requires an integer held-out subject")
        subject, dm_cls = int(subject), table[2]
    elif subject == "all":
        dm_cls = table[1]
    else:
        subject, dm_cls = int(subject), table[0]

    dm_kwargs = dict(preprocessing_dict=preprocessing_dict, subject_id=subject)
    if subject != "all":
        dm_kwargs["include_subject_ids"] = True if loso else include_subject_ids
        if cross_subject_normalize and dataset_name == "bcicha" and not loso:
            dm_kwargs["cross_subject_normalize"] = True
    if val_split is not None:
        if dataset_name != "bcic2a":
            raise ValueError("--val_split is only implemented for bcic2a; bcicha has a fixed "
                             "validation split, bcic2b uses the test session")
        dm_kwargs["val_split"] = val_split
    dm = dm_cls(**dm_kwargs)
    dm.setup()
    return dm, meta


def get_dataloaders_from_config(config, subject, batch_size, **kwargs):
    """Returns: trainloader, validloader, testloader, in_channels, num_classes, num_subjects"""
    dm, meta = build_datamodule(config, subject, batch_size, **kwargs)
    return (dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader(),
            meta["channels"], meta["classes"], meta["subjects"])


# ----------------------------------------------------------------- optimiser
def get_param_groups(model, lr_manifold, weight_decay_manifold):
    """Parameter groups for RiemannianAdamW: Euclidean, no-decay (``scale``),
    manifold parameters (prototypes), and the curvature k."""
    no_decay = ["scale"]
    k_params = ".k"
    named = list(model.named_parameters())
    return [
        {"params": [p for n, p in named if p.requires_grad and not any(nd in n for nd in no_decay)
                    and not isinstance(p, geoopt.ManifoldParameter) and not n.endswith(k_params)],
         "name": "1"},
        {"params": [p for n, p in named if p.requires_grad and any(nd in n for nd in no_decay)
                    and not isinstance(p, geoopt.ManifoldParameter) and not n.endswith(k_params)],
         "weight_decay": 0.0, "name": "no_decay"},
        {"params": [p for n, p in named if isinstance(p, geoopt.ManifoldParameter)],
         "lr": lr_manifold, "weight_decay": weight_decay_manifold, "name": "manifold"},
        {"params": [p for n, p in named if p.requires_grad and n.endswith(k_params)],
         "weight_decay": weight_decay_manifold, "lr": lr_manifold, "name": "k_group"},
    ]


def clip_gradients_value(model, clip_value, losses=None):
    for _, p in model.named_parameters():
        if p.grad is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)


def count_params(model, verbose=True):
    """Print and return trainable parameters per top-level module."""
    component_params, total = {}, 0
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        component_params[name] = n
        total += n
        if verbose:
            print(f"{name}: {n:,} trainable parameters")
    child_ids = {id(p) for _, m in model.named_children() for p in m.parameters()}
    root = sum(p.numel() for p in model.parameters() if p.requires_grad and id(p) not in child_ids)
    if root:
        component_params["__root__"] = root
        total += root
    if verbose:
        print(f"\nTotal trainable parameters: {total:,}")
    return component_params, total
