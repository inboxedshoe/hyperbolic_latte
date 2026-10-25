import fnmatch

import numpy as np
import pandas as pd
import torch

from baselines.manifold import third_party

third_party("TSMNet")

from spdnets.models.tsmnet import TSMNet                                   # noqa: E402
from spdnets.batchnorm import (MomentumBatchNormScheduler,                 # noqa: E402
                               BatchNormTestStatsMode)
from library.utils.torch import (CachedDomainDataset,                      # noqa: E402
                                 StratifiedDomainDataLoader, DomainIndex)

# TSMNet architecture and training configuration.
DEFAULTS = dict(
    temporal_filters=4, spatial_filters=40, subspacedims=20, temp_cnn_kernel=25,
    epochs=50, batch_size_train=50, domains_per_batch=5,
    lr=1e-3, weight_decay=1e-4,
)
# SPDDSMBN momentum schedule: epochs - 10, bs = 50/5, bs0 = 50.
SCHED_DSMBN = dict(tau0=0.85, bs_div=True)
# SPDMBN momentum schedule:   bs = batch_size_train, bs0 = bs * 10.
SCHED_MBN = dict(tau0=0.35, bs_div=False)


class _NetShim:
    """Expose `module` as `net.module_` for `MomentumBatchNormScheduler`."""

    def __init__(self, module):
        self.module_ = module


DTYPE = torch.float32


def build_dataset(X, y, d, dtype=DTYPE):
    """Wrap the arrays in a `CachedDomainDataset`. `metadata` only needs to carry the length."""
    return CachedDomainDataset(
        torch.from_numpy(np.ascontiguousarray(X)),
        labels=torch.from_numpy(y).long(),
        domains=torch.from_numpy(d).long(),
        metadata=pd.DataFrame(index=range(len(y))),
        training=False, dtype=dtype,
    )


def make_optimizer(model, lr, weight_decay):
    """RiemannianAdam with weight-decay param groups selected by fnmatch.

    Both `spd*.mean` and `spdnet.*.W` are excluded from weight decay: the first
    is an SPD bias, the second a Stiefel frame.
    """
    from geoopt.optim import RiemannianAdam

    patterns = ["spd*.mean", "spdnet.*.W"]
    zero_wd, rest = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (zero_wd if any(fnmatch.fnmatch(name, pat) for pat in patterns) else rest).append(p)
    groups = [dict(params=rest)]
    if zero_wd:
        groups.append(dict(params=zero_wd, weight_decay=0.0))
    return RiemannianAdam(groups, lr=lr, weight_decay=weight_decay)


def make_scheduler(arm, epochs, batch_size_train, domains_per_batch):
    """`MomentumBatchNormScheduler`, with the per-arm `bs`/`bs0`/`tau0` resolved.

    The schedule spans `epochs - 10`, so it reaches its floor ten epochs before
    training ends.
    """
    if arm == "default":
        bs = batch_size_train / domains_per_batch
        bs0 = batch_size_train
        tau0 = SCHED_DSMBN["tau0"]
    else:
        bs = batch_size_train
        bs0 = batch_size_train * 10
        tau0 = SCHED_MBN["tau0"]
    sched = MomentumBatchNormScheduler(epochs=epochs - 10, bs=bs, bs0=bs0, tau0=tau0)
    sched.initialize()
    return sched


def build(data, arm, device, cfg=None):
    """Instantiate TSMNet for the chosen arm, in fp32."""
    cfg = {**DEFAULTS, **(cfg or {})}

    if arm == "default":
        # Register a branch for every domain that appears anywhere in the cell.
        # For LOSO that includes the held-out subject, whose branch stays at
        # its initialisation until `domainadapt_finetune` fits it from
        # unlabeled target data.
        domains = torch.as_tensor(
            np.union1d(data["train_domains"], data["test_domains"])).long()
        bnorm, dispersion = "spddsbn", "SCALAR"
    elif arm == "inductive":
        domains = torch.as_tensor(data["train_domains"]).long()
        bnorm, dispersion = "spdbn", "SCALAR"
    else:
        raise ValueError(arm)

    model = TSMNet(
        temporal_filters=cfg["temporal_filters"],
        spatial_filters=cfg["spatial_filters"],
        subspacedims=cfg["subspacedims"],
        temp_cnn_kernel=cfg["temp_cnn_kernel"],
        bnorm=bnorm, bnorm_dispersion=dispersion,
        nclasses=data["n_classes"], nchannels=data["n_channels"],
        nsamples=data["n_samples"], nbands=1, domains=domains,
        input_shape=(1, data["n_channels"], data["n_samples"]),
        device=device, spd_dtype=DTYPE,
    )
    model.to(device)
    check_fp32(model)
    return model, cfg


def check_fp32(model):
    """Fail if any floating parameter or buffer is not float32."""
    bad = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
           if t.is_floating_point() and t.dtype != torch.float32]
    if bad:
        raise RuntimeError(
            f"TSMNet has non-fp32 tensors ({', '.join(bad[:5])}...). Use the "
            "copy in baselines/external/TSMNet (spd_dtype).")


@torch.no_grad()
def predict(model, ds, batch=512):
    """Per-domain inference.

    Predicts one domain at a time because the domain-specific BN dispatches on
    the domain index.
    """
    model.eval()
    n = len(ds)
    y_hat = np.empty(n, dtype=np.int64)
    logits = None
    for du in ds.domains.unique():
        ixs = np.flatnonzero(ds.domains.numpy() == du.item())
        if ixs.size == 0:
            continue
        for s in range(0, ixs.size, batch):
            sel = ixs[s:s + batch]
            x = ds.get_features(sel).to(dtype=ds.dtype)
            d = ds.domains[sel]
            out = model(x, d)
            lg = (out[0] if isinstance(out, (list, tuple)) else out).float().cpu()
            if logits is None:
                logits = np.empty((n, lg.shape[1]), dtype=np.float32)
            logits[sel] = lg.numpy()
            y_hat[sel] = lg.argmax(1).numpy()
    return y_hat, logits


def train_loader(ds, train_idx, cfg, n_train_domains):
    """`StratifiedDomainDataLoader` over a `Subset` of the training indices.

    The batch size is rounded to a multiple of the domains per batch.
    """
    dpb = int(min(cfg["domains_per_batch"], n_train_domains))
    bs = int(max(round(cfg["batch_size_train"] / dpb), 2) * dpb)
    sub = torch.utils.data.Subset(ds, np.asarray(train_idx))
    loader = StratifiedDomainDataLoader(
        sub, batch_size=bs, domains_per_batch=dpb, shuffle=True)
    return loader, bs, dpb


def domain_adapt(model, ds, device):
    """Label-free target refit: `TSMNet.domainadapt_finetune`."""
    model.eval()
    x = ds.features.to(device=device)
    model.domainadapt_finetune(x=x, y=ds.labels, d=ds.domains, target_domains=None)


def source_finetune(model, ds, device):
    """`FineTuneableModel.finetune` on *source* data.

    Refits the single shared BN's running statistics from the training pool.
    Uses no target data.
    """
    if not (hasattr(model, "spdbnorm") or hasattr(model, "tsbnorm")):
        return
    model.eval()
    model.finetune(x=ds.features.to(device=device), y=ds.labels, d=ds.domains)
