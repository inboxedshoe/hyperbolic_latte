import numpy as np
import torch

from baselines.manifold import third_party

third_party("HEEGNet")

from nets.model import HEEGNet                                        # noqa: E402
from nets.trainer import Trainer                                      # noqa: E402
from nets.callbacks import (Callback, EarlyStopping,                  # noqa: E402
                            MomentumBatchNormScheduler)
from nets.utils.data import DomainDataset, StratifiedDomainDataLoader  # noqa: E402
import nets.batchnorm as hbn                                          # noqa: E402
import nets.functionals as hfn                                        # noqa: E402

# HEEGNet configuration.
DEFAULTS = dict(
    epochs=100, min_epochs=70, patience=20,
    batch_size_train=50, domains_per_batch=5,
    lr=1e-3, weight_decay=1e-3,
    swd_weight=0.01,
    F1=8, F2=16, D=2, kernel_1=64, kernel_2=16, dropout=0.25,
    tau0=0.85, sched_offset=10,
)

DTYPE = torch.float32


def euclidean_align(X, d, domains):
    """`nets.functionals.euler_align`, applied per domain over that domain's trials.

    Called on the *union* of splits for a domain: `X[domain == i]` is aligned
    over the whole array before any CV split. For
    the held-out LOSO subject the alignment is fitted on its unlabeled test
    trials.
    """
    Xa = np.array(X, dtype=np.float32, copy=True)
    for i in domains:
        m = (d == i)
        if m.sum() < 2:
            continue
        Xa[m] = np.asarray(hfn.euler_align(X[m]), dtype=np.float32)
    return Xa


def build_dataset(X, y, d, dtype=DTYPE):
    return DomainDataset(
        torch.from_numpy(np.ascontiguousarray(X)).to(dtype=dtype),
        torch.from_numpy(y).long(),
        torch.from_numpy(d).long(),
    )


def build(data, arm, device, cfg=None, dtype=DTYPE):
    cfg = {**DEFAULTS, **(cfg or {})}

    if arm == "default":
        domains = torch.as_tensor(
            np.union1d(data["train_domains"], data["test_domains"])).long()
        domain_adaptation = True
    elif arm == "inductive":
        domains = torch.as_tensor(data["train_domains"]).long()
        domain_adaptation = False
    else:
        raise ValueError(arm)

    model = HEEGNet(
        chunk_size=data["n_samples"], num_electrodes=data["n_channels"],
        F1=cfg["F1"], F2=cfg["F2"], D=cfg["D"],
        bnorm_dispersion=hbn.BatchNormDispersion.SCALAR,
        num_classes=data["n_classes"],
        kernel_1=cfg["kernel_1"], kernel_2=cfg["kernel_2"], dropout=cfg["dropout"],
        domains=domains, domain_adaptation=domain_adaptation,
        device=device, dtype=dtype,
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    ).to(device=device, dtype=dtype)
    bad = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
           if t.is_floating_point() and t.dtype != dtype]
    if bad:
        raise RuntimeError(f"HEEGNet tensors not in {dtype}: {', '.join(bad[:5])}")
    return model, cfg


def make_trainer(cfg, device, dtype, extra_callbacks=()):
    """Trainer + EarlyStopping + MomentumBatchNormScheduler."""
    es = EarlyStopping(metric="val_loss", higher_is_better=False,
                       patience=cfg["patience"], verbose=False)
    sched = MomentumBatchNormScheduler(
        epochs=cfg["epochs"] - cfg["sched_offset"],
        bs0=cfg["batch_size_train"],
        bs=cfg["batch_size_train"] / cfg["domains_per_batch"],
        tau0=cfg["tau0"],
    )
    trainer = Trainer(
        max_epochs=cfg["epochs"], min_epochs=cfg["min_epochs"],
        callbacks=[es, sched, *extra_callbacks],
        loss=torch.nn.CrossEntropyLoss(),
        device=device, dtype=dtype,
        swd_weight=cfg["swd_weight"], lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    return trainer, es


def train_loader(ds, cfg, n_train_domains):
    """`StratifiedDomainDataLoader`."""
    dpb = int(min(cfg["domains_per_batch"], n_train_domains))
    return StratifiedDomainDataLoader(
        ds, cfg["batch_size_train"], domains_per_batch=dpb,
        shuffle=True, drop_last=False), dpb


def eval_loader(ds, batch=512):
    return torch.utils.data.DataLoader(ds, batch_size=min(batch, max(len(ds), 1)),
                                       shuffle=False)


def domain_adapt(model, ds, device, dtype):
    """`HEEGNet.domainadapt_finetune`."""
    model.eval()
    model.domainadapt_finetune(
        ds.features.to(dtype=dtype, device=device),
        ds.labels.to(device=device), ds.domains, None)


class SnapshotEval(Callback):
    """Records per-epoch validation score and keeps a CPU snapshot of the weights.

    `EarlyStopping` selects on best validation loss; the snapshots let the peak 5-point-smoothed validation
    score be recorded next to it.
    """

    def __init__(self):
        self.scores = []
        self.losses = []
        self.states = []

    def on_train_epoch_end(self, trainer, net):
        rec = None
        for r in trainer.records[::-1]:
            if r.get("epoch") == trainer.current_epoch and "val_score" in r:
                rec = r
                break
        if rec is None:
            return
        self.scores.append(float(rec["val_score"]))
        self.losses.append(float(rec["val_loss"]))
        self.states.append({k: v.detach().to("cpu").clone()
                            for k, v in net.state_dict().items()})
