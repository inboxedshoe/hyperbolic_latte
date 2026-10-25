import numpy as np
import torch

from latte.geometry.manifold import CustomLorentz
from latte.geometry.optim import RiemannianAdamW
from latte.geometry.layers.batchnorm import (
    set_bn_epoch, finalize_bn_domain_stats, refit_bn_domains)
from latte.models.latte import TCFormerDeviationHeadLora
from latte.training.utils import get_param_groups
from latte.training.lr_scheduler import linear_warmup_cosine_decay

DEFAULTS = dict(
    conv_type="base", batch_type="gyro", fc_type="plfc_scaled",
    pool_type="clean", lora_type="boost", windows=2, dropout=0.4,
    features=450 / 4, lora_lr=0.1, proc_lora_lr=2.0, bias=False,
    learn_k=True, curvature=1.0,
    encoder_adaptive=True,
    lr=5e-5, weight_decay=1e-2, epochs=200,
    warmup_epochs=3,
    domains_per_batch=8, obs_per_domain=8,
    sub_rank=32, decoder_rank=32,
    interaug=True,
)

ROWS = {
    "base":        dict(stats_mode="shared",     affine_mode="per_subject", momentum_mode="fixed"),
    "adaptmom":    dict(stats_mode="shared",     affine_mode="per_subject", momentum_mode="adaptive"),
    "domain":      dict(stats_mode="per_domain", affine_mode="shared",      momentum_mode="adaptive"),
    "domain_subj": dict(stats_mode="per_domain", affine_mode="per_subject", momentum_mode="adaptive"),
}


def build(data, row, device, cfg=None, dataset_key="bcic2a"):
    """Instantiate LAtte with the BN knob set to `row`.

    `dataset_key` selects the architecture's per-dataset config (channel count,
    subject-slot count). It has to match the data actually being fed.
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    knob = ROWS[row]

    if cfg.get("sub_rank") is not None or cfg.get("decoder_rank") is not None:
        from latte.models.configs import DEFAULT_MODEL_CONFIGS
        entry = DEFAULT_MODEL_CONFIGS.get(dataset_key, [{}])[0]
        if cfg.get("sub_rank") is not None:
            entry["sub_rank"] = int(cfg["sub_rank"])
        if cfg.get("decoder_rank") is not None:
            entry["decoder_rank"] = int(cfg["decoder_rank"])

    manifold = CustomLorentz(k=cfg["curvature"], learnable=cfg["learn_k"])
    bn_kwargs = dict(knob)
    if knob["momentum_mode"] == "adaptive":
        bn_kwargs["momentum_K_frac"] = 0.8

    net = TCFormerDeviationHeadLora(
        manifold=manifold, n_classes=data["n_classes"], features=cfg["features"],
        dataset=dataset_key, learn_k=cfg["learn_k"], pool_type=cfg["pool_type"],
        conv_type=cfg["conv_type"], batch_type=cfg["batch_type"],
        fc_type=cfg["fc_type"], dropout=cfg["dropout"], windows=cfg["windows"],
        lora_lr=cfg["lora_lr"], lora_type=cfg["lora_type"], bias=cfg["bias"],
        proc_lora_lr=cfg["proc_lora_lr"],
        encoder_adaptive=cfg["encoder_adaptive"], bn_kwargs=bn_kwargs,
    ).to(device)

    n_adaptive = sum(1 for _, m in net.named_modules()
                     if type(m).__name__.startswith("GyroLBNCustomAdaptive"))
    if n_adaptive == 0:
        raise RuntimeError(
            f"row '{row}' produced no subject-conditioned BN layers. Check "
            "encoder_adaptive and bn_kwargs.")
    return net, cfg, n_adaptive


def make_optimizer(net, cfg, iterations):
    """RiemannianAdamW with a linear-warmup cosine-decay schedule."""
    from torch.optim.lr_scheduler import LambdaLR

    groups = get_param_groups(net, cfg["lr"], cfg["weight_decay"])
    opt = RiemannianAdamW(groups, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = LambdaLR(opt, linear_warmup_cosine_decay(cfg["warmup_epochs"], iterations))
    return opt, sched


class _DS(torch.utils.data.Dataset):
    """Dataset of (x, subject_id, y) tuples."""

    def __init__(self, X, s, y):
        self.X = torch.from_numpy(np.ascontiguousarray(X)).float()
        self.s = torch.from_numpy(s).long()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.s[i], self.y[i]


def loaders(data, cfg):
    """Train loader uses the domain-stratified sampler; eval loaders are plain.

    `DomainStratifiedBatchSampler` composes every batch as D domains x N
    observations and derives the batch size as D*N.
    """
    from baselines.manifold.sampler import DomainStratifiedBatchSampler

    tr = _DS(data["X_train"], data["s_train"], data["y_train"])
    va = _DS(data["X_val"], data["s_val"], data["y_val"])
    te = _DS(data["X_test"], data["s_test"], data["y_test"])

    d = int(min(cfg["domains_per_batch"], len(np.unique(data["s_train"]))))
    sampler = DomainStratifiedBatchSampler(
        domain_labels=data["s_train"], domains_per_batch=d,
        obs_per_domain=cfg["obs_per_domain"])
    train_loader = torch.utils.data.DataLoader(
        tr, batch_sampler=sampler, num_workers=0)
    mk = lambda x: torch.utils.data.DataLoader(  # noqa: E731
        x, batch_size=512, shuffle=False, num_workers=0)
    return train_loader, mk(va), mk(te), sampler


def augment(x, y, cfg, n_classes):
    """Identity: no augmentation."""
    return x, y


@torch.no_grad()
def predict(net, loader, device):
    net.eval()
    yh, yt = [], []
    for x, s, y in loader:
        x, s = x.to(device), s.to(device)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        out = net(x, s)
        out = out[0] if isinstance(out, (list, tuple)) else out
        yh.append(out.argmax(1).cpu().numpy())
        yt.append(y.numpy())
    return np.concatenate(yt), np.concatenate(yh)


def epoch_step(net, loader, opt, sched, CE, device, cfg, n_classes):
    net.train()
    tot = n = 0
    for x, s, y in loader:
        x, s, y = x.to(device), s.to(device), y.to(device)
        x, y = augment(x, y, cfg, n_classes)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        out = net(x, s)
        out = out[0] if isinstance(out, (list, tuple)) else out
        loss = CE(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += float(loss.item()) * y.shape[0]
        n += y.shape[0]
    if sched is not None:
        sched.step()
    return tot / max(n, 1)


@torch.no_grad()
def bn_refit(net, loader, device, verbose=False):
    """Label-free target refit.

    Gathers the held-out subject's trials from `loader` (labels are never read)
    and re-estimates every adaptive BN's statistics for the slot those trials
    route to. Call *after* `load_state_dict`, so the refit is scoped to one
    evaluation: the state carries buffers, so each `evaluate()` starts from the
    trained statistics and the refit never leaks into the other selection rule.

    The slot this writes is whichever the test trials index.
    """
    xs, ss = [], []
    for x, s, _ in loader:
        xs.append(x)
        ss.append(s)
    if not xs:
        return 0
    x = torch.cat(xs).to(device)
    s = torch.cat(ss).to(device)
    return refit_bn_domains(net, x, s, verbose=verbose)


def bn_epoch(net, epoch, total):
    set_bn_epoch(net, epoch, total)


def bn_finalize(net):
    finalize_bn_domain_stats(net)
