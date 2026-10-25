from typing import Any

import torch
import torch.nn as nn

from . import functional
from .functional import gr_identity_onb
from .grassmann_gyro import GrassmannianOBN, GyroGrAttention
from .spd_gyro import GyroSPDAttention, SPDOperator


# --------------------------------------------------------------------------
# Euclidean -> manifold front end
# --------------------------------------------------------------------------

def patch_len(n, epochs):
    """Split ``n`` samples into ``epochs`` near-equal patches."""
    base, remainder = n // epochs, n % epochs
    return [base + 1 if i < remainder else base for i in range(epochs)]


class Signal2Spd(nn.Module):
    """Covariance of one temporal patch, with a small ridge for conditioning."""

    def __init__(self, ridge=1e-5, tr_norm=False):
        super().__init__()
        self.ridge = ridge
        self.tr_norm = tr_norm

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(-2)                    # (B, C, 1, T) -> (B, C, T)
        x = x - x.mean(dim=-1, keepdim=True)
        cov = x @ x.transpose(-1, -2) / (x.shape[-1] - 1)
        if self.tr_norm:
            cov = cov / cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1, keepdim=True).unsqueeze(-1)
        return cov + self.ridge * torch.eye(cov.shape[-1], device=x.device, dtype=x.dtype)


class E2R(nn.Module):
    """Euclidean signal -> a sequence of SPD covariances (one per patch)."""

    def __init__(self, epochs, dim=-1, ridge=1e-5):
        super().__init__()
        self.epochs = epochs
        self.dim = dim
        self.signal2spd = Signal2Spd(ridge=ridge)

    def forward(self, x):
        list_patch = patch_len(x.shape[self.dim], self.epochs)
        x = torch.stack([self.signal2spd(p) for p in torch.split(x, list_patch, dim=self.dim)])
        return x.permute(1, 0, 2, 3)             # (P, B, n, n) -> (B, P, n, n)


class ProjMap(nn.Module):
    def forward(self, x):
        return x @ x.transpose(-1, -2)


class TrilEmbed(nn.Module):
    """Vectorise the lower triangle (optionally including the diagonal)."""

    def __init__(self, ndim, tril=False):
        super().__init__()
        if tril:
            ixs_lower = torch.tril_indices(ndim, ndim, offset=-1)
            ixs_diag = torch.arange(0, ndim, dtype=torch.long)
            ixs = torch.cat((ixs_diag[None, :].tile((2, 1)), ixs_lower), dim=1)
        else:
            ixs = torch.tril_indices(ndim, ndim, offset=-1)
        self.register_buffer("ixs", ixs)
        self.ndim = ndim

    def forward(self, X):
        return X[..., self.ixs[0], self.ixs[1]]


class PowLog(nn.Module):
    def __init__(self, power):
        super().__init__()
        self.register_buffer("power", torch.tensor(float(power)), persistent=False)

    def forward(self, x):
        return 1 / self.power * functional.sym_powm.apply(x, self.power)


# --------------------------------------------------------------------------
# PSPD composition (Grassmann x SPD)
# --------------------------------------------------------------------------

class PSPDComposition(GrassmannianOBN):
    """Split each covariance into a Grassmann frame and an SPD core.

    ``torch.linalg.eigh`` returns eigenvalues in *ascending* order, so
    ``[..., :p]`` selects the p smallest.
    """

    def __init__(self, n, p, momentum=0.1):
        super().__init__(n, p)
        self.p, self.momentum = p, momentum
        self.register_buffer("running_mean", gr_identity_onb(n, p))

    def gr_mean(self, x: torch.Tensor, karcher_steps: int = 1) -> torch.Tensor:
        with torch.no_grad():
            x = x.reshape(-1, x.shape[-2], x.shape[-1])
            mean_gr = x[0, :, :]
            for _ in range(karcher_steps):
                mean_log = self.logmap(mean_gr, x).mean(dim=-3)
                mean_gr = self.expmap(mean_gr, mean_log)
            return mean_gr

    @staticmethod
    def gr_canonicalize(gr, spd, mean):
        y, _, v = torch.linalg.svd(gr.transpose(-1, -2) @ mean)
        gr_canon = gr @ y
        spd_canon = v.transpose(-1, -2) @ gr_canon.transpose(-1, -2) @ spd @ gr_canon @ v
        return gr_canon, spd_canon

    def forward(self, x):
        s, u = torch.linalg.eigh(x)
        gr = u[..., :self.p]
        val = s[..., :self.p]
        pspd = gr * val.unsqueeze(-2) @ gr.transpose(-1, -2)
        with torch.no_grad():
            mean = self.gr_mean(gr) if self.training else self.running_mean.data
            if self.training:
                self.running_mean.data = self.geodesic(self.running_mean.data, mean, self.momentum)
        return self.gr_canonicalize(gr, pspd, mean)


class GyroAttention(nn.Module):
    """Dispatch to the SPD-only or the Grassmann x SPD attention block."""

    def __init__(self, n: int, p: int, manifold='PSPD', spd_metric='LEM',
                 activation_power=None, spd_power=None):
        super().__init__()
        self.n, self.p, self.manifold = n, p, manifold
        # `activation_power` is stored but not forwarded. The fuse power of
        # `GyroSPDAttention` is `spd_power` (power * activation_power), or its
        # default 1.0 when `spd_power` is None.
        self.activation_power = activation_power
        pw = {} if spd_power is None else {"power": spd_power}
        if manifold == 'SPD':
            self.SPD = GyroSPDAttention(n, spd_metric, **pw)
        elif manifold == 'Gr':
            self.Gr = GyroGrAttention(n, p)
        elif manifold == 'PSPD':
            self.Gr = GyroGrAttention(n, p)
            self.SPD = GyroSPDAttention(p, spd_metric, **pw)
        else:
            raise ValueError(f"unknown manifold {manifold!r}")

    def forward(self, *args):
        if self.manifold == 'SPD':
            return self.SPD(args[0])
        if self.manifold == 'Gr':
            return self.Gr(args[0])
        return self.forward_pspd(*args)

    def forward_pspd(self, x_gr, x_spd):
        q_gr, k_gr, v_gr = self.Gr.transform(x_gr)
        q_spd, k_spd, v_spd = self.SPD.transform(x_spd)
        att_energy = (self.SPD.dist(q_spd.unsqueeze(1), k_spd.unsqueeze(2))
                      + self.Gr.dist(q_gr.unsqueeze(1), k_gr.unsqueeze(2)))
        att_prob = nn.Softmax(dim=-2)(1 / (1 + torch.log(1 + att_energy))).transpose(-1, -2)
        return self.Gr.fuse(v_gr, att_prob), self.SPD.fuse(v_spd, att_prob)


# --------------------------------------------------------------------------
# Task models
# --------------------------------------------------------------------------

class GyroAttMamem(nn.Module):
    """MAMEM-SSVEP-II model."""

    def __init__(self, args: Any):
        super().__init__()
        dim1 = 20
        n_ch = getattr(args, "n_channels", 8)
        n_cls = getattr(args, "num_classes", 5)
        self.manifold = getattr(args, "manifold", "PSPD")

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, dim1, (n_ch, 1)),
            nn.BatchNorm2d(dim1), nn.ELU(),
            nn.Conv2d(dim1, dim1, (1, 36), bias=False, groups=dim1, padding=(0, 36 // 2)),
            nn.Conv2d(dim1, dim1, (1, 1), bias=False, padding=(0, 0)),
            nn.BatchNorm2d(dim1), nn.ELU(),
            nn.Conv2d(dim1, args.in_size, (1, 72), padding=(0, 36)),
            nn.BatchNorm2d(args.in_size), nn.ELU(),
        )

        self.E2R = E2R(args.epochs)
        self.att1 = GyroAttention(args.in_size, args.p, manifold=self.manifold,
                                  spd_metric=args.spd_metric,
                                  activation_power=args.activation_power * args.power,
                                  spd_power=getattr(args, "spd_power", None))

        if self.manifold == 'PSPD':
            self.pspd = PSPDComposition(args.in_size, args.p)
            self.cls_gr = nn.Sequential(
                ProjMap(), TrilEmbed(args.in_size, True), nn.Flatten(),
                nn.Linear(int(args.epochs * args.in_size * (args.in_size + 1) // 2), n_cls, bias=True))
            self.cls_spd = nn.Sequential(
                TrilEmbed(args.p, True), nn.Flatten(),
                nn.Linear(int(args.epochs * args.p * (args.p + 1) // 2), n_cls, bias=False))
        else:
            self.cls_spd = nn.Sequential(
                TrilEmbed(args.in_size, True), nn.Flatten(),
                nn.Linear(int(args.epochs * args.in_size * (args.in_size + 1) // 2), n_cls, bias=True))

        self.w = args.linear_weight

    def forward(self, x):
        x = self.conv_block(x)
        x = self.E2R(x)
        if self.manifold == 'PSPD':
            x_gr, x_spd = self.pspd(x)
            x_gr, x_spd = self.att1(x_gr, x_spd)
            return self.cls_gr(x_gr) + self.w * self.cls_spd(x_spd)
        return self.cls_spd(self.att1(x))


class GyroAttBCI(nn.Module):
    """Motor-imagery model for BCIC-IV-2a / 2b.

    The input channel count and class count are arguments. ``manifold`` selects the PSPD composition or the pure-SPD branch.
    """

    def __init__(self, args: Any):
        super().__init__()
        dim1 = 22
        n_ch = args.n_channels
        n_cls = args.num_classes
        self.manifold = getattr(args, "manifold", "PSPD")

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, dim1, (n_ch, 1)), nn.BatchNorm2d(dim1), nn.ELU())

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(dim1, dim1, (1, 13), bias=False, groups=dim1, padding=(0, 12 // 2)),
            nn.Conv2d(dim1, dim1, (1, 1), bias=False, padding=(0, 0)),
            nn.BatchNorm2d(dim1), nn.ELU(),
            nn.Conv2d(dim1, args.in_size, (1, 12), padding=(0, 6)),
            nn.BatchNorm2d(args.in_size), nn.ELU())

        self.E2R = E2R(args.epochs)
        self.att = GyroAttention(args.in_size, args.p, manifold=self.manifold,
                                 spd_metric=args.spd_metric,
                                 activation_power=args.activation_power * args.power,
                                 spd_power=getattr(args, "spd_power", None))

        self.flat = nn.Flatten()
        self.linear_pow = PowLog(args.power)

        if self.manifold == 'PSPD':
            self.pspd = PSPDComposition(args.in_size, args.p)
            self.proj = ProjMap()
            self.linear = nn.Linear(
                args.epochs * args.p * args.p + args.epochs * args.in_size * args.in_size,
                n_cls, bias=True)
        else:
            self.linear = nn.Linear(args.epochs * args.in_size * args.in_size, n_cls, bias=True)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.E2R(x)

        if self.manifold == 'PSPD':
            x_gr, x_spd = self.pspd(x)
            x_gr, x_spd = self.att(x_gr, x_spd)
            x_gr = self.flat(self.proj(x_gr))
            x_spd = self.flat(self.linear_pow(x_spd))
            return self.linear(torch.cat([x_gr, x_spd], dim=1))

        x_spd = self.att(x)
        return self.linear(self.flat(self.linear_pow(x_spd)))
