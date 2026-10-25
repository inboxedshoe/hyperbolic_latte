import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt import ManifoldParameter


class LorentzPrototypeDecoder(nn.Module):

    def __init__(self, manifold, in_channels, n_classes, init="lift", std=None):
        super().__init__()
        if init not in ("lift", "wrapped_normal"):
            raise ValueError(f"prototype init must be lift/wrapped_normal, got {init!r}")
        self.manifold = manifold
        self.n_classes = n_classes

        noise = torch.randn((n_classes, in_channels), device=self.manifold.k.device)
        if init == "lift":
            prototypes = self.manifold.add_time(noise if std is None else std * noise)
        else:
            if std is None:
                std = math.acosh(math.sqrt(1 + in_channels)) / math.sqrt(in_channels)
            prototypes = self.manifold.expmap0(F.pad(std * noise, (1, 0)))
        self.prototypes = ManifoldParameter(prototypes, self.manifold)

    def forward(self, x):
        prototypes = self.manifold.projx(self.prototypes)
        return -self.manifold.sqdist(x.unsqueeze(-2), prototypes)
