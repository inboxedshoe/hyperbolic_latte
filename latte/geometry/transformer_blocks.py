import torch
import torch.nn as nn
import torch.nn.functional as F

from latte.geometry.layers.linear import LORENTZ_FC_TYPES
from .manifold import CustomLorentz


class LorentzMultiHeadAttention(nn.Module):
    """Lorentz attention: squared-Lorentz-distance scores,
    Lorentzian-centroid aggregation, all on the hyperboloid."""

    def __init__(self, manifold: CustomLorentz, num_features, num_patches, heads,
                 dropout=0.0, learn_scale=False, bias=False, out_features=None,
                 fc_type="base"):
        super().__init__()

        self.manifold = manifold
        self.out_features = out_features if out_features is not None else num_features
        self.num_features = num_features
        self.num_patches = num_patches
        self.heads = heads
        self.head_dim = (num_features - 1) // heads
        self.temperature = nn.Parameter(torch.ones(1))
        self.scale = nn.Parameter(self.head_dim ** (-0.5) * torch.ones((1, heads, 1, 1)),
                                  requires_grad=learn_scale)

        def fc(i, o):
            return LORENTZ_FC_TYPES[fc_type](manifold=manifold, in_features=i,
                                             out_features=o, bias=bias)

        self.q = fc(num_features, self.out_features)
        self.k = fc(num_features, self.out_features)
        self.v = fc(num_features, self.out_features)
        self.o = fc(self.out_features, self.out_features)

    def forward(self, x):
        b, n, _ = x.size()

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        cs_dist = self.manifold.csqdist(q, k)
        if self.heads == 1:
            cs_dist = cs_dist.unsqueeze(1)
        score = F.softmax(-cs_dist * self.scale.abs() / self.temperature.abs(), dim=-1)

        # Lorentzian centroid aggregation
        attn = self.manifold.centroid(v.unsqueeze(1), score)

        # Lorentz direct concatenation of the heads
        attn_space = attn.narrow(-1, 1, attn.shape[-1] - 1).reshape(b, n, -1)
        attn_time = attn.narrow(-1, 0, 1).reshape(b, n, -1)
        attn_time = torch.sqrt(
            torch.sum(attn_time ** 2, dim=-1, keepdim=True) - ((self.heads - 1) * self.manifold.k))
        attn = torch.concat((attn_time, attn_space), dim=-1)

        return self.o(attn)
