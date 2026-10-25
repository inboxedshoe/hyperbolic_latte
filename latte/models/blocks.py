import torch
from torch import nn

from latte.geometry.layers.misc.LModules import LorentzAct, QuickDirtyMaxPool
from latte.geometry.layers.misc import LORENTZ_POOL1D_TYPES
from latte.geometry.layers.convolution import LORENTZ_CONV1D_TYPES
from latte.geometry.layers.batchnorm import LORENTZ_BN1D_TYPES


def pass_through(x):
    return x


class TangentMaxPool(nn.Module):
    """Max pool in the tangent space at the origin: logmap0, Euclidean max
    pool on the tangent vectors, expmap0 back."""

    def __init__(self, manifold, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.manifold = manifold
        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        v = self.manifold.logmap0(x)[..., 1:]                    # (B, L, D)
        v = self.pool(v.permute(0, 2, 1)).permute(0, 2, 1)
        return self.manifold.expmap0(nn.functional.pad(v, (1, 0)))


def build_pool(manifold, pool_type, pool_tau=1.0):
    """pool_type:
      dirty          per-coordinate max over the space components, then lift
      clean          the point in each window farthest from the origin
      clean_centroid the point in each window farthest from the window's centroid
      soft           centroid of the window weighted by softmax(radius / tau), tau learnable
      tangent        logmap0 -> max pool -> expmap0
      average        Lorentz average pool (reference branch)
    """
    if pool_type == "dirty":
        return QuickDirtyMaxPool(manifold, nn.MaxPool1d(kernel_size=3, stride=1, padding=1))
    if pool_type == "tangent":
        return TangentMaxPool(manifold)
    if pool_type == "soft":
        return LORENTZ_POOL1D_TYPES[pool_type](manifold, kernel_size=3, padding=1, stride=1, tau=pool_tau)
    return LORENTZ_POOL1D_TYPES[pool_type](manifold, kernel_size=3, padding=1, stride=1)


class LorentzInceptionBlock(nn.Module):
    """Lorentz InceptionTime block.

    With batch_type / activation None the block applies no normalisation;
    normalisation is then applied once per branch in the encoder. Passing
    them applies HBN + HAct inside the block.
    """

    def __init__(self,
                 manifold,
                 in_channels,
                 n_filters=32,
                 kernel_sizes=(9, 19, 39),
                 bottleneck_channels=8,
                 activation=None,
                 conv_type="base",
                 batch_type=None,
                 fc_type="base",
                 pool_type="dirty",
                 pool_tau=1.0,
                 dropout=0,
                 bias=False):
        super().__init__()
        self.manifold = manifold

        conv_layer = LORENTZ_CONV1D_TYPES[conv_type]

        n_filters = n_filters + 1
        bottleneck_channels = bottleneck_channels + 1
        in_channels = in_channels + 1

        if in_channels > 1:
            self.bottleneck = conv_layer(self.manifold, in_channels=in_channels,
                                         out_channels=bottleneck_channels, fc_type=fc_type,
                                         kernel_size=1, stride=1, bias=bias)
        else:
            self.bottleneck = pass_through
            bottleneck_channels = 1

        def branch(k):
            return conv_layer(self.manifold, in_channels=bottleneck_channels,
                              out_channels=n_filters, fc_type=fc_type,
                              kernel_size=k, stride=1, padding=k // 2, bias=bias)

        self.conv_from_bottleneck_1 = branch(kernel_sizes[0])
        self.conv_from_bottleneck_2 = branch(kernel_sizes[1])
        self.conv_from_bottleneck_3 = branch(kernel_sizes[2])

        self.max_pool = build_pool(self.manifold, pool_type, pool_tau)

        self.conv_from_maxpool = conv_layer(self.manifold, in_channels=in_channels,
                                            out_channels=n_filters, kernel_size=1,
                                            stride=1, padding=0, bias=False)

        out_features = 4 * (n_filters - 1) + 1
        self.batch_norm = (LORENTZ_BN1D_TYPES[batch_type](manifold, num_features=out_features)
                           if batch_type is not None else nn.Sequential())
        self.activation = LorentzAct(activation, manifold) if activation is not None else nn.Identity()
        self.drop = LorentzAct(nn.Dropout(p=dropout), manifold) if dropout != 0 else nn.Identity()

    def forward(self, x):
        z_bottleneck = self.bottleneck(x)
        z_maxpool = self.manifold.rescale_to_max(self.max_pool(x))

        z1 = self.conv_from_bottleneck_1(z_bottleneck)
        z2 = self.conv_from_bottleneck_2(z_bottleneck)
        z3 = self.conv_from_bottleneck_3(z_bottleneck)
        z4 = self.conv_from_maxpool(z_maxpool)

        z = self.manifold.direct_concat(
            torch.cat([z1.unsqueeze(-2), z2.unsqueeze(-2), z3.unsqueeze(-2), z4.unsqueeze(-2)], dim=-2))
        z = self.activation(self.batch_norm(z))
        z = self.drop(z)
        return self.manifold.rescale_to_max(z)

