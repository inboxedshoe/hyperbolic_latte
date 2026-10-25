import torch
import torch.nn as nn

from latte.geometry.transformer_blocks import LorentzMultiHeadAttention
from latte.geometry.layers.batchnorm import MaybeAdaptiveBN, build_lorentz_bn1d
from latte.models.blocks import LorentzInceptionBlock


class BaselineDeviationEncoder(nn.Module):
    def __init__(self,
                 manifold,
                 in_channel=1,
                 features=18 * 18,
                 kernel_sizes=(9, 19, 39),
                 inception_channels=8,
                 conv_type="base",
                 batch_type="gyro",
                 fc_type="base",
                 pool_type="dirty",
                 pool_tau=1.0,
                 dropout=0,
                 bias=False,
                 num_subjects=None,
                 adaptive_bn=False,
                 bn_kwargs=None,
                 reference="block",
                 block_norm="encoder"):
        super().__init__()
        if reference not in ("block", "none", "none_matched"):
            raise ValueError(f"reference must be block/none/none_matched, got {reference!r}")
        if block_norm not in ("encoder", "block"):
            raise ValueError(f"block_norm must be encoder/block, got {block_norm!r}")
        self.reference = reference
        self.block_norm = block_norm
        self.manifold = manifold

        # adaptive_bn: per-subject affine on the branch-level GyroBN.
        bn_type = "gyro_custom_adaptive" if adaptive_bn else batch_type
        bn_kwargs = dict(bn_kwargs or {})

        inner_bn = batch_type if block_norm == "block" else None
        inner_act = nn.SiLU() if block_norm == "block" else None
        task_bottleneck = inception_channels * (2 if reference == "none_matched" else 1)

        def block(pool, bottleneck):
            return LorentzInceptionBlock(self.manifold,
                                         in_channels=in_channel,
                                         n_filters=int(features / 4),
                                         kernel_sizes=kernel_sizes,
                                         bottleneck_channels=bottleneck,
                                         activation=inner_act,
                                         conv_type=conv_type,
                                         batch_type=inner_bn,
                                         fc_type=fc_type,
                                         pool_type=pool,
                                         pool_tau=pool_tau,
                                         dropout=dropout,
                                         bias=bias)

        self.inception_block = block(pool_type, task_bottleneck)
        if reference == "block":
            self.baseline_block = block("average", inception_channels)
        elif reference == "none":
            block("average", inception_channels)

        self.att = LorentzMultiHeadAttention(manifold=self.manifold,
                                             num_features=features + 1,
                                             num_patches=1,
                                             heads=1,
                                             out_features=features + 1,
                                             fc_type=fc_type,
                                             bias=False)
        torch.rand(features + 1)

        def outer_bn():
            if bn_type is None or block_norm == "block":
                return MaybeAdaptiveBN(nn.Sequential())
            return build_lorentz_bn1d(bn_type, self.manifold, features + 1,
                                      num_subjects=num_subjects, wrap=True, **bn_kwargs)

        self.norm_layer_individual = outer_bn()
        if reference == "block":
            self.norm_layer_baseline = outer_bn()

        self.activation = nn.SiLU()

    def forward(self, x, x_sub=None):
        incepted = self.inception_block(x)

        if self.reference == "block":
            baseline = self.norm_layer_baseline(self.baseline_block(x), x_sub)
            incepted = self.norm_layer_individual(incepted, x_sub)
            # log at the reference, transport to the origin, exp back
            incepted = self.manifold.fused_logmap_transp0back(baseline, incepted)
            incepted = self.manifold.expmap0(incepted)
            incepted = self.manifold.rescale_to_max(incepted)

        incepted = self.manifold.add_time(self.activation(incepted[..., 1:]))
        return self.att(incepted)
