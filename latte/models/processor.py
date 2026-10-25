import torch
import torch.nn as nn
import torch.nn.functional as F


class SubjectBatchNorm2d(nn.Module):
    """BatchNorm2d with shared running statistics but per-subject affine parameters.

    Normalization (mean/var) is global across all subjects — so unseen subjects
    still get normalized at eval time.  The per-subject
    adaptation is limited to the affine transform (gamma/beta), which defaults to
    identity (gamma=1, beta=0) for unseen subjects.
    """

    def __init__(self, num_features, num_subjects, momentum=0.1, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.num_subjects = num_subjects

        # Shared normalization (global running stats, no built-in affine)
        self.bn = nn.BatchNorm2d(num_features, momentum=momentum, eps=eps, affine=False)

        # Per-subject affine: (num_subjects+1, C)
        # Initialized to identity (gamma=1, beta=0) so unseen subjects pass through unchanged
        self.weight = nn.Parameter(torch.ones(num_subjects + 1, num_features))
        self.bias = nn.Parameter(torch.zeros(num_subjects + 1, num_features))

    def forward(self, x, subject_ids):
        """
        x: (B, C, H, W)
        subject_ids: (B,) int tensor; if None, falls back to the unseen slot
        (last index, identity affine).
        """
        # Shared normalization across all subjects
        x = self.bn(x)

        if subject_ids is None:
            subject_ids = torch.full(
                (x.size(0),), self.num_subjects,
                device=x.device, dtype=torch.long,
            )

        # Per-subject affine transform
        B, C = x.shape[0], x.shape[1]
        gamma = self.weight[subject_ids].view(B, C, 1, 1)
        beta = self.bias[subject_ids].view(B, C, 1, 1)
        return gamma * x + beta


class MaybeAdaptiveBNEuc(nn.Module):
    """Uniform call site for `nn.BatchNorm2d` and `SubjectBatchNorm2d`.

    Plain BNs ignore `subject_ids`; adaptive BNs
    consume them (and fall back to the unseen slot when `None`).
    """

    def __init__(self, bn: nn.Module):
        super().__init__()
        self.bn = bn
        self.adaptive = isinstance(bn, SubjectBatchNorm2d)

    def forward(self, x, subject_ids=None):
        if self.adaptive:
            return self.bn(x, subject_ids)
        return self.bn(x)


def build_eu_bn2d(num_features, num_subjects=None, adaptive=False, **bn_kwargs):
    """Construct a 2-D Euclidean BN, returning it pre-wrapped in
    `MaybeAdaptiveBNEuc` so callers always invoke `bn(x, x_sub)`.
    """
    if adaptive:
        assert num_subjects is not None, "adaptive=True requires num_subjects"
        bn = SubjectBatchNorm2d(num_features, num_subjects, **bn_kwargs)
    else:
        bn = nn.BatchNorm2d(num_features, **bn_kwargs)
    return MaybeAdaptiveBNEuc(bn)


class Conv2dLora(nn.Module):
    def __init__(self, conv_layer, rank=3, num_subjects=1, recon=True, proc=None, alpha=1.0,
                 depthwise=False):
        """
        Wraps a Conv2d layer with LoRA adapters.
        Args:
            conv_layer: existing nn.Conv2d layer
            rank: low-rank dimension K
            num_subjects: number of subjects
            depthwise: if True, the LoRA correction mirrors the depthwise structure of
                       conv_layer (R has in_c=1, forward uses groups=B*in_channels).
        """
        super().__init__()
        self.shared_conv = conv_layer
        self.rank = rank
        self.alpha = alpha
        self.num_subjects = num_subjects
        self.depthwise = depthwise

        # Q: (out_channels, rank, 1, 1)
        # R: (rank, in_channels, kH, kW)  — or (rank, 1, kH, kW) for depthwise mode
        r_in_channels = 1 if depthwise else conv_layer.in_channels
        self.Q = nn.Parameter(torch.zeros(num_subjects+1,
                                          conv_layer.out_channels,
                                          rank, 1, 1))
        self.R = nn.Parameter(torch.zeros(num_subjects+1,
                                          rank,
                                          r_in_channels,
                                          *conv_layer.kernel_size))

        nn.init.normal_(self.Q)
        nn.init.zeros_(self.R)

        # LoRA adapters: initialized to zero
        if recon:
            nn.init.normal_(self.Q)
            nn.init.zeros_(self.R)
        elif proc == False:
            nn.init.normal_(self.Q)
            nn.init.zeros_(self.R)

    def forward(self, x, subject_idx):
        """
        x: [B, C, H, W]
        subject_idx: [B] tensor of subject indices
        """
        out = self.shared_conv(x)

        Qs = self.Q[subject_idx]  # [B, out_c, rank, 1, 1]
        Rs = self.R[subject_idx]  # [B, rank, in_c or 1, kH, kW]
        lora_weight = torch.einsum("borhw,bricd->boicd", Qs, Rs)  # [B, out_c, in_c or 1, kH, kW]

        B = lora_weight.shape[0]
        out_c = lora_weight.shape[1]
        in_c = x.shape[1]
        kH, kW = lora_weight.shape[3], lora_weight.shape[4]

        x_reshaped = x.view(1, B * in_c, *x.shape[2:])  # [1, B*in_c, H, W]

        if self.depthwise:
            # Keep depthwise structure: each input channel is processed independently.
            # groups = B * in_c so each of the B*in_c channels forms its own group.
            w_reshaped = lora_weight.view(B * out_c, 1, kH, kW)
            groups = B * in_c
        else:
            w_reshaped = lora_weight.view(B * out_c, in_c, kH, kW)
            groups = B

        lora_out = F.conv2d(x_reshaped,
                            w_reshaped,
                            bias=None,
                            stride=self.shared_conv.stride,
                            padding=self.shared_conv.padding,
                            dilation=self.shared_conv.dilation,
                            groups=groups)

        lora_out = lora_out.view(B, out_c, *lora_out.shape[2:])
        return out + self.alpha * lora_out


def _glorot_weight_zero_bias(model):
    for module in model.modules():
        if hasattr(module, "weight"):
            if "norm" not in module.__class__.__name__.lower():
                nn.init.xavier_uniform_(module.weight)
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class _ChannelGroupAttention(nn.Module):
    """Channel group attention."""

    def __init__(self, in_channels, num_groups, reduction=4):
        super().__init__()
        assert in_channels % num_groups == 0
        self.num_groups = num_groups
        self.group_size = in_channels // num_groups
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.att_fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, groups=num_groups, bias=False)
        self.att_fc2 = nn.Conv2d(in_channels // reduction, num_groups, kernel_size=1, groups=num_groups, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        pooled = self.pool(x)
        att = self.att_fc2(self.relu(self.att_fc1(pooled)))  # [B, G, 1, 1]
        att = self.sigmoid(att)
        x_grouped = x.view(B, self.num_groups, self.group_size, H, W)
        out = x_grouped * att.view(B, self.num_groups, 1, 1, 1)
        return out.view(B, C, H, W)


class TCFormerPreprocessorLora(nn.Module):
    """TCFormerPreprocessor with per-subject LoRA adapters on temporal conv layers.

    LoRA is applied to the two temporal convolution stages (multi-kernel front-end
    and grouped temporal conv).

    Input:  (B, n_channels, T),  x_sub: (B,) or (B, 1, 1, T) subject indices
    Output: (B, d_model, T')
    """

    def __init__(
        self,
        n_channels: int,
        temp_kernel_lengths: tuple = (20, 32, 64),
        F1: int = 32,
        D: int = 2,
        pool_length_1: int = 8,
        pool_length_2: int = 7,
        dropout: float = 0.4,
        d_group: int = 16,
        use_group_attn: bool = True,
        num_subjects: int = 1,
        rank: int = 3,
        proc_lora_lr: float = 1.0,
        adaptive_bn: bool = False,
        dw_lora: bool = False,
    ):
        super().__init__()
        n_groups = len(temp_kernel_lengths)
        self.d_model = d_group * n_groups
        self.adaptive_bn = adaptive_bn
        self.dw_lora = dw_lora

        def _bn(num_features):
            return build_eu_bn2d(num_features, num_subjects=num_subjects, adaptive=adaptive_bn)

        # Temporal conv stage 1: per-kernel LoRA conv + per-kernel BN + padding
        self.temp_pads = nn.ModuleList([
            nn.ConstantPad2d(
                (k // 2 - 1, k // 2, 0, 0) if k % 2 == 0 else (k // 2, k // 2, 0, 0), 0
            )
            for k in temp_kernel_lengths
        ])
        self.temp_conv_loras = nn.ModuleList([
            Conv2dLora(nn.Conv2d(1, F1, (1, k), bias=False), rank=rank, num_subjects=num_subjects, alpha=proc_lora_lr)
            for k in temp_kernel_lengths
        ])
        self.temp_bns = nn.ModuleList([_bn(F1) for _ in temp_kernel_lengths])

        # Depthwise spatial conv. When dw_lora=True it gets a depthwise-compatible
        # LoRA adapter (R has in_c=1, groups=B*in_channels in the forward pass).
        F2 = F1 * n_groups * D
        _dw_base = nn.Conv2d(F1 * n_groups, F2, (n_channels, 1), bias=False, groups=F1 * n_groups)
        if dw_lora:
            self.channel_dw_conv = Conv2dLora(
                _dw_base, rank=rank, num_subjects=num_subjects, alpha=proc_lora_lr, depthwise=True
            )
        else:
            self.channel_dw_conv = _dw_base
        self.channel_dw_bn = _bn(F2)
        self.channel_dw_act = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, pool_length_1))
        self.drop1 = nn.Dropout(dropout)

        # Optional channel reduction (shared).
        self.has_channel_reduction = self.d_model != F2
        if self.has_channel_reduction:
            self.channel_reduction_conv = nn.Conv2d(
                F2, self.d_model, (1, 1), bias=False, groups=n_groups
            )
            self.channel_reduction_bn = _bn(self.d_model)
        else:
            self.channel_reduction_conv = None
            self.channel_reduction_bn = None

        # Temporal conv stage 2: LoRA conv + BN
        # base_tc2 = nn.Conv2d(self.d_model, self.d_model, (1, 8), padding="same", bias=False, groups=n_groups)
        base_tc2 = nn.Conv2d(self.d_model, self.d_model, (1, 16), padding="same", bias=False, groups=n_groups)
        self.temporal_conv2_lora = Conv2dLora(base_tc2, rank=rank, num_subjects=num_subjects, alpha=proc_lora_lr)
        self.temporal_conv2_bn = _bn(self.d_model)
        self.temporal_conv2_act = nn.ELU()

        self.use_group_attn = False if n_groups == 1 else use_group_attn
        if self.use_group_attn:
            self.group_attn = _ChannelGroupAttention(self.d_model, n_groups)

        self.pool2 = nn.AvgPool2d((1, pool_length_2))
        self.drop2 = nn.Dropout(dropout)

        _glorot_weight_zero_bias(self)

    def forward(self, x, x_sub):
        subj_idx = x_sub.int() if len(x_sub.shape) == 1 else x_sub[:, 0, 0, 0].int()

        if len(x.shape)==3:
            x = x.unsqueeze(1)  # (B, 1, C, T)
        feats = [
            bn(lora(pad(x), subj_idx), subj_idx)
            for pad, lora, bn in zip(self.temp_pads, self.temp_conv_loras, self.temp_bns)
        ]
        x = torch.cat(feats, dim=1)  # (B, F1*n_groups, C, T)

        dw_out = self.channel_dw_conv(x, subj_idx) if self.dw_lora else self.channel_dw_conv(x)
        x = self.channel_dw_act(self.channel_dw_bn(dw_out, subj_idx))
        x = self.pool1(x)
        x = self.drop1(x)

        if self.has_channel_reduction:
            x = self.channel_reduction_bn(self.channel_reduction_conv(x), subj_idx)

        x = self.temporal_conv2_act(
            self.temporal_conv2_bn(self.temporal_conv2_lora(x, subj_idx), subj_idx)
        )

        if self.use_group_attn:
            x = x + self.group_attn(x)

        x = self.pool2(x)
        x = self.drop2(x)

        return x.squeeze(2)  # (B, d_model, T')
