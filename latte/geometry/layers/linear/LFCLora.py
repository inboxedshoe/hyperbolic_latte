import torch
import torch.nn as nn
import torch.nn.functional as F



class LorentzFullyConnectedLora(nn.Module):
    """
    Lorentz fully connected layer with subject-specific low-rank adapters (LoRA).

    Args:
        manifold: Instance of Lorentz manifold
        in_features, out_features, bias: Same as nn.Linear
        init_scale: Scale parameter for internal normalization
        learn_scale: If scale parameter should be learnable
        normalize: If internal normalization should be applied
        activation: Optional nonlinearity
        dropout: Dropout probability
        nheads: Number of heads for splitting
        num_subjects: Number of subject-specific adapters
        rank: LoRA rank (low-rank dimension)
        lora_alpha: Scaling factor for the LoRA update
    """

    def __init__(
        self,
        manifold,
        in_features,
        out_features,
        bias=False,
        init_scale=None,
        learn_scale=False,
        normalize=False,
        activation=None,
        dropout=0.0,
        nheads=1,
        num_subjects=None,
        rank=0,
        lora_alpha=1e-6
    ):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.normalize = normalize
        self.nheads = nheads
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.rank = rank
        self.lora_alpha = lora_alpha

        # Shared base weight
        self.weight = nn.Linear(self.in_features - 1, self.out_features - 1, bias=bias)

        # LoRA adapters: lora_A projects down (in -> rank), lora_B projects up (rank -> out)
        self.lora_A = nn.Parameter(torch.zeros(num_subjects + 1, self.in_features - 1, rank))
        self.lora_B = nn.Parameter(torch.zeros(num_subjects + 1, self.out_features - 1, rank))

        # Scale for internal normalization
        if init_scale is not None:
            self.scale = nn.Parameter(torch.ones(()) * init_scale, requires_grad=learn_scale)
        else:
            self.scale = nn.Parameter(torch.ones(()) * 2.3, requires_grad=learn_scale)

        # Initialization
        self.init_std = 0.02
        self.reset_parameters()
        nn.init.uniform_(self.lora_A, -self.init_std, self.init_std)
        nn.init.zeros_(self.lora_B)

    def reset_parameters(self):
        nn.init.uniform_(self.weight.weight, -self.init_std, self.init_std)
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)

    def forward(self, x, subject_ids):
        # Extract subject IDs — handle both 1D (B,) and multi-dim (B,1,1,1) inputs
        if subject_ids.ndim > 1:
            subject_ids = subject_ids[..., 0].squeeze().int()
        else:
            subject_ids = subject_ids.int()

        # Drop Lorentz time dimension -> (B, in_features-1)
        x_space = x.narrow(-1, 1, x.shape[-1] - 1)

        # Shared base linear transform
        x_base = F.linear(x_space, self.weight.weight)  # (B, out_features-1)

        # Subject-specific LoRA adapters
        batch_A = self.lora_A[subject_ids]  # (B, in_features-1, r)
        batch_B = self.lora_B[subject_ids]  # (B, out_features-1, r)

        # Low-rank LoRA update: B @ A^T @ x_space
        lora_intermediate = torch.einsum("bir,bi->br", batch_A, x_space)   # (B, r)
        lora_out = torch.einsum("bir,br->bi", batch_B, lora_intermediate)  # (B, out_features-1)
        # Combine base + LoRA
        x_space = x_base + self.lora_alpha * lora_out

        # Nonlinearity and dropout
        if self.activation is not None:
            x_space = self.activation(x_space)
        x_space = self.dropout(x_space)

        # Multi-head reshaping
        if self.nheads > 1:
            x_space = x_space.view(
                x_space.size(0),
                x_space.size(1),
                self.nheads,
                self.out_features // self.nheads,
            ).transpose(1, 2)

        # Normalization or Lorentzian time augmentation
        if self.normalize:
            scale = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp()
            square_norm = (x_space * x_space).sum(dim=-1, keepdim=True)
            mask = square_norm <= 1e-10
            square_norm[mask] = 1
            unit_length = x_space / torch.sqrt(square_norm)
            x_space = scale * unit_length
            x_time = torch.sqrt(scale**2 + self.manifold.k + 1e-5)
            x_time = x_time.masked_fill(mask, self.manifold.k.sqrt())
            mask = ~mask
            x_space = x_space * mask
            x = torch.cat([x_time, x_space], dim=-1)
        else:
            x = self.manifold.add_time(x_space)

        return x


class LorentzFullyConnectedBoostLora(nn.Module):
    """
    Lorentz fully connected layer with subject-specific low-rank adaptation via Lorentz boosts.
    
    This layer applies a composition of subject-specific Lorentz boosts.

    Args:
        manifold: Instance of Lorentz manifold
        in_features, out_features, bias: Same as nn.Linear
        init_scale: Scale parameter for internal normalization
        learn_scale: If scale parameter should be learnable
        normalize: If internal normalization should be applied
        activation: Optional nonlinearity
        dropout: Dropout probability
        num_subjects: Number of subject-specific adapters
        rank: Number of composed boosts per subject (boost rank)
        alpha: Scaling factor for boost magnitudes
    """

    def __init__(
        self,
        manifold,
        in_features,
        out_features,
        bias=False,
        init_scale=None,
        learn_scale=False,
        normalize=False,
        activation=None,
        dropout=0.0,
        num_subjects=None,
        rank=4,
        alpha=0.1
    ):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.normalize = normalize
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.rank = rank
        self.alpha = alpha
        self.num_subjects = num_subjects

        self.weight = nn.Linear(self.in_features - 1, self.out_features - 1, bias=bias)

        # Subject-specific boost direction parameters
        # Each subject has `rank` boost directions, each of dimension (out_features - 1)
        self.boost_dirs = nn.Parameter(
            torch.zeros(num_subjects + 1, rank, self.out_features - 1)
        )
        
        # Subject-specific boost magnitude
        self.boost_mags = nn.Parameter(
            torch.zeros(num_subjects + 1, rank)
        )

        if init_scale is not None:
            self.scale = nn.Parameter(torch.ones(()) * init_scale, requires_grad=learn_scale)
        else:
            self.scale = nn.Parameter(torch.ones(()) * 2.3, requires_grad=learn_scale)

        self.init_std = 0.02
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight.weight, -self.init_std, self.init_std)
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)
        nn.init.uniform_(self.boost_dirs, -self.init_std, self.init_std)
        nn.init.zeros_(self.boost_mags)

    def _apply_boost(self, x, direction, magnitude):
        x_t = x[..., 0:1]  # Time component (B, 1)
        x_s = x[..., 1:]   # Space component (B, out_features-1)
        
        # Normalize direction
        dir_norm = torch.norm(direction, dim=-1, keepdim=True).clamp(min=1e-8)
        v_unit = direction / dir_norm
        
        # Scale magnitude
        mag = magnitude.unsqueeze(-1) * self.alpha  # (B, 1)
        
        # Compute boost components
        cosh_mag = torch.cosh(mag)
        sinh_mag = torch.sinh(mag)

        vx = (v_unit * x_s).sum(dim=-1, keepdim=True)  # (B, 1)

        xt_new = x_t * cosh_mag + vx * sinh_mag
        xs_new = x_s + v_unit * ((cosh_mag - 1) * vx + x_t * sinh_mag)
        
        return torch.cat([xt_new, xs_new], dim=-1)

    def forward(self, x, subject_ids):
        # Handle both 1D (B,) and multi-dim (B,1,1,1) subject IDs
        if subject_ids.ndim > 1:
            subject_ids = subject_ids[..., 0].squeeze().int()
        else:
            subject_ids = subject_ids.int()


        x_space = x.narrow(-1, 1, x.shape[-1] - 1)
        x_space = self.weight(x_space)  # (B, out_features-1)

        # Nonlinearity and dropout (applied before adding time)
        if self.activation is not None:
            x_space = self.activation(x_space)
        x_space = self.dropout(x_space)

        x_hyp = self.manifold.add_time(x_space)  # (B, out_features)

        batch_dirs = self.boost_dirs[subject_ids]  # (B, rank, out_features-1)
        batch_mags = self.boost_mags[subject_ids]  # (B, rank)

        for r in range(self.rank):
            x_hyp = self._apply_boost(
                x_hyp,
                batch_dirs[:, r, :],  # (B, out_features-1)
                batch_mags[:, r]       # (B,)
            )

        x_hyp = self.manifold.projx(x_hyp)

        return x_hyp


