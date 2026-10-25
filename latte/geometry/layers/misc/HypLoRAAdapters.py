import math
import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.backends.cuda.preferred_linalg_library("cusolver")


def cayley_lowrank_apply(x, U, V, scale):
    """
    Apply the orthogonal matrix

        Q = (I - S)(I + S)^{-1},    S = scale * (U V^T - V U^T)

    to ``x`` without ever forming a D x D matrix.

    S is skew-symmetric by construction, so the Cayley transform Q is exactly
    orthogonal (Q^T Q = I, det Q = +1) for *any* real U, V and any scale. I + S is always
    invertible for skew S: its eigenvalues are 1 +/- i*lambda, so
    |det(I + S)| >= 1, and since S is normal the system is well conditioned
    (cond = sqrt(1 + lambda_max^2)).

    Note Q = I - 2 S (I + S)^{-1}, i.e. identity plus a correction of rank at
    most 2r.

    The Woodbury identity turns the D x D inverse into a 2r x 2r one:
        (I + P R^T)^{-1} = I - P (I_2r + R^T P)^{-1} R^T,
    with P = scale * [U, V] and R = [V, -U] so that P R^T = S.

    Args:
        x:     (B, ..., D) — vectors to rotate (no Lorentz time coordinate)
        U, V:  (B, D, r)   — per-sample low-rank factors
        scale: float       — multiplier on S (rotation magnitude)

    Returns:
        (B, ..., D) — Q x, with ||Q x|| == ||x|| up to float error.
    """
    P = torch.cat([U, V], dim=-1) * scale          # (B, D, 2r)
    R = torch.cat([V, -U], dim=-1)                 # (B, D, 2r)

    two_r = P.shape[-1]
    eye = torch.eye(two_r, device=P.device, dtype=P.dtype)
    M = eye + torch.einsum("bdk,bdj->bkj", R, P)   # I + R^T P   (B, 2r, 2r)

    # z = (I + S)^{-1} x, via a batched solve.
    t = torch.einsum("b...d,bdk->b...k", x, R)     # R^T x   (B, ..., 2r)
    rhs = t.reshape(t.shape[0], -1, two_r).transpose(1, 2)   # (B, 2r, N)
    w = torch.linalg.solve(M, rhs).transpose(1, 2).reshape(t.shape)
    z = x - torch.einsum("b...k,bdk->b...d", w, P)

    # Q x = (I - S) z
    Rz = torch.einsum("b...d,bdk->b...k", z, R)
    return z - torch.einsum("b...k,bdk->b...d", Rz, P)


def boost_coefficients(r2, alpha):
    """
    Smooth per-sample boost coefficients from the *squared* generator norm.

    For a boost generator vector w (rapidity m = alpha * ||w||, direction
    w / ||w||), the boost of a point (y_t, y_s) is

        y_t' = y_t * ch + <w, y_s> * c1
        y_s' = y_s + w * (c2 * <w, y_s> + y_t * c1)

    with   ch = cosh(m),   c1 = sinh(m)/||w||,   c2 = (cosh(m) - 1)/||w||^2.

    c1 and c2 are analytic *even* functions of ||w||, so the map w -> Boost(w)
    is smooth at w = 0 (where it is exactly the identity: c1 -> alpha,
    c2 -> alpha^2/2). This routine computes them so that both value and
    gradient are finite and fp32-accurate everywhere:

      - exact branch: cosh(m) - 1 is evaluated as 2*sinh(m/2)^2, which does
        not cancel catastrophically for small m;
      - small-||w|| branch (r2 < 1e-6): even Taylor series in r2, so no sqrt
        at zero.

    Both branches are finite for every input, so torch.where is safe for
    gradients (no NaN * 0).

    Args:
        r2:    (..., 1) squared norm ||w||^2 (non-negative)
        alpha: float — rapidity per unit generator norm

    Returns:
        (ch, c1, c2), each shaped like r2.
    """
    small = r2 < 1e-6
    r2_safe = r2.clamp_min(1e-6)
    r = torch.sqrt(r2_safe)
    m = alpha * r

    ch_e = torch.cosh(m)
    c1_e = torch.sinh(m) / r
    c2_e = 2.0 * torch.sinh(0.5 * m).pow(2) / r2_safe

    # Even Taylor series in s = (alpha*||w||)^2, nested Horner form.
    s = (alpha * alpha) * r2
    ch_t = 1.0 + 0.5 * s * (1.0 + s / 12.0)
    c1_t = alpha * (1.0 + (s / 6.0) * (1.0 + s / 20.0))
    c2_t = 0.5 * (alpha * alpha) * (1.0 + (s / 12.0) * (1.0 + s / 30.0))

    ch = torch.where(small, ch_t, ch_e)
    c1 = torch.where(small, c1_t, c1_e)
    c2 = torch.where(small, c2_t, c2_e)
    return ch, c1, c2


def apply_boost_from_generator(y, w, alpha):
    """
    Apply the exact Lorentz boost exp(K(w)) with rapidity alpha*||w||.

    y: (..., D) point on the Lorentz manifold (time first)
    w: (..., D-1) boost generator vector (unnormalised; w=0 -> identity)

    Returns (..., D), exactly on the manifold up to float error.
    """
    y_t = y[..., 0:1]
    y_s = y[..., 1:]

    r2 = (w * w).sum(dim=-1, keepdim=True)
    ch, c1, c2 = boost_coefficients(r2, alpha)

    wy = (w * y_s).sum(dim=-1, keepdim=True)
    yt_new = y_t * ch + wy * c1
    ys_new = y_s + w * (c2 * wy + y_t * c1)
    return torch.cat([yt_new, ys_new], dim=-1)


class GeodesicLoRAAdapter(nn.Module):
    """
    Hyperbolic LoRA via geodesic displacement.

    Constructs a subject-specific low-rank tangent vector at the origin from the
    current output, parallel-transports it to T_y, then expmaps onto the manifold.

    At initialisation R=0 -> tangent=0 -> output == y (exact identity).

    Parameter count: 2 * (num_subjects+1) * (features-1) * rank

    Args:
        manifold:     CustomLorentz instance shared with the preceding layer
        features:     D = out_channels or out_features (time coord included)
        rank:         low-rank bottleneck dimension
        num_subjects: table size = num_subjects + 1
        lora_scaling: tangent multiplier (default 1/rank)
    """

    def __init__(self, manifold, features, rank, num_subjects, lora_scaling=None):
        super().__init__()
        self.manifold = manifold
        self.features = features
        self.rank = rank
        self.lora_scaling = (1.0 / rank) if lora_scaling is None else lora_scaling

        N = num_subjects + 1
        D = features - 1  # space dimension

        # Q maps rank -> space; R maps space -> rank.
        # Zero-initialising R ensures zero tangent (and therefore identity) at start.
        self.Q = nn.Parameter(torch.empty(N, D, rank))
        self.R = nn.Parameter(torch.empty(N, D, rank))
        nn.init.kaiming_uniform_(self.Q, a=math.sqrt(5))
        nn.init.zeros_(self.R)

    def forward(self, y, subject_ids):
        """
        y:           (B, ..., D) — point on Lorentz manifold
        subject_ids: (B,) or (B, 1) — integer subject indices
        returns:     (B, ..., D) — point on Lorentz manifold
        """
        sid = subject_ids.view(-1).int()
        Q = self.Q[sid]   # (B, D-1, rank)
        R = self.R[sid]   # (B, D-1, rank)

        y_s = y[..., 1:]                                                         # (B, ..., D-1)
        mid = torch.einsum("b...d,bdr->b...r", y_s, R)                          # (B, ..., rank)
        delta = self.lora_scaling * torch.einsum("b...r,bdr->b...d", mid, Q)    # (B, ..., D-1)

        # Tangent at origin requires v_time = 0; pad a zero on the left.
        tangent = F.pad(delta, (1, 0))                                           # (B, ..., D)

        out = self.manifold.fused_transp0_expmap(y, tangent)
        return self.manifold.projx(out)


class HypLoRAAdapter(nn.Module):
    """
    Direct Lorentz Low-Rank Transformation (LLR).

    The operator applies the low-rank product to the
    space-like part only and recomputes the time coordinate so the result stays
    on the hyperboloid:

        LLR(BA, x^H) = ( sqrt( ||BA x_s^H||^2 + K ),  BA x_s^H )

    with x_s^H = x^H_[1:n] the space-like component and K the curvature
    constant of L_K^n = {x : <x,x>_L = -K, x_0 > 0}.

    The layer output is already a point on the manifold, so LLR is applied to
    it directly, with no exp/log projections around it.

    The transform is made per-subject and residual, M_i = I + s * B_i A_i, so
    that B=0 at initialisation gives M_i = I and the adapter is an exact
    identity.

    Parameter count: 2 * (num_subjects+1) * (features-1) * rank.

    Args:
        manifold:     CustomLorentz instance shared with the preceding layer
        features:     D = out_channels or out_features (time coord included)
        rank:         low-rank bottleneck dimension
        num_subjects: table size = num_subjects + 1
        lora_scaling: multiplier on the low-rank term (default 1/rank)
    """

    def __init__(self, manifold, features, rank, num_subjects, lora_scaling=None):
        super().__init__()
        self.manifold = manifold
        self.features = features
        self.rank = rank
        self.lora_scaling = (1.0 / rank) if lora_scaling is None else lora_scaling

        N = num_subjects + 1
        D = features - 1  # space-like dimension

        # A: D -> rank,  B: rank -> D.  B zero-init => M = I => exact identity.
        self.A = nn.Parameter(torch.empty(N, D, rank))
        self.B = nn.Parameter(torch.zeros(N, rank, D))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, y, subject_ids):
        """
        y:            (B, ..., D) point on the Lorentz manifold
        subject_ids:  (B,) long
        returns:      (B, ..., D) point on the same manifold
        """
        y_s = y[..., 1:]                                    # (B, ..., D-1)

        A = self.A[subject_ids]                             # (B, D-1, r)
        B = self.B[subject_ids]                             # (B, r, D-1)
        # Broadcast the per-sample matrices over any inner dims (e.g. time).
        for _ in range(y.dim() - 2):
            A = A.unsqueeze(1)
            B = B.unsqueeze(1)

        # s * B A y_s, computed right-to-left.
        proj = torch.einsum("...dr,...d->...r", A, y_s)     # (B, ..., r)
        delta = torch.einsum("...rd,...r->...d", B, proj)   # (B, ..., D-1)
        ys_new = y_s + self.lora_scaling * delta

        K = self.manifold.k.to(ys_new.dtype)
        yt_new = torch.sqrt((ys_new * ys_new).sum(dim=-1, keepdim=True) + K)

        return torch.cat([yt_new, ys_new], dim=-1)


