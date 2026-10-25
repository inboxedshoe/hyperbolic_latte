import math
from typing import Any, Callable, Tuple

import torch
from torch.autograd import Function
from torch.functional import Tensor

# Eigenvalue floor, keyed by dtype.  fp32 only.
EPS = {torch.float32: 1e-6}

UPSTREAM_EPS = {torch.float32: 1e-4}


def set_eps(mapping):
    """Replace the eigenvalue floor table in place (affects every op below)."""
    EPS.clear()
    EPS.update(mapping)


def trace(A):
    return torch.einsum("...ii->...", A)


def ensure_sym(A: Tensor) -> Tensor:
    return 0.5 * (A + A.transpose(-1, -2))


class sym_modeig:
    """Modify the eigenvalues of a batch of symmetric matrices elementwise.

    Forward and backward are hand-written, so autograd never
    differentiates through ``svd``/``eigh`` itself.
    """

    @staticmethod
    def forward(M: Tensor, fun: Callable[[Tensor], Tensor], fun_param: Tensor = None,
                ensure_symmetric: bool = False, ensure_psd: bool = False,
                decom_mode='svd') -> Tensor:
        if ensure_symmetric:
            M = ensure_sym(M)

        if decom_mode == 'eigh':
            s, U = torch.linalg.eigh(M)
        elif decom_mode == 'svd':
            U, s, _ = torch.linalg.svd(M)
        else:
            raise ValueError(f"unknown decom_mode {decom_mode!r}")

        if ensure_psd:
            s = s.clamp(min=EPS[s.dtype])

        smod = fun(s, fun_param)
        X = U @ torch.diag_embed(smod) @ U.transpose(-1, -2)
        return X, s, smod, U

    @staticmethod
    def backward(dX: Tensor, s: Tensor, smod: Tensor, U: Tensor,
                 fun_der: Callable[[Tensor], Tensor], fun_der_param: Tensor = None) -> Tensor:
        # Loewner matrix
        L_den = s[..., None] - s[..., None].transpose(-1, -2)
        is_eq = L_den.abs() < EPS[s.dtype]
        L_den[is_eq] = 1.0
        L_num_ne = smod[..., None] - smod[..., None].transpose(-1, -2)
        L_num_ne[is_eq] = 0
        sder = fun_der(s, fun_der_param)
        L_num_eq = 0.5 * (sder[..., None] + sder[..., None].transpose(-1, -2))
        L_num_eq[~is_eq] = 0
        L = (L_num_ne + L_num_eq) / L_den
        return U @ (L * (U.transpose(-1, -2) @ ensure_sym(dX) @ U)) @ U.transpose(-1, -2)


class sym_logm(Function):
    """Matrix logarithm of a batch of SPD matrices, with clamped eigenvalues."""

    @staticmethod
    def value(s: Tensor, param: Tensor = None) -> Tensor:
        return s.clamp(min=EPS[s.dtype]).log()

    @staticmethod
    def derivative(s: Tensor, param: Tensor = None) -> Tensor:
        sder = s.reciprocal()
        sder[s <= EPS[s.dtype]] = 0          # subgradient 0 on the clamped part
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric: bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_logm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_logm.derivative), None


class sym_expm(Function):
    """Matrix exponential of a batch of symmetric matrices."""

    @staticmethod
    def value(s: Tensor, param: Tensor = None) -> Tensor:
        return s.exp()

    @staticmethod
    def derivative(s: Tensor, param: Tensor = None) -> Tensor:
        return s.exp()

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric: bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_expm.value,
                                           ensure_symmetric=ensure_symmetric, decom_mode='eigh')
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_expm.derivative), None


class sym_powm(Function):
    """Matrix power of a batch of symmetric matrices."""

    @staticmethod
    def value(s: Tensor, exponent: Tensor) -> Tensor:
        return s.pow(exponent=exponent)

    @staticmethod
    def derivative(s: Tensor, exponent: Tensor) -> Tensor:
        return exponent * s.pow(exponent=exponent - 1.)

    @staticmethod
    def forward(ctx: Any, M: Tensor, exponent: Tensor, ensure_symmetric: bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_powm.value, exponent,
                                           ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U, exponent)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U, exponent = ctx.saved_tensors
        dM = sym_modeig.backward(dX, s, smod, U, sym_powm.derivative, exponent)
        dXs = (U.transpose(-1, -2) @ ensure_sym(dX) @ U).diagonal(dim1=-1, dim2=-2)
        dexp = dXs * smod * s.log()
        return dM, dexp, None


class sym_sqrtm(Function):
    """Matrix square root of a batch of SPD matrices."""

    @staticmethod
    def value(s: Tensor, param: Tensor = None) -> Tensor:
        return s.clamp(min=EPS[s.dtype]).sqrt()

    @staticmethod
    def derivative(s: Tensor, param: Tensor = None) -> Tensor:
        sder = 0.5 * s.rsqrt()
        sder[s <= EPS[s.dtype]] = 0
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric: bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_sqrtm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_sqrtm.derivative), None


class sym_expm_sqrt(Function):
    """exp(S)^{1/2}, i.e. exp(S/2), on a batch of symmetric matrices.

    ``derivative`` returns ``0.5*exp(s)`` where the true derivative of
    ``exp(s)^{1/2}`` is ``0.5*exp(s/2)``.
    """

    @staticmethod
    def value(s: Tensor, param: Tensor = None) -> Tensor:
        return s.exp().sqrt()

    @staticmethod
    def derivative(s: Tensor, param: Tensor = None) -> Tensor:
        return 0.5 * s.exp()

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric: bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_expm_sqrt.value,
                                           ensure_symmetric=ensure_symmetric, decom_mode='eigh')
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_expm_sqrt.derivative), None


class sym_expm_power(Function):
    """``exp(S)^{1/2}`` on a batch of symmetric matrices, called with a ``power``.

    Two properties of this function:

      1. ``forward`` takes ``(M, ensure_symmetric)``, but every call site is
         ``sym_expm_power.apply(x, self.power)`` -- so ``power`` lands in the
         ``ensure_symmetric`` slot and the matrix is symmetrised whenever the
         power is non-zero.
      2. The elementwise map used is ``sym_expm_sqrt.value`` (= exp(s/2)), not a
         power-parameterised one, so the ``power`` argument never reaches the
         eigenvalue map.  At ``power=0.5`` the forward value equals
         ``exp(S)^power``.
    """

    @staticmethod
    def value(s: Tensor, exponent: Tensor) -> Tensor:
        return s.exp().pow(exponent=exponent)

    @staticmethod
    def derivative(s: Tensor, exponent: Tensor) -> Tensor:
        return exponent * s.exp()

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric: bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_expm_sqrt.value,
                                           ensure_symmetric=ensure_symmetric, decom_mode='eigh')
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_expm_sqrt.derivative), None


class sym_invsqrtm(Function):
    """Inverse matrix square root of a batch of SPD matrices."""

    @staticmethod
    def value(s: Tensor, param: Tensor = None) -> Tensor:
        return s.clamp(min=EPS[s.dtype]).rsqrt()

    @staticmethod
    def derivative(s: Tensor, param: Tensor = None) -> Tensor:
        sder = -0.5 * s.pow(-1.5)
        sder[s <= EPS[s.dtype]] = 0
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric: bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_invsqrtm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_invsqrtm.derivative), None


# --------------------------------------------------------------------------
# Grassmann helpers (needed only by the PSPD composition)
# --------------------------------------------------------------------------

def calculate_k(s: torch.Tensor) -> torch.Tensor:
    """K_ij = 1 / (s_i - s_j), with non-finite entries zeroed."""
    sr, sc = s.unsqueeze(-2), s.unsqueeze(-1)
    K = 1.0 / (sc - sr)
    K[~torch.isfinite(K)] = 0
    return K


def gr_identity_onb(*shape, dtype=None, device=None) -> torch.Tensor:
    """Grassmannian identity [..., n, p]: I_p stacked on top of zeros."""
    *batch_dims, n, p = shape
    if p >= n:
        raise ValueError("p must be <= n.")
    dtype = torch.get_default_dtype() if dtype is None else dtype
    out = torch.zeros(*batch_dims, n, p, dtype=dtype, device=device)
    idx = torch.arange(p, device=device)
    out[..., idx, idx] = 1.0
    return out


def gr_identity_pp(*shape, dtype=None, device=None) -> torch.Tensor:
    """[..., n, n] matrix whose first p diagonal entries are 1."""
    *batch_dims, n, p = shape
    if p > n:
        raise ValueError("p must be <= n.")
    dtype = torch.get_default_dtype() if dtype is None else dtype
    out = torch.zeros(*batch_dims, n, n, dtype=dtype, device=device)
    idx = torch.arange(p, device=device)
    out[..., idx, idx] = 1.0
    return out


def commutator(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b - b @ a


def cayley_map(x: torch.Tensor) -> torch.Tensor:
    """Cayley transform (I - X) @ inv(I + X)."""
    Id = torch.eye(x.size(-1), dtype=x.dtype, device=x.device)
    return (Id - x) @ torch.inverse(Id + x)


def gr_pp2onb(x: torch.Tensor, p: int) -> torch.Tensor:
    return OrthMapFunction.apply(x, p)


def b2gr_tangent(x: torch.Tensor) -> torch.Tensor:
    """Embed a [..., p, q] block into a skew-symmetric [..., p+q, p+q]."""
    *batch_dims, p, q = x.shape
    a = torch.zeros(*batch_dims, p + q, p + q, device=x.device, dtype=x.dtype)
    a[..., :p, p:] = x
    a[..., p:, :p] = -torch.transpose(x, -1, -2)
    return a


def matrix2skew(x: torch.Tensor) -> torch.Tensor:
    return x - x.transpose(-1, -2)


class OrthMapFunction(Function):
    @staticmethod
    def forward(ctx, x, p):
        S, U = torch.linalg.eigh(x)
        S, idx = torch.sort(S, descending=True)
        U = torch.gather(U, -1, idx.unsqueeze(-2).expand_as(U))
        ctx.save_for_backward(U, S)
        return U[..., :p]

    @staticmethod
    def backward(ctx, grad_output):
        U, S = ctx.saved_tensors
        *batch_dims, h, w = grad_output.shape
        p = h - w
        pad_zero = torch.zeros(*batch_dims, h, p,
                               device=grad_output.device, dtype=grad_output.dtype)
        grad_output = torch.cat((grad_output, pad_zero), -1)
        Ut = U.transpose(-1, -2)
        K = calculate_k(S)
        mid_1 = K.transpose(-1, -2) * torch.matmul(Ut, grad_output)
        mid_2 = torch.matmul(U, mid_1)
        return torch.matmul(mid_2, Ut), None


def matrix2dial_on(x: torch.Tensor, y: torch.Tensor, mode="cayley") -> torch.Tensor:
    """Block-diagonal composition of two orthogonal matrices."""
    if mode == "cayley":
        x = cayley_map(x)
        y = cayley_map(y)
    elif mode == "qr":
        x, _ = torch.linalg.qr(x)
        y, _ = torch.linalg.qr(y)
    else:
        raise ValueError("Unsupported mode")
    *batch_dims, _, p = x.shape
    n = x.shape[-1] + y.shape[-1]
    out = torch.zeros(*batch_dims, n, n, device=x.device, dtype=x.dtype)
    out[..., :p, :p] = x
    out[..., p:, p:] = y
    return out


def inner_product(a, b):
    return torch.einsum("...ij,...ij->...", a, b)
