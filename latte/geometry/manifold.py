import torch
import torch.nn as nn


from latte.geometry.manifold_parent import Lorentz
from .manifold_parent import math

MIN_NORM = 1e-8

eps = {"16": 4.9 * 1e-4,
       "32": 1.2 * 1e-7}


def set_limits_mode(mode):
    """Select the radial-bound formula for every CustomLorentz (see below)."""
    if mode not in ("constant", "curvature"):
        raise ValueError(f"manifold limits must be 'constant' or 'curvature', got {mode!r}")
    CustomLorentz.limits_mode = mode


class CustomLorentz(Lorentz):
    def __init__(self, k=1.0, learnable=False, max_norm=None):
        super(CustomLorentz, self).__init__(k=k, learnable=learnable, max_norm=max_norm)
        
        self.s = nn.Parameter(torch.tensor(4), requires_grad=False)
        # r is a scaler of the maximum distance
        self.r = nn.Parameter(torch.tensor(1.2), requires_grad=False)

    #: How the radial bound used by ``rescale_to_max`` is computed.
    #:   "constant"  (default) 0.5 * arccosh(1 / (2 * eps32)) * r ~= 9.56.
    #:   "curvature" sqrt(k) * arccosh(2e3 / k) * r, which follows the learnable
    #:               curvature (~9.95 at k = 1).
    #: Set process-wide with ``set_limits_mode``.
    limits_mode = "constant"

    #: Derived properties; these keys are dropped from loaded state dicts.
    _LEGACY_LIMIT_KEYS = ("max_dist", "tanh_factor")

    @property
    def max_dist(self):
        k = self.k.detach().clamp_min(eps["32"])
        if CustomLorentz.limits_mode == "constant":
            return 0.5 * torch.arccosh(torch.tensor(1 / (2 * eps["32"]), device=k.device)) * self.r
        arg = (2e3 / k).clamp_min(1.0)
        return (torch.sqrt(k) * torch.arccosh(arg)).clamp_min(5) * self.r

    @property
    def tanh_factor(self):
        return torch.atanh(torch.tensor(0.99, device=self.k.device)) / (self.max_dist * self.s)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for key in self._LEGACY_LIMIT_KEYS:
            state_dict.pop(prefix + key, None)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def fused_logmap_transp0back(self, mean, x):
        """Fused logmap(mean, x) + transp0back(mean, result)."""
        return math.fused_logmap_transp0back(mean, x, self.k)

    def fused_transp0_expmap(self, y, v):
        """Fused transp0(y, v) + expmap(y, result)."""
        return math.fused_transp0_expmap(y, v, self.k)

    def projx(self, x, *, dim=-1):
        dn = x.size(dim) - 1
        right_ = x.narrow(dim, 1, dn)
        
        if self.max_norm:
            self.max_norm = self.max_dist
            if right_.ndim >= 2:
                right_ = torch.renorm(right_, 2, -2, self.max_norm)
            else:
                # 1-D tensor (single point): clamp spatial norm directly.
                norm = right_.norm(p=2).clamp_min(1e-8)
                right_ = right_ * (self.max_norm / norm).clamp(max=1.0)
        left_ = torch.sqrt(
            self.k + (right_ * right_).sum(dim=dim, keepdim=True)
        )
        x = torch.cat((left_, right_), dim=dim)
        return x

    def rescale_to_max(self, hyperbolic_vector):
        """Fused single-pass using sinh ratio."""
        x_norm = self.dist0(hyperbolic_vector, dim=-1, keepdim=True)
        sqrt_k = torch.sqrt(self.k)

        # Target norm via tanh capping
        new_norms = self.max_dist * torch.tanh(self.tanh_factor.to(device=self.k.device) * x_norm)

        # Scale space components by sinh(new/sqrt_k) / sinh(old/sqrt_k)
        old_sinh = torch.sinh(x_norm / sqrt_k).clamp_min(MIN_NORM)
        new_sinh = torch.sinh(new_norms / sqrt_k)
        ratio = new_sinh / old_sinh

        space = hyperbolic_vector[..., 1:] * ratio
        return self.add_time(space)

    def rescale_to_max_euclid(self, euclid_vector):
        x_norm = torch.norm(euclid_vector, dim=-1, keepdim=True)
        new_norms = self.max_dist * torch.tanh(self.tanh_factor.to(device=self.k.device) * x_norm)
        return new_norms * euclid_vector / (x_norm + MIN_NORM)

    def direct_concat(self, xs: torch.Tensor):
        """
        Perform Lorentz direct concatenation.

        Args:
            xs: list of tensors to concatenate. Assumes the last dimension is the dimension along which the tensor lies on the manifold, and the second last dimension is the dimension to concatenate over.


        """
        time_sq_sum = ((xs[..., 0]) ** 2).sum(dim=-1, keepdim=True)
        sqrt_arg = time_sq_sum - (xs.shape[-2] - 1) * self.k
        eps = 1e-7
        time = torch.sqrt(torch.clamp(sqrt_arg, min=eps))
        space = xs[..., 1:].reshape(*xs.shape[:-2], -1)
        out = torch.cat([time, space], dim=-1)
        return out

    def sqdist(self, x, y, dim=-1):
        """ Squared Lorentzian distance. """
        return (-2 * self.k - 2 * math.inner(x, y, keepdim=False, dim=dim)).clamp_min(0)

    def add_time(self, space):
        """ Concatenates time component to given space component. """
        time = self.calc_time(space)
        return torch.cat([time, space], dim=-1)

    def calc_time(self, space):
        """ Calculates time component from given space component. """
        return torch.sqrt(torch.norm(space, dim=-1, keepdim=True)**2+self.k)

    def centroid(self, x, w=None, eps=1e-8, dim=-2):
        """ Lorentzian centroid. """
        if w is not None:
            avg = w.matmul(x)
        else:
            avg = x.mean(dim=dim)

        denom = (-self.inner(avg, avg, keepdim=True))
        denom = denom.abs().clamp_min(eps).sqrt()

        centroid = torch.sqrt(self.k) * avg / denom

        return centroid

    def csqdist(self, x, y):
        """ Squared Lorentzian distance between all points. """
        return -2*self.k - 2 * self.cinner(x, y)

    def cinner(self, x, y):
        """ Lorentzian inner product between all points. """
        x = x.clone()
        x.narrow(-1, 0, 1).mul_(-1)
        return x @ y.transpose(-1, -2)

    def mobius_add(self, x, y):
        u = self.logmap0(y)
        v = self.transp0(x, u)
        return self.expmap(x, v)

    def lorentz_activation(self, x: torch.Tensor, activation, add_time: bool=True) -> torch.Tensor:
        """ Implements activation directly on the manifold. """
        x = activation(x.narrow(-1, 1, x.shape[-1] - 1))
        if add_time:
            x = self.add_time(x)
        return x
