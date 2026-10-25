import torch
import torch.nn as nn
import torch.nn.functional as F

from geoopt import ManifoldParameter
from ...manifold import CustomLorentz


class LorentzPLFCCustom(nn.Module):
    """Point-to-hyperplane Fully Connected layer.

    Absorbs the time-component of the Lorentz inner product into the V matrix.
    Forward simplifies to ``y = add_time(x @ V)``
    with V columns of unit Lorentz norm.
    """

    def __init__(
        self,
        manifold: CustomLorentz,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        in_spatial = in_features - 1
        out_spatial = out_features - 1

        # Spatial direction vectors (columns = one direction per output neuron).
        self.z = nn.Parameter(torch.empty(in_spatial, out_spatial))
        # Offset / rapidity parameters.
        self.a = nn.Parameter(torch.zeros(out_spatial))

        if bias:
            self.bias = ManifoldParameter(
                self.manifold.origin(out_features), manifold=self.manifold
            )

        self._init_weights(init_std)

    # ------------------------------------------------------------------
    def _init_weights(self, std: float) -> None:
        nn.init.normal_(self.z, 0.0, std)
        nn.init.zeros_(self.a)

    # ------------------------------------------------------------------
    def _hyperplane_normals(self) -> torch.Tensor:
        """Build spacelike normal vectors V of shape (in_features, out_spatial).

        V_k = [−sinh(a_k/√k),  cosh(a_k/√k) · ẑ_k]

        where ẑ_k = z_k / ‖z_k‖_2.
        """
        sqrt_k = self.manifold.k.sqrt()
        # Normalise each column of z to a unit spatial direction.
        z_hat = F.normalize(self.z, p=2, dim=0)  # (in_spatial, out_spatial)
        ak = self.a / sqrt_k  # (out_spatial,)
        # Time row: −‖ẑ‖·sinh(a/√k) = −sinh(a/√k)  (since ‖ẑ‖=1)
        t_row = -torch.sinh(ak).unsqueeze(0)          # (1, out_spatial)
        s_rows = torch.cosh(ak).unsqueeze(0) * z_hat  # (in_spatial, out_spatial)
        return torch.cat([t_row, s_rows], dim=0)       # (in_features, out_spatial)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., in_features) — points on the Lorentz manifold.

        Returns:
            (..., out_features) — points on the Lorentz manifold.
        """
        V = self._hyperplane_normals()       # (in_features, out_spatial)
        y_s = x @ V                          # (..., out_spatial)
        y = self.manifold.add_time(y_s)      # (..., out_features)

        if self.use_bias:
            # Gyro-additive bias: y ← y ⊕ bias
            y = self.manifold.mobius_add(y, self.bias)

        return y


class LorentzPLFCScaled(LorentzPLFCCustom):
    """PLFC with a per-output learnable spatial scale.

    This subclass multiplies `y_s = x @ V` by a learnable
    positive scalar per output dim.

    Equivalent to scaling each hyperplane normal `ν_k` by a learnable factor
    `s_k > 0`.
    """

    def __init__(
        self,
        manifold,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_std: float = 0.05,
    ):
        super().__init__(manifold, in_features, out_features, bias=bias, init_std=init_std)
        # log-parameterized so scale stays positive
        self.log_scale = nn.Parameter(torch.zeros(out_features - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        V = self._hyperplane_normals()
        y_s = (x @ V) * torch.exp(self.log_scale)
        y = self.manifold.add_time(y_s)
        if self.use_bias:
            y = self.manifold.mobius_add(y, self.bias)
        return y


