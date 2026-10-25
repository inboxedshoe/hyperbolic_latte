import math

import torch
import torch.nn as nn

from . import functional
from .functional import cayley_map, matrix2skew


class SPDOperator(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def expm(x): return functional.sym_expm.apply(x)

    @staticmethod
    def logm(x): return functional.sym_logm.apply(x)

    @staticmethod
    def sqrtm(x): return functional.sym_sqrtm.apply(x)

    def powm(self, x, exponent): return functional.sym_powm.apply(x, exponent)

    @staticmethod
    def matrix2sym(x): return 0.5 * (x + x.transpose(-1, -2))

    @staticmethod
    def invsqrtm(x): return functional.sym_invsqrtm.apply(x)

    @staticmethod
    def expm_sqrt(x): return functional.sym_expm_sqrt.apply(x)


class GyroSPDAttention(SPDOperator):
    """Gyro self-attention on the SPD manifold under AIM / LEM / LCM.

    The transform is a gyro homomorphism ``O X O^T`` with ``O`` orthogonal via a
    Cayley map, applied to ``X``, ``log X`` or the Cholesky-log depending on the
    metric.  Scoring and aggregation:

        A_ij = Softmax( (1 + log(1 + d(Q_i, K_j)))^{-1} )
        R_i  = WeightedFrechetMean(A_i, {V_j})
        phi(R) = sigma(B + R)
    """

    def __init__(self, n, metric="LEM", power=1.0):
        super().__init__()
        self.n, self.metric = n, metric
        self.register_buffer("power", torch.tensor(float(power)), persistent=False)

        self.q_morphism = nn.Parameter(torch.randn(n, n) * 2 - 1.0)
        self.k_morphism = nn.Parameter(torch.randn(n, n) * 2 - 1.0)
        self.v_morphism = nn.Parameter(torch.randn(n, n) * 2 - 1.0)
        self.v_bias = nn.Parameter((torch.randn(n, n) * 2 - 1.0) * 1e-3)

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

        if metric == "AIM":
            self.transform, self.dist, self.fuse = self.transform_aim, self.dist_aim, self.fuse_aim
        elif metric == "LEM":
            self.transform, self.dist, self.fuse = self.transform_lem, self.dist_lem, self.fuse_lem
        elif metric == "LCM":
            self.transform, self.dist, self.fuse = self.transform_lcm, self.dist_lcm, self.fuse_lcm
        else:
            raise NotImplementedError(f"Not implemented metric {metric}.")

    # ---- gyro homomorphism -------------------------------------------------
    @staticmethod
    def morphism(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        orth_mat = cayley_map(matrix2skew(x))
        return orth_mat @ y @ orth_mat.transpose(-1, -2)

    @staticmethod
    def chol_log(x: torch.Tensor) -> torch.Tensor:
        chol_x = torch.linalg.cholesky(x)
        diag = chol_x.diagonal(dim1=-2, dim2=-1).log()
        return chol_x.tril(-1) + chol_x.tril(-1).transpose(-1, -2) + diag.diag_embed()

    @staticmethod
    def chol_log_inv(x: torch.Tensor) -> torch.Tensor:
        l = x.tril(-1) + x.diagonal(dim1=-2, dim2=-1).exp().diag_embed()
        return l @ l.transpose(-1, -2)

    def transform_base(self, x):
        return (self.morphism(self.q_morphism, x),
                self.morphism(self.k_morphism, x),
                self.morphism(self.v_morphism, x))

    def transform_aim(self, x): return self.transform_base(x)

    def transform_lem(self, x): return self.transform_base(self.logm(x))

    def transform_lcm(self, x): return self.transform_base(self.chol_log(x))

    # ---- geodesic distances ------------------------------------------------
    @staticmethod
    def dist_lem(a, b, keepdim=False):
        return torch.norm(a - b, dim=[-1, -2], keepdim=keepdim)

    @staticmethod
    def dist_lcm(a, b, keepdim=False):
        return torch.norm(a.tril(0) - b.tril(0), dim=[-1, -2], keepdim=keepdim)

    def dist_aim(self, a, b, keepdim=False):
        a_sqrt = self.sqrtm(a)
        return 0.5 * torch.norm(self.logm(a_sqrt @ b @ a_sqrt), dim=[-1, -2], keepdim=keepdim)

    # ---- weighted Frechet mean + gyro bias + activation --------------------
    def cal_w_mean(self, W, x):
        """Closed-form weighted mean in the metric's flat coordinates.

        LEM and LCM are log-Euclidean, so the weighted Frechet mean is this
        linear combination -- no Karcher-flow iteration is needed.
        """
        W = W.unsqueeze(-1).unsqueeze(-1)
        mean = x.unsqueeze(2) * W
        return mean.sum(dim=1)

    def left_add_tan_aim(self, x, y, is_inverse: bool = False):
        x = -x if is_inverse else x
        x = self.expm_sqrt(x)
        return x @ y @ x

    def fuse_lem(self, x, weight):
        x = self.cal_w_mean(weight, x)                    # weighted Frechet mean
        x = self.matrix2sym(self.v_bias) + x              # gyro bias
        x = 1 / self.power * functional.sym_expm_power.apply(x, self.power)
        return x

    def fuse_aim(self, x, weight):
        x = self.expm(self.cal_w_mean(weight, self.logm(x)))
        x = self.left_add_tan_aim(self.matrix2sym(self.v_bias), x)
        x = 1 / self.power * self.powm(x, self.power)
        return x

    def fuse_lcm(self, x, weight):
        x = self.cal_w_mean(weight, x)
        x = self.chol_log_inv(self.matrix2sym(self.v_bias) + x)
        x = 1 / self.power * self.powm(x, self.power)
        return x

    def forward(self, x):
        q, k, v = self.transform(x)
        att_energy = self.dist(q.unsqueeze(1), k.unsqueeze(2))
        att_prob = nn.Softmax(dim=-2)(1 / (1 + torch.log(1 + att_energy))).transpose(-1, -2)
        # att_prob = nn.Softmax(dim=-2)(-self.alpha.exp() * att_energy - self.beta).transpose(-1, -2)
        return self.fuse(v, att_prob)
