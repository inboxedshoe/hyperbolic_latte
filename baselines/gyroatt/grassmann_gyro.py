import torch
import torch.nn as nn

from .functional import (b2gr_tangent, cayley_map, gr_identity_onb,
                         matrix2dial_on, matrix2skew)


class GrassmannianOBN(nn.Module):
    def __init__(self, n: int, p: int, eps: float = 1e-8):
        super().__init__()
        self.n, self.p, self.eps = n, p, eps
        self.register_buffer('identity', gr_identity_onb(n, p))

    @staticmethod
    def qr_flipped(x: torch.Tensor) -> torch.Tensor:
        Q, R = torch.linalg.qr(x)
        flip = torch.where(torch.diagonal(R, dim1=-2, dim2=-1) < 0, -1.0, 1.0)
        return Q * flip.unsqueeze(-2)

    @staticmethod
    def dist(x: torch.Tensor, y: torch.Tensor, keepdim=False, eps=1e-3):
        s = torch.linalg.svdvals(x.transpose(-2, -1) @ y).clamp(-1 + eps, 1 - eps)
        return torch.norm(s.acos(), dim=-1, keepdim=keepdim)

    def logmap(self, x: torch.Tensor, y: torch.Tensor, is_return_tuple=False):
        """Log_x(y) = U Theta V^T, with U S V = thinSVD(x^T y) and Theta = atan(S)."""
        if torch.allclose(x, y, atol=self.eps):
            delta = y - y.detach()
            if is_return_tuple:
                eye = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
                return eye, torch.zeros_like(x[..., -1]), eye
            return delta

        Bt = torch.linalg.pinv(y.transpose(-2, -1) @ x) @ (
            y.transpose(-2, -1) - y.transpose(-2, -1) @ x @ x.transpose(-2, -1))
        u, s, vh = torch.linalg.svd(Bt.transpose(-2, -1), full_matrices=False)
        return (u, s.atan(), vh) if is_return_tuple else u * s.atan().unsqueeze(-2) @ vh

    def logmap_id(self, x, is_lower_part=False):
        if torch.allclose(x, self.identity, atol=self.eps):
            delta = x - x.detach()
            return delta[..., x.shape[-1]:, :] if is_lower_part else delta

        p = x.shape[-1]
        Bt = x[..., p:, :] @ torch.linalg.pinv(x[..., :p, :])
        u, s, vh = torch.linalg.svd(Bt, full_matrices=False)
        delta = u * torch.atan(s).unsqueeze(-2) @ vh
        if is_lower_part:
            return delta
        upper = torch.zeros(*x.shape[:-2], p, p, dtype=x.dtype, device=x.device)
        return torch.cat([upper, delta], dim=-2)

    def expmap(self, x, y):
        u, s, vh = torch.linalg.svd(y, full_matrices=False)
        return self.qr_flipped(
            ((x @ vh.transpose(-2, -1)) * s.cos().unsqueeze(-2) + u * s.sin().unsqueeze(-2)) @ vh)

    def expmap_id(self, x):
        return self.expmap(self.identity, x)

    def geodesic(self, u0: torch.Tensor, u1: torch.Tensor, t: float) -> torch.Tensor:
        u, s, vh = self.logmap(u0, u1, is_return_tuple=True)
        s_new = t * s
        return ((u0 @ vh.transpose(-2, -1)) * s_new.cos().unsqueeze(-2)
                + u * s_new.sin().unsqueeze(-2)) @ vh


class GrassmannianOBNGyro(GrassmannianOBN):
    def __init__(self, n, p, eps=1e-8):
        super().__init__(n=n, p=p, eps=eps)

    @staticmethod
    def left_addition_skew(x, y, is_inverse: bool = False):
        return cayley_map(-x if is_inverse else x) @ y

    def get_omega(self, U: torch.Tensor) -> torch.Tensor:
        return b2gr_tangent(self.logmap_id(U, is_lower_part=True))


class GyroGrAttention(GrassmannianOBNGyro):
    """Gyro self-attention on the Grassmannian."""

    def __init__(self, n, p):
        super().__init__(n, p)
        self.q_morphism_1 = nn.Parameter(torch.rand(p, p) * 2 - 1.0)
        self.q_morphism_2 = nn.Parameter(torch.rand(n - p, n - p) * 2 - 1.0)
        self.k_morphism_1 = nn.Parameter(torch.rand(p, p) * 2 - 1.0)
        self.k_morphism_2 = nn.Parameter(torch.rand(n - p, n - p) * 2 - 1.0)
        self.v_morphism_1 = nn.Parameter(torch.rand(p, p) * 2 - 1.0)
        self.v_morphism_2 = nn.Parameter(torch.rand(n - p, n - p) * 2 - 1.0)
        self.v_bias = nn.Parameter(torch.randn(p, n - p) * 2 - 1.0)

    @staticmethod
    def morphism(x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        orth_mat = matrix2dial_on(matrix2skew(x1), matrix2skew(x2))
        return orth_mat @ y

    def transform(self, x):
        return (self.morphism(self.q_morphism_1, self.q_morphism_2, x),
                self.morphism(self.k_morphism_1, self.k_morphism_2, x),
                self.morphism(self.v_morphism_1, self.v_morphism_2, x))

    def cal_w_mean(self, W, V, karcher_steps=1):
        """Weighted Frechet mean by Karcher flow."""
        output = torch.zeros_like(V)
        W = W.unsqueeze(-1).unsqueeze(-1)
        for i in range(V.shape[1]):
            output[:, i] = self.cal_w_mean_1(W[:, i], V, i, karcher_steps)
        return output

    def cal_w_mean_1(self, weight, V, i, karcher_steps=1):
        w_mean = V[:, i]
        for _ in range(karcher_steps):
            x_log = torch.zeros_like(V)
            for j in range(V.shape[1]):
                x_log[:, j, :, :] = self.logmap(w_mean, V[:, j, :, :])
            w_mean_log = torch.sum(x_log * weight, dim=1)
            w_mean = self.expmap(w_mean, w_mean_log)
        return w_mean

    def fuse(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        output = self.cal_w_mean(weight, x)
        return self.left_addition_skew(b2gr_tangent(self.v_bias), output)

    def forward(self, x):
        q, k, v = self.transform(x)
        att_energy = self.dist(q.unsqueeze(1), k.unsqueeze(2))
        att_prob = nn.Softmax(dim=-2)(1 / (1 + torch.log(1 + att_energy))).transpose(-1, -2)
        return self.fuse(v, att_prob)
