import torch.jit
from typing import Tuple, Optional


EXP_MAX_NORM = 100

def arcosh(x: torch.Tensor):
    # fp32 throughout. (x - 1)(x + 1) avoids the cancellation in x^2 - 1 near
    # x = 1.
    x = x.clamp_min(1.0)
    z = torch.sqrt(torch.clamp_min((x - 1.0) * (x + 1.0), 1e-15))
    return torch.log(x + z)

def inner(u, v, *, keepdim=False, dim=-1):
    r"""
    Minkowski inner product.

    .. math::
        \langle\mathbf{u}, \mathbf{v}\rangle_{\mathcal{L}}:=-u_{0} v_{0}+u_{1} v_{1}+\ldots+u_{d} v_{d}

    Parameters
    ----------
    u : tensor
        vector in ambient space
    v : tensor
        vector in ambient space
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        inner product
    """
    return _inner(u, v, keepdim=keepdim, dim=dim)


# @torch.compile
def _inner(u, v, keepdim: bool = False, dim: int = -1):
    d = u.size(dim) - 1
    uv = u * v
    if keepdim is False:
        return -uv.narrow(dim, 0, 1).sum(dim=dim, keepdim=False) + uv.narrow(
            dim, 1, d
        ).sum(dim=dim, keepdim=False)
    else:
        return torch.cat((-uv.narrow(dim, 0, 1), uv.narrow(dim, 1, d)), dim=dim).sum(
            dim=dim, keepdim=True
        )


# def _inner(u, v, keepdim: bool = False, dim: int = -1):
#     uv = u * v
#
#     if keepdim is False:
#         return uv.sum(dim=dim) - 2 * uv.narrow(dim, 0, 1).sum(dim=dim)
#     else:
#         return uv.sum(dim=dim) - 2 * uv.narrow(dim, 0, 1).sum(dim=dim).squeeze()


def inner0(v, *, k, keepdim=False, dim=-1):
    r"""
    Minkowski inner product with zero vector.

    Parameters
    ----------
    v : tensor
        vector in ambient space
    k : tensor
        manifold negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        inner product
    """
    return _inner0(v, k=k, keepdim=keepdim, dim=dim)


# @torch.compile
def _inner0(v, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    res = -v.narrow(dim, 0, 1) * torch.sqrt(k)
    if keepdim is False:
        res = res.squeeze(dim)
    return res


def dist(x, y, *, k, keepdim=False, dim=-1):
    r"""
    Compute geodesic distance on the Hyperboloid.

    .. math::

        d_{\mathcal{L}}^{k}(\mathbf{x}, \mathbf{y})=\sqrt{k} \operatorname{arcosh}\left(-\frac{\langle\mathbf{x}, \mathbf{y}\rangle_{\mathcal{L}}}{k}\right)

    Parameters
    ----------
    x : tensor
        point on Hyperboloid
    y : tensor
        point on Hyperboloid
    k : tensor
        manifold negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        geodesic distance between :math:`x` and :math:`y`
    """
    return _dist(x, y, k=k, keepdim=keepdim, dim=dim)


# @torch.compile
def _dist(x, y, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    d = -_inner(x, y, dim=dim, keepdim=keepdim)
    #print(arcosh(d / k).isnan().sum())
    return (torch.sqrt(k) * arcosh(d / k))


def dist0(x, *, k, keepdim=False, dim=-1):
    r"""
    Compute geodesic distance on the Hyperboloid to zero point.

    .. math::

    Parameters
    ----------
    x : tensor
        point on Hyperboloid
    k : tensor
        manifold negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        geodesic distance between :math:`x` and zero point
    """
    return _dist0(x, k=k, keepdim=keepdim, dim=dim)


# @torch.compile
def _dist0(x, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    d = -_inner0(x, k=k, dim=dim, keepdim=keepdim)
    return torch.sqrt(k) * arcosh(d / k)


def project(x, *, k, dim=-1):
    r"""
    Projection on the Hyperboloid.

    .. math::

        \Pi_{\mathbb{R}^{d+1} \rightarrow \mathbb{H}^{d, 1}}(\mathbf{x}):=\left(\sqrt{k+\left\|\mathbf{x}_{1: d}\right\|_{2}^{2}}, \mathbf{x}_{1: d}\right)

    Parameters
    ----------
    x: tensor
        point in Rn
    k: tensor
        hyperboloid negative curvature
    dim : int
        reduction dimension to compute norm

    Returns
    -------
    tensor
        projected vector on the manifold
    """
    return _project(x, k=k, dim=dim)


def project(x, *, k, dim=-1, max_norm=None):
    r"""
    Projection on the Hyperboloid.

    .. math::

        \Pi_{\mathbb{R}^{d+1} \rightarrow \mathbb{H}^{d, 1}}(\mathbf{x}):=\left(\sqrt{k+\left\|\mathbf{x}_{1: d}\right\|_{2}^{2}}, \mathbf{x}_{1: d}\right)

    Parameters
    ----------
    x: tensor
        point in Rn
    k: tensor
        hyperboloid negative curvature
    dim : int
        reduction dimension to compute norm

    Returns
    -------
    tensor
        projected vector on the manifold
    """
    return _project(x, k=k, dim=dim, max_norm=max_norm if max_norm is not None else 0)


def _project(x, k: torch.Tensor, dim: int = -1, max_norm: float = 1.5):
    dn = x.size(dim) - 1
    right_ = x.narrow(dim, 1, dn)
    if max_norm:
        right_ = torch.renorm(right_, 2, -2, max_norm)
    left_ = torch.sqrt(
        k + (right_ * right_).sum(dim=dim, keepdim=True)
    )
    x = torch.cat((left_, right_), dim=dim)
    return x


def project_polar(x, *, k, dim=-1):
    r"""
    Projection on the Hyperboloid from polar coordinates.

    ... math::
        \pi((\mathbf{d}, r))=(\sqrt{k} \sinh (r/\sqrt{k}) \mathbf{d}, \cosh (r / \sqrt{k}))

    Parameters
    ----------
    x: tensor
        point in Rn
    k: tensor
        hyperboloid negative curvature
    dim : int
        reduction dimension to compute norm

    Returns
    -------
    tensor
        projected vector on the manifold
    """
    return _project_polar(x, k=k, dim=dim)


# @torch.compile
def _project_polar(x, k: torch.Tensor, dim: int = -1):
    dn = x.size(dim) - 1
    d = x.narrow(dim, 0, dn)
    r = x.narrow(dim, -1, 1)
    res = torch.cat(
        (
            torch.cosh(r / torch.sqrt(k)),
            torch.sqrt(k) * torch.sinh(r / torch.sqrt(k)) * d,
        ),
        dim=dim,
    )
    return res


def project_u(x, v, *, k, dim=-1):
    r"""
    Projection of the vector on the tangent space.

    ... math::

        \Pi_{\mathbb{R}^{d+1} \rightarrow \mathcal{T}_{\mathbf{x}} \mathbb{H}^{d, 1}(\mathbf{v})}:=\mathbf{v}+\langle\mathbf{x}, \mathbf{v}\rangle_{\mathcal{L}} \mathbf{x} / k

    Parameters
    ----------
    x: tensor
        point on the Hyperboloid
    v: tensor
        vector in Rn
    k: tensor
        hyperboloid negative curvature
    dim : int
        reduction dimension to compute norm

    Returns
    -------
    tensor
        projected vector on the manifold
    """
    return _project_u(x, v, k=k, dim=dim)


# @torch.compile
def _project_u(x, v, k: torch.Tensor, dim: int = -1):
    return v.addcmul(_inner(x, v, dim=dim, keepdim=True), x / k)


def norm(u, *, keepdim=False, dim=-1):
    r"""
    Compute vector norm on the tangent space w.r.t Riemannian metric on the Hyperboloid.

    .. math::

        \|\mathbf{v}\|_{\mathcal{L}}=\sqrt{\langle\mathbf{v}, \mathbf{v}\rangle_{\mathcal{L}}}

    Parameters
    ----------
    u : tensor
        tangent vector on Hyperboloid
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        norm of vector
    """
    return _norm(u, keepdim=keepdim, dim=dim)


# @torch.compile
def _norm(u, keepdim: bool = False, dim: int = -1):
    return torch.sqrt(_inner(u, u, keepdim=keepdim) + 1e-8)


def expmap(x, u, *, k, dim=-1):
    r"""
    Compute exponential map on the Hyperboloid.

    .. math::

        \exp _{\mathbf{x}}^{k}(\mathbf{v})=\cosh \left(\frac{\|\mathbf{v}\|_{\mathcal{L}}}{\sqrt{k}}\right) \mathbf{x}+\sqrt{k} \sinh \left(\frac{\|\mathbf{v}\|_{\mathcal{L}}}{\sqrt{k}}\right) \frac{\mathbf{v}}{\|\mathbf{v}\|_{\mathcal{L}}}


    Parameters
    ----------
    x : tensor
        point on Hyperboloid
    u : tensor
        unit speed vector on Hyperboloid
    k: tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        :math:`\gamma_{x, u}(1)` end point
    """
    return _expmap(x, u, k=k, dim=dim)


# @torch.compile
def _expmap(x, u, k: torch.Tensor, dim: int = -1):
    nomin = _norm(u, keepdim=True, dim=dim)

    u = u / nomin

    nomin = (nomin / torch.sqrt(k)).clamp_max(EXP_MAX_NORM)

    p = (
        torch.cosh(nomin) * x
        + torch.sqrt(k) * torch.sinh(nomin) * u
    )
    return p


def expmap_scaled_logmap0(y, s, k, dim: int = -1):
    # Step 1: Compute the inner product and the hyperbolic distance
    in_ = _inner0(y, k, keepdim=True, dim=dim)  # Inner product in the hyperbolic space
    dist_ = torch.sqrt(k) * torch.arccosh(-in_ / k)  # Hyperbolic distance from the origin

    # Step 2: Compute the nomin (numerator) of logmap0
    nomin_ = 1.0 / k * in_ * torch.sqrt(k)
    dn = y.size(dim) - 1
    nomin = torch.cat((nomin_ + y.narrow(dim, 0, 1), y.narrow(dim, 1, dn)), dim)
    denom = _norm(nomin, keepdim=True)

    # Step 3: Compute the unit vector direction of logmap0
    unit_vector = nomin / denom

    # Step 4: Compute the scaled distance
    scaled_dist = s * dist_

    # Step 5: Compute the left and right components of the exponential map
    cosh_term = torch.cosh(scaled_dist / torch.sqrt(k)) * torch.sqrt(k)
    sinh_term = torch.sinh(scaled_dist / torch.sqrt(k)) * unit_vector

    # Step 6: Construct the result of the expmap0
    dn = sinh_term.size(dim) - 1
    p = torch.cat((cosh_term + sinh_term.narrow(dim, 0, 1), sinh_term.narrow(dim, 1, dn)), dim)
    return p


def expmap_scaled_logmap(x, y, s, k, dim: int = -1):
    in_ = _inner(x, y, keepdim=True)
    dist_ = s * torch.sqrt(k) * arcosh(-in_ / k)

    nomin = y + (1.0 / k) * in_ * x
    denom = _norm(nomin, keepdim=True)

    direction = nomin / denom

    p = (
        torch.cosh(dist_ / torch.sqrt(k)) * x
        + torch.sqrt(k) * torch.sinh(dist_ / torch.sqrt(k)) * direction
    )
    return p


def optimized_recenter(x, y, m, s, k, dim=-1):
    sqrt_k = torch.sqrt(k)
    d = x.size(dim) - 1

    # Helper to compute the inner product directly
    def inner_product(u, v):
        uv = u * v
        return -uv[..., 0:1].sum(dim=dim, keepdim=True) + uv[..., 1:].sum(dim=dim, keepdim=True)

    # Step 1: Compute inner products
    inner_xm = inner_product(x, m)
    inner_xy = inner_product(x, y)
    inner_yx = inner_product(y, x)

    # Step 2: Compute distances using arcosh
    dist_xm = sqrt_k * arcosh(-inner_xm / k)
    dist_xy = sqrt_k * arcosh(-inner_xy / k)
    dist_yx = sqrt_k * arcosh(-inner_yx / k)

    # Step 3: Compute nominators for logmaps
    nomin_m = m + (inner_xm / k) * x
    nomin_y = y + (inner_xy / k) * x
    nomin_yx = x + (inner_yx / k) * y

    # Step 4: Compute norms and clamp for stability
    norm_m_sq = inner_product(nomin_m, nomin_m).clamp_min(1e-12)
    norm_y_sq = inner_product(nomin_y, nomin_y).clamp_min(1e-12)
    norm_yx_sq = inner_product(nomin_yx, nomin_yx).clamp_min(1e-12)

    # Step 5: Compute logmap vectors and scale logmap_xm by s
    logmap_xm = s * (dist_xm / torch.sqrt(norm_m_sq)) * nomin_m
    logmap_xy = (dist_xy / torch.sqrt(norm_y_sq)) * nomin_y
    logmap_yx = (dist_yx / torch.sqrt(norm_yx_sq)) * nomin_yx

    # Step 6: Parallel transport the scaled logmap_xm from x to y
    inner_lmap_xm = inner_product(logmap_xy, logmap_xm)
    norm_lmap_xy_sq = inner_product(logmap_xy, logmap_xy).clamp_min(1e-12)
    transport_factor = inner_lmap_xm / norm_lmap_xy_sq
    transported_u = logmap_xm - transport_factor * (logmap_xy + logmap_yx)

    # Step 7: Exponential map at y
    norm_transported_u_sq = inner_product(transported_u, transported_u).clamp_min(1e-12)
    norm_transported_u = torch.sqrt(norm_transported_u_sq)
    unit_transported_u = transported_u / norm_transported_u
    norm_clamped = (norm_transported_u / sqrt_k).clamp_max(EXP_MAX_NORM)

    # Step 8: Compute the result using the exponential map formula
    result = (
            torch.cosh(norm_clamped) * y
            + sqrt_k * torch.sinh(norm_clamped) * unit_transported_u
    )
    return result



def expmap0(u, *, k, dim=-1):
    r"""
    Compute exponential map for Hyperboloid from :math:`0`.

    Parameters
    ----------
    u : tensor
        speed vector on Hyperboloid
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        :math:`\gamma_{0, u}(1)` end point
    """
    return _expmap0(u, k, dim=dim)


# @torch.compile
def _expmap0(u, k: torch.Tensor, dim: int = -1):
    nomin = _norm(u, keepdim=True, dim=dim)
    u = u / nomin
    nomin = (nomin / torch.sqrt(k)).clamp_max(EXP_MAX_NORM)
    l_v = torch.cosh(nomin) * torch.sqrt(k)
    r_v = torch.sqrt(k) * torch.sinh(nomin) * u
    dn = r_v.size(dim) - 1
    p = torch.cat((l_v + r_v.narrow(dim, 0, 1), r_v.narrow(dim, 1, dn)), dim)
    return p


def logmap(x, y, *, k, dim=-1):
    r"""
    Compute logarithmic map for two points :math:`x` and :math:`y` on the manifold.

    .. math::

        \log _{\mathbf{x}}^{k}(\mathbf{y})=d_{\mathcal{L}}^{k}(\mathbf{x}, \mathbf{y})
            \frac{\mathbf{y}+\frac{1}{k}\langle\mathbf{x},
            \mathbf{y}\rangle_{\mathcal{L}} \mathbf{x}}{\left\|
            \mathbf{y}+\frac{1}{k}\langle\mathbf{x},
            \mathbf{y}\rangle_{\mathcal{L}} \mathbf{x}\right\|_{\mathcal{L}}}

    The result of Logarithmic map is a vector such that

    .. math::

        y = \operatorname{Exp}^c_x(\operatorname{Log}^c_x(y))


    Parameters
    ----------
    x : tensor
        starting point on Hyperboloid
    y : tensor
        target point on Hyperboloid
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector that transports :math:`x` to :math:`y`
    """
    return _logmap(x, y, k=k, dim=dim)


# @torch.compile
def _logmap(x, y, k, dim: int = -1):
    in_ = _inner(x, y, keepdim=True)
    dist_ = torch.sqrt(k) * arcosh(-in_ / k)

    nomin = y + 1.0 / k * in_ * x
    denom = _norm(nomin, keepdim=True)
    return dist_ * nomin / denom


def math_check_point_on_manifold(x: torch.Tensor, k, *, atol=1e-5, rtol=1e-5, dim=-1
) -> Tuple[bool, Optional[str]]:
    dn = x.size(dim) - 1
    x = x**2
    quad_form = -x.narrow(dim, 0, 1) + x.narrow(dim, 1, dn).sum(
        dim=dim, keepdim=True
    )
    ok = torch.allclose(quad_form, -k, atol=atol, rtol=rtol)
    if not ok:
        reason = f"'x' minkowski quadratic form is not equal to {-k.item()}"
    else:
        reason = None
    return ok, reason

def math_check_vector_on_tangent(x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5, dim=-1
) -> Tuple[bool, Optional[str]]:
    inner_ = inner(u, x, dim=dim)
    ok = torch.allclose(inner_, torch.zeros(1, device=inner_.device), atol=atol, rtol=rtol)
    if not ok:
        reason = "Minkowski inner product is not equal to zero"
    else:
        reason = None
    return ok, reason


def logmap0(y, *, k, dim=-1):
    r"""
    Compute logarithmic map for :math:`y` from :math:`0` on the manifold.

    Parameters
    ----------
    y : tensor
        target point on Hyperboloid
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector that transports :math:`0` to :math:`y`
    """
    return _logmap0(y, k=k, dim=dim)


# @torch.compile
def _logmap0(y, k, dim: int = -1):
    in_ = _inner0(y, k=k, keepdim=True)
    dist_ = torch.sqrt(k) * arcosh(-in_ / k)

    nomin_ = 1.0 / k * in_ * torch.sqrt(k)
    dn = y.size(dim) - 1
    nomin = torch.cat((nomin_ + y.narrow(dim, 0, 1), y.narrow(dim, 1, dn)), dim)
    denom = _norm(nomin, keepdim=True)
    return dist_ * nomin / denom


def logmap0back(x, *, k, dim=-1):
    r"""
    Compute logarithmic map for :math:`0` from :math:`x` on the manifold.

    Parameters
    ----------
    x : tensor
        target point on Hyperboloid
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector that transports :math:`0` to :math:`y`
    """
    return _logmap0back(x, k=k, dim=dim)


# @torch.compile
def _logmap0back(x, k, dim: int = -1):
    in_ = _inner0(x, k=k, keepdim=True)

    dist_ = torch.sqrt(k) * arcosh(-in_ / k)
    nomin_ = 1.0 / k * in_ * x
    dn = nomin_.size(dim) - 1
    nomin = torch.cat(
        (nomin_.narrow(dim, 0, 1) + torch.sqrt(k), nomin_.narrow(dim, 1, dn)), dim
    )
    denom = _norm(nomin, keepdim=True)
    return dist_ * nomin / denom


def egrad2rgrad(x, grad, *, k, dim=-1):
    r"""
    Translate Euclidean gradient to Riemannian gradient on tangent space of :math:`x`.

    .. math::

        \Pi_{\mathbb{R}^{d+1} \rightarrow \mathcal{T}_{\mathbf{x}} \mathbb{H}^{d, k}(\mathbf{v})}:=\mathbf{v}+\langle\mathbf{x}, \mathbf{v}\rangle_{\mathcal{L}} \frac{\mathbf{x}}{k}

    Parameters
    ----------
    x : tensor
        point on the Hyperboloid
    grad : tensor
        Euclidean gradient for :math:`x`
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Riemannian gradient :math:`u\in `
    """
    return _egrad2rgrad(x, grad, k=k, dim=dim)


# @torch.compile
def _egrad2rgrad(x, grad, k, dim: int = -1):
    grad.narrow(-1, 0, 1).mul_(-1)
    grad = grad.addcmul(_inner(x, grad, dim=dim, keepdim=True), x / k)
    return grad


def parallel_transport(x, y, v, *, k, dim=-1):
    r"""
    Perform parallel transport on the Hyperboloid.

    Parameters
    ----------
    x : tensor
        starting point
    y : tensor
        end point
    v : tensor
        tangent vector to be transported
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        transported vector
    """
    return _custom_parallel_transport(x, y, v, k=k, dim=dim)


# @torch.compile
def _parallel_transport(x, y, v, k, dim: int = -1):
    lmap = _logmap(x, y, k=k, dim=dim)
    nom = _inner(lmap, v, keepdim=True)
    denom = _dist(x, y, k=k, dim=dim, keepdim=True) ** 2
    p = v - nom / denom * (lmap + _logmap(y, x, k=k, dim=dim))
    return p

# @torch.compile
def _custom_parallel_transport (x,y,v,k, dim=-1):
    inner_xy = _inner(x, y, keepdim=True)
    inner_yv = _inner(y, v, keepdim=True)

    denom = (k - inner_xy).clamp_min(1e-8)

    return v + (inner_yv/denom)*(x+y)

def parallel_transport0(y, v, *, k, dim=-1):
    r"""
    Perform parallel transport from zero point.

    Parameters
    ----------
    y : tensor
        end point
    v : tensor
        tangent vector to be transported
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        transported vector
    """
    return _custom_parallel_transport0(y, v, k=k, dim=dim)


# @torch.compile
def _parallel_transport0(y, v, k, dim: int = -1):
    lmap = _logmap0(y, k=k, dim=dim)
    nom = _inner(lmap, v, keepdim=True)
    denom = _dist0(y, k=k, dim=dim, keepdim=True) ** 2
    p = v - nom / denom * (lmap + _logmap0back(y, k=k, dim=dim))
    return p

# @torch.compile
def _custom_parallel_transport0(y, v, k, dim):
    inner_xy = _inner0(y, k=k, keepdim=True)
    inner_yv = _inner(y, v, keepdim=True)

    denom = (k - inner_xy).clamp_min(1e-8)

    zero_point = torch.zeros_like(y)
    zero_point[..., 0] = torch.sqrt(k)

    return v + (inner_yv / denom) * (zero_point + y)


def parallel_transport0back(x, v, *, k, dim: int = -1):
    r"""
    Perform parallel transport to the zero point.

    Special case parallel transport with last point at zero.

    Parameters
    ----------
    x : tensor
        target point
    v : tensor
        vector to be transported
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
    """
    return _custom_parallel_transport0back(x, v, k=k, dim=dim)


# @torch.compile
def _parallel_transport0back(x, v, k, dim: int = -1):
    lmap = _logmap0back(x, k=k, dim=dim)
    nom = _inner(lmap, v, keepdim=True)
    denom = _dist0(x, k=k, dim=dim, keepdim=True) ** 2
    p = v - nom / denom * (lmap + _logmap0(x, k=k, dim=dim))
    return p

# @torch.compile
def _custom_parallel_transport0back(x, v, k, dim):
    inner_xy = _inner0(x, k=k, keepdim=True)
    inner_yv = _inner0(v, k=k, keepdim=True)

    denom = k-inner_xy

    zero_point = torch.zeros_like(x)
    zero_point[..., 0] = torch.sqrt(k)

    return v + (inner_yv / denom) * (zero_point + x)

# @torch.compile
def logmap_then_transp0back(x, y, k, dim: int = -1):

    # dist(x,y)
    d = -_inner(x, y, dim=dim, keepdim=True)
    dist_ = arcosh(d)
    ##########################################
    nomin = y + d * x
    denom = _norm(nomin, keepdim=True)

    v= dist_ * nomin / denom

    # dist0
    d = -_inner0(x, k=k, dim=dim, keepdim=True)
    dist_ = arcosh(d)
    #############################################
    nomin_ = d * x
    dn = nomin_.size(dim) - 1
    nomin = torch.cat(
        (nomin_.narrow(dim, 0, 1), nomin_.narrow(dim, 1, dn)), dim
    )
    denom = _norm(nomin, keepdim=True)
    lmap = dist_ * nomin / denom


    nom = _inner(lmap, v, keepdim=True)
    denom = dist_ ** 2
    p = v - nom / denom * (lmap + _logmap0(x, k=k, dim=dim))

    return p


def fused_logmap_transp0back(x, y, k, dim: int = -1):
    """
    Fused logmap(x, y) followed by transp0back(x, v).
    Equivalent to:  transp0back(x, logmap(x, y))
    but shares intermediate inner products to reduce computation.
    """
    # --- logmap(x, y) ---
    in_xy = _inner(x, y, keepdim=True)
    dist_xy = torch.sqrt(k) * arcosh(-in_xy / k)
    nomin = y + (1.0 / k) * in_xy * x
    denom = _norm(nomin, keepdim=True)
    v = dist_xy * nomin / denom  # this is logmap(x, y), tangent at x

    # --- custom_parallel_transport0back(x, v) ---
    # Transport v from T_x to T_origin using the formula:
    #   v + (inner(origin, v) / (k - inner(origin, x))) * (origin + x)
    # inner(origin, x) = -sqrt(k) * x[..., 0]
    # inner(origin, v) = -sqrt(k) * v[..., 0]
    sqrt_k = torch.sqrt(k)
    inner_0x = -sqrt_k * x.narrow(dim, 0, 1)  # keepdim via narrow
    inner_0v = -sqrt_k * v.narrow(dim, 0, 1)

    transport_coeff = inner_0v / (k - inner_0x)

    # origin + x: origin is [sqrt(k), 0, 0, ...], so just add sqrt(k) to x[..., 0]
    dn = x.size(dim) - 1
    ox = torch.cat((x.narrow(dim, 0, 1) + sqrt_k, x.narrow(dim, 1, dn)), dim)

    return v + transport_coeff * ox


def fused_transp0_expmap(y, v, k, dim: int = -1):
    """
    Fused transp0(y, v) followed by expmap(y, transported_v).
    Equivalent to:  expmap(y, transp0(y, v))
    but shares intermediate computations.
    """
    # --- custom_parallel_transport0(y, v) ---
    # Transport v from T_origin to T_y using the formula:
    #   v + (inner(y, v) / (k - inner(origin, y))) * (origin + y)
    sqrt_k = torch.sqrt(k)
    inner_0y = -sqrt_k * y.narrow(dim, 0, 1)
    inner_yv = _inner(y, v, keepdim=True)

    transport_coeff = inner_yv / (k - inner_0y)

    dn = y.size(dim) - 1
    oy = torch.cat((y.narrow(dim, 0, 1) + sqrt_k, y.narrow(dim, 1, dn)), dim)

    transported = v + transport_coeff * oy

    # --- expmap(y, transported) ---
    nomin = _norm(transported, keepdim=True, dim=dim)
    u_hat = transported / nomin
    nomin = (nomin / sqrt_k).clamp_max(EXP_MAX_NORM)

    p = torch.cosh(nomin) * y + sqrt_k * torch.sinh(nomin) * u_hat
    return p


def fused_transp_expmap(x, y, v, k, dim: int = -1):
    """
    Fused transp(x, y, v) followed by expmap(y, transported_v).
    Equivalent to:  expmap(y, transp(x, y, v))
    Uses the efficient custom parallel transport formula.
    """
    # --- custom_parallel_transport(x, y, v) ---
    inner_xy = _inner(x, y, keepdim=True)
    inner_yv = _inner(y, v, keepdim=True)

    transported = v + (inner_yv / (k - inner_xy)) * (x + y)

    # --- expmap(y, transported) ---
    sqrt_k = torch.sqrt(k)
    nomin = _norm(transported, keepdim=True, dim=dim)
    u_hat = transported / nomin
    nomin = (nomin / sqrt_k).clamp_max(EXP_MAX_NORM)

    p = torch.cosh(nomin) * y + sqrt_k * torch.sinh(nomin) * u_hat
    return p


def geodesic_unit(t, x, u, *, k):
    r"""
    Compute unit speed geodesic at time :math:`t` starting from :math:`x` with direction :math:`u/\|u\|_x`.

    .. math::

        \gamma_{\mathbf{x} \rightarrow \mathbf{u}}^{k}(t)=\cosh \left(\frac{t}{\sqrt{k}}\right) \mathbf{x}+\sqrt{k} \sinh \left(\frac{t}{\sqrt{k}}\right) \mathbf{u}

    Parameters
    ----------
    t : tensor
        travelling time
    x : tensor
        initial point
    u : tensor
        unit direction vector
    k : tensor
        manifold negative curvature

    Returns
    -------
    tensor
        the point on geodesic line
    """
    return _geodesic_unit(t, x, u, k=k)


# @torch.compile
def _geodesic_unit(t, x, u, k):
    return (
        torch.cosh(t / torch.sqrt(k)) * x
        + torch.sqrt(k) * torch.sinh(t / torch.sqrt(k)) * u
    )

# @torch.compile
def lorentz_to_poincare(x, k, dim=-1):
    r"""
    Diffeomorphism that maps from Hyperboloid to Poincare disk.

    .. math::

        \Pi_{\mathbb{H}^{d, 1} \rightarrow \mathbb{D}^{d, 1}\left(x_{0}, \ldots, x_{d}\right)}=\frac{\left(x_{1}, \ldots, x_{d}\right)}{x_{0}+\sqrt{k}}

    Parameters
    ----------
    x : tensor
        point on Hyperboloid
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        points on the Poincare disk
    """
    dn = x.size(dim) - 1
    return x.narrow(dim, 1, dn) / (x.narrow(-dim, 0, 1) + torch.sqrt(k))

# @torch.compile
def poincare_to_lorentz(x, k, dim=-1, eps=1e-6):
    r"""
    Diffeomorphism that maps from Poincare disk to Hyperboloid.

    .. math::

        \Pi_{\mathbb{D}^{d, k} \rightarrow \mathbb{H}^{d d, 1}}\left(x_{1}, \ldots, x_{d}\right)=\frac{\sqrt{k} \left(1+|| \mathbf{x}||_{2}^{2}, 2 x_{1}, \ldots, 2 x_{d}\right)}{1-\|\mathbf{x}\|_{2}^{2}}

    Parameters
    ----------
    x : tensor
        point on Poincare ball
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        points on the Hyperboloid
    """
    x_norm_square = torch.sum(x * x, dim=dim, keepdim=True)
    res = (
        torch.sqrt(k)
        * torch.cat((1 + x_norm_square, 2 * x), dim=dim)
        / (1.0 - x_norm_square + eps)
    )
    return res
