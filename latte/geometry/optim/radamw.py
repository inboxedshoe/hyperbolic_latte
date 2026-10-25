import math
import torch.optim
import warnings

from .mixin import OptimMixin
from geoopt.tensor import ManifoldParameter, ManifoldTensor
from .utils import move_parameters_scale
from latte.geometry.manifold_parent import Lorentz

__all__ = ["RiemannianAdamW"]


class RiemannianAdamW(OptimMixin, torch.optim.AdamW):
    r"""
    Riemannian AdamW.

    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining
        parameter groups
    lr : float (optional)
        learning rate (default: 1e-3)
    betas : Tuple[float, float] (optional)
        coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.999))
    eps : float (optional)
        term added to the denominator to improve
        numerical stability (default: 1e-8)
    weight_decay : float (optional)
        weight decay (L2 penalty) (default: 0)
    amsgrad : bool (optional)
        whether to use the AMSGrad variant of this algorithm
        (default: False)

    Other Parameters
    ----------------
    stabilize : int
        Stabilize parameters if they are off-manifold due to numerical
        reasons every ``stabilize`` steps (default: ``None`` -- no stabilize)
    max_grad_norm : float (optional)
        Maximum gradient norm for gradient clipping (default: None)
    use_centroid_weight_decay : bool (optional)
        Use centroid-based weight decay for hyperbolic parameters (default: True)


    """

    def euclid_step(self, point, lr, weight_decay, eps, betas, amsgrad, group):
        """Optimization step for Euclidean parameters."""
        if point.grad is None:
            return point

        grad = point.grad
        if grad.is_sparse:
            raise RuntimeError('AdamW does not support sparse gradients')

        # Apply gradient clipping if specified
        if group.get('max_grad_norm') is not None:
            torch.nn.utils.clip_grad_norm_([grad], group['max_grad_norm'])

        # Apply weight decay
        if weight_decay != 0:
            point.mul_(1 - lr * weight_decay)

        state = self.state[point]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(point, memory_format=torch.preserve_format)
            state['exp_avg_sq'] = torch.zeros_like(point, memory_format=torch.preserve_format)
            if amsgrad:
                state['max_exp_avg_sq'] = torch.zeros_like(point, memory_format=torch.preserve_format)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = betas

        state['step'] += 1
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        # Update moments
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # Compute denominator
        if amsgrad:
            max_exp_avg_sq = state['max_exp_avg_sq']
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        # Update parameters
        point.addcdiv_(exp_avg, denom, value=-step_size)

        return point

    def hyperbolic_step(self, point, lr, weight_decay, eps, betas, amsgrad, group):
        """Optimization step for hyperbolic parameters."""
        grad = point.grad

        if grad is None:
            return point

        if grad.is_sparse:
            raise RuntimeError(
                "RiemannianAdam does not support sparse gradients, use SparseRiemannianAdam instead"
            )

        # Apply gradient clipping if specified
        if group.get('max_grad_norm') is not None:
            torch.nn.utils.clip_grad_norm_([grad], group['max_grad_norm'])

        manifold = point.manifold

        # Track Lorentz manifold parameters for curvature updates
        if isinstance(manifold, Lorentz):
            self.man_params.append([point, manifold.k.clone()])

        state = self.state[point]

        # State initialization
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(point, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(point, memory_format=torch.preserve_format)
            if amsgrad:
                state["max_exp_avg_sq"] = torch.zeros_like(point, memory_format=torch.preserve_format)

        state["step"] += 1
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]

        # Apply weight decay (centroid-based or retraction-based)
        # use_centroid = group.get('use_centroid_weight_decay', True)
        use_centroid = True
        if weight_decay != 0 and use_centroid:
            # Centroid-based weight decay
            means = torch.concat(
                (point.unsqueeze(-2), point.manifold.origin(point.shape).unsqueeze(-2).to(point.device)),
                dim=-2)
            point = point.manifold.centroid(means, w=torch.tensor(
                ((1 - lr * weight_decay), lr * weight_decay), dtype=means.dtype,
                device=means.device))
        elif weight_decay != 0:
            # Retraction-based weight decay
            weight_decay_grad = manifold.egrad2rgrad(point, point)
            weight_decay_step = -lr * weight_decay * weight_decay_grad
            point, _ = manifold.retr_transp(point, weight_decay_step, exp_avg)

        # Convert Euclidean gradient to Riemannian gradient
        grad = manifold.egrad2rgrad(point, grad)

        # Update moments
        exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
        exp_avg_sq.mul_(betas[1]).add_(
            manifold.component_inner(point, grad), alpha=1 - betas[1]
        )

        bias_correction1 = 1 - betas[0] ** state["step"]
        bias_correction2 = 1 - betas[1] ** state["step"]

        # Compute denominator
        if amsgrad:
            max_exp_avg_sq = state["max_exp_avg_sq"]
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            denom = max_exp_avg_sq.div(bias_correction2).sqrt_()
        else:
            denom = exp_avg_sq.div(bias_correction2).sqrt_()

        # Compute update direction
        direction = exp_avg.div(bias_correction1) / denom.add_(eps)

        # Apply retraction and parallel transport
        new_point, exp_avg_new = manifold.retr_transp(
            point, -lr * direction, exp_avg
        )

        # Update point and momentum
        # if self._validate_hyperbolic_point(new_point, manifold):
        point.copy_(new_point)
        exp_avg.copy_(exp_avg_new)
        # else:
        #     warnings.warn(f"Invalid hyperbolic point detected at step {state['step']}, skipping update")

        return point

    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        self.man_params = []

        k_group = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            try:
                # Process regular parameter groups
                for group in self.param_groups:
                    if group.get("name") == "k_group":
                        k_group = group
                        continue

                    if "step" not in group:
                        group["step"] = 0
                    group["step"] += 1

                    betas = group["betas"]
                    weight_decay = group["weight_decay"]
                    eps = group["eps"]
                    learning_rate = group["lr"]
                    amsgrad = group["amsgrad"]

                    for point in group["params"]:
                        if point.grad is not None and torch.isnan(point.grad).any():
                            warnings.warn("NaN gradient detected, skipping parameter update")
                            continue

                        if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                            new_point = self.hyperbolic_step(point, learning_rate, weight_decay, eps, betas, amsgrad,
                                                             group)
                        else:
                            new_point = self.euclid_step(point, learning_rate, weight_decay, eps, betas, amsgrad, group)

                        # Only update if the new point is valid
                        if not torch.isnan(new_point).any():
                            point.copy_(new_point)

                # Process curvature parameters (k_group)
                if k_group is not None:
                    betas = k_group["betas"]
                    weight_decay = k_group["weight_decay"]
                    eps = k_group["eps"]
                    learning_rate = k_group["lr"]
                    amsgrad = k_group["amsgrad"]

                    for point in k_group["params"]:
                        if point.grad is not None and torch.isnan(point.grad).any():
                            warnings.warn("NaN gradient detected in k_group, skipping parameter update")
                            continue

                        new_point = self.euclid_step(point, learning_rate, weight_decay, eps, betas, amsgrad, k_group)
                        # Clamp curvature parameters
                        if not torch.isnan(new_point).any():
                            point.copy_(new_point.clamp(0.01, 50.0))

                # Move parameters between manifolds if needed
                if self.man_params:
                    move_parameters_scale(self.man_params, self.state)

            except Exception as e:
                warnings.warn(f"Optimizer step failed: {e}")
                # Attempt to recover by stabilizing parameters
                self._stabilize_all_parameters()

        return loss

    def _validate_hyperbolic_point(self, point, manifold, atol=1e-1, rtol=1e-1):
        """Validate that a point satisfies hyperbolic manifold constraints."""
        try:
            if hasattr(manifold, 'check_point_on_manifold'):
                on_manifold, reason = manifold.check_point_on_manifold(point, atol=atol, rtol=1e-1)
                return on_manifold
            # Fallback: check Lorentz constraint for common hyperbolic manifolds
            if hasattr(point, 'manifold') and hasattr(point.manifold, 'k'):
                # Basic Lorentz constraint check
                lorentz_constraint = -point[..., 0:1].pow(2) + point[..., 1:].pow(2).sum(dim=-1, keepdim=True)
                expected = -point.manifold.k
                return torch.allclose(lorentz_constraint, expected, atol=atol, rtol=1e-1)
            return True
        except Exception:
            return False

    def _stabilize_all_parameters(self):
        """Stabilize all manifold parameters to recover from numerical issues."""
        for group in self.param_groups:
            for point in group["params"]:
                if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                    self.stabilize_param(point)

    @torch.no_grad()
    def stabilize_param(self, p):
        """Stabilize a single parameter to ensure it remains on the manifold."""
        state = self.state.get(p)
        if not state:  # due to None grads
            return
        manifold = p.manifold
        exp_avg = state.get("exp_avg")

        # Project parameter back to manifold
        p.copy_(manifold.projx(p))

        # Project momentum if it exists
        if exp_avg is not None:
            exp_avg.copy_(manifold.proju(p, exp_avg))



