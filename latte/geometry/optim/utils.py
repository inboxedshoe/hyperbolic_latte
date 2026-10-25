import torch
import warnings
from latte.geometry.manifold_parent.math import logmap0, project_u, project, parallel_transport0back
from geoopt.tensor import ManifoldParameter, ManifoldTensor


@torch.no_grad()
def move_parameters_scale(man_params, state):
    """Move parameters between manifolds with different curvatures using scaling approach."""
    for point_data in man_params:
        point, old_k = point_data[0], point_data[1]
        new_k = point.manifold.k

        if old_k != new_k:
            if new_k == 0:
                warnings.warn("Manifold curvature is zero, which may cause numerical issues")

            param_state = state.get(point)
            if param_state is None or "exp_avg" not in param_state:
                warnings.warn(f"No state found for parameter during move_parameters_scale")
                continue

            exp_avg = param_state["exp_avg"]

            # Stabilize and project to old manifold
            # point.copy_(project(point, k=old_k))
            # exp_avg.copy_(project_u(point, exp_avg, k=old_k))

            # Scale parameter and momentum to new curvature
            # The scaling factor sqrt(new_k / old_k) comes from the isometry between
            # Lorentz manifolds of different curvatures
            scale_factor = torch.sqrt(new_k / old_k)
            point.copy_(point * scale_factor)
            exp_avg.copy_(exp_avg * scale_factor)

        #stabilize_param(state, point)


@torch.no_grad()
def move_parameters(man_params, state):
    """Move parameters between manifolds with different curvatures using exponential map approach."""
    for point_data in man_params:
        point, old_k = point_data[0], point_data[1]
        new_k = point.manifold.k

        if old_k != new_k:
            if new_k == 0:
                warnings.warn("Manifold curvature is zero, which may cause numerical issues")

            param_state = state.get(point)
            if param_state is None or "exp_avg" not in param_state:
                warnings.warn(f"No state found for parameter during move_parameters")
                continue

            exp_avg = param_state["exp_avg"]

            # Stabilize and project to old manifold
           # point.copy_(project(point, k=old_k))
            #exp_avg.copy_(project_u(point, exp_avg, k=old_k))

            # Transport momentum to origin in old manifold
            exp_avg.copy_(parallel_transport0back(point, exp_avg, k=old_k))

            # Map to tangent space at origin in old manifold
            tangent = logmap0(point, k=old_k, dim=-1)

            # Optional scaling
            # tangent = point.manifold.rescale_to_max(tangent).squeeze()

            # Map back to manifold with new curvature
            point.copy_(point.manifold.expmap0(tangent, dim=-1))

            # Transport momentum to new point in new manifold
            exp_avg.copy_(point.manifold.transp0(point, exp_avg))

        #stabilize_param(state, point)


@torch.no_grad()
def stabilize_group(state, group):
    """Stabilize all parameters in a group to ensure they remain on their manifolds."""
    for p in group["params"]:
        if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
            continue

        param_state = state.get(p)
        if param_state is None or not param_state:  # due to None grads
            continue

        manifold = p.manifold
        exp_avg = param_state.get("exp_avg")

        # Project parameter back to manifold
        p.copy_(manifold.projx(p))

        # Project momentum if it exists
        if exp_avg is not None:
            exp_avg.copy_(manifold.proju(p, exp_avg))


@torch.no_grad()
def stabilize_param(state, p):
    """Stabilize a single parameter to ensure it remains on its manifold."""
    param_state = state.get(p)
    if param_state is None or not param_state:  # due to None grads
        return

    manifold = p.manifold
    exp_avg = param_state.get("exp_avg")

    # Project parameter back to manifold
    p.copy_(manifold.projx(p))

    # Project momentum if it exists
    if exp_avg is not None:
        exp_avg.copy_(manifold.proju(p, exp_avg))


@torch.no_grad()
def validate_manifold_parameters(state, group):
    """Validate that all parameters in a group satisfy their manifold constraints."""
    invalid_params = []
    for p in group["params"]:
        if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
            continue

        manifold = p.manifold
        if hasattr(manifold, 'check_point_on_manifold'):
            on_manifold, reason = manifold.check_point_on_manifold(p)
            if not on_manifold:
                invalid_params.append((p, reason))
        else:
            # Fallback validation for common manifolds
            try:
                # For Lorentz manifolds, check the constraint
                if hasattr(manifold, 'k'):
                    lorentz_constraint = -p[..., 0:1].pow(2) + p[..., 1:].pow(2).sum(dim=-1, keepdim=True)
                    expected = -manifold.k
                    if not torch.allclose(lorentz_constraint, expected, atol=1e-5, rtol=1e-5):
                        invalid_params.append((p, "Lorentz constraint violated"))
            except Exception as e:
                invalid_params.append((p, f"Validation error: {e}"))

    return invalid_params


@torch.no_grad()
def move_parameters_rsgd_scale(man_params, param_groups, state):
    """Move parameters for RSGD optimizer using scaling approach."""
    for point_data in man_params:
        point, old_k = point_data[0], point_data[1]
        new_k = point.manifold.k

        if old_k != new_k:
            if new_k == 0:
                warnings.warn("Manifold curvature is zero, which may cause numerical issues")

            # Get momentum from parameter groups
            momentum = param_groups[1].get("momentum", 0) if len(param_groups) > 1 else 0

            # Stabilize and project to old manifold
            point.copy_(project(point, k=old_k))

            if momentum > 0:
                param_state = state.get(point)
                if param_state is None:
                    continue
                if "momentum_buffer" in param_state:
                    buf = param_state["momentum_buffer"]
                    buf.copy_(project_u(point, buf, k=old_k))

            # Scale parameter and momentum buffer to new curvature
            scale_factor = torch.sqrt(new_k / old_k)
            point.copy_(point * scale_factor)

            if momentum > 0:
                param_state = state.get(point)
                if param_state is None:
                    continue
                if "momentum_buffer" in param_state:
                    buf = param_state["momentum_buffer"]
                    buf.copy_(buf * scale_factor)

            stabilize_param_rsgd(point, momentum, state)


@torch.no_grad()
def move_parameters_rsgd(man_params, param_groups, state):
    """Move parameters for RSGD optimizer between manifolds with different curvatures."""
    for point_data in man_params:
        point, old_k = point_data[0], point_data[1]
        new_k = point.manifold.k

        if old_k != new_k:
            if new_k == 0:
                warnings.warn("Manifold curvature is zero, which may cause numerical issues")

            # Get momentum from parameter groups (assuming second group has momentum)
            momentum = param_groups[1].get("momentum", 0) if len(param_groups) > 1 else 0

            # Stabilize and project to old manifold
            point.copy_(project(point, k=old_k))

            if momentum > 0:
                param_state = state.get(point)
                if param_state is None:
                    continue
                if "momentum_buffer" in param_state:
                    buf = param_state["momentum_buffer"]
                    buf.copy_(project_u(point, buf, k=old_k))
                    buf.copy_(parallel_transport0back(point, buf, k=old_k))

            # Map to tangent space and back with new curvature
            tangent = logmap0(point, k=old_k, dim=-1)
            # Optional scaling: tangent = point.manifold.rescale_to_max(tangent)

            point.copy_(point.manifold.expmap0(tangent, dim=-1))

            if momentum > 0:
                param_state = state.get(point)
                if param_state is None:
                    continue
                if "momentum_buffer" in param_state:
                    buf = param_state["momentum_buffer"]
                    # Optional scaling: buf = point.manifold.rescale_to_max(buf)
                    buf.copy_(point.manifold.transp0(point, buf))

            # stabilize_param_rsgd(point, momentum, state)


@torch.no_grad()
def stabilize_group_rsgd(group, state):
    """Stabilize all parameters in an RSGD group to ensure they remain on their manifolds."""
    for p in group["params"]:
        if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
            continue

        manifold = p.manifold
        momentum = group.get("momentum", 0)

        # Project parameter back to manifold
        p.copy_(manifold.projx(p))

        if momentum > 0:
            param_state = state.get(p)
            if param_state is None or not param_state:
                continue
            if "momentum_buffer" in param_state:
                buf = param_state["momentum_buffer"]
                buf.copy_(manifold.proju(p, buf))


@torch.no_grad()
def stabilize_param_rsgd(p, momentum, state):
    """Stabilize a single parameter for RSGD optimizer."""
    if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
        return

    manifold = p.manifold

    # Project parameter back to manifold
    p.copy_(manifold.projx(p))

    if momentum > 0:
        param_state = state.get(p)
        if param_state is None or not param_state:
            return
        if "momentum_buffer" in param_state:
            buf = param_state["momentum_buffer"]
            buf.copy_(manifold.proju(p, buf))


def get_param_groups(model, lr_manifold, weight_decay_manifold):
    no_decay = ["scale"]
    k_params = ".k"

    parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                   and not any(nd in n for nd in no_decay)
                   and not isinstance(p, ManifoldParameter)
                   and not n.endswith(k_params)
            ],
            "name": "1"
        },
        {
            # Params matching `no_decay` substrings (e.g. `log_scale` in
            # LorentzPLFCScaled), optimized without weight decay.
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                   and any(nd in n for nd in no_decay)
                   and not isinstance(p, ManifoldParameter)
                   and not n.endswith(k_params)
            ],
            "weight_decay": 0.0,
            "name": "no_decay"
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if isinstance(p, ManifoldParameter)
            ],
            'lr': lr_manifold,
            "weight_decay": weight_decay_manifold,
            "name": "manifold"
        },
        {  # k parameters
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                   and n.endswith(k_params)
            ],
            "weight_decay": weight_decay_manifold,
            "lr": lr_manifold,
            "name": "k_group"
        }
    ]

    return parameters
