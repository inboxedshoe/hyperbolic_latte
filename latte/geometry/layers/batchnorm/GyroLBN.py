import warnings

import torch
import torch.nn as nn

from geoopt import ManifoldParameter
from ...manifold import CustomLorentz
from ...manifold_parent.math import _inner, arcosh


def tsmnet_momentum(k: int, K: int, gamma_min: float = 0.2) -> float:
    """Adaptive momentum schedule.

        γ_train(k) = 1 − γ_min^{(1/(K−1))·max(K−k, 0)} + γ_min

    Decays from 1 at k=1 to `gamma_min` at k=K.
    """
    if K <= 1:
        return 1.0
    expo = (1.0 / (K - 1)) * max(K - k, 0)
    g = 1.0 - (gamma_min ** expo) + gamma_min
    # k=0 evaluates marginally above 1; clamp.
    return float(min(1.0, max(gamma_min, g)))


class AdaptiveMomentum(nn.Module):
    """Epoch-aware momentum source, shared by every domain-aware BN variant.

    Holds the epoch counter as a buffer so it survives checkpointing, and
    resolves `K` either from an explicit value or as a fraction of the total
    epoch budget.

    Args:
        mode:      "fixed" uses the constant `momentum`;
                   "adaptive" uses `tsmnet_momentum`.
        momentum:  the constant used by "fixed", and the *starting* value of the
                   eval-buffer schedule under "adaptive" (see `eval_momentum`).
        gamma_min: floor of the schedule.
        K:         schedule length in epochs. None → derived as
                   `K_frac * total_epochs`.
        K_frac:    fraction of the epoch budget used as `K`.
        eval_bs_ratio: `bs/bs0`, the batch size relative to a reference batch
                   size. Sets how far the eval-buffer momentum decays.
    """

    def __init__(self, mode="fixed", momentum=0.1, gamma_min=0.2,
                 K=None, K_frac=0.8, eval_bs_ratio=0.5):
        super().__init__()
        assert mode in ("fixed", "adaptive"), f"unknown momentum mode '{mode}'"
        self.mode = mode
        self.momentum = float(momentum)
        self.gamma_min = float(gamma_min)
        self.K = None if K is None else int(K)
        self.K_frac = float(K_frac)
        self.eval_bs_ratio = float(eval_bs_ratio)
        self._warned_no_budget = False
        # Non-persistent: these are kept out of the state_dict.
        self.register_buffer("epoch", torch.zeros((), dtype=torch.long),
                             persistent=False)
        self.register_buffer("total_epochs", torch.zeros((), dtype=torch.long),
                             persistent=False)

    def set_epoch(self, epoch, total_epochs=None):
        self.epoch.fill_(int(epoch))
        if total_epochs is not None:
            self.total_epochs.fill_(int(total_epochs))

    def resolved_K(self) -> int:
        if self.K is not None:
            return max(2, self.K)
        total = int(self.total_epochs.item())
        if total > 0:
            return max(2, int(round(self.K_frac * total)))
        if not self._warned_no_budget:
            self._warned_no_budget = True
            warnings.warn(
                "AdaptiveMomentum: neither K nor a total-epoch budget was set; "
                "falling back to K=40. Call set_bn_epoch(model, epoch, total_epochs) "
                "from the train loop so K scales with the actual budget."
            )
        return 40

    def train_momentum(self) -> float:
        if self.mode == "fixed":
            return self.momentum
        return tsmnet_momentum(int(self.epoch.item()), self.resolved_K(),
                               self.gamma_min)

    def eval_momentum(self) -> float:
        """Momentum for the separate test-statistics buffer (`eta_test`).

        Only consulted in "adaptive" mode, since `has_eval_stats` gates the
        buffer's existence. It is scheduled *alongside* the training momentum
        and recomputed every epoch:

            w         = max(K − k, 0) / (K − 1)
            η_test(k) = 1 − (1 − momentum) ** (bs/bs0 · (1 − w) + w)

        so it decays from `momentum` at k=1 to `1 − (1 − momentum)^(bs/bs0)` at
        k=K.

        The training schedule only touches `running_*`, which `_eval_stats`
        ignores whenever `has_eval_stats` is set.
        """
        if self.mode == "fixed":
            return self.momentum
        K = self.resolved_K()
        if K <= 1:
            return self.momentum
        w = max(K - int(self.epoch.item()), 0) / (K - 1)
        tau = 1.0 - self.momentum
        expo = self.eval_bs_ratio * (1.0 - w) + w
        return float(min(1.0, max(0.0, 1.0 - tau ** expo)))


class GyroLBNCustom(nn.Module):
    """Custom Gyrogroup Lorentz Batch Normalization.

    Gyro batch normalisation with a per-spatial-feature γ and
    Fréchet-variance-based σ².

        x̃_i ← β ⊕ [(γ / √(σ_B + ε)) ⊙ (⊖μ_B ⊕ x_i)]

    where γ is per-spatial-feature, σ² = (1/m) Σ d²(x_i, μ_B) is the
    Fréchet variance, and β is a `ManifoldParameter` updated via Riemannian
    optimization.
    """

    def __init__(
        self,
        manifold: CustomLorentz,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.manifold = manifold
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        num_spatial = num_features - 1

        # Learnable bias point on the manifold (β).
        self.beta = ManifoldParameter(
            self.manifold.origin(num_features), manifold=self.manifold
        )
        # Per-spatial-feature scale (γ). Initialised to 1.
        self.gamma = nn.Parameter(torch.ones(num_spatial))

        # Running statistics stored as tangent vectors at origin for mean
        # (avoids needing to track a manifold-valued running mean).
        self.register_buffer("running_mean_tangent", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(1))

    # ------------------------------------------------------------------
    def _batch_frechet_variance(
        self, x: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        """Fréchet variance: σ² = (1/m) Σ d²(x_i, μ)."""
        d_sq = self.manifold.dist(x, mu.unsqueeze(-2)).pow(2)
        return d_sq.mean()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim >= 2, "Input must have at least 2 dimensions."

        beta = self.beta

        if self.training:
            mu = self.manifold.centroid(x)
            if x.ndim == 3:
                mu = self.manifold.centroid(mu)

            var = self._batch_frechet_variance(x, mu)
            sigma = var.sqrt()

            with torch.no_grad():
                mu_tangent = self.manifold.logmap0(mu.detach())
                self.running_mean_tangent.copy_(
                    (1 - self.momentum) * self.running_mean_tangent
                    + self.momentum * mu_tangent
                )
                self.running_var.copy_(
                    (1 - self.momentum) * self.running_var
                    + self.momentum * sigma.detach()
                )
        else:
            mu = self.manifold.expmap0(self.running_mean_tangent)
            sigma = self.running_var

        # ⊖μ ⊕ x via fused logmap+transp0back: returns tangent v at origin with ‖v‖ = d(μ, x).
        v = self.manifold.fused_logmap_transp0back(mu, x)

        scale = self.gamma / (sigma + self.eps)
        v_s = v[..., 1:] * scale
        v = torch.cat([v[..., :1], v_s], dim=-1)
        v = self.manifold.rescale_to_max_euclid(v)

        output = self.manifold.fused_transp0_expmap(beta, v)
        return output


class GyroLBNCustom1d(GyroLBNCustom):
    """1-D GyroLBNCustom wrapper (batch, num_features)."""

    def __init__(self, manifold: CustomLorentz, num_features: int, **kwargs):
        super().__init__(manifold, num_features, **kwargs)


class GyroLBNCustomAdaptive(nn.Module):
    """Domain-conditioned `GyroLBNCustom`, with both conditioning sides switchable.

    The normalization pipeline is `GyroLBNCustom`'s:

        x̃ = β ⊕ [(γ / (σ + ε)) ⊙ (⊖μ ⊕ x)]

    implemented as `fused_logmap_transp0back(μ, ·)` → spatial rescale →
    `fused_transp0_expmap(β, ·)`. The modes differ only
    in **which side of the layer is keyed by domain**:

    | `stats_mode`  | `affine_mode`  |                                          |
    |---------------|----------------|------------------------------------------|
    | `shared`      | `per_subject`  | default — *personalisation*              |
    | `per_domain`  | `shared`       | *alignment*                              |
    | `per_domain`  | `per_subject`  | both, composed                           |
    | `shared`      | `shared`       | plain GyroLBNCustom                      |

    Per-domain *estimated* statistics with a shared affine whiten every domain
    to a common origin. A per-subject *learned* affine with shared statistics
    maps each subject to its own output distribution.

    A learned affine has no entry for an unseen subject, so the
    fallback slot is an identity. Estimated statistics can be computed for
    any subject from unlabeled data in a forward pass.

    Layout (`S = num_subjects + 1`, indexed directly by the raw subject id;
    slot `num_subjects` is the fallback used when `subject_ids=None`):

        γ : (S | 1, num_features-1)  init 1
        β : (S | 1, num_features)    init 0   (tangent at origin → expmap0)
        running_mean_tangent : (num_features,) | (S, num_features)
        running_var          : (1,)            | (S, 1)      — holds a *std*

    Unseen-domain fallback. Under `stats_mode="per_domain"` any slot that never
    received an update reads the **pooled** statistics instead: the Fréchet mean
    over all seen domain slots, refreshed by `finalize_pooled_stats()`. The
    fallback keys off "this slot was never updated", not off a reserved index.

    Args:
        stats_mode:     "shared" | "per_domain" — where the running statistics live.
        affine_mode:    "per_subject" | "shared" — where γ/β live.
        momentum_mode:  "fixed" | "adaptive" (epoch schedule, see `AdaptiveMomentum`).
        min_domain_obs: warn when a domain slot updates from fewer than this many
                        observations. 0 disables the check.

    forward(x, subject_ids):
        x            : (B, [seq,] num_features)
        subject_ids  : (B,) long tensor, or None to use the fallback slot.
    """

    def __init__(
        self,
        manifold: CustomLorentz,
        num_features: int,
        num_subjects: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        stats_mode: str = "shared",
        affine_mode: str = "per_subject",
        momentum_mode: str = "fixed",
        gamma_min: float = 0.2,
        momentum_K: int = None,
        momentum_K_frac: float = 0.8,
        eval_bs_ratio: float = 0.5,
        min_domain_obs: int = 4,
    ):
        super().__init__()
        assert stats_mode in ("shared", "per_domain"), \
            f"unknown stats_mode '{stats_mode}'"
        assert affine_mode in ("per_subject", "shared"), \
            f"unknown affine_mode '{affine_mode}'"

        self.manifold = manifold
        self.num_features = num_features
        self.num_subjects = num_subjects
        self.eps = eps
        self.momentum = momentum
        self.stats_mode = stats_mode
        self.affine_mode = affine_mode
        self.min_domain_obs = int(min_domain_obs)

        num_spatial = num_features - 1
        S = num_subjects + 1
        self.num_slots = S
        self._unseen_idx = num_subjects

        self.mom = AdaptiveMomentum(
            mode=momentum_mode, momentum=momentum, gamma_min=gamma_min,
            K=momentum_K, K_frac=momentum_K_frac,
            eval_bs_ratio=eval_bs_ratio,
        )
        # Two running-statistic sets, tracked with *two different decaying
        # schedules*: `eta` for the training buffer and `eta_test`
        # for the one used at test.
        # The second set is only allocated when the schedule is on, so "fixed"
        # mode has no eval buffers in its state_dict.
        self.has_eval_stats = (momentum_mode == "adaptive")

        # ---- affine -------------------------------------------------------
        # "shared" collapses to a single row so every domain is mapped to the
        # same origin.
        A = S if affine_mode == "per_subject" else 1
        self.gamma = nn.Parameter(torch.ones(A, num_spatial))
        self.beta_tangent = nn.Parameter(torch.zeros(A, num_features))

        # ---- running statistics -------------------------------------------
        if stats_mode == "shared":
            self.register_buffer("running_mean_tangent", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(1))
            if self.has_eval_stats:
                self.register_buffer("eval_mean_tangent", torch.zeros(num_features))
                self.register_buffer("eval_var", torch.ones(1))
        else:
            self.register_buffer("running_mean_tangent", torch.zeros(S, num_features))
            self.register_buffer("running_var", torch.ones(S, 1))
            if self.has_eval_stats:
                self.register_buffer("eval_mean_tangent", torch.zeros(S, num_features))
                self.register_buffer("eval_var", torch.ones(S, 1))
            # Pooled fallback for domains never seen in training.
            self.register_buffer("pooled_mean_tangent", torch.zeros(num_features))
            self.register_buffer("pooled_var", torch.ones(1))
            self.register_buffer("eval_pooled_mean_tangent", torch.zeros(num_features))
            self.register_buffer("eval_pooled_var", torch.ones(1))
            # Per-slot bookkeeping: number of update steps and total observations.
            self.register_buffer("domain_updates", torch.zeros(S, dtype=torch.long))
            self.register_buffer("domain_obs", torch.zeros(S, dtype=torch.long))

        self._step = 0
        self._starved_warnings = 0

    # ------------------------------------------------------------------
    # epoch / schedule plumbing
    # ------------------------------------------------------------------
    def set_epoch(self, epoch, total_epochs=None):
        self.mom.set_epoch(epoch, total_epochs)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _align(m: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Reshape a per-sample statistic (B, F) to broadcast against x.

        Needed because per-domain μ is per-sample, whereas the shared μ is
        (F,) and broadcasts against any rank. For x of rank R the statistic
        becomes (B, 1, ..., 1, F) with R-2 singleton axes.
        """
        if x.ndim == 2:
            return m
        return m.reshape(m.shape[0], *([1] * (x.ndim - 2)), m.shape[-1])

    def _per_sample_points(self, x: torch.Tensor) -> torch.Tensor:
        """Collapse every axis between batch and feature to one point per sample.

        Mirrors `GyroLBNCustom`'s two-stage `centroid(centroid(x))`: project
        after each reduction.
        """
        if x.ndim == 2:
            return x
        return self.manifold.centroid(x.flatten(1, -2), dim=-2)

    def _project(self, avg: torch.Tensor) -> torch.Tensor:
        """Lift a Euclidean average back onto the hyperboloid.

        Reuses `manifold.centroid` on a singleton reduction axis so the exact
        library projection (and its eps clamping) is applied.
        """
        return self.manifold.centroid(avg.unsqueeze(-2), dim=-2)

    # ------------------------------------------------------------------
    # statistics
    # ------------------------------------------------------------------
    def _batch_frechet_variance(
        self, x: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        d_sq = self.manifold.dist(x, mu.unsqueeze(-2)).pow(2)
        return d_sq.mean()

    def _domain_stats(self, x: torch.Tensor, slot_ids: torch.Tensor):
        """Per-domain Fréchet mean and dispersion, vectorised over domains.

        Domain-by-domain this is identical to the shared path — reduce to one
        point per sample, average, project, then take the root-mean-square
        geodesic distance to that mean. It is computed with two scatter-adds.

        Returns:
            mu_slot  : (S, F) domain centroid; the origin for absent domains.
            sig_slot : (S,)   root Fréchet variance (a *std*, as stored
                              in `running_var`); 1 if absent.
            counts   : (S,)   observations contributing to each slot.
        """
        S, F = self.num_slots, self.num_features
        pts = self._per_sample_points(x)                              # (B, F)
        dev, dt = pts.device, pts.dtype

        ones = torch.ones(pts.shape[0], device=dev, dtype=dt)
        counts = torch.zeros(S, device=dev, dtype=dt).index_add(0, slot_ids, ones)
        sums = torch.zeros(S, F, device=dev, dtype=dt).index_add(0, slot_ids, pts)

        present = counts > 0
        avg = sums / counts.clamp_min(1.0).unsqueeze(-1)
        mu_slot = self._project(avg)
        # Absent slots have avg = 0, which is not a manifold point; the origin
        # is the meaningful placeholder and it is never read (present masks it).
        origin = self.manifold.origin(F).to(device=dev, dtype=dt)
        mu_slot = torch.where(present.unsqueeze(-1), mu_slot, origin)

        # Fréchet variance of each sample against *its own* domain's mean.
        mu_b = self._align(mu_slot.index_select(0, slot_ids), x)
        d_sq = self.manifold.dist(x, mu_b).pow(2)
        if d_sq.ndim > 1:
            d_sq = d_sq.flatten(1).mean(dim=1)                        # (B,)
        var_sums = torch.zeros(S, device=dev, dtype=dt).index_add(0, slot_ids, d_sq)
        sig_slot = (var_sums / counts.clamp_min(1.0)).sqrt()
        sig_slot = torch.where(present, sig_slot, torch.ones_like(sig_slot))

        return mu_slot, sig_slot, counts

    def _warn_if_starved(self, counts: torch.Tensor) -> None:
        """Warn when slots update from fewer than `min_domain_obs` samples.

        `counts.any()` forces a device sync, so this samples rather than runs
        every step: the first 20 steps and every 200th step after that.
        """
        if self.min_domain_obs <= 0:
            return
        self._step += 1
        if self._step > 20 and (self._step % 200) != 0:
            return
        starved = (counts > 0) & (counts < self.min_domain_obs)
        if not bool(starved.any()):
            return
        self._starved_warnings += 1
        if self._starved_warnings <= 3 or (self._starved_warnings % 50) == 0:
            ids = starved.nonzero(as_tuple=True)[0].tolist()
            got = counts[starved].tolist()
            warnings.warn(
                f"{type(self).__name__}: per-domain statistics updated from "
                f"fewer than min_domain_obs={self.min_domain_obs} samples "
                f"(step {self._step}, slots {ids} got {got}). Per-domain "
                f"Fréchet means estimated from groups this small are noise — "
                f"enable the domain-stratified sampler "
                f"(preprocessing_dict['domain_stratified']=True) or raise "
                f"obs_per_domain. Occurrence {self._starved_warnings}."
            )

    @torch.no_grad()
    def _update_running(self, mu_slot, sig_slot, counts) -> None:
        """Scatter-update only the domain slots present in this batch.

        Domains absent from a batch keep their previous statistics.

        A slot's first update copies the batch statistic outright (momentum 1).
        """
        idx = (counts > 0).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return

        mu_tan = self.manifold.logmap0(mu_slot.index_select(0, idx).detach())
        sig = sig_slot.index_select(0, idx).detach().unsqueeze(-1)      # (P, 1)
        first = (self.domain_updates.index_select(0, idx) == 0).unsqueeze(-1)

        def blend(buf, m_const):
            m = torch.full_like(sig, m_const)
            m = torch.where(first, torch.ones_like(m), m)
            cur_mean = buf[0].index_select(0, idx)
            cur_var = buf[1].index_select(0, idx)
            buf[0][idx] = (1 - m) * cur_mean + m * mu_tan
            buf[1][idx] = (1 - m) * cur_var + m * sig

        blend((self.running_mean_tangent, self.running_var),
              self.mom.train_momentum())
        if self.has_eval_stats:
            blend((self.eval_mean_tangent, self.eval_var),
                  self.mom.eval_momentum())

        self.domain_updates[idx] += 1
        self.domain_obs[idx] += counts.index_select(0, idx).long()

    @torch.no_grad()
    def _update_running_shared(self, mu, sigma) -> None:
        m = self.mom.train_momentum()
        mu_tangent = self.manifold.logmap0(mu.detach())
        self.running_mean_tangent.copy_(
            (1 - m) * self.running_mean_tangent + m * mu_tangent)
        self.running_var.copy_(
            (1 - m) * self.running_var + m * sigma.detach())
        if self.has_eval_stats:
            me = self.mom.eval_momentum()
            self.eval_mean_tangent.copy_(
                (1 - me) * self.eval_mean_tangent + me * mu_tangent)
            self.eval_var.copy_(
                (1 - me) * self.eval_var + me * sigma.detach())

    @torch.no_grad()
    def finalize_pooled_stats(self) -> None:
        """Refresh the unseen-domain fallback from the seen domain slots.

        The pooled point is the Fréchet mean over the seen domains' running
        means, weighted uniformly by *domain* (not by trial count). Costs one
        `centroid` over at most `S` points per BN layer.

        The unseen slot thus normalises a held-out subject with the pooled
        statistics.
        """
        if self.stats_mode != "per_domain":
            return
        seen = self.domain_updates > 0
        if not bool(seen.any()):
            return

        def pool(mean_buf, var_buf, out_mean, out_var):
            pts = self.manifold.expmap0(mean_buf[seen])            # (P, F)
            mu = self.manifold.centroid(pts.unsqueeze(0), dim=-2).squeeze(0)
            out_mean.copy_(self.manifold.logmap0(mu))
            out_var.copy_(var_buf[seen].mean(dim=0))

        pool(self.running_mean_tangent, self.running_var,
             self.pooled_mean_tangent, self.pooled_var)
        if self.has_eval_stats:
            pool(self.eval_mean_tangent, self.eval_var,
                 self.eval_pooled_mean_tangent, self.eval_pooled_var)
        else:
            self.eval_pooled_mean_tangent.copy_(self.pooled_mean_tangent)
            self.eval_pooled_var.copy_(self.pooled_var)

    def _train_norm_shared(self, mu, sigma):
        """The post-update running statistic, used as the training normaliser.

        The raw batch statistic is not used as the normaliser. The running mean
        is interpolated toward it by the scheduled momentum `t` (in the tangent
        space at the origin) and the result normalises the batch; the same value
        is then stored as the new running mean, so the normaliser is the
        post-update statistic. At `t = 1.0` this is exactly the batch mean;
        as the schedule decays `t` the normaliser slides toward the
        accumulated mean.
        Differentiable through `mu`: only the buffer write is detached.
        """
        t = self.mom.train_momentum()
        rm_tan = ((1 - t) * self.running_mean_tangent
                  + t * self.manifold.logmap0(mu))
        rv = (1 - t) * self.running_var + t * sigma
        return self.manifold.expmap0(rm_tan), rv

    def _train_norm_domain(self, mu_slot, sig_slot, counts):
        """Per-slot version of `_train_norm_shared`.

        Mirrors `_update_running`'s blend exactly, including the momentum-1
        first update, so the value returned here is what the buffer will hold
        after `_update_running` runs. Absent slots keep their buffer (they are
        never indexed, but the tensor has to stay finite).
        """
        t = self.mom.train_momentum()
        present = (counts > 0).unsqueeze(-1)                       # (S, 1)
        first = (self.domain_updates == 0).unsqueeze(-1)           # (S, 1)
        m = torch.full_like(self.running_var, float(t))            # (S, 1)
        m = torch.where(first, torch.ones_like(m), m)

        mu_tan = self.manifold.logmap0(mu_slot)                    # (S, F)
        rm_tan = (1 - m) * self.running_mean_tangent + m * mu_tan
        rv = (1 - m) * self.running_var + m * sig_slot.unsqueeze(-1)

        rm_tan = torch.where(present, rm_tan, self.running_mean_tangent)
        rv = torch.where(present, rv, self.running_var)
        return self.manifold.expmap0(rm_tan), rv

    @torch.no_grad()
    def refit_slots(self, mu_slot, sig_slot, counts) -> int:
        """Overwrite the named slots' statistics outright (no blending).

        Marks the slot seen so `_eval_stats` reads it instead of substituting
        the pooled fallback.
        """
        if self.stats_mode != "per_domain":
            return 0
        idx = (counts > 0).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return 0
        mu_tan = self.manifold.logmap0(mu_slot.index_select(0, idx).detach())
        sig = sig_slot.index_select(0, idx).detach().unsqueeze(-1)
        self.running_mean_tangent[idx] = mu_tan
        self.running_var[idx] = sig
        if self.has_eval_stats:
            self.eval_mean_tangent[idx] = mu_tan
            self.eval_var[idx] = sig
        self.domain_updates[idx] = self.domain_updates[idx].clamp_min(1)
        self.domain_obs[idx] += counts.index_select(0, idx).long()
        return int(idx.numel())

    @torch.no_grad()
    def refit_shared(self, mu, sigma) -> int:
        """`refit_slots` for the shared-statistics parameterisation."""
        if self.stats_mode != "shared":
            return 0
        mu_tan = self.manifold.logmap0(mu.detach())
        if mu_tan.ndim > 1:
            mu_tan = mu_tan.reshape(-1, mu_tan.shape[-1]).mean(0)
        sig = sigma.detach().reshape(-1)[:1]
        self.running_mean_tangent.copy_(mu_tan)
        self.running_var.copy_(sig)
        if self.has_eval_stats:
            self.eval_mean_tangent.copy_(mu_tan)
            self.eval_var.copy_(sig)
        return 1

    def set_refit(self, on: bool = True) -> None:
        """Toggle REFIT: in eval, re-estimate stats from the batch before using
        them."""
        self.refit_mode = bool(on)

    def _eval_stats(self, slot_ids: torch.Tensor):
        """Gather (μ, σ) for inference, substituting pooled stats for unseen slots."""
        if self.has_eval_stats:
            mean_buf, var_buf = self.eval_mean_tangent, self.eval_var
            p_mean, p_var = (self.eval_pooled_mean_tangent, self.eval_pooled_var) \
                if self.stats_mode == "per_domain" else (None, None)
        else:
            mean_buf, var_buf = self.running_mean_tangent, self.running_var
            p_mean, p_var = (self.pooled_mean_tangent, self.pooled_var) \
                if self.stats_mode == "per_domain" else (None, None)

        if self.stats_mode == "shared":
            return self.manifold.expmap0(mean_buf), var_buf

        seen = (self.domain_updates > 0).unsqueeze(-1)                # (S, 1)
        mean_slot = torch.where(seen, mean_buf, p_mean.unsqueeze(0))
        var_slot = torch.where(seen, var_buf, p_var.unsqueeze(0))
        mu = self.manifold.expmap0(mean_slot.index_select(0, slot_ids))
        sigma = var_slot.index_select(0, slot_ids)                    # (B, 1)
        return mu, sigma

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, subject_ids: torch.Tensor = None
    ) -> torch.Tensor:
        assert x.ndim >= 2, "Input must have at least 2 dimensions."
        B = x.size(0)

        if subject_ids is None:
            slot_ids = torch.full(
                (B,), self._unseen_idx, device=x.device, dtype=torch.long
            )
        else:
            slot_ids = subject_ids.reshape(B, -1)[:, 0].to(
                device=x.device, dtype=torch.long)

        # ---- statistics ---------------------------------------------------
        per_sample_stats = False
        adaptive_norm = self.mom.mode == "adaptive"
        if self.training:
            if self.stats_mode == "per_domain":
                mu_slot, sig_slot, counts = self._domain_stats(x, slot_ids)
                self._warn_if_starved(counts)
                # Blend for the normaliser *before* the buffer write, so it stays
                # differentiable through the batch mean (see `_train_norm_domain`).
                if adaptive_norm:
                    mu_n, sig_n = self._train_norm_domain(
                        mu_slot, sig_slot, counts)
                else:
                    mu_n, sig_n = mu_slot, sig_slot.unsqueeze(-1)
                self._update_running(mu_slot, sig_slot, counts)
                mu = mu_n.index_select(0, slot_ids)                   # (B, F)
                sigma = sig_n.index_select(0, slot_ids)               # (B, 1)
                per_sample_stats = True
            else:
                mu = self.manifold.centroid(x)
                if x.ndim == 3:
                    mu = self.manifold.centroid(mu)
                sigma = self._batch_frechet_variance(x, mu).sqrt()
                if adaptive_norm:
                    mu_n, sigma_n = self._train_norm_shared(mu, sigma)
                    self._update_running_shared(mu, sigma)
                    mu, sigma = mu_n, sigma_n
                else:
                    self._update_running_shared(mu, sigma)
        else:
            if getattr(self, "refit_mode", False):
                if self.stats_mode == "per_domain":
                    self.refit_slots(*self._domain_stats(x, slot_ids))
                else:
                    mu_r = self.manifold.centroid(x)
                    if x.ndim == 3:
                        mu_r = self.manifold.centroid(mu_r)
                    self.refit_shared(mu_r,
                                      self._batch_frechet_variance(x, mu_r).sqrt())
            mu, sigma = self._eval_stats(slot_ids)
            per_sample_stats = (self.stats_mode == "per_domain")

        # ---- affine -------------------------------------------------------
        if self.affine_mode == "per_subject":
            gamma_b = self.gamma.index_select(0, slot_ids)             # (B, F-1)
            beta_b = self.manifold.expmap0(
                self.beta_tangent.index_select(0, slot_ids))           # (B, F)
        else:
            gamma_b = self.gamma.expand(B, -1)
            beta_b = self.manifold.expmap0(self.beta_tangent).expand(B, -1)

        # ---- gyro path (⊖μ ⊕ x, rescale, β ⊕ ·) ---------------------------
        v = self.manifold.fused_logmap_transp0back(
            self._align(mu, x) if per_sample_stats else mu, x)

        scale = gamma_b / (sigma + self.eps)                           # (B, F-1)
        if v.ndim > 2:
            scale = self._align(scale, v)
            beta_b = self._align(beta_b, v)

        v_s = v[..., 1:] * scale
        v = torch.cat([v[..., :1], v_s], dim=-1)
        v = self.manifold.rescale_to_max_euclid(v)

        return self.manifold.fused_transp0_expmap(beta_b, v)


class GyroLBNCustomAdaptive1d(GyroLBNCustomAdaptive):
    """1-D wrapper (batch, num_features)."""

    def __init__(
        self,
        manifold: CustomLorentz,
        num_features: int,
        num_subjects: int,
        **kwargs,
    ):
        super().__init__(manifold, num_features, num_subjects, **kwargs)


class MaybeAdaptiveBN(nn.Module):
    """Uniform call site for subject-conditioned and plain BN modules.

    Wraps any Lorentz BN so `bn(x, subject_ids)` can be called
    without branching at every site on whether the underlying module is
    subject-aware. Adaptive modules receive both arguments; plain modules
    are called with `x` alone, and `subject_ids` is silently ignored.

    Wrapping is idempotent.
    """

    def __init__(self, bn: nn.Module):
        super().__init__()
        self.bn = bn
        self.adaptive = isinstance(bn, (GyroLBNCustomAdaptive, MaybeAdaptiveBN))

    def forward(
        self, x: torch.Tensor, subject_ids: torch.Tensor = None
    ) -> torch.Tensor:
        if self.adaptive:
            return self.bn(x, subject_ids)
        return self.bn(x)


# ----------------------------------------------------------------------
# Model-level helpers for the domain-specific BN
# ----------------------------------------------------------------------
def domain_bn_modules(model: nn.Module):
    """Every subject-conditioned Lorentz BN in `model`."""
    return [m for m in model.modules() if isinstance(m, GyroLBNCustomAdaptive)]


@torch.no_grad()
def refit_bn_domains(model: nn.Module, x: torch.Tensor,
                     subject_ids: torch.Tensor, verbose: bool = False) -> int:
    """Re-estimate the target domain's BN statistics from unlabeled trials.

    Puts every domain BN in refit mode and runs one forward pass per domain in
    `subject_ids` over that domain's trials. No loss, no backward, no labels.

    Without this the held-out subject reads the pooled Fréchet-mean fallback.

    Returns the number of BN layers refit. Leaves the model in eval mode.
    """
    mods = domain_bn_modules(model)
    if not mods:
        return 0
    model.eval()
    for m in mods:
        m.set_refit(True)
    try:
        for du in subject_ids.unique():
            sel = subject_ids == du
            if not bool(sel.any()):
                continue
            xi = x[sel]
            if xi.dim() == 3:
                xi = xi.unsqueeze(1)
            model(xi, subject_ids[sel])
            if verbose:
                print(f"  refit domain {int(du)} from {int(sel.sum())} trials",
                      flush=True)
    finally:
        for m in mods:
            m.set_refit(False)
    return len(mods)


def set_bn_epoch(model: nn.Module, epoch: int, total_epochs: int = None) -> int:
    """Publish the current epoch to every adaptive BN; returns how many were set.

    Required by the adaptive momentum schedule, which is indexed by epoch rather
    than by step, and by `K` when it is derived from the epoch budget rather
    than pinned.
    """
    mods = domain_bn_modules(model)
    for m in mods:
        m.set_epoch(epoch, total_epochs)
    return len(mods)


def finalize_bn_domain_stats(model: nn.Module) -> int:
    """Refresh every per-domain BN's pooled unseen-domain fallback."""
    n = 0
    for m in domain_bn_modules(model):
        if m.stats_mode == "per_domain":
            m.finalize_pooled_stats()
            n += 1
    return n


def domain_bn_coverage(model: nn.Module):
    """Diagnostic: which domain slots have statistics, and from how many obs.

    Returns a list of `(qualified_name, n_slots_seen, n_slots_total, min_obs)`.
    """
    out = []
    for name, m in model.named_modules():
        if isinstance(m, GyroLBNCustomAdaptive) and m.stats_mode == "per_domain":
            seen = m.domain_updates > 0
            obs = m.domain_obs[seen]
            out.append((name, int(seen.sum()), int(m.num_slots),
                        int(obs.min()) if obs.numel() else 0))
    return out
