import torch
from latte.training.helpers import slice_time_series


@torch.no_grad()
def _pre_decoder_embeds(model, x):
    """Forward through processor + encoder + manifold-centroid, skipping
    the subject-conditional pre_decoder. Returns (B, D+1) Lorentz points.
    """
    if x.dim() == 3:
        x = x.unsqueeze(1)
    return _forward_to_centroid(model, x)


def _forward_to_centroid(model, x, x_sub=None):
    if x_sub is None:
        # neutral routing — slot 0 (unknown).
        x_sub = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    proc = model.processor(x, x_sub)
    proc = model.to_hyperbolic(proc)
    proc = slice_time_series(proc, model.windows)
    emb = model.encoder(proc)
    emb = emb.reshape(x.shape[0], model.windows, -1, emb.shape[-1]).permute(0, 2, 1, 3)
    emb = model.manifold.direct_concat(emb)
    return model.manifold.centroid(emb)  # (B, D+1)


def _lorentz_centroid(points, k, eps=1e-8):
    """Lorentz centroid of (N, D+1) points on a hyperboloid with curvature k."""
    avg = points.mean(dim=0)
    inner = -avg[0] ** 2 + (avg[1:] ** 2).sum()
    denom = inner.abs().clamp(min=eps).sqrt()
    return k.sqrt() * avg / denom


def _lorentz_distance(x, y, k):
    inner = -x[0] * y[0] + (x[1:] * y[1:]).sum()
    arg = (-inner / k).clamp(min=1.0 + 1e-6)
    return k.sqrt() * arg.acosh()


@torch.no_grad()
def find_nearest_training_subject(
    model,
    train_loader,
    heldout_loader,
    heldout_id,
    device,
    max_samples_per_subject=512,
    use_actual_subject_routing=True,
):
    """Compute per-subject Lorentz centroids and return the trained subject
    whose centroid is closest to the held-out subject's centroid.

    Args:
        model: classification model with `get_decoder` already called once.
        train_loader: yields (x, x_sub, y) for the training subjects.
        heldout_loader: yields (x, x_sub, y) for the held-out subject only.
        heldout_id: the held-out subject id (excluded from candidates).
        max_samples_per_subject: cap on samples accumulated per subject.
        use_actual_subject_routing: if True, pass each sample's true subject
                                    id into the model so trained subjects are
                                    measured through their own LoRA slot.
                                    Set False to measure under neutral slot 0.

    Returns:
        (nearest_id: int, distances: dict[int, float])
    """
    was_training = model.training
    model.eval()
    try:
        per_sub_emb = {}  # int -> list[Tensor] (cpu)

        def _accumulate(loader, only_id=None):
            for x, sub, _ in loader:
                x = x.to(device, non_blocking=True)
                sub = sub.to(device, non_blocking=True)
                if x.dim() == 3:
                    x = x.unsqueeze(1)
                if use_actual_subject_routing:
                    z = _forward_to_centroid(model, x, sub)
                else:
                    z = _forward_to_centroid(model, x, None)
                z = z.detach().cpu()
                sub_cpu = sub.detach().cpu()
                for s in torch.unique(sub_cpu).tolist():
                    s_int = int(s)
                    if only_id is not None and s_int != only_id:
                        continue
                    bucket = per_sub_emb.setdefault(s_int, [])
                    have = sum(t.shape[0] for t in bucket)
                    if have >= max_samples_per_subject:
                        continue
                    pick = (sub_cpu == s)
                    take = min(int(pick.sum()), max_samples_per_subject - have)
                    if take > 0:
                        idx = torch.nonzero(pick, as_tuple=False).squeeze(-1)[:take]
                        bucket.append(z[idx])
                # early exit when every needed subject is full
                if only_id is not None:
                    if sum(t.shape[0] for t in per_sub_emb.get(only_id, [])) >= max_samples_per_subject:
                        return

        _accumulate(train_loader)
        _accumulate(heldout_loader, only_id=heldout_id)

        if heldout_id not in per_sub_emb:
            raise RuntimeError(
                f"held-out subject {heldout_id} produced no samples in heldout_loader")

        k = model.manifold.k.detach()
        centroids = {s: _lorentz_centroid(torch.cat(ts, dim=0).to(device), k)
                     for s, ts in per_sub_emb.items() if len(ts) > 0}

        ho_centroid = centroids[heldout_id]
        candidates = {s: c for s, c in centroids.items()
                      if s != heldout_id and s != 0}
        if not candidates:
            raise RuntimeError(
                f"no candidate trained subjects found (got: {list(centroids)})")

        distances = {s: _lorentz_distance(ho_centroid, c, k).item()
                     for s, c in candidates.items()}
        nearest = min(distances, key=distances.get)
        return nearest, distances
    finally:
        if was_training:
            model.train()


# ── SPD covariance metric (model-free) ────────────────────────────────────────

@torch.no_grad()
def _subject_avg_covariance(loader, target_id=None, max_trials=None,
                            reg=1e-4, device="cpu"):
    """Average per-trial channel covariance for each subject in `loader`.

    Returns dict[int, Tensor (C, C)] of regularised SPD matrices on `device`.
    If `target_id` is set, only that subject is collected.
    """
    sums = {}     # int -> Tensor (C, C)
    counts = {}   # int -> int

    for batch in loader:
        if not isinstance(batch, (list, tuple)) or len(batch) != 3:
            continue
        x, sub, _ = batch
        # x: (B, C, T) or (B, 1, C, T) — squeeze the singleton if present
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        x = x.to(device, dtype=torch.float32)
        sub = sub.to("cpu")
        # centre per trial along time
        x = x - x.mean(dim=-1, keepdim=True)
        # per-trial covariance (B, C, C)
        T = x.shape[-1]
        covs = torch.einsum("bct,bdt->bcd", x, x) / max(T - 1, 1)

        for s in torch.unique(sub).tolist():
            s_int = int(s)
            if target_id is not None and s_int != target_id:
                continue
            if max_trials is not None and counts.get(s_int, 0) >= max_trials:
                continue
            mask = (sub == s)
            take = int(mask.sum().item())
            if max_trials is not None:
                take = min(take, max_trials - counts.get(s_int, 0))
                if take <= 0:
                    continue
                idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)[:take]
                contrib = covs[idx].sum(dim=0)
            else:
                contrib = covs[mask].sum(dim=0)
            sums[s_int] = sums.get(s_int, 0) + contrib
            counts[s_int] = counts.get(s_int, 0) + take

    out = {}
    for s, total in sums.items():
        avg = total / max(counts[s], 1)
        # Tikhonov regularisation: add small multiple of trace*I for SPD safety
        c = avg.shape[0]
        eps = reg * (avg.diagonal().mean().abs().item() + 1e-12)
        out[s] = avg + eps * torch.eye(c, dtype=avg.dtype, device=avg.device)
    return out


def _spd_riemann_distance(A, B, eps=1e-8):
    """Affine-invariant Riemannian distance on SPD matrices:
        d(A, B) = || log( eigvals( A^{-1} B ) ) ||_2
    Computed via a generalised eigenvalue formulation:
    eigenvalues of A^{-1}B equal generalised eigvals of (B, A) and are equal
    to eigvals of (L^{-1} B L^{-T}) with A = L L^T (Cholesky).
    """
    L = torch.linalg.cholesky(A)
    Linv = torch.linalg.solve_triangular(L, torch.eye(A.shape[0],
                                                      dtype=A.dtype,
                                                      device=A.device),
                                         upper=False)
    M = Linv @ B @ Linv.T
    # M is similar to A^{-1}B and is symmetric since A,B SPD
    eigs = torch.linalg.eigvalsh((M + M.T) / 2)
    eigs = eigs.clamp(min=eps)
    return torch.sqrt((torch.log(eigs) ** 2).sum())


@torch.no_grad()
def find_nearest_training_subject_spd(
    train_loader,
    heldout_loader,
    heldout_id,
    max_trials_per_subject=None,
    reg=1e-4,
    device="cpu",
):
    """Model-free routing: nearest training subject by Riemannian SPD distance
    on average channel covariance matrices.

    Returns (nearest_id: int, distances: dict[int, float]).
    """
    train_covs = _subject_avg_covariance(
        train_loader, target_id=None,
        max_trials=max_trials_per_subject, reg=reg, device=device,
    )
    ho_covs = _subject_avg_covariance(
        heldout_loader, target_id=heldout_id,
        max_trials=max_trials_per_subject, reg=reg, device=device,
    )
    if heldout_id not in ho_covs:
        raise RuntimeError(
            f"held-out subject {heldout_id} produced no samples in heldout_loader")

    candidates = {s: c for s, c in train_covs.items()
                  if s != heldout_id and s != 0}
    if not candidates:
        raise RuntimeError(
            f"no candidate trained subjects found (got: {list(train_covs)})")

    ho = ho_covs[heldout_id]
    distances = {s: _spd_riemann_distance(c, ho).item()
                 for s, c in candidates.items()}
    nearest = min(distances, key=distances.get)
    return nearest, distances
