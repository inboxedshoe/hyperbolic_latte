from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _four_point_delta_batch(
    D: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    k: torch.Tensor,
    l: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-quadruple δ for batches of index quadruples (i, j, k, l).

    δ(i,j,k,l) = (S_1 - S_2) / 2 where S_1 ≥ S_2 are the two largest
    of the three sums { d(i,j)+d(k,l), d(i,k)+d(j,l), d(i,l)+d(j,k) }.

    Parameters
    ----------
    D : (N, N) pairwise distance matrix
    i, j, k, l : (Q,) index tensors

    Returns
    -------
    (Q,) δ values for each quadruple
    """
    s1 = D[i, j] + D[k, l]
    s2 = D[i, k] + D[j, l]
    s3 = D[i, l] + D[j, k]

    # Stack and sort descending along last dim
    sums = torch.stack([s1, s2, s3], dim=-1)          # (Q, 3)
    sums_sorted, _ = sums.sort(dim=-1, descending=True)

    delta = (sums_sorted[:, 0] - sums_sorted[:, 1]) / 2.0
    return delta


def gromov_delta(
    features: torch.Tensor,
    n_quadruples: int = 50_000,
    batch_size: int = 10_000,
    seed: int = 42,
    quantile: float = 1.0,
    dist_matrix: Optional[torch.Tensor] = None,
    bootstrap_ci: bool = True,
    n_bootstrap: int = 200,
) -> Dict[str, float]:
    """
    Estimate Gromov δ-hyperbolicity from a set of feature vectors.

    Parameters
    ----------
    features : (N, D) tensor of feature vectors (Euclidean L2 distance used)
    n_quadruples : int
        Number of random quadruples to sample.
    batch_size : int
        Process this many quadruples at a time to manage memory.
    seed : int
        RNG seed for reproducibility.
    quantile : float
        Which quantile of δ values to report as the "global" δ.
        1.0 = max (true supremum estimate).
    dist_matrix : (N, N) tensor, optional
        Precomputed distance matrix. If None, Euclidean L2 is computed from
        `features`.
    bootstrap_ci : bool
        If True, compute a 95% bootstrap CI for delta_q95 and delta_rel_q95
        by resampling the sampled quadruples. Captures within-seed
        quadruple-sampling uncertainty (complements the across-seed std
        reported by gromov_delta_robust).
    n_bootstrap : int
        Number of bootstrap resamples.

    Returns
    -------
    dict with keys:
        delta_max            : max δ over sampled quadruples
        delta_q95            : 95th-percentile δ
        delta_mean           : mean δ
        delta_std            : std of δ distribution
        diameter             : max pairwise distance in the sample
        delta_rel            : delta_max / diameter  (normalized)
        delta_rel_q95        : delta_q95 / diameter  (normalized)
        n_quadruples         : actual number of quadruples evaluated
        n_points             : N
        delta_q95_boot_lo    : lower 2.5% bootstrap bound for delta_q95
        delta_q95_boot_hi    : upper 97.5% bootstrap bound for delta_q95
        delta_rel_q95_boot_lo: lower 2.5% bootstrap bound for delta_rel_q95
        delta_rel_q95_boot_hi: upper 97.5% bootstrap bound for delta_rel_q95
    """
    N = features.shape[0]
    device = features.device

    if N < 4:
        raise ValueError(f"Need at least 4 points, got {N}.")

    # Build distance matrix if not provided
    if dist_matrix is None:
        # cdist can be memory-heavy for large N; chunk if needed
        if N > 5000:
            warnings.warn(
                f"N={N} is large. Building {N}x{N} distance matrix may use "
                f"{N*N*4/1e6:.0f} MB. Consider subsampling first."
            )
        dist_matrix = torch.cdist(features.float(), features.float(), p=2)

    diameter = dist_matrix.max().item()
    if diameter < 1e-8:
        warnings.warn("All points appear identical (diameter ≈ 0). δ is trivially 0.")
        return {
            "delta_max": 0.0, "delta_q95": 0.0, "delta_mean": 0.0,
            "delta_std": 0.0, "diameter": 0.0, "delta_rel": 0.0,
            "delta_rel_q95": 0.0, "n_quadruples": 0, "n_points": N,
            "delta_q95_boot_lo": 0.0, "delta_q95_boot_hi": 0.0,
            "delta_rel_q95_boot_lo": 0.0, "delta_rel_q95_boot_hi": 0.0,
        }

    # Sample random quadruples
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    # Rejection-sample distinct quadruples in batches
    all_deltas = []
    total_sampled = 0

    while total_sampled < n_quadruples:
        remaining = n_quadruples - total_sampled
        current_batch = min(batch_size, remaining)

        # Sample 4 indices per quadruple; re-roll duplicates (rare for large N)
        idx = torch.randint(0, N, (current_batch, 4), generator=rng, device=device)

        # Filter out degenerate quadruples (any repeated index)
        # Sort within each row and check for duplicates
        idx_sorted, _ = idx.sort(dim=1)
        distinct = (idx_sorted[:, 1:] != idx_sorted[:, :-1]).all(dim=1)
        idx = idx[distinct]

        if idx.shape[0] == 0:
            continue

        i, j, k, l = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
        deltas = _four_point_delta_batch(dist_matrix, i, j, k, l)
        all_deltas.append(deltas)
        total_sampled += idx.shape[0]

    all_deltas = torch.cat(all_deltas)
    n_actual = all_deltas.shape[0]

    delta_max = all_deltas.max().item()
    delta_q95 = torch.quantile(all_deltas, 0.95).item()
    delta_mean = all_deltas.mean().item()
    delta_std = all_deltas.std().item()

    result = {
        "delta_max": delta_max,
        "delta_q95": delta_q95,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "diameter": diameter,
        "delta_rel": delta_max / diameter,
        "delta_rel_q95": delta_q95 / diameter,
        "n_quadruples": n_actual,
        "n_points": N,
    }

    if bootstrap_ci and n_actual >= 10:
        # Bootstrap resample the quadruple δ values to get a 95% CI for
        # delta_q95.
        cpu_deltas = all_deltas.cpu()
        rng_boot = torch.Generator()
        rng_boot.manual_seed(seed + 99999)
        boot_q95s: List[torch.Tensor] = []
        done = 0
        chunk = 50  # keep peak memory around 50×n_actual×4 bytes
        while done < n_bootstrap:
            c = min(chunk, n_bootstrap - done)
            idx_b = torch.randint(0, n_actual, (c, n_actual), generator=rng_boot)
            boot_q95s.append(torch.quantile(cpu_deltas[idx_b], 0.95, dim=1))
            done += c
        boot_q95_tensor = torch.cat(boot_q95s)
        lo = torch.quantile(boot_q95_tensor, 0.025).item()
        hi = torch.quantile(boot_q95_tensor, 0.975).item()
        result["delta_q95_boot_lo"] = lo
        result["delta_q95_boot_hi"] = hi
        result["delta_rel_q95_boot_lo"] = lo / diameter
        result["delta_rel_q95_boot_hi"] = hi / diameter

    return result


# ---------------------------------------------------------------------------
# Multi-seed wrapper (sampling-variance estimate of δ)
# ---------------------------------------------------------------------------

_AGG_KEYS = (
    "delta_max", "delta_q95", "delta_mean", "delta_std",
    "diameter", "delta_rel", "delta_rel_q95",
    # bootstrap CI keys (present only when bootstrap_ci=True)
    "delta_q95_boot_lo", "delta_q95_boot_hi",
    "delta_rel_q95_boot_lo", "delta_rel_q95_boot_hi",
)


def gromov_delta_robust(
    features: torch.Tensor,
    n_quadruples: int = 50_000,
    n_seeds: int = 5,
    base_seed: int = 42,
    batch_size: int = 10_000,
    dist_matrix: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Run gromov_delta with `n_seeds` different RNG seeds and report mean/std
    across runs. The std across seeds estimates the
    finite-sample uncertainty in δ_rel and δ_rel_q95 due to quadruple sampling.

    The pairwise distance matrix is computed once and reused across seeds.

    Returns
    -------
    dict with the same keys as gromov_delta, plus '{key}_std' giving the
    across-seed standard deviation. Also adds 'n_seeds' and 'feature_dim'.
    """
    N, D = features.shape
    if dist_matrix is None:
        if N > 5000:
            warnings.warn(
                f"N={N} is large; building {N}x{N} distance matrix may use "
                f"{N*N*4/1e6:.0f} MB."
            )
        dist_matrix = torch.cdist(features.float(), features.float(), p=2)

    runs: List[Dict[str, float]] = []
    for k in range(n_seeds):
        runs.append(gromov_delta(
            features,
            n_quadruples=n_quadruples,
            batch_size=batch_size,
            seed=base_seed + k,
            dist_matrix=dist_matrix,
        ))

    out: Dict[str, float] = {}
    for key in _AGG_KEYS:
        if not all(key in r for r in runs):
            continue  # skip optional keys absent from some runs
        vals = torch.tensor([r[key] for r in runs], dtype=torch.float32)
        out[key] = vals.mean().item()
        out[f"{key}_std"] = vals.std(unbiased=True).item() if n_seeds > 1 else 0.0

    out["n_seeds"] = n_seeds
    out["n_quadruples"] = runs[0]["n_quadruples"]
    out["n_points"] = N
    out["feature_dim"] = D
    return out


# ---------------------------------------------------------------------------
# Comparison against a Euclidean random baseline
# ---------------------------------------------------------------------------

def _build_baseline(
    features: torch.Tensor,
    mode: str,
    n_baseline: int,
    seed: int,
) -> torch.Tensor:
    """
    Construct a null-distribution feature matrix that destroys data structure
    while controlling for ambient scale.

    Modes
    -----
    "gaussian"
        Independent Gaussian samples with per-dimension std matched to the
        data (an isotropic baseline of matched scale).
    "shuffle"
        Per-dimension permutation: each feature column is independently
        shuffled across rows. Preserves marginal feature distributions
        exactly.
    """
    N, D = features.shape
    device = features.device
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    feats = features.float()

    if mode == "gaussian":
        std_per_dim = feats.std(dim=0)                 # (D,)
        z = torch.randn(n_baseline, D, generator=rng, device=device, dtype=torch.float32)
        return z * std_per_dim
    elif mode == "shuffle":
        if n_baseline != N:
            warnings.warn(
                f"shuffle baseline yields exactly N={N} points; ignoring "
                f"n_baseline_samples={n_baseline}."
            )
        # Vectorised per-column permutation: argsort of fresh random uniform
        # gives an iid permutation per column.
        rand_keys = torch.rand(N, D, generator=rng, device=device)
        perm = rand_keys.argsort(dim=0)                 # (N, D)
        return feats.gather(0, perm)
    else:
        raise ValueError(f"Unknown baseline mode: {mode!r}")


def gromov_delta_with_baseline(
    features: torch.Tensor,
    n_quadruples: int = 50_000,
    n_seeds: int = 5,
    n_baseline_samples: Optional[int] = None,
    seed: int = 42,
    baseline_mode: str = "gaussian",
    batch_size: int = 10_000,
) -> Dict[str, float]:
    """
    Compute δ for the real features and compare to a structureless baseline.

    Each side is averaged over `n_seeds` quadruple-samplings so Δ comes with
    an across-seed std (assuming independence between data and baseline runs).

    Parameters
    ----------
    features : (N, D) feature tensor
    n_quadruples : int
    n_seeds : int
        Number of independent quadruple-samplings per side.
    n_baseline_samples : int, optional
        Baseline cardinality. Defaults to N. (Ignored for "shuffle" mode.)
    seed : int
    baseline_mode : {"gaussian", "shuffle"}
        See _build_baseline.
    batch_size : int

    Returns
    -------
    dict merging real and baseline results (baseline keys prefixed
    'baseline_'), plus `delta_rel_vs_baseline` and `delta_rel_vs_baseline_std`.
    """
    N, D = features.shape
    n_bl = n_baseline_samples or N

    result = gromov_delta_robust(
        features,
        n_quadruples=n_quadruples,
        n_seeds=n_seeds,
        base_seed=seed,
        batch_size=batch_size,
    )

    if baseline_mode == "shuffle":
        # Build a fresh shuffle per seed so that the across-seed std captures
        # both quadruple-sampling variance AND shuffle-realization variance.
        baseline_runs: List[Dict[str, float]] = []
        for k in range(n_seeds):
            bl_k = _build_baseline(features, "shuffle", n_bl, seed=seed + 1 + k)
            bl_dist = torch.cdist(bl_k.float(), bl_k.float(), p=2)
            baseline_runs.append(gromov_delta(
                bl_k,
                n_quadruples=n_quadruples,
                batch_size=batch_size,
                seed=seed + 1000 + k,
                dist_matrix=bl_dist,
            ))
        baseline_result: Dict[str, float] = {}
        for key in _AGG_KEYS:
            if not all(key in r for r in baseline_runs):
                continue
            vals = torch.tensor([r[key] for r in baseline_runs], dtype=torch.float32)
            baseline_result[key] = vals.mean().item()
            baseline_result[f"{key}_std"] = (
                vals.std(unbiased=True).item() if n_seeds > 1 else 0.0
            )
        baseline_result["n_seeds"] = n_seeds
        baseline_result["n_quadruples"] = baseline_runs[0]["n_quadruples"]
        baseline_result["n_points"] = baseline_runs[0]["n_points"]
        baseline_result["feature_dim"] = D
    else:
        baseline_feats = _build_baseline(features, baseline_mode, n_bl, seed=seed + 1)
        baseline_result = gromov_delta_robust(
            baseline_feats,
            n_quadruples=n_quadruples,
            n_seeds=n_seeds,
            base_seed=seed + 1000,
            batch_size=batch_size,
        )

    combined = dict(result)
    for key, val in baseline_result.items():
        combined[f"baseline_{key}"] = val

    # Δ = real - baseline.
    # Two variants: max-based (delta_rel_vs_baseline) and q95-based
    # (based on delta_rel_q95).
    diff_max = result["delta_rel"] - baseline_result["delta_rel"]
    diff_max_std = (result.get("delta_rel_std", 0.0) ** 2
                   + baseline_result.get("delta_rel_std", 0.0) ** 2) ** 0.5
    combined["delta_rel_vs_baseline"] = diff_max          # max-based
    combined["delta_rel_vs_baseline_std"] = diff_max_std

    diff_q95 = result["delta_rel_q95"] - baseline_result["delta_rel_q95"]
    diff_q95_std = (result.get("delta_rel_q95_std", 0.0) ** 2
                   + baseline_result.get("delta_rel_q95_std", 0.0) ** 2) ** 0.5
    combined["delta_rel_q95_vs_baseline"] = diff_q95      # q95-based
    combined["delta_rel_q95_vs_baseline_std"] = diff_q95_std
    combined["baseline_mode"] = baseline_mode

    return combined


# ---------------------------------------------------------------------------
# Per-class analysis
# ---------------------------------------------------------------------------

def gromov_delta_per_class(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_quadruples: int = 20_000,
    n_seeds: int = 3,
    seed: int = 42,
    batch_size: int = 10_000,
) -> Dict[int, Dict[str, float]]:
    """
    Compute δ separately within each class, with multi-seed uncertainty.

    Returns
    -------
    dict mapping class_id -> gromov_delta_robust result dict
    """
    classes = labels.unique().tolist()
    results = {}
    for cls in classes:
        mask = labels == cls
        cls_feats = features[mask]
        if cls_feats.shape[0] < 4:
            warnings.warn(f"Class {cls} has fewer than 4 samples; skipping.")
            continue
        results[int(cls)] = gromov_delta_robust(
            cls_feats,
            n_quadruples=n_quadruples,
            n_seeds=n_seeds,
            base_seed=seed,
            batch_size=batch_size,
        )
    return results
