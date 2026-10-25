import numpy as np

__all__ = ["fit_subject_alignment", "apply_subject_alignment", "align_splits"]


def _inv_sqrtm_psd(cov, ridge=1e-6):
    """inv(sqrtm(cov)) for a symmetric PSD matrix, via eigendecomposition."""
    cov = np.asarray(cov, dtype=np.float32)
    cov = 0.5 * (cov + cov.T)
    w, V = np.linalg.eigh(cov)
    floor = max(ridge, ridge * float(w.max())) if w.size and w.max() > 0 else ridge
    w = np.clip(w, floor, None)
    return (V * (w ** -0.5)) @ V.T


def fit_subject_alignment(X, subjects, ridge=1e-6):
    """Per-subject whitening transforms fitted on the trials given.

    Args:
        X:        (n_trials, n_channels, n_times)
        subjects: (n_trials,) subject id per trial
    Returns:
        {subject_id: (n_channels, n_channels) transform}
    """
    X = np.asarray(X)
    subjects = np.asarray(subjects).reshape(-1)
    transforms = {}
    for s in np.unique(subjects):
        Xs = X[subjects == s]
        if len(Xs) == 0:
            continue
        # Mean of per-trial covariances.
        covs = np.einsum("nct,ndt->ncd", Xs, Xs, optimize=True) / max(Xs.shape[-1] - 1, 1)
        transforms[int(s)] = _inv_sqrtm_psd(covs.mean(axis=0), ridge=ridge)
    return transforms


def apply_subject_alignment(X, subjects, transforms, fallback="identity"):
    """Apply per-subject transforms; subjects with no fitted transform are left
    alone (`identity`) or given the mean of the fitted transforms (`grand`).

    `identity` (the default) applies no transform to subjects absent from
    training.
    """
    X = np.asarray(X)
    subjects = np.asarray(subjects).reshape(-1)
    out = np.array(X, dtype=np.float32, copy=True)
    grand = (np.mean(list(transforms.values()), axis=0)
             if transforms and fallback == "grand" else None)
    unaligned = []
    for s in np.unique(subjects):
        m = subjects == s
        T = transforms.get(int(s))
        if T is None:
            if grand is None:
                unaligned.append(int(s))
                continue
            T = grand
        out[m] = np.einsum("cd,ndt->nct", T, X[m].astype(np.float32),
                           optimize=True).astype(np.float32)
    return out, unaligned


def align_splits(X_train, s_train, others, ridge=1e-6, fallback="identity"):
    """Fit on train, apply to train and to every (X, s) pair in `others`.

    Returns (X_train_aligned, [X_other_aligned, ...], info).
    """
    transforms = fit_subject_alignment(X_train, s_train, ridge=ridge)
    Xtr, _ = apply_subject_alignment(X_train, s_train, transforms, fallback)
    outs, unaligned_all = [], []
    for Xo, so in others:
        Xa, un = apply_subject_alignment(Xo, so, transforms, fallback)
        outs.append(Xa)
        unaligned_all.extend(un)
    info = dict(n_subjects_fitted=len(transforms), mode="fit_on_train",
                unaligned_subjects=sorted(set(unaligned_all)))
    return Xtr, outs, info


def align_splits_ea_all_trials(splits, ridge=1e-6):
    """EA per subject, over every trial present.

    The splits are pooled and each subject's transform is fitted on all of that
    subject's trials, so a held-out subject's whitening matrix is estimated
    from that subject's own trials.

    Args:
        splits: ordered list of (X, subject_ids) pairs, e.g. [train, val, test].
    Returns:
        ([X_aligned, ...], info)
    """
    Xs = [np.asarray(X) for X, _ in splits]
    ss = [np.asarray(s).reshape(-1) for _, s in splits]
    pooled_X = np.concatenate(Xs, axis=0)
    pooled_s = np.concatenate(ss, axis=0)

    transforms = fit_subject_alignment(pooled_X, pooled_s, ridge=ridge)
    outs, unaligned_all = [], []
    for X, s in zip(Xs, ss):
        Xa, un = apply_subject_alignment(X, s, transforms, fallback="identity")
        outs.append(Xa)
        unaligned_all.extend(un)
    info = dict(n_subjects_fitted=len(transforms), mode="ea_all_trials",
                unaligned_subjects=sorted(set(unaligned_all)))
    return outs, info
