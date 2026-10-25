from .GyroLBN import (
    GyroLBNCustom1d,
    GyroLBNCustomAdaptive1d,
    MaybeAdaptiveBN,
    domain_bn_modules,
    set_bn_epoch,
    finalize_bn_domain_stats,
    domain_bn_coverage,
    refit_bn_domains,
)

# batch_type -> 1-D Lorentz batch norm
#   gyro                  per-feature gamma and Frechet variance (default)
#   gyro_custom_adaptive  shared statistics, per-subject affine
LORENTZ_BN1D_TYPES = {
    "gyro":        GyroLBNCustom1d,
    "gyro_custom_adaptive": GyroLBNCustomAdaptive1d,
}

_ADAPTIVE_BN_TYPES = {"gyro_custom_adaptive"}

# Accepted only by the adaptive BN; dropped for the plain variants.
_ADAPTIVE_ONLY_KWARGS = {
    "stats_mode", "affine_mode", "momentum_mode", "gamma_min",
    "momentum_K", "momentum_K_frac", "eval_bs_ratio", "min_domain_obs",
}


def is_adaptive_bn(batch_type) -> bool:
    """True when `batch_type` names a subject-conditioned BN."""
    return batch_type in _ADAPTIVE_BN_TYPES


def build_lorentz_bn1d(batch_type, manifold, num_features, num_subjects=None,
                       wrap=False, **kwargs):
    """Construct a 1-D Lorentz BN, threading `num_subjects` only when needed.
    With wrap=True the result is wrapped in MaybeAdaptiveBN so every call site
    can use ``bn(x, subject_ids)``."""
    cls = LORENTZ_BN1D_TYPES[batch_type]
    if is_adaptive_bn(batch_type):
        assert num_subjects is not None, f"BN type '{batch_type}' requires num_subjects."
        bn = cls(manifold, num_features, num_subjects=num_subjects, **kwargs)
    else:
        for k in set(kwargs) & _ADAPTIVE_ONLY_KWARGS:
            kwargs.pop(k)
        bn = cls(manifold, num_features, **kwargs)
    return MaybeAdaptiveBN(bn) if wrap else bn
