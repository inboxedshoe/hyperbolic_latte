# TSMNet and HEEGNet

Copies of the authors' code, used by `baselines/manifold/`.

| Repo | Commit | License |
|---|---|---|
| [TSMNet](https://github.com/rkobler/TSMNet) (Kobler et al., NeurIPS 2022) | `90293b9` | BSD-3-Clause |
| [HEEGNet](https://github.com/fightlesliefigt/HEEGNet) (Li et al., ICLR 2026) | `5ad0df2` | none |

Extra dependencies: `hydra-core`, `omegaconf`, `moabb`, `mne`.

## Changes

TSMNet
- `spdnets/models/tsmnet.py`: `spd_dtype` argument in place of the hard-coded
  `torch.double` (default unchanged), so the baseline runs in fp32.

HEEGNet
- `lib/geoopt/optim/rlinesearch.py`: import fix for scipy >= 1.11.
- `nets/trainer.py`: `float(...)` instead of `.item()` for scikit-learn >= 1.6.
- `nets/model.py`: `forward` handles `domain_adaptation=False`.
