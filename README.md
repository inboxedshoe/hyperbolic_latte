# LAtte: Hyperbolic Lorentz Attention for Joint-Subject EEG Classification

Official implementation of LAtte, a hyperbolic model for cross-subject EEG
classification. A Euclidean multi-kernel CNN with per-subject LoRA feeds a
Lorentz contrast encoder (a reference and a task Lorentz InceptionTime block,
combined by parallel transport, then Lorentz attention). A projection with a
per-subject Lorentz Boost Adapter and a Lorentz prototype decoder follow.

## Installation

```bash
conda create -n latte python=3.11
conda activate latte
pip install -r requirements.txt
```

All code runs in float32.

## Data

| `--dataset` | Dataset | Source |
|---|---|---|
| `bcic2a` | BCI Competition IV 2a | downloaded by MOABB on first use |
| `bcic2b` | BCI Competition IV 2b | downloaded by MOABB on first use |
| `bcicha` | BCI Challenge (ERN) | `Data_S02_Sess.mat` ... in `./data/BCIcha/` |

The BCI Challenge files follow the preprocessing and splits of
MAtt [Pan et al., 2022]. Pass `--data_path` or set `data_path` in
`configs/datasets.yaml` to use another location. Preprocessing for every
dataset is set in `configs/datasets.yaml`.

## Training

The hyperparameters of every paper run are in `configs/<task>/<dataset>.cfg`
(one `key = value` per line, keys are the command-line options). Pass one with
`--cfg`; any flag given on the command line overrides the file.

```bash
# the paper's subject-conditional configuration for BCIC-IV 2a
python scripts/train.py --cfg configs/sc/bcic2a.cfg --sub all --seed 3330

# subject-conditional (one model for all subjects)
python scripts/train.py --dataset bcic2a --sub all --seed 3330

# subject-specific, from scratch
python scripts/train.py --dataset bcic2a --sub 3 --seed 3330

# subject-conditional + fine-tuning, starting from a subject-conditional checkpoint
python scripts/finetune_subject.py --dataset bcic2a \
    --checkpoint runs/<run>/checkpoints/best_val_suball.pt --iterations 50

# leave-one-subject-out, with the slot-0 generalist adapter; the head embeddings of
# each batch are centred on their mean (--head_center batch, the LOSO default)
python scripts/train_loso.py --dataset bcic2a --subject_dropout_p 0.2 --mask_test_subject true
```

`python scripts/<script>.py --help` lists every option. Each run writes
`params.json`, the per-epoch curves in `results.csv` and the best-validation
checkpoint to `runs/<timestamp>_<run_name>/`.

### Epoch selection

A reported number is chosen from `results.csv` after training
(`latte/training/selection.py`). With a validation split, the rule takes the
test accuracy at the peak of the 5-epoch centred moving average of validation
accuracy. Without one (the validation loader falls back to the test loader),
the rule takes the best test epoch.

```bash
python scripts/aggregate.py runs/ --filter tag=table3_loso
```

BCIC-IV 2b has no validation split separate from the test session.
BCIC-IV 2a has one only with `--val_split`.

## Reproducing the paper

Each script in `experiments/` prints the commands for one table, one
`--cfg configs/<task>/<dataset>.cfg` per dataset plus seeds and run names. Run
them with the job queue, then aggregate:

```bash
bash experiments/table3_15_loso.sh > jobs.txt
python scripts/run_queue.py --gpus 0,1,2,3 --per_gpu 2 jobs.txt
python scripts/aggregate.py runs/ --filter tag=table3_loso
```

| Paper | Script |
|---|---|
| Tables 1-2, LAtte SC and SS | `experiments/table1_2_latte.sh` |
| Tables 1-2, LAtte SC + FT | `experiments/table1_2_finetune.sh runs/` (after the SC runs) |
| Table 3, Table 15, LOSO | `experiments/table3_15_loso.sh` |
| Table 4, LOSO with Euclidean Alignment | `experiments/table4_ea_all_trials.sh` |
| Table 5, HEEGNet protocol | `baselines/manifold/experiments.sh` |

## Citation

```bibtex
@inproceedings{bdeir2026latte,
  title     = {{LAtte}: Hyperbolic Lorentz Attention for Joint-Subject {EEG} Classification},
  author    = {Bdeir, Ahmad and Burchert, Johannes and Hanika, Tom and Schmidt-Thieme, Lars and Landwehr, Niels},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgements

This code builds on TCFormer [Altaheri et al., 2025], the Lorentz layers of
Bdeir et al. [2024], the PLFC layer and Lorentz GyroBN of Shi et al. [2025], MAtt [Pan et al., 2022], Burchert et al. [2024], and
TSMNet, HEEGNet and GyroAtt for the baselines (see `baselines/`).

## License

MIT, see [LICENSE](LICENSE). The copies of TSMNet and HEEGNet in
`baselines/external/` remain under their authors' terms (see
`baselines/external/README.md`).
