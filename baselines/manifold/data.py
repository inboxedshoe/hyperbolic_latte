import numpy as np
import torch
import yaml


def _tensors(loader):
    """(X, subject_ids, y) out of the loader's dataset, without its collate."""
    ds = loader.dataset
    if hasattr(ds, "tensors"):
        t = ds.tensors
    else:
        dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)
        parts = list(zip(*dl))
        t = tuple(torch.cat(p) for p in parts)
    if len(t) == 3:
        X, sid, y = t
    else:                                  # include_subject_ids was False
        X, y = t
        sid = torch.full((len(y),), -1, dtype=torch.long)
    return (X.float().numpy(), sid.numpy().astype(np.int64),
            y.numpy().astype(np.int64))


def load_loaders(config, subject, batch_size, loso, interaug=False, z_scale=True):
    """`build_datamodule`'s loaders.

    Callers read the split tensors out of `loader.dataset` and never iterate
    these loaders, so no worker process is ever started. The worker count is
    pinned to one.
    """
    from latte.training.utils import build_datamodule

    dm, meta = build_datamodule(config, subject, batch_size, interaug=interaug,
                                z_scale=z_scale, include_subject_ids=True, loso=loso)
    dm.preprocessing_dict["num_workers"] = 1
    return (dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader(),
            meta["channels"], meta["classes"], meta["subjects"])


def load_setting(dataset, setting, heldout, config="configs/datasets.yaml",
                 interaug=False, z_scale=True, bs=48, data_path=None):
    """Return the arrays and index sets for one (dataset, setting, fold) cell.

    Args:
        setting: "ss"   per-subject   (train = subject's train session(s))
                 "sc"   all subjects, test subject present in the training pool
                 "loso" held-out subject, never seen during training
        heldout: the SS subject, or the LOSO held-out subject. Ignored for SC.

    Returns a dict with X/y/domain arrays for train, val and test, plus the
    domain-id bookkeeping the domain-specific batch norms need.

    `interaug` (LAtte's augmentation) defaults to False.
    """
    with open(config) as f:
        cfg = yaml.safe_load(f)
    cfg["dataset_name"] = dataset
    if data_path:
        cfg["data_path"] = data_path

    if setting == "sc":
        subject, loso = "all", False
    elif setting == "ss":
        subject, loso = int(heldout), False
    elif setting == "loso":
        subject, loso = int(heldout), True
    else:
        raise ValueError(setting)

    trainloader, validloader, testloader, n_ch, n_cls, n_subj = load_loaders(
        cfg, subject, bs, loso, interaug=interaug, z_scale=z_scale)

    Xtr, dtr, ytr = _tensors(trainloader)
    Xva, dva, yva = _tensors(validloader)
    Xte, dte, yte = _tensors(testloader)

    val_is_test = (
        Xva.shape == Xte.shape
        and bool(np.array_equal(yva, yte))
        and bool(np.array_equal(Xva, Xte))
    )

    # Domain = subject, encoded densely over every subject in this cell.
    subjects = np.unique(np.concatenate([dtr, dva, dte]))
    remap = {int(s): i for i, s in enumerate(subjects)}
    to_dom = np.vectorize(lambda s: remap[int(s)], otypes=[np.int64])

    train_domains = np.unique(to_dom(dtr))
    test_domains = np.unique(to_dom(dte))

    return dict(
        X_train=Xtr.astype(np.float32), y_train=ytr, d_train=to_dom(dtr),
        X_val=Xva.astype(np.float32),   y_val=yva,   d_val=to_dom(dva),
        X_test=Xte.astype(np.float32),  y_test=yte,  d_test=to_dom(dte),
        n_channels=int(Xtr.shape[1]), n_samples=int(Xtr.shape[2]),
        n_classes=int(n_cls), n_domains=int(len(subjects)),
        subject_of_domain={int(v): int(k) for k, v in remap.items()},
        train_domains=train_domains, test_domains=test_domains,
        unseen_domains=np.setdiff1d(test_domains, train_domains),
        val_is_test=val_is_test, z_scale=z_scale,
    )


def summarise(d):
    return (f"train {d['X_train'].shape} / val {d['X_val'].shape} / "
            f"test {d['X_test'].shape} | {d['n_classes']}cl {d['n_channels']}ch "
            f"{d['n_samples']}samp | domains: {len(d['train_domains'])} train, "
            f"{len(d['unseen_domains'])} unseen at test | "
            f"val_is_test={d['val_is_test']}")
