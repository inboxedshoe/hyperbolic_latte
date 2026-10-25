import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from baselines.gyroatt.models import GyroAttBCI


DEFAULTS = dict(
    epochs=130,
    batch_size=50,
    lr=1e-3,
    weight_decay=1e-2,
    in_size=20,
    p=8,
    patches=3,
    spd_metric="LEM",
    power=0.5,
    activation_power=0.5,
    manifold="SPD",
)


class _ModelArgs:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def build(data, device, cfg=None):
    cfg = {**DEFAULTS, **(cfg or {})}
    margs = _ModelArgs(
        n_channels=data["n_channels"], num_classes=data["n_classes"],
        in_size=cfg["in_size"], p=cfg["p"], epochs=cfg["patches"],
        linear_weight=0.2, spd_metric=cfg["spd_metric"],
        power=cfg["power"], activation_power=cfg["activation_power"],
        manifold=cfg["manifold"],
        spd_power=cfg["power"] * cfg["activation_power"],
    )
    net = GyroAttBCI(margs).to(device)
    return net, cfg


def loaders(data, cfg, device):
    def ds(split):
        return TensorDataset(
            torch.from_numpy(np.ascontiguousarray(data[f"X_{split}"])).float(),
            torch.from_numpy(data[f"y_{split}"]).long())
    tr = DataLoader(ds("train"), batch_size=cfg["batch_size"], shuffle=True,
                    num_workers=0, drop_last=False)
    va = DataLoader(ds("val"), batch_size=512, shuffle=False, num_workers=0)
    te = DataLoader(ds("test"), batch_size=512, shuffle=False, num_workers=0)
    return tr, va, te


@torch.no_grad()
def predict(net, loader, device):
    net.eval()
    ys, logits = [], []
    for x, y in loader:
        x = x.to(device)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        out = net(x)
        logits.append(out.cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(logits)
