import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import LambdaLR

from latte.data.cutfill import CutFillConfig, apply_cut_and_fill
from latte.geometry.optim import RiemannianAdamW
from latte.training.lr_scheduler import linear_warmup_cosine_decay
from latte.training.save_results import save_output
from latte.training.utils import get_param_groups, clip_gradients_value, count_params

try:
    import wandb
except ImportError:
    wandb = None


# Cut-and-fill: one span of 5-15 % of the trial, all channels,
# filled with the per-channel mean.
CUTFILL = CutFillConfig(min_frac=0.05, max_frac=0.15, fill_value="mean")


def trainNetworkMultiple(net, trainloader, validloader, testloader, dataset=None,
                         iterations=500, lr=5e-4, wd=None, sub=None,
                         verbose=True, hyperbolic=False, clip_grad=0, num_classes=0,
                         scheduler=False, cutfill=False, run_dir=None, checkpoints_dir=None,
                         epoch_end_callback=None, device="cuda:0", **kwargs):
    best_test = 0
    best_val = 0
    test_from_best_val = 0
    test_from_best_val_loss = 0
    test_from_best_train_loss = 0
    best_train_loss = 1e10
    update_best_test = 0

    CE = nn.CrossEntropyLoss()

    if not hyperbolic:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                     lr=lr, betas=(0.9, 0.999), weight_decay=wd)
    else:
        optimizer = RiemannianAdamW(get_param_groups(net, lr, wd), lr=lr, weight_decay=wd)

    scheduler = LambdaLR(optimizer, linear_warmup_cosine_decay(3, iterations)) if scheduler else None

    bestLoss = 1e10
    train_losses, train_accs, val_losses, val_accs, test_accs, test_losses = [], [], [], [], [], []

    count_params(net)

    for ite in range(iterations):
        net.train()
        acc_val = acc_tr = tr_len = val_len = 0

        # ------------------------------------------------------------ train
        for x_orig, x_sub, yb in trainloader:
            tr_len += yb.shape[0]
            if len(x_orig.shape) == 3:
                x_orig = x_orig.unsqueeze(1)
            if cutfill:
                x_orig, _, _ = apply_cut_and_fill(x_orig, CUTFILL)

            x_orig, x_sub, yb = x_orig.to(device), x_sub.to(device), yb.to(device)

            out = net(x_orig, x_sub)
            loss = CE(out, yb)

            optimizer.zero_grad()
            loss.backward()
            if clip_grad > 0:
                clip_gradients_value(net, clip_grad, losses=[loss])
            optimizer.step()
            acc_tr += (torch.max(out, 1).indices == yb).sum().item()

        if loss.item() < best_train_loss:
            best_train_loss = loss.item()
            update_best_train_loss = True
        else:
            update_best_train_loss = False

        if scheduler is not None:
            scheduler.step()

        # --------------------------------------------------------- validate
        net.eval()
        TL = 0
        for x_orig, x_sub, yb in validloader:
            x_orig, x_sub, yb = x_orig.to(device), x_sub.to(device), yb.to(device)
            if len(x_orig.shape) == 3:
                x_orig = x_orig.unsqueeze(1)
            val_len += yb.shape[0]
            with torch.no_grad():
                out = net(x_orig, x_sub)
                acc_val += (torch.max(out, 1).indices == yb).sum().item()
                TL += CE(out, yb)

        if verbose:
            print('')
            print(f'Iteration{ite}=====')
            print(f'train_loss:{loss:.4f}    val_loss:{TL / len(validloader):.4f}')
            print(f'train_acc:{acc_tr / tr_len:.4f}    val_acc:{acc_val / val_len:.4f}')
            if hasattr(net, 'manifold') and hasattr(net.manifold, 'k'):
                print(f'curvature k: {net.manifold.k.item():.4f}')

        if acc_val / val_len > best_val:
            best_val = acc_val / val_len
            update_best_test = 1
            ckpt_base = checkpoints_dir if checkpoints_dir else "./checkpoints"
            os.makedirs(ckpt_base, exist_ok=True)
            torch.save(net, os.path.join(ckpt_base, f"best_val_sub{sub}.pt"))

        train_losses.append(loss.detach().cpu())
        train_accs.append(acc_tr / tr_len)
        val_losses.append(TL / len(validloader))
        val_accs.append(acc_val / val_len)

        # ------------------------------------------------------------- test
        # BCI Challenge: AUC, everything else: accuracy.
        if dataset.startswith('bcicha'):
            test_acc, test_loss = testNetwork_auc(net, testloader, device)
        else:
            test_acc, test_loss = testNetwork(net, testloader, device)
        test_accs.append(test_acc)
        test_losses.append(test_loss)

        if update_best_test == 1:
            test_from_best_val = test_acc
        update_best_test = 0
        if update_best_train_loss:
            test_from_best_train_loss = test_acc
        best_test = max(best_test, test_acc)
        if TL / len(validloader) < bestLoss:
            bestLoss = (TL / len(validloader)).item()
            test_from_best_val_loss = test_acc

        if verbose:
            print(f'test_acc:{test_acc.__round__(3)}')
            print(f"test_from_best_val_acc:{test_from_best_val.__round__(3)}")
            print(f"test_from_best_val_loss:{test_from_best_val_loss.__round__(3)}")

        if wandb is not None and wandb.run is not None:
            log_dict = {
                'train/loss': loss.item(), 'train/acc': acc_tr / tr_len,
                'val/loss': (TL / len(validloader)).item(), 'val/acc': acc_val / val_len,
                'test/acc': test_acc, 'test/loss': test_loss,
                'best/test_from_best_val': test_from_best_val,
                'optim/lr': optimizer.param_groups[0]['lr'],
            }
            if hasattr(net, 'manifold') and hasattr(net.manifold, 'k'):
                log_dict['manifold/k'] = net.manifold.k.item()
            wandb.log(log_dict, step=ite)

        if epoch_end_callback is not None:
            epoch_end_callback(net=net, epoch=ite, trainloader=trainloader,
                               validloader=validloader, testloader=testloader)

    if run_dir:
        def _np(xs):
            return np.array([x.cpu().item() if isinstance(x, torch.Tensor) else x for x in xs])
        save_output(run_dir=run_dir,
                    results=[_np(train_losses), _np(train_accs), _np(val_losses),
                             _np(val_accs), _np(test_losses), _np(test_accs)])

    return (TL / len(validloader), acc_val / val_len, test_from_best_val_loss,
            test_from_best_train_loss, test_accs[-1] if test_accs else 0.0)


def testNetwork(net, testloader, device):
    net.eval()
    acc = test_len = 0
    total_loss = 0.0
    CE = nn.CrossEntropyLoss()
    for x_orig, x_sub, yb in testloader:
        x_orig, x_sub, yb = x_orig.to(device), x_sub.to(device), yb.to(device)
        test_len += yb.shape[0]
        with torch.no_grad():
            pred = net(x_orig, x_sub)
            acc += (torch.max(pred, 1).indices == yb).sum().item()
            total_loss += CE(pred, yb).item()
    return acc / test_len, total_loss / len(testloader)


def testNetwork_auc(net, testloader, device):
    auc, loss, _, _ = testNetwork_auc_preds(net, testloader, device)
    return auc, loss


def testNetwork_auc_preds(net, testloader, device):
    """AUC plus the per-sample positive-class scores and labels (numpy), so a
    caller can pool predictions across loaders."""
    net.eval()
    y_pred, y_true = [], []
    CE = nn.CrossEntropyLoss()
    total_loss = 0.0
    with torch.no_grad():
        for x_orig, x_sub, yb in testloader:
            x_orig, x_sub, yb = x_orig.to(device), x_sub.to(device), yb.to(device)
            pred = net(x_orig, x_sub)
            total_loss += CE(pred, yb).item()
            y_pred.append(pred[:, 1])
            y_true.append(yb)
    y_pred_np = torch.cat(y_pred).cpu().numpy()
    y_true_np = torch.cat(y_true).cpu().numpy()
    return roc_auc_score(y_true_np, y_pred_np), total_loss / len(testloader), y_pred_np, y_true_np
