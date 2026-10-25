import torch
import torch.nn as nn
import unfoldNd


from ...manifold import CustomLorentz


class LorentzAct(nn.Module):
    """ Implementation of a general Lorentz Activation on space components. 
    """
    def __init__(self, activation, manifold: CustomLorentz):
        super(LorentzAct, self).__init__()
        self.manifold = manifold
        self.activation = activation  # e.g. torch.relu

    def forward(self, x):
        if type(x) == tuple:
            return self.manifold.lorentz_activation(x[0], self.activation), x[1]
        else:
            return self.manifold.lorentz_activation(x, self.activation)


class QuickDirtyMaxPool(nn.Module):
    def __init__(self, manifold, pool_layer):
        super(QuickDirtyMaxPool, self).__init__()

        self.manifold = manifold
        self.pool_layer = pool_layer

    def forward(self, x, return_indices=False):
        x_temp = x[..., 1:].permute(0, 2, 1)
        if return_indices:
            z_maxpool, indices = self.pool_layer(x_temp)
        else:
            z_maxpool = self.pool_layer(x_temp)
        z_maxpool = self.manifold.add_time(z_maxpool.permute(0, 2, 1))
        z_maxpool = self.manifold.rescale_to_max(z_maxpool)

        if return_indices:
            return z_maxpool, indices
        return z_maxpool


class LorentzAvgPool1d(nn.Module):
    def __init__(self, manifold: CustomLorentz, kernel_size, padding=1, stride=1):
        super(LorentzAvgPool1d, self).__init__()

        self.manifold = manifold

        self.unfold = unfoldNd.UnfoldNd(
            (kernel_size), padding=padding, stride=stride
        )

    def forward(self, x):
        bs, n, c = x.shape
        x = x.permute(0, 2, 1)

        # (B, CxW, N')
        unfolded = self.unfold(x)
        unfolded = unfolded.reshape(bs, c, -1, unfolded.shape[-1]).permute(0, 3, 2, 1)
        out = self.manifold.centroid(unfolded)

        return out


def replicate_pad_time(x, padding):
    if padding <= 0:
        return x
    return torch.nn.functional.pad(x.permute(0, 2, 1), (padding, padding), mode="replicate").permute(0, 2, 1)


def unfold_time(x, kernel_size, stride, padding):
    x_pad = replicate_pad_time(x, padding)
    return x_pad.unfold(1, kernel_size, stride).permute(0, 1, 3, 2)


class LorentzCentroidMaxPool1D(nn.Module):
    def __init__(self, manifold, kernel_size, stride=None, padding=0):
        super().__init__()
        self.manifold = manifold
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        windows = unfold_time(x, self.kernel_size, self.stride, self.padding)
        centroid = self.manifold.centroid(windows)
        distances = self.manifold.dist(windows, centroid.unsqueeze(2))
        best = distances.argmax(dim=-1)
        best = best.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, x.shape[-1])
        return windows.gather(2, best).squeeze(2)


class LorentzSoftMaxPool1D(nn.Module):
    def __init__(self, manifold, kernel_size, stride=None, padding=0, tau=1.0):
        super().__init__()
        self.manifold = manifold
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.log_tau = nn.Parameter(torch.tensor(float(tau)).log())

    def forward(self, x):
        windows = unfold_time(x, self.kernel_size, self.stride, self.padding)
        radius = self.manifold.dist0(windows)
        w = torch.softmax(radius / self.log_tau.exp(), dim=-1).unsqueeze(-2)
        return self.manifold.centroid(windows, w).squeeze(-2)


class LorentzMaxPool1D(nn.Module):
    def __init__(self,
                 manifold,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1,
                 wrt_centroid=False):
        super(LorentzMaxPool1D, self).__init__()

        self.manifold = manifold
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.wrt_centroid = wrt_centroid

        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                                    return_indices=True)

    def forward(self, x):
        if self.wrt_centroid:
            stride = self.stride if self.stride is not None else self.kernel_size
            effective_kernel = self.dilation * (self.kernel_size - 1) + 1

            x_pad = replicate_pad_time(x, self.padding)

            # (B, L', C, effective_kernel) -> (B, L', effective_kernel, C)
            windows = x_pad.unfold(1, effective_kernel, stride).permute(0, 1, 3, 2)
            if self.dilation > 1:
                windows = windows[:, :, ::self.dilation, :]  # (B, L', kernel_size, C)

            centroid = self.manifold.centroid(windows)           # (B, L', C)
            distances = self.manifold.dist(windows, centroid.unsqueeze(2))  # (B, L', kernel_size)

            best_idx = distances.argmax(dim=-1)                  # (B, L')
            best_idx_exp = best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, x.shape[-1])
            return windows.gather(2, best_idx_exp).squeeze(2)    # (B, L', C)

        distances = self.manifold.dist0(x, keepdim=True).permute(0, 2, 1)

        pooled, indices = self.maxpool(distances)

        # Get flat indices of norms before pooling
        B, _, L = pooled.shape
        unpooled_shape = distances.shape  # (B, 1, L)
        _, _, L = unpooled_shape

        # Prepare to gather from x
        x_flat = x.permute(0, 2, 1)

        # Indices from maxpool are w.r.t. (L), so we use them directly
        indices_flat = indices.view(B, -1)  # (B, L')

        # Gather values from original tensor
        gathered = torch.gather(x_flat, dim=2,
                                index=indices_flat.unsqueeze(1).expand(-1, x.shape[-1], -1))  # (B, C, L')

        # Reshape if needed to (B, L', C)
        output = gathered.permute(0, 2, 1)

        return output

