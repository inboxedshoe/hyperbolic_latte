import torch
import torch.nn as nn

from ...manifold import CustomLorentz


def lorentz_init_(tensor: torch.Tensor, scheme: str, fan_in: int) -> None:
    """In-place Lorentz weight initialization.

    Args:
        tensor: 2-D weight tensor to initialize in-place
        scheme: One of 'eye', 'xavier', 'lorentz_kaiming', 'kaiming', 'mlr', 'mlr_eye'
        fan_in: Number of input features (time coordinate excluded)
    """
    fan_out = tensor.numel() // fan_in
    with torch.no_grad():
        if scheme in ("eye", "mlr_eye"):
            tensor.data.copy_(0.5 * torch.eye(*tensor.shape))
        elif scheme == "xavier":
            tensor.data.normal_(0, (1.0 / (fan_in + fan_out)) ** 0.5)
        elif scheme == "lorentz_kaiming":
            tensor.data.normal_(0, (1.0 / fan_in) ** 0.5)
        elif scheme == "kaiming":
            tensor.data.normal_(0, (2.0 / fan_in) ** 0.5)
        elif scheme == "mlr":
            tensor.data.normal_(0, (5.0 / fan_in) ** 0.5)
        else:
            raise KeyError(f"Unknown Lorentz init scheme: {scheme!r}")


class LorentzFullyConnectedChen(nn.Module):
    """
        Lorentz fully connected layer.

        Code modified from https://github.com/chenweize1998/fully-hyperbolic-nn

        args:
            manifold: Instance of Lorentz manifold
            in_features, out_features, bias: Same as nn.Linear
            init_scale: Scale parameter for internal normalization
            learn_scale: If scale parameter should be learnable
            normalize: If internal normalization should be applied
    """

    def __init__(
            self,
            manifold: CustomLorentz,
            in_features,
            out_features,
            bias=False,
            init_scale=None,
            learn_scale=False,
            normalize=False,
            init_method=None,
            reset_params=None,
            a_default=None,
            activation=None,
            do_mlr=False,
            mlr_init: str | None = None,
            use_weight_norm: bool = False,
            **kwargs,
    ):

        super(LorentzFullyConnectedChen, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.normalize = normalize

        self.weight = nn.Linear(self.in_features - 1, self.out_features - 1, bias=bias)

        self.init_std = 0.02
        self.reset_parameters(reset_params)

        # Scale for internal normalization
        if init_scale is not None:
            self.scale = nn.Parameter(torch.ones(()) * init_scale, requires_grad=learn_scale)
        else:
            self.scale = nn.Parameter(torch.ones(()) * 2.3, requires_grad=learn_scale)

    def forward(self, x):

        x_space = x.narrow(-1, 1, x.shape[-1] - 1)
        x_space = self.weight(x_space)

        if self.normalize:
            scale = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp()
            square_norm = (x_space * x_space).sum(dim=-1, keepdim=True)

            mask = square_norm <= 1e-10

            square_norm = torch.where(mask, torch.ones_like(square_norm), square_norm)
            unit_length = x_space / torch.sqrt(square_norm)
            x_space = scale * unit_length

            x_time = torch.sqrt(scale ** 2 + self.manifold.k)
            x_time = x_time.masked_fill(mask, self.manifold.k.sqrt())

            x_space = x_space * ~mask

            x = torch.cat([x_time, x_space], dim=-1)
        else:
            x = self.manifold.add_time(x_space)

        return x

    def reset_parameters(self, scheme=None):
        if scheme is not None:
            # fan_in is weight.shape[1] since nn.Linear stores (out, in)
            lorentz_init_(self.weight.weight, scheme, fan_in=self.weight.weight.shape[1])
        else:
            nn.init.uniform_(self.weight.weight, -self.init_std, self.init_std)

        if self.bias:
            nn.init.constant_(self.weight.bias, 0)

