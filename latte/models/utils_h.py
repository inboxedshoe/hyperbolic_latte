import torch.nn as nn


class ToHyperbolic(nn.Module):
    def __init__(self, manifold, norm=False, tangent_based=False):
        super(ToHyperbolic, self).__init__()
        self.manifold = manifold

        self.norm = nn.Sequential() if not norm else self.manifold.rescale_to_max
        self.euclid_norm = nn.Sequential() if not norm else self.manifold.rescale_to_max_euclid

        self.to_hyperbolic = self.project_tangents if tangent_based else self.project_basic

    def project_tangents(self, x):
        x = nn.functional.pad(x, (1, 0, 0, 0))
        x = self.euclid_norm(x)
        return self.manifold.expmap0(x)

    def project_basic(self, x):
        x = self.manifold.add_time(x)
        return self.norm(x)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.to_hyperbolic(x)


