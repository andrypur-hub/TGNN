import torch
import torch.nn as nn


class TGNN(nn.Module):

    def __init__(self, dim, feature_dim=165):
        super().__init__()
        self.dim = dim
        self.feature_dim = feature_dim

        self.mlp = nn.Sequential(
            nn.Linear(dim * 3 + feature_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, hu, hv, neigh_u, neigh_v, x):

        zu = torch.cat([hu, hv, neigh_u, x], dim=1)
        hu_new = self.norm(hu + self.mlp(zu))

        zv = torch.cat([hv, hu, neigh_v, x], dim=1)
        hv_new = self.norm(hv + self.mlp(zv))

        return hu_new, hv_new, None
